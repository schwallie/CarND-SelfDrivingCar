import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle

import config


def load_data(path='data/full_driving_log.csv'):  # altered_driving_log.csv
    final_df, steering = return_final_df(path=path)
    X_data = final_df.img_path.values
    y_data = final_df[steering].values
    y_data = np.float32(y_data)
    # Shuffle since I'm not doing validation train_test_split
    return shuffle(X_data, y_data)


def return_final_df(path='data/full_driving_log.csv'):
    drive_df = pd.read_csv(path)
    drive_df, steering = smoothe_steering_or_nah(drive_df)
    final_df = make_final_df(drive_df, steering)
    final_df = cut_out_pieces_of_final_df(final_df, steering)
    print_data_makeup(final_df, steering)
    return final_df, steering


def cut_out_pieces_of_final_df(final_df, steering):
    begin = len(final_df)
    if not config.KEEP_PERTURBED_ANGLES:
        final_df = del_perturbed_angles(final_df)
    if config.TAKE_OUT_BRIGHT_IMGS:
        final_df = del_bright_images(final_df)
    if config.TAKE_OUT_LOW_THROTTLE:
        final_df = cut_out_low_throttle(final_df)
    if config.EVEN_OUT_LR_STEERING_ANGLES:
        final_df = even_out_steering_angles(final_df, steering)
    if config.TAKE_OUT_NONCENTER_TRANSLATED_IMAGES:
        final_df = take_out_noncenter_translated(final_df)
    if config.TAKE_OUT_TRANSLATED_IMGS:
        final_df = take_out_all_translations(final_df)
    if config.TAKE_OUT_FLIPPED_0_STEERING:
        final_df = take_out_flipped_0_steering(final_df, steering)
    if config.TAKE_OUT_FLIPPED:
        final_df = take_out_flipped(final_df)
    # Take out some of the 'steering' angles of 0
    if not config.KEEP_ALL_0_STEERING_VALS:
        final_df = take_out_some_0_steer_vals(final_df)
    # Keep specific cameras only
    if config.CAMERAS_TO_USE != 3:
        final_df = use_specific_cameras(final_df, config.CAMERAS_TO_USE)
    # Deleting some bad data
    if config.DEL_IMAGES:
        final_df = take_out_specific_bad_images(final_df)
    end = len(final_df)
    print('Length of Final DF After Cutting: {0} ({1})'.format(end, end - begin))
    return final_df


def del_bright_images(final_df):
    begin = len(final_df)
    final_df = final_df[~final_df.img_path.str.contains('BRIGHT')]
    end = len(final_df)
    print('Took out Brightness Changes: {0} ({1})'.format(end, end - begin))
    return final_df


def del_perturbed_angles(final_df):
    begin = len(final_df)
    final_df = final_df[pd.isnull(final_df.PERT)]
    end = len(final_df)
    print('Took out Perturbed Angles: {0} ({1})'.format(end, end - begin))
    return final_df


def cut_out_low_throttle(final_df):
    begin = len(final_df)
    final_df = final_df[final_df.throttle > .25]
    end = len(final_df)
    print('Took out low throttle: {0} ({1})'.format(end, end - begin))
    return final_df


def make_final_df(drive_df, steering):
    drive_df['left_steering'] = drive_df[steering]
    drive_df['right_steering'] = drive_df[steering]
    drive_df.index = range(0, len(drive_df))
    drive_df = drive_df.rename(columns={steering: 'center_steering'})
    final_df = pd.concat(
        [drive_df[pd.notnull(drive_df['left'])][['left', 'left_steering', 'PERT']],
         drive_df[pd.notnull(drive_df['center'])][['center', 'center_steering', 'PERT']],
         drive_df[pd.notnull(drive_df['right'])][['right', 'right_steering', 'PERT']]])
    final_df = final_df.rename(columns={'center': 'img_path', 'center_steering': steering})
    final_df['img_path'] = final_df['img_path'].fillna(final_df['right'])
    final_df['img_path'] = final_df['img_path'].fillna(final_df['left'])
    final_df[steering] = final_df[steering].fillna(final_df['left_steering'])
    final_df[steering] = final_df[steering].fillna(final_df['right_steering'])
    for to_del in ['left', 'right', 'left_steering', 'right_steering']:
        if to_del in final_df.columns:
            del final_df[to_del]
    # Editing L/R Angles
    final_df = edit_lr_steering_angles(final_df, steering)
    final_df.to_csv('data/final_df.csv')
    return final_df


def smoothe_steering_or_nah(drive_df):
    if config.SMOOTH_STEERING:
        print('WARN**** USING SMOOTHED STEERING, BEWARE')
        drive_df['steering_smoothed'] = drive_df['steering'].rolling(center=False,
                                                                     window=config.STEER_SMOOTHING_WINDOW).mean()
        drive_df['steering_smoothed'] = drive_df['steering_smoothed'].fillna(0)
        steering = 'steering_smoothed'
    else:
        drive_df['steering_smoothed'] = drive_df['steering']
        steering = 'steering'
    return drive_df, steering


def edit_lr_steering_angles(final_df, steering):
    """
    Need to edit the L/R steering angles to make them feel like they are from the center
    :param final_df:
    :return:
    """
    final_df.loc[final_df.img_path.str.contains('left'), steering] = final_df.loc[final_df.img_path.str.contains(
        'left'), steering] + config.L_STEERING_ADJUSTMENT
    final_df.loc[final_df.img_path.str.contains('right'), steering] = final_df.loc[final_df.img_path.str.contains(
        'right'), steering] - config.R_STEERING_ADJUSTMENT
    final_df.index = range(0, len(final_df))
    return final_df


def take_out_specific_bad_images(final_df):
    """
    Udacity data has some bad steering angles on some images
    :param final_df:
    :return:
    """
    begin = len(final_df)
    for del_img in config.DEL_IMAGES:
        final_df = final_df[~(final_df.img_path.str.contains(del_img))]
    end = len(final_df)
    print('Deleted specifically annotated BAD IMAGES: {0} ({1})'.format(end, end - begin))
    return final_df


def take_out_all_translations(final_df):
    """
    Takes out all translated images
    :param final_df:
    :return:
    """
    begin = len(final_df)
    final_df = final_df[~final_df.img_path.str.contains('TRANS')]
    end = len(final_df)
    print('Took out translations: {0} ({1})'.format(end, end - begin))
    return final_df


def take_out_flipped_0_steering(final_df, steering):
    begin = len(final_df)
    final_df = final_df[~((final_df.img_path.str.contains('FLIPPED')) & (final_df[steering] == 0))]
    end = len(final_df)
    print('Took out FLIPPED 0 steering: len: {0} ({1})'.format(end, end - begin))
    return final_df


def take_out_flipped(final_df):
    begin = len(final_df)
    final_df = final_df[~(final_df.img_path.str.contains('FLIPPED'))]
    end = len(final_df)
    print('Took out FLIPPED: len: {0} ({1})'.format(end, end - begin))
    return final_df


def use_specific_cameras(final_df, cameras_to_use):
    """
    # TODO: Allow only L/R
    L/R/C are included. Can take the L/R out if I want
    :param final_df:
    :param cameras_to_use:
    :return:
    """
    begin = len(final_df)
    if cameras_to_use == 1:
        final_df = final_df[final_df.img_path.str.contains('center')]
        end = len(final_df)
        print('Only allowed CENTER values: {0} ({1})'.format(end, end - begin))
    return final_df


def take_out_some_0_steer_vals(final_df, angle=.04):
    """
    Takes out some of the zero steering vals, so the car doesn't tend towards going straight
    :param final_df:
    :return:
    """
    final_df.index = range(0, len(final_df))
    piece = final_df[(abs(final_df.steering) <= angle)]
    begin = len(piece)
    steer_0s = piece.index
    # Cut out a certain portion of 0 steers and keep only the leftovers
    take_out = len(steer_0s) - int(len(steer_0s) / config.KEEP_1_OVER_X_0_STEERING_VALS)
    deleted = np.random.choice(steer_0s, size=take_out, replace=False)
    final_df.loc[:, 'ix'] = final_df.index
    final_df = final_df[~(final_df['ix'].isin(deleted))]
    del final_df['ix']
    end = len(final_df[(abs(final_df.steering) <= angle)])
    print('Took out {0} of 0 steering values...Current len of 0s: {1} ({2})'.format(take_out, end, end - begin))
    return final_df


def take_out_noncenter_translated(final_df):
    """
    Realized there was a bug in my translation code, shouldn't need to use this anymore
    :param final_df:
    :return:
    """
    begin = len(final_df)
    final_df = final_df[
        (~final_df.img_path.str.contains('TRANS_left')) & (~final_df.img_path.str.contains('TRANS_right'))]
    end = len(final_df)
    print('Took out non-center translations: len: {0} ({1})'.format(end, end - begin))
    return final_df


def even_out_steering_angles(final_df, steering, bins=config.EVEN_BINS):
    """
    The idea behind this is we don't want an uneven distribution,
    aka 25k Left turns and 10k right turns
    :param final_df:
    :param steering:
    :param bins:
    :return:
    """
    pos_begin = len(final_df[(final_df[steering] > 0)])
    neg_begin = len(final_df[(final_df[steering] < 0)])
    for rng in bins:
        final_df.index = range(0, len(final_df))
        pos = final_df[(final_df[steering] > rng[0]) & (final_df[steering] <= rng[1])]
        neg = final_df[(final_df[steering] < -rng[0]) & (final_df[steering] >= -rng[1])]
        # Taking out small angles only that are L or R
        if len(pos) > len(neg):
            diff = len(pos) - len(neg)
            options = pos.index
        else:
            diff = len(neg) - len(pos)
            options = neg.index
        deleted = np.random.choice(options, size=diff, replace=False)
        final_df.loc[:, 'ix'] = final_df.index
        final_df = final_df[~(final_df['ix'].isin(deleted))]
        del final_df['ix']
        pos_end = len(final_df[(final_df[steering] > rng[0]) & (final_df[steering] <= rng[1])])
        neg_end = len(final_df[(final_df[steering] < -rng[0]) & (final_df[steering] >= -rng[1])])
        print('END: {0}, Positive Steering: {1} ({2}), Negative Steering: {3} ({4})'.format(rng,
                                                                                            len(pos),
                                                                                            pos_end - len(pos),
                                                                                            len(neg),
                                                                                            neg_end - len(neg)))
    pos = len(final_df[(final_df[steering] > 0)])
    neg = len(final_df[(final_df[steering] < 0)])
    print('FINAL: Positive Steering: {0} ({1}), Negative Steering: {2} ({3})'.format(pos, pos - pos_begin,
                                                                                     neg, neg - neg_begin))
    return final_df


def print_data_makeup(final_df, steering):
    pos = final_df[(final_df[steering] > 0)]
    neg = final_df[(final_df[steering] < 0)]
    zero = final_df[final_df[steering] == 0]
    print('FINAL: Positive Steering: {0}, Negative Steering: {1}, 0 Angle Steering: {2}'.format(len(pos),
                                                                                                len(neg), len(zero)))
    print('Low Angle Steering: <.05, {0}'.format(len(final_df[abs(final_df[steering]) < .05])))
    print('High Angle Steering: >.25, {0}'.format(len(final_df[abs(final_df[steering]) > .25])))
    print('Translated Images: {0}'.format(len(final_df[final_df.img_path.str.contains('TRANS')])))
    print('Brightened Images: {0}'.format(len(final_df[final_df.img_path.str.contains('BRIGHT')])))
    print('Flipped Images: {0}'.format(len(final_df[final_df.img_path.str.contains('FLIP')])))
    print('Left Images: {0}, Right Images: {1}, Center Images: {2}'.format(
        len(final_df[final_df.img_path.str.contains('left')]), len(final_df[final_df.img_path.str.contains('right')]),
        len(final_df[final_df.img_path.str.contains('center')])))


def return_validation(path='data/driving_log.csv'):
    X_data, y_data = load_data(path=path)
    return train_test_split(X_data, y_data, test_size=.05, random_state=43)


def load_drive_df(path='data/full_driving_log.csv'):
    steering = 'steering'
    drive_df = pd.read_csv(path, index_col=0)
    # Changing specific images in Udacity dataset that have bad steering angles:
    """print('Altering bad steering angles')
    drive_df.loc[drive_df['center'].str.contains('center_2016_12_01_13_38_02'), steering] = -.05
    drive_df.loc[drive_df['center'].str.contains('center_2016_12_01_13_40_43'), steering] = -.05
    drive_df.loc[drive_df['center'].str.contains('center_2016_12_01_13_40_44'), steering] = -.05
    drive_df.loc[
        (drive_df['center'].str.contains('center_2016_12_01_13_41_12')) & (drive_df.steering == 0), steering] = -.05
    drive_df.loc[drive_df['center'].str.contains('center_2016_12_01_13_41_17'), steering] = 0
    drive_df.loc[drive_df['center'].str.contains('center_2016_12_01_13_41_20'), steering] = -.05
    drive_df.loc[
        (drive_df['center'].str.contains('center_2016_12_01_13_41_21')) & (drive_df.steering == 0), steering] = -.05
    drive_df.loc[
        (drive_df['center'].str.contains('center_2016_12_01_13_41_22')) & (drive_df.steering == 0), steering] = -.05"""
    return drive_df
