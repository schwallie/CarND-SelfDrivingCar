import os

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle

import config

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


def load_data(path='data/full_driving_log.csv', final_df='data/final_df.csv'):  # altered_driving_log.csv
    #if final_df is not None and os.path.exists(final_df):
    #    final_df = pd.read_csv(final_df)
    #else:
    drive_df = pd.read_csv(path)
    # drive_df = drive_df[drive_df.throttle > .25]
    if config.SMOOTH_STEERING:
        print('WARN**** USING SMOOTHED STEERING, BEWARE')
        drive_df['steering_smoothed'] = drive_df['steering'].rolling(center=False,
                                                                     window=config.STEER_SMOOTHING_WINDOW).mean()
        drive_df['steering_smoothed'] = drive_df['steering_smoothed'].fillna(0)
    else:
        drive_df['steering_smoothed'] = drive_df['steering']
    drive_df['left_steering'] = drive_df['steering_smoothed']  # + config.L_STEERING_ADJUSTMENT
    drive_df['right_steering'] = drive_df['steering_smoothed']  # - config.R_STEERING_ADJUSTMENT
    drive_df.index = range(0, len(drive_df))
    drive_df = drive_df.rename(columns={'steering_smoothed': 'center_steering'})
    final_df = pd.concat(
        [drive_df[pd.notnull(drive_df['left'])][['left', 'left_steering', 'steering']],
         drive_df[pd.notnull(drive_df['center'])][['center', 'center_steering', 'steering']],
         drive_df[pd.notnull(drive_df['right'])][['right', 'right_steering', 'steering']]])
    final_df = final_df.rename(columns={'center': 'img_path', 'center_steering': 'steering_smoothed'})
    final_df['img_path'] = final_df['img_path'].fillna(final_df['right'])
    final_df['img_path'] = final_df['img_path'].fillna(final_df['left'])
    final_df['steering_smoothed'] = final_df['steering_smoothed'].fillna(final_df['left_steering'])
    final_df['steering_smoothed'] = final_df['steering_smoothed'].fillna(final_df['right_steering'])
    if config.SMOOTH_STEERING:
        steering = 'steering_smoothed'
    else:
        steering = 'steering'
    for to_del in ['left', 'right', 'left_steering', 'right_steering']:
        if to_del in final_df.columns:
            del final_df[to_del]
    # Editing L/R Angles
    final_df.loc[final_df.img_path.str.contains('left'), steering] = final_df.loc[final_df.img_path.str.contains(
        'left'), steering] + config.L_STEERING_ADJUSTMENT
    final_df.loc[final_df.img_path.str.contains('right'), steering] = final_df.loc[final_df.img_path.str.contains(
        'right'), steering] - config.R_STEERING_ADJUSTMENT
    final_df.index = range(0, len(final_df))
    final_df.to_csv('data/final_df.csv')
    print('Length of Final DF Before Cutting: {0}'.format(len(final_df)))
    if config.EVEN_OUT_LR_STEERING_ANGLES:
        for rng in [[0, .1], [.1, .2], [.2, .5]]:
            pos = final_df[(final_df[steering] > rng[0]) & (final_df[steering] <= rng[1])]
            neg = final_df[(final_df[steering] < -rng[0]) & (final_df[steering] >= -rng[1])]
            print('{0}, Positive Steering: {1}, Negative Steering: {2}'.format(rng, len(pos),
                                                                          len(neg)))
            # Taking out small angles only that are L or R
            if len(pos) > len(neg):
                diff = len(pos) - len(neg)
                options = pos.index
            else:
                diff = len(neg) - len(pos)
                options = neg.index
            deleted = np.random.choice(options, size=diff)
            final_df['ix'] = final_df.index
            final_df = final_df[~(final_df['ix'].isin(deleted))]
            del final_df['ix']
            pos = final_df[(final_df[steering] > rng[0]) & (final_df[steering] <= rng[1])]
            neg = final_df[(final_df[steering] < -rng[0]) & (final_df[steering] >= -rng[1])]
            print('END: {0}, Positive Steering: {1}, Negative Steering: {2}'.format(rng, len(pos),
                                                                          len(neg)))
        pos = final_df[(final_df[steering] > 0)]
        neg = final_df[(final_df[steering] < 0)]
        print('FINAL: Positive Steering: {0}, Negative Steering: {1}'.format(len(pos),
                                                                             len(neg)))
    if config.TAKE_OUT_NONCENTER_TRANSLATED_IMAGES:
        final_df = final_df[
            (~final_df.img_path.str.contains('TRANS_left')) & (~final_df.img_path.str.contains('TRANS_right'))]
        print('Took out non-center translations: len: {0}'.format(len(final_df)))
    if config.TAKE_OUT_TRANSLATED_IMGS:
        final_df = final_df[~final_df.img_path.str.contains('TRANS')]
        print('Took out translations: len: {0}'.format(len(final_df)))
    if config.TAKE_OUT_FLIPPED_0_STEERING:
        final_df = final_df[~((final_df.img_path.str.contains('FLIPPED')) & (final_df['steering'] == 0))]
        print('Took out FLIPPED 0 steering: len: {0}'.format(len(final_df)))
    # Take out some of the 'steering' angles of 0
    if not config.KEEP_ALL_0_STEERING_VALS:
        print(
            'Taking out a lot of 0 steering values...Current len of 0s: {0}'.format(
                len(final_df[final_df.steering == 0])))
        steer_0s = final_df[final_df.steering == 0].index
        # Cut out a certain portion of 0 steers and keep only the leftovers
        to_keep = int(len(steer_0s) / config.KEEP_1_OVER_X_0_STEERING_VALS)
        kept = np.random.choice(steer_0s, size=to_keep)
        final_df['ix'] = final_df.index
        final_df = final_df[(final_df['ix'].isin(kept)) | (final_df['steering'] != 0)]
        del final_df['ix']
        print('Taking out a lot of 0 steering values...New len of 0s: {0}, Total: {1}'.format(
            len(final_df[final_df.steering == 0]), len(final_df)))
    # Keep specific cameras only
    if config.CAMERAS_TO_USE == 1:
        final_df = final_df[final_df.img_path.str.contains('center')]
        print('Only allowed CENTER values: {0}'.format(len(final_df)))
        # TODO: Allow only L/R
    # Deleting some bad data
    for del_img in config.DEL_IMAGES:
        final_df = final_df[~(final_df.img_path.str.contains(del_img))]
    print('Deleted specifically annotated BAD IMAGES: {0}'.format(len(final_df)))
    ####
    #
    # Done adjusting images
    #
    ####
    print_data_makeup(final_df, steering)
    X_data = final_df.img_path.values
    y_data = final_df[steering].values
    y_data = np.float32(y_data)
    # Shuffle since I'm not doing validation
    return shuffle(X_data, y_data)


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
