import os

import cv2
import numpy as np
import pandas as pd
from keras.optimizers import Adam

pd.set_option('display.height', 1000)
pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)

IMAGE_HEIGHT_CROP = 108
IMAGE_WIDTH_CROP = 320
CHANNELS = 3
STEERING_ADJUSTMENT = 1
AUTONOMOUS_THROTTLE = .2
# (200, 66) <-- Original NVIDIA Paper
# (160, 320) <-- Original Comma
IMAGE_WIDTH = 64
IMAGE_HEIGHT = 32
LR = 1e-5
OPTIMIZER = Adam(lr=LR)
LOSS = 'mse'
NB_EPOCH = 10
BATCH_SIZE = 256

####
#
# This section is refered to in load_data.py
# Keep i-0908e9b5131608d26
#
####

# Mean smoothing for the steering column
# TODO: Found bug on steer smoothing, where it has to be ordered by time
SMOOTH_STEERING = False
STEER_SMOOTHING_WINDOW = 3

TAKE_OUT_FLIPPED_0_STEERING = True
TAKE_OUT_TRANSLATED_IMGS = True
TAKE_OUT_NONCENTER_TRANSLATED_IMAGES = True
# Too many vals at 0 steering, need to take some out to prevent driving straight
KEEP_ALL_0_STEERING_VALS = False
KEEP_1_OVER_X_0_STEERING_VALS = 2  # Lower == More kept images at 0 steering
CAMERAS_TO_USE = 3  # 1 for Center, 3 for L/R/C
# Steering adjustmenet for L/R images
L_STEERING_ADJUSTMENT = .20
R_STEERING_ADJUSTMENT = .20

# Even out skew on L/R steering angles
EVEN_OUT_LR_STEERING_ANGLES = False

DEL_IMAGES = ['center_2016_12_01_13_38_02']


def full_train(path_orig='data/driving_log.csv',
               path_altered='data/altered_driving_log.csv', path_altered_plus='data/altered_plus_driving_log.csv',
               path_bright='data/brightness_driving_log.csv', path_angle_augment='data/altered_angle.csv',
               path_full='data/full_driving_log.csv', override=False, path_save=None):
    if not os.path.isfile(path_angle_augment) or override:
        print('Creating Slightly Altered Angles')
        drive_df = pd.read_csv(path_orig, index_col=0)
        path_save = path_angle_augment
        add_augment_steering_angles(drive_df, path_save)
    if not os.path.isfile(path_altered) or override:
        print("Creating Altered Files")
        drive_df = pd.read_csv(path_save, index_col=0)
        path_save = path_altered
        add_flipped_images(drive_df, path_save)
    if not os.path.isfile(path_altered_plus) or override:
        print('Creating Translated Files')
        drive_df = pd.read_csv(path_save, index_col=0)
        path_save = path_altered_plus
        add_translated_images(drive_df, path_save)
    if not os.path.isfile(path_bright) or override:
        print('Creating Brightness')
        drive_df = pd.read_csv(path_save, index_col=0)
        path_save = path_bright
        add_brightness_augmented_images(drive_df, path_save)
    if not path_save:
        path_save = path_bright
    drive_df = pd.read_csv(path_save)
    drive_df.to_csv(path_full)
    ### Need to add
    import model
    model.train(path=path_full, checkpoint_path="models/aug_angles_no_transl_upsample_lg_ang_256_batch-{epoch:02d}.h5")


def return_image(img, color_change=True):
    # Take out the dash and horizon
    img_shape = img.shape
    crop_img = img[int(img_shape[0] / 5):img_shape[0] - 20, 0:img_shape[1]]
    assert crop_img.shape[0] == IMAGE_HEIGHT_CROP
    assert crop_img.shape[1] == IMAGE_WIDTH_CROP
    if color_change:
        img = cv2.cvtColor(crop_img, cv2.COLOR_BGR2RGB)
    img = (cv2.resize(img, (IMAGE_WIDTH, IMAGE_HEIGHT), interpolation=cv2.INTER_AREA))
    return np.float32(img)


def add_augment_steering_angles(drive_df, path):
    original = drive_df[(pd.notnull(drive_df['left'])) & (~drive_df['center'].str.contains('BRIGHT'))]
    original = original[abs(original['steering']) > .1]
    original['steering2'] = original.apply(lambda x: x['steering'] + np.random.uniform(-1, 1) / 40, axis=1)
    del original['steering']
    original = original.rename(columns={'steering2': 'steering'})
    original.index = range(len(drive_df) + 1, len(drive_df) + 1 + len(original))
    drive_df = drive_df.append(original)
    drive_df.to_csv(path)


def create_altered_drive_df(path):
    import load_data
    drive_df = load_data.load_drive_df('data/driving_log.csv')
    drive_df.to_csv(path)


def add_flipped_images(drive_df, path):
    maxidx = max(drive_df.index)
    addition = {}
    for idx, row in drive_df.iterrows():
        if row['steering'] == 0:
            continue
        # rnd = np.random.randint(2)
        # if rnd == 1:
        maxidx += 1
        # Flip and created a new image
        img_path = 'data/{0}'.format(row['center'])
        new_path = 'IMG/FLIPPED_{0}'.format(row['center'].split('/')[-1])
        img = cv2.imread(img_path)
        img = np.array(img)
        img = np.fliplr(img)
        cv2.imwrite('data/{0}'.format(new_path), img)
        steer = -row['steering']
        addition[maxidx] = {'center': new_path, 'steering': steer}
    new_df = pd.DataFrame.from_dict(addition, orient='index')
    drive_df = drive_df.append(new_df)
    drive_df.to_csv(path)


def trans_image(image, steer, trans_range):
    # Translation
    rows, cols, channels = image.shape
    tr_x = trans_range * np.random.uniform() - trans_range / 2
    steer_ang = steer + tr_x / trans_range * 2 * .2
    tr_y = 40 * np.random.uniform() - 40 / 2
    # tr_y = 0
    Trans_M = np.float32([[1, 0, tr_x], [0, 1, tr_y]])
    image_tr = cv2.warpAffine(image, Trans_M, (cols, rows))
    return image_tr, steer_ang


def add_translated_images(drive_df, path, translated_image_per_image=5):
    maxidx = max(drive_df.index)
    addition = {}
    choices = ['center']  # 'left', 'right'
    # I only want to translate the original images with all 3 images avail, not flipped images
    for idx, row in drive_df[pd.notnull(drive_df['left'])].iterrows():
        addition[maxidx] = {}
        for choice in choices:
            img_path = 'data/{0}'.format(row[choice].strip())
            new_path = 'IMG/TRANS_{0}'.format(row[choice].split('/')[-1])
            img = cv2.imread(img_path)
            # ERROR: If this is L/R I need to use row['steering'] +- ANGLE!!
            # TODO: Be able to translate non-center
            img, steer = trans_image(img, row['steering'], 150)
            cv2.imwrite('data/{0}'.format(new_path), img)
            addition[maxidx][choice] = new_path
            if choice == 'center':
                addition[maxidx]['steering'] = steer
        maxidx += 1
    new_df = pd.DataFrame.from_dict(addition, orient='index')
    drive_df = drive_df.append(new_df)
    drive_df.to_csv(path)


def add_random_shadow(image):
    top_y = 320 * np.random.uniform()
    top_x = 0
    bot_x = 160
    bot_y = 320 * np.random.uniform()
    image_hls = cv2.cvtColor(image, cv2.COLOR_RGB2HLS)
    shadow_mask = 0 * image_hls[:, :, 1]
    X_m = np.mgrid[0:image.shape[0], 0:image.shape[1]][0]
    Y_m = np.mgrid[0:image.shape[0], 0:image.shape[1]][1]
    shadow_mask[((X_m - top_x) * (bot_y - top_y) - (bot_x - top_x) * (Y_m - top_y) >= 0)] = 1
    # random_bright = .25+.7*np.random.uniform()
    if np.random.randint(2) == 1:
        random_bright = .5
        cond1 = shadow_mask == 1
        cond0 = shadow_mask == 0
        if np.random.randint(2) == 1:
            image_hls[:, :, 1][cond1] = image_hls[:, :, 1][cond1] * random_bright
        else:
            image_hls[:, :, 1][cond0] = image_hls[:, :, 1][cond0] * random_bright
    image = cv2.cvtColor(image_hls, cv2.COLOR_HLS2RGB)
    return image


def add_shadowed_images(drive_df, path):
    maxidx = max(drive_df.index)
    addition = {}
    choices = ['center', 'left', 'right']
    # I only want to translate the original images with all 3 images avail, not flipped images
    for idx, row in drive_df[pd.notnull(drive_df['left'])].iterrows():
        addition[maxidx] = {'steering': row['steering']}
        for choice in choices:
            img_path = 'data/{0}'.format(row[choice].strip())
            new_path = 'IMG/SHADOW_{0}'.format(row[choice].split('/')[-1])
            img = cv2.imread(img_path)
            img = add_random_shadow(img)
            cv2.imwrite('data/{0}'.format(new_path), img)
            addition[maxidx][choice] = new_path
        maxidx += 1
    new_df = pd.DataFrame.from_dict(addition, orient='index')
    drive_df = drive_df.append(new_df)
    drive_df.to_csv(path)


def augment_brightness_camera_images(image):
    image1 = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    random_bright = .25 + np.random.uniform()
    image1[:, :, 2] = image1[:, :, 2] * random_bright
    image1 = cv2.cvtColor(image1, cv2.COLOR_HSV2RGB)
    return image1


def add_brightness_augmented_images(drive_df, path):
    maxidx = max(drive_df.index)
    addition = {}
    choices = ['center', 'left', 'right']
    # I only want to translate the original images with all 3 images avail, not flipped images
    for idx, row in drive_df[pd.notnull(drive_df['left'])].iterrows():
        addition[maxidx] = {'steering': row['steering']}
        for choice in choices:
            img_path = 'data/{0}'.format(row[choice].strip())
            new_path = 'IMG/BRIGHT_{0}'.format(row[choice].split('/')[-1])
            img = cv2.imread(img_path)
            img = augment_brightness_camera_images(img)
            cv2.imwrite('data/{0}'.format(new_path), img)
            addition[maxidx][choice] = new_path
        maxidx += 1
    new_df = pd.DataFrame.from_dict(addition, orient='index')
    drive_df = drive_df.append(new_df)
    drive_df.to_csv(path)


def vis(df=None, rn=None, img_view='center', img=None):
    """
    Bad learning images:
    IMG/center_2016_12_01_13_38_02_790.jpg
    3200 - 3205 on full_driving, or IMG/center_2016_12_01_13_38_02_*
    :param df:
    :param rn:
    :param img_view:
    :param img:
    :return:
    """
    import matplotlib.pyplot as plt
    if df is None:
        # df = df[df.throttle > .25]
        df = pd.read_csv('data/data/full_driving_log.csv')
        df['steering_smoothed'] = pd.rolling_mean(df['steering'], 3)
        df['steering_smoothed'] = df['steering_smoothed'].fillna(0)
        df['left_steering'] = df['steering_smoothed'] + L_STEERING_ADJUSTMENT
        df['right_steering'] = df['steering_smoothed'] - R_STEERING_ADJUSTMENT
        df = df.rename(columns={'steering_smoothed': 'center_steering'})
    if rn is None and img is None:
        # RN is just a random number to choose an img to show
        rn = np.random.randint(len(df))
    elif img:
        rn = df[df[img_view] == img].index[0]
    path = 'data/data/{0}'.format(df.iloc[rn][img_view].strip())
    print('{0}: {1}'.format(rn, path))
    print(df.iloc[rn]['{0}_steering'.format(img_view)])
    img = np.array(cv2.imread(path))
    img_shape = img.shape
    img = img[int(img_shape[0] / 5):img_shape[0] - 20, 0:img_shape[1]]
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    plt.imshow(img)
    plt.show()
    return df


"""
Office Hours:
He used the comma.ai model and these transformations

from img_transformations import translate, brighten_or_darken, cropout_sky_hood

img = Image.open(center_img_filepath)
img = brighten_or_darken(img, brightness_factor_min=0.25)
img, steering_angle = translate(img, steering_angle)
img = cropout_sky_hood(img)
print('translated steering angle', steering_angle)
img = img.resize((64, 32))
plt.imshow(img)


as for the bridge part, you can subsample data to remove data with small angles, so the bias for going straight is reduced


"""
