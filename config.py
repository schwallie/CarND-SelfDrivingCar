import os

import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from keras.optimizers import Adam
pd.set_option('display.height', 1000)
pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)


IMAGE_HEIGHT_CROP = 108
IMAGE_WIDTH_CROP = 320
CHANNELS = 3
STEERING_ADJUSTMENT = 1
AUTONOMOUS_THROTTLE = .1
# (200, 66) <-- Original NVIDIA Paper
IMAGE_WIDTH = 64
IMAGE_HEIGHT = 32
LR = 1e-5
OPTIMIZER = Adam(lr=LR)
LOSS = 'mse'
NB_EPOCH = 15
BATCH_SIZE = 256

####
#
# This section is refered to in load_data.py
#
####

# Mean smoothing for the steering column
SMOOTH_STEERING = True
STEER_SMOOTHING_WINDOW = 3

TAKE_OUT_FLIPPED_0_STEERING = True
TAKE_OUT_TRANSLATED_IMGS = True
# Too many vals at 0 steering, need to take some out to prevent driving straight
KEEP_ALL_0_STEERING_VALS = False
KEEP_1_OVER_X_0_STEERING_VALS = 5
CAMERAS_TO_USE = 1
# Steering adjustmenet for L/R images
LR_STEERING_ADJUSTMENT = .08

DEL_IMAGES = ['center_2016_12_01_13_38_02']

def return_image(img, color_change=True):
    # Take out the dash and horizon
    img_shape = img.shape
    crop_img = img[int(img_shape[0] / 5):img_shape[0] - 20, 0:img_shape[1]]
    assert crop_img.shape[0] == IMAGE_HEIGHT_CROP
    assert crop_img.shape[1] == IMAGE_WIDTH_CROP
    if color_change:
        img = cv2.cvtColor(crop_img, cv2.COLOR_BGR2RGB)
    # img = augment_brightness_camera_images(img)
    img = (cv2.resize(img, (IMAGE_WIDTH, IMAGE_HEIGHT), interpolation=cv2.INTER_AREA))
    return np.float32(img)


def create_altered_drive_df(path):
    drive_df = pd.read_csv('data/driving_log.csv')
    drive_df.to_csv(path)


def add_flipped_images(path):
    drive_df = pd.read_csv(path)
    maxidx = max(drive_df.index)
    addition = {}
    for idx, row in drive_df.iterrows():
        rnd = np.random.randint(2)
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
    choices = ['center', 'left', 'right']
    # I only want to translate the original images with all 3 images avail, not flipped images
    for idx, row in drive_df[pd.notnull(drive_df['left'])].iterrows():
        addition[maxidx] = {}
        for choice in choices:
            img_path = 'data/{0}'.format(row[choice].strip())
            new_path = 'IMG/TRANS_{0}'.format(row[choice].split('/')[-1])
            img = cv2.imread(img_path)
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


def create_and_train_with_altered_images(path='data/altered_driving_log.csv'):
    import os
    if not os.path.isfile(path):
        print("Creating Altered Files")
        create_altered_drive_df(path)
        add_flipped_images(path)
    import model
    model.train(path=path, checkpoint_path="models/altered_comma_model_no_validate-{epoch:02d}.h5")


def train_altered_and_translated_train(path_altered='data/altered_driving_log.csv',
                                       path_altered_plus='data/altered_plus_driving_log.csv'):
    if not os.path.isfile(path_altered):
        print("Creating Altered Files")
        create_altered_drive_df(path_altered)
        add_flipped_images(path_altered)
    if not os.path.isfile(path_altered_plus):
        print('Creating Translated Files')
        drive_df = pd.read_csv(path_altered)
        add_translated_images(drive_df, path_altered_plus)
    import model
    model.train(path=path_altered_plus, checkpoint_path="models/full_comma_model_no_validate-{epoch:02d}.h5")


def full_train(path_altered='data/altered_driving_log.csv', path_altered_plus='data/altered_plus_driving_log.csv',
               path_full='data/full_driving_log.csv'):
    if not os.path.isfile(path_altered):
        print("Creating Altered Files")
        create_altered_drive_df(path_altered)
        add_flipped_images(path_altered)
    if not os.path.isfile(path_altered_plus):
        print('Creating Translated Files')
        drive_df = pd.read_csv(path_altered)
        add_translated_images(drive_df, path_altered_plus)
    if not os.path.isfile(path_full):
        print('Creating Brightness')
        drive_df = pd.read_csv(path_altered_plus)
        add_brightness_augmented_images(drive_df, path_full)
    import model
    model.train(path=path_full, checkpoint_path="models/full_new_load_only_center-{epoch:02d}.h5")


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
    if df is None:
        # df = df[df.throttle > .25]
        df = pd.read_csv('data/data/full_driving_log.csv')
        df['steering_smoothed'] = pd.rolling_mean(df['steering'], 3)
        df['steering_smoothed'] = df['steering_smoothed'].fillna(0)
        df['left_steering'] = df['steering_smoothed'] + .05
        df['right_steering'] = df['steering_smoothed'] - .05
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
