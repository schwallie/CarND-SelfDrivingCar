import os

import cv2
import numpy as np
import pandas as pd
from keras.optimizers import Adam
from keras.preprocessing.image import img_to_array, load_img

import model

pd.set_option('display.height', 1000)
pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)
# Bridges = IMG/FLIPPED_center_2016_12_01_13_35_39_514.jpg, IMG/left_2016_12_01_13_35_36_686.jpg
# Original Image: (320, 160, 3)
IMAGE_HEIGHT_CROP = 108
IMAGE_WIDTH_CROP = 320
STEERING_ADJUSTMENT = 1
AUTONOMOUS_THROTTLE = .2
# (200, 66) <-- Original NVIDIA Paper
# (320, 160) <-- Original Comma
IMAGE_WIDTH = 160
IMAGE_HEIGHT = 80
CHANNELS = 3
LR = 1e-5
OPTIMIZER = Adam(lr=LR)
LOSS = 'mse'
NB_EPOCH = 60
BATCH_SIZE = 256

####
#
# This section is referred to in load_data.py
#
####
CHECKPOINT_PATH = "models/comma_256_batch_some_trans-{epoch:02d}.h5"
TAKE_OUT_TRANSLATED_IMGS = True
TAKE_OUT_BRIGHT_IMGS = True
TAKE_OUT_FLIPPED = True
EVEN_OUT_LR_STEERING_ANGLES = False
KEEP_ALL_0_STEERING_VALS = True
KEEP_1_OVER_X_0_STEERING_VALS = 3  # Lower == More kept images at 0 steering
KEEP_PERTURBED_ANGLES = True
#### DEPRECATED, Were used to fix bugs!
TAKE_OUT_NONCENTER_TRANSLATED_IMAGES = False
TAKE_OUT_FLIPPED_0_STEERING = False

# Mean smoothing for the steering column
# TODO: Found bug on steer smoothing, where it has to be ordered by time (duh)
SMOOTH_STEERING = False
STEER_SMOOTHING_WINDOW = 3
# Take only images with throttle being used
# If wanting to activate, set to some threshold(i.e., .25), if not, use False
TAKE_OUT_LOW_THROTTLE = False
# Too many vals at 0 steering, need to take some out to prevent driving straight
CAMERAS_TO_USE = 3  # 1 for Center, 3 for L/R/C
# Steering adjustmenet for L/R images
L_STEERING_ADJUSTMENT = .25
R_STEERING_ADJUSTMENT = .25
# Even out skew on L/R steering angles
DEL_IMAGES = ['2016_12_01_13_38_02']
# Keep Perturbed Angles
PERTURBED_ANGLE = np.random.uniform(-1, 1) / 40
PERTURBED_ANGLE_MIN = .05


def full_train(path_full='data/full_driving_log.csv', prev_model=False):
    new_model = model.comma_model()
    if prev_model:
        new_model = model.load_saved_model(prev_model, new_model)
    model.train(model=new_model, path=path_full, checkpoint_path=CHECKPOINT_PATH)


def get_augmented(x, y):
    steering = y
    image = load_img("data/{0}".format(x))
    image = img_to_array(image)
    flip = np.random.choice([0, 1])
    if flip == 1:
        steering *= -1
        image = cv2.flip(image, 1)
    image = augment_brightness_camera_images(image)
    trans = np.random.random()
    if trans > .8:
        # Translate 20% of the images
        image, steering = trans_image(image, steering, 100)
    image = return_image(image)
    return image, steering


def build_augmented_files(path_orig='data/driving_log.csv',
                          path_flips='data/flipped_driving_log.csv',
                          path_translations='data/translations_driving_log.csv',
                          path_bright='data/brightness_driving_log.csv',
                          path_angle_augment='data/altered_angle_driving_log.csv',
                          path_full='data/full_driving_log.csv', override=False):
    drive_df = pd.read_csv(path_orig, index_col=0)
    print('Original Len: {0}'.format(len(drive_df)))
    if override or not os.path.isfile(path_angle_augment):
        print('Creating Slightly Altered Angles from Original')
        augmented = add_augment_steering_angles(drive_df)
        augmented.to_csv(path_angle_augment)
    else:
        augmented = pd.read_csv(path_angle_augment)
    print('Augmented: {0}'.format(len(augmented)))
    if override or not os.path.isfile(path_flips):
        print("Creating Flipped Files")
        flipped = add_flipped_images(drive_df)
        flipped.to_csv(path_flips)
    else:
        flipped = pd.read_csv(path_flips)
    print('Flipped: {0}'.format(len(flipped)))
    if override or not os.path.isfile(path_translations):
        print('Creating Translated Files')
        trans = add_translated_images(drive_df)
        trans.to_csv(path_translations)
    else:
        trans = pd.read_csv(path_translations)
    print('Translations: {0}'.format(len(trans)))
    if override or not os.path.isfile(path_bright):
        print('Creating Brightness')
        bright = add_brightness_augmented_images(drive_df)
        bright.to_csv(path_bright)
    else:
        bright = pd.read_csv(path_bright)
    print('Brightness: {0}'.format(len(bright)))
    # Concatenate everything
    for add_in in [augmented, flipped, trans, bright]:
        drive_df = drive_df.append(add_in)
    print('Final LEN: {0}'.format(len(drive_df)))
    drive_df.to_csv(path_full)
    return drive_df


def return_image(img, color_change=True):
    # Take out the dash and horizon
    img_shape = img.shape
    img = img[60:img_shape[0] - 25, 0:img_shape[1]]
    # assert crop_img.shape[0] == IMAGE_HEIGHT_CROP
    # assert crop_img.shape[1] == IMAGE_WIDTH_CROP
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = (cv2.resize(img, (IMAGE_WIDTH, IMAGE_HEIGHT), interpolation=cv2.INTER_AREA))
    return np.float32(img)


def add_augment_steering_angles(drive_df):
    """
    The idea is to teach the Model that we can have slighlty adjusted angles
    Also adds more to the dataset
    :param drive_df:
    :param path:
    :return:
    """
    begin = len(drive_df)
    # for ix in range(0, 1):
    original = drive_df[abs(drive_df['steering']) > PERTURBED_ANGLE_MIN]
    original['steering2'] = original.apply(lambda x: x['steering'] + PERTURBED_ANGLE, axis=1)
    del original['steering']
    original = original.rename(columns={'steering2': 'steering'})
    original.index = range(len(drive_df) + 1, len(drive_df) + 1 + len(original))
    original['PERT'] = 1
    # drive_df = drive_df.append(original)
    end = len(drive_df) + len(original)
    print('Augmented Steering Angles: {0} ({1})'.format(end, end - begin))
    return original


def add_flipped_images(drive_df):
    begin = len(drive_df)
    maxidx = max(drive_df.index)
    addition = {}
    for idx, row in drive_df.iterrows():
        # rnd = np.random.randint(2)
        # if rnd == 1:
        # Flip and created a new image
        for path in ['center', 'left', 'right']:
            """if row['steering'] == 0 and path == 'center':
                # Flipping 0 steering doesn't make any sense?
                continue"""
            maxidx += 1
            steer = -row['steering']
            img_path = 'data/{0}'.format(row[path].strip())
            new_path = 'IMG/FLIPPED_{0}'.format(row[path].split('/')[-1])
            img = cv2.imread(img_path)
            img = np.array(img)
            img = np.fliplr(img)
            cv2.imwrite('data/{0}'.format(new_path), img)
            addition[maxidx] = {'steering': steer, path: new_path,
                                'throttle': row['throttle'],
                                'speed': row['speed']}
    new_df = pd.DataFrame.from_dict(addition, orient='index')
    # drive_df = drive_df.append(new_df)
    end = len(drive_df) + len(new_df)
    print('Flipped Images: {0} ({1})'.format(end, end - begin))
    return new_df


def trans_image(image, steer, trans_range):
    # Translation
    rows, cols, channels = image.shape
    tr_x = trans_range * np.random.uniform() - trans_range / 2
    steer_ang = steer + tr_x / trans_range * 2 * .2
    tr_y = 10 * np.random.uniform() - 10 / 2
    # tr_y = 0
    Trans_M = np.float32([[1, 0, tr_x], [0, 1, tr_y]])
    image_tr = cv2.warpAffine(image, Trans_M, (cols, rows))

    return image_tr, steer_ang


def add_translated_images(drive_df):
    begin = len(drive_df)
    maxidx = max(drive_df.index)
    addition = {}
    choices = ['center', 'left', 'right']
    # I only want to translate the original images with all 3 images avail, not flipped images
    for idx, row in drive_df.iterrows():
        for choice in choices:
            addition[maxidx] = {}
            img_path = 'data/{0}'.format(row[choice].strip())
            new_path = 'IMG/TRANS_{0}'.format(row[choice].split('/')[-1])
            img = cv2.imread(img_path)
            img, steer = trans_image(img, row['steering'], 150)
            cv2.imwrite('data/{0}'.format(new_path), img)
            addition[maxidx] = {choice: new_path, 'steering': steer,
                                'throttle': row['throttle'],
                                'speed': row['speed']}
            maxidx += 1
    new_df = pd.DataFrame.from_dict(addition, orient='index')
    # drive_df = drive_df.append(new_df)
    end = len(drive_df) + len(new_df)
    print('Translations: {0} ({1})'.format(end, end - begin))
    return new_df


def augment_brightness_camera_images(image):
    image1 = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    random_bright = .25 + np.random.uniform()
    image1[:, :, 2] = image1[:, :, 2] * random_bright
    image1 = cv2.cvtColor(image1, cv2.COLOR_HSV2RGB)
    return image1


def add_brightness_augmented_images(drive_df):
    begin = len(drive_df)
    maxidx = max(drive_df.index)
    addition = {}
    choices = ['center', 'left', 'right']
    # I only want to translate the original images with all 3 images avail, not flipped images
    for idx, row in drive_df.iterrows():
        addition[maxidx] = {'steering': row['steering'],
                            'throttle': row['throttle'],
                            'speed': row['speed']}
        for choice in choices:
            img_path = 'data/{0}'.format(row[choice].strip())
            new_path = 'IMG/BRIGHT_{0}'.format(row[choice].split('/')[-1])
            img = cv2.imread(img_path)
            img = augment_brightness_camera_images(img)
            cv2.imwrite('data/{0}'.format(new_path), img)
            addition[maxidx][choice] = new_path
        maxidx += 1
    new_df = pd.DataFrame.from_dict(addition, orient='index')
    # drive_df = drive_df.append(new_df)
    end = len(drive_df) + len(new_df)
    print('Brightness Augment: {0} ({1})'.format(end, end - begin))
    return new_df


def vis(df=None, rn=None, img_view='img_path', img=None):
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
    img = img[100:img_shape[0] - 20, 0:img_shape[1]]
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    plt.imshow(img)
    plt.show()
    return df


def plot_camera_images(df):
    import matplotlib.pyplot as plt
    images = []
    for ix in ['FLIPPED', 'BRIGHT', 'TRANS', 'left']:
        piece = df[df.img_path.str.contains(ix)]
        rnd = np.random.choice(piece.index, 1)
        images.append([piece.loc[rnd]['steering'],
                       return_image(cv2.imread('data/{0}'.format(piece.loc[rnd]['img_path'].iloc[0].strip()))),
                       piece.loc[rnd]['img_path'].iloc[0].strip()])
    for ix, image in enumerate(images):
        plt.subplot(2, 2, ix + 1)
        steering = image[0]
        img_path = image[-1]
        print(img_path)
        plt.imshow(image[1], aspect='auto')
        plt.axis('off')
        plt.title("%s, steering %.2f" % (img_path, steering))


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
