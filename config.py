import os

import cv2
import numpy as np
import pandas as pd
from keras.optimizers import Adam

IMAGE_HEIGHT_CROP = 108
IMAGE_WIDTH_CROP = 320
CHANNELS = 3
# IMAGE_HEIGHT = 64
# IMAGE_WIDTH = 64
STEERING_ADJUSTMENT = 1
AUTONOMOUS_THROTTLE = .15
# (200, 66) <-- Original NVIDIA Paper
IMAGE_WIDTH = 64
IMAGE_HEIGHT = 32
LR = 1e-5
OPTIMIZER = Adam(lr=LR)
LOSS = 'mse'
NB_EPOCH = 15
BATCH_SIZE = 256


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


def create_and_train_with_altered_images(path='data/altered_driving_log.csv'):
    import os
    if not os.path.isfile(path):
        print("Creating Altered Files")
        create_altered_drive_df(path)
        add_flipped_images(path)
    import model
    model.train(path=path, checkpoint_path="models/altered_comma_model_no_validate-{epoch:02d}.h5")
    

def full_train(path_altered='data/altered_driving_log.csv', path_full='data/full_driving_log.csv'):
    if not os.path.isfile(path_altered):
        print("Creating Altered Files")
        create_altered_drive_df(path_altered)
        add_flipped_images(path_altered)
    if not os.path.isfile(path_full):
        print('Creating Translated Files')
        drive_df = pd.read_csv(path_altered)
        add_translated_images(drive_df, path_full)
    import model
    model.train(path=path_full, checkpoint_path="models/altered_comma_model_no_validate-{epoch:02d}.h5")


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
