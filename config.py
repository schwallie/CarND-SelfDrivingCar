import cv2
import numpy as np
import pandas as pd
from keras.optimizers import Adam

IMAGE_HEIGHT_CROP = 108
IMAGE_WIDTH_CROP = 320
IMAGE_HEIGHT = 64
IMAGE_WIDTH = 64
# (200, 66) <-- Original NVIDIA Paper
# IMAGE_WIDTH = 200
# IMAGE_HEIGHT = 66
LR = 1e-4
OPTIMIZER = Adam(lr=LR)
LOSS = 'mse'
NB_EPOCH = 10
BATCH_SIZE = 256


def return_image(img):
    # Take out the dash and horizon
    img_shape = img.shape
    crop_img = img[int(img_shape[0] / 5):img_shape[0] - 20, 0:img_shape[1]]
    # resize_img = cv2.resize(crop_img, (320, 108), interpolation=cv2.INTER_AREA)
    assert crop_img.shape[0] == IMAGE_HEIGHT_CROP
    assert crop_img.shape[1] == IMAGE_WIDTH_CROP
    img = cv2.cvtColor(crop_img, cv2.COLOR_BGR2RGB)
    # img = np.array(img)/255. - 0.5
    img = (cv2.resize(img, (IMAGE_WIDTH, IMAGE_HEIGHT), interpolation=cv2.INTER_AREA))
    return np.float32(img)

def flip_image():
    pass

def add_flipped_images(path = 'data/altered_drive_df.csv'):
    drive_df = pd.read_csv(path)
    maxidx = max(drive_df.index)
    addition = {}
    for idx, row in drive_df.iterrows():
        rnd = np.random.randint(2)
        if rnd == 1:
            maxidx += 1
            # Flip and created a new image
            img_path = 'data/{0}'.format(row['center'])
            new_path = 'data/IMG/FLIPPED_{0}'.format(row['center'].split('/')[-1])
            img = cv2.imread(img_path)
            img = np.array(img)
            print(img)
            img = cv2.flip(img, 1)
            cv2.imwrite(new_path, img)
            steer = -row['steering']
            addition[maxidx] = {'center': new_path, 'steering': steer}
    drive_df = pd.concat([drive_df, pd.DataFrame(addition)])
    drive_df.to_csv(path)
