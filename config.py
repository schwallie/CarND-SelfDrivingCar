import cv2
import numpy as np
from keras.optimizers import Adam

IMAGE_HEIGHT = 108
IMAGE_WIDTH = 320
LR = 1e-5
OPTIMIZER = Adam(lr=LR)
LOSS = 'mse'
NB_EPOCH = 10
BATCH_SIZE = 256


def return_image(img):
    # Take out the dash and horizon
    img_shape = img.shape
    crop_img = img[int(img_shape[0] / 5):img_shape[0] - 20, 0:img_shape[1]]
    # resize_img = cv2.resize(crop_img, (320, 108), interpolation=cv2.INTER_AREA)
    assert crop_img.shape[0] == IMAGE_HEIGHT
    assert crop_img.shape[1] == IMAGE_WIDTH
    img = cv2.cvtColor(crop_img, cv2.COLOR_BGR2RGB)
    img = np.array(img)/255. - 0.5
    return np.float32(img)


def normalize_image(image_set):
    return (image_set - image_set.mean()) / np.std(image_set)
