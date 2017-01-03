import cv2
import numpy as np
import math
import pandas as pd
from sklearn.model_selection import train_test_split


def load_data(path='data/'):
    drive_df = pd.read_csv('{0}driving_log.csv'.format(path))
    drive_df['left_steering'] = drive_df['steering'] + .25
    drive_df['right_steering'] = drive_df['steering'] - .25
    drive_df = drive_df.rename(columns={'steering': 'center_steering'})
    X_data = []
    y_data = []
    for cam_type in ['center']: #'left', 'right'
        drive_df[cam_type] = drive_df[cam_type].str.strip()
        vals = drive_df[cam_type].values
        arr = [return_image(f) for f in vals]
        X_data.extend(arr)
        y_data.extend(drive_df['{0}_steering'.format(cam_type)].values)
    X_data = np.float32(X_data)
    y_data = np.float32(y_data)
    X_data /= 255.
    X_data -= 0.5
    return X_data, y_data


def normalize_image(image_set):
    return (image_set - image_set.mean()) / np.std(image_set)


def return_image(f):
    path = 'data/{0}'.format(f)
    img = cv2.imread(path, 1)
    # Take out the dash and horizon
    img_shape = img.shape
    crop_img = img[int(img_shape[0]/5):img_shape[0]-20, 0:img_shape[1]]
    # resize_img = cv2.resize(crop_img, (320, 108), interpolation=cv2.INTER_AREA)
    img = cv2.cvtColor(crop_img, cv2.COLOR_BGR2RGB)
    return np.array(img)


def return_validation():
    X_data, y_data = load_data()
    return train_test_split(X_data, y_data, test_size=.2, random_state=43)
