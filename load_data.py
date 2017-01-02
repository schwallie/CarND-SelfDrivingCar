import cv2
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


def load_data(path='data/'):
    drive_df = pd.read_csv('{0}driving_log.csv'.format(path))
    X_data = [np.float32(cv2.resize(cv2.imread('{0}{1}'.format(path, f), 1)[32:140, 0:320], (200, 66))) / 255.0 for f in drive_df.center]
    y_data = drive_df['steering'].values
    X_data = clean_data(X_data)
    return X_data, y_data

def clean_data(X_data):
    return X_data


def return_validation():
    X_data, y_data = load_data()
    return train_test_split(X_data, y_data, test_size=.2, random_state=43)
