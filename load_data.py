import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle


def load_data(path='data/driving_log.csv'):  # altered_driving_log.csv
    drive_df = pd.read_csv(path)
    drive_df['steering_smoothed'] = pd.rolling_mean(drive_df['steering'], 3)
    drive_df['steering_smoothed'] = drive_df['steering_smoothed'].fillna(0)
    # TODO: Try .08 instead of .25
    drive_df['left_steering'] = drive_df['steering_smoothed'] + .08
    drive_df['right_steering'] = drive_df['steering_smoothed'] - .08
    drive_df = drive_df.rename(columns={'steering_smoothed': 'center_steering'})
    X_data = []
    y_data = []
    for cam_type in ['center', 'left', 'right']:
        drive_df[cam_type] = drive_df[cam_type].str.strip()
        vals = drive_df[pd.notnull(drive_df[cam_type])][cam_type].values
        arr = ['data/{0}'.format(f) for f in vals]
        X_data.extend(arr)
        y_cam_type = '{0}_steering'.format(cam_type)
        y_data.extend(drive_df[pd.notnull(drive_df[cam_type])][y_cam_type].values)
    y_data = np.float32(y_data)
    # Shuffle since I'm not doing validation
    # TODO: Make sure I didn't screw up anything with shuffle
    return shuffle(X_data, y_data)


def return_validation(path='data/driving_log.csv'):
    X_data, y_data = load_data(path=path)
    return train_test_split(X_data, y_data, test_size=.05, random_state=43)



