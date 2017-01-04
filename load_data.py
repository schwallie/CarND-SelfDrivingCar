import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


def load_data(path='data/driving_log.csv'):  # altered_driving_log.csv
    drive_df = pd.read_csv(path)
    # TODO: Try .08 instead of .25
    # drive_df['left_steering'] = drive_df['steering'] + .08
    # drive_df['right_steering'] = drive_df['steering'] - .08
    drive_df = drive_df.rename(columns={'steering': 'center_steering'})
    X_data = []
    y_data = []
    for cam_type in ['center']: # , 'left', 'right']:
        drive_df[cam_type] = drive_df[cam_type].str.strip()
        vals = drive_df[cam_type].values
        arr = ['data/{0}'.format(f) for f in vals]
        X_data.extend(arr)
        y_data.extend(drive_df['{0}_steering'.format(cam_type)].values)
    y_data = np.float32(y_data)
    return X_data, y_data


def return_validation(path='data/driving_log.csv'):
    X_data, y_data = load_data(path=path)
    return train_test_split(X_data, y_data, test_size=.2, random_state=43)
