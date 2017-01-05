import json
import math

import cv2
import numpy as np
from keras.callbacks import ModelCheckpoint
from keras.layers import Convolution2D, ELU
from keras.layers import Dense, Flatten
from keras.layers.core import Lambda, Dropout
from keras.models import Sequential
from keras.models import model_from_json

import config
import load_data


def steering_net():
    model = Sequential()
    # Vivek, color space conversion layer so the model automatically figures out the best color space
    model.add(Lambda(lambda x: x / 255. - .5,
                     input_shape=(config.IMAGE_HEIGHT, config.IMAGE_WIDTH, 3)))
    # model.add(Convolution2D(3, 1, 1, border_mode='same', name='color_conv'))
    # Subsample == stride
    # keras.layers.convolutional.Convolution2D(nb_filter, nb_row, nb_col, border_mode='valid')
    model.add(Convolution2D(24, 5, 5, init='he_normal', activation='elu',
                            subsample=(2, 2), name='conv1'))
    model.add(Convolution2D(36, 5, 5, init='he_normal', activation='elu',
                            subsample=(2, 2), name='conv2'))
    model.add(Convolution2D(48, 5, 5, init='he_normal', activation='elu',
                            subsample=(2, 2), name='conv3'))
    model.add(Convolution2D(64, 3, 3, init='he_normal', activation='elu',
                            subsample=(1, 1), name='conv4'))
    model.add(Convolution2D(64, 3, 3, init='he_normal', activation='elu',
                            subsample=(1, 1), name='conv5'))
    model.add(Flatten())
    # We think NVIDIA has an error and actually meant the flatten == 1152, so no Dense 1164 layer
    # model.add(Dense(1164, init='he_normal', name="dense_1164", activation='elu'))
    model.add(Dense(100, init='he_normal', name="dense_100", activation='elu'))
    model.add(Dense(50, init='he_normal', name="dense_50", activation='elu'))
    model.add(Dense(10, init='he_normal', name="dense_10", activation='elu'))
    model.add(Dense(1, init='he_normal', name="dense_1"))
    return model


def comma_model():
    model = Sequential()
    model.add(Lambda(lambda x: x / 127.5 - 1.,
                     input_shape=(config.IMAGE_HEIGHT, config.IMAGE_WIDTH, config.CHANNELS),
                     output_shape=(config.IMAGE_HEIGHT, config.IMAGE_WIDTH, config.CHANNELS)))
    model.add(Convolution2D(16, 8, 8, subsample=(4, 4), border_mode="same", activation='elu'))
    model.add(Convolution2D(32, 5, 5, subsample=(2, 2), border_mode="same", activation='elu'))
    model.add(Convolution2D(64, 5, 5, subsample=(2, 2), border_mode="same"))
    model.add(Flatten())
    model.add(Dropout(.2))
    model.add(ELU())
    model.add(Dense(512))
    model.add(Dropout(.5))
    model.add(ELU())
    model.add(Dense(1))
    return model


def get_model():
    model = steering_net()
    model.compile(loss=config.LOSS, optimizer=config.OPTIMIZER)
    return model

def get_comma_model():
    model = comma_model()
    model.compile(loss=config.LOSS, optimizer=config.OPTIMIZER)
    return model



def generate_arrays(X_train, y_train):
    while 1:
        for ix in range(math.floor(len(X_train) / config.BATCH_SIZE)):
            paths = X_train[ix * config.BATCH_SIZE:(ix + 1) * config.BATCH_SIZE]
            imgs = [config.return_image(cv2.imread(f)) for f in paths]
            yield np.array(imgs), np.array(y_train[ix * config.BATCH_SIZE:(ix + 1) * config.BATCH_SIZE])


def train(path='data/driving_log.csv', checkpoint_path="models/comma_model_no_validate-{epoch:02d}.h5"):
    model = get_comma_model()
    print("Loaded model")
    X_train, y_train = load_data.load_data(path=path)
    print(model.summary())
    print("Training..")
    checkpoint = ModelCheckpoint(checkpoint_path, verbose=1, save_best_only=False, save_weights_only=True, mode='auto')
    model.fit_generator(generate_arrays(X_train, y_train),
                        samples_per_epoch=10000,
                        nb_epoch=config.NB_EPOCH, verbose=1, callbacks=[checkpoint])


def load_saved_model(path):
    model = steering_net()
    model.load_weights(path)
    model.compile(loss=config.LOSS, optimizer=config.OPTIMIZER)
    return model


def load_saved_comma_model():
    # with open('model.json', 'r') as jfile:
    #    loaded = json.load(jfile)
    #    model = model_from_json(loaded)
    # model = model_from_json(json.load(open('model.json')))
    # model.compile("adam", "mse")
    model = get_comma_model()
    model.load_weights('model.h5')
    return model


def save_comma_model(path):
    model = get_comma_model()
    model.load_weights(path)
    json_string = model.to_json()
    with open('model.json', 'w') as f:
        json.dump(json_string, f)
    model.save_weights('model.h5')

def save_model(model):
    import json
    json_string = model.to_json()
    model.save_weights('model.h5')
    json.dump(json_string, open('model.json', 'w'))




