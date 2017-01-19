import math

import cv2
import numpy as np
from keras.callbacks import ModelCheckpoint
from keras.layers import Convolution2D, ELU
from keras.layers import Dense, Flatten
from keras.layers.core import Lambda, Dropout
from keras.models import Sequential

import config
import load_data


def steering_net(dropout=.25):
    print('NVIDIA Model...')
    model = Sequential()
    # Vivek, color space conversion layer so the model automatically figures out the best color space
    model.add(Lambda(lambda x: x / 255. - .5,
                     input_shape=(config.IMAGE_HEIGHT, config.IMAGE_WIDTH, config.CHANNELS)))
    model.add(Convolution2D(3, 1, 1, border_mode='same', name='color_conv'))
    # Subsample == stride
    # keras.layers.convolutional.Convolution2D(nb_filter, nb_row, nb_col, border_mode='valid')
    model.add(Convolution2D(24, 5, 5, init='he_normal', activation='elu',
                            subsample=(2, 2), name='conv1'))
    model.add(Dropout(dropout))
    model.add(Convolution2D(36, 5, 5, init='he_normal', activation='elu',
                            subsample=(2, 2), name='conv2'))
    model.add(Dropout(dropout))
    model.add(Convolution2D(48, 5, 5, init='he_normal', activation='elu',
                            subsample=(2, 2), name='conv3'))
    model.add(Dropout(dropout))
    model.add(Convolution2D(64, 3, 3, init='he_normal', activation='elu',
                            subsample=(1, 1), name='conv4'))
    model.add(Dropout(dropout))
    model.add(Convolution2D(64, 3, 3, init='he_normal', activation='elu',
                            subsample=(1, 1), name='conv5'))
    model.add(Dropout(dropout))
    model.add(Flatten())
    # We think NVIDIA has an error and actually meant the flatten == 1152, so no Dense 1164 layer
    # model.add(Dense(1164, init='he_normal', name="dense_1164", activation='elu'))
    model.add(Dense(100, init='he_normal', name="dense_100", activation='elu'))
    model.add(Dropout(dropout))
    model.add(Dense(50, init='he_normal', name="dense_50", activation='elu'))
    model.add(Dropout(dropout))
    model.add(Dense(10, init='he_normal', name="dense_10", activation='elu'))
    model.add(Dropout(dropout))
    model.add(Dense(1, init='he_normal', name="dense_1"))
    model.compile(loss=config.LOSS, optimizer=config.OPTIMIZER)
    return model


def comma_model():
    print('Comma Model...')
    model = Sequential()
    # Color conversion
    model.add(Lambda(lambda x: x / 127.5 - 1.,
                     input_shape=(config.IMAGE_HEIGHT, config.IMAGE_WIDTH, config.CHANNELS),
                     output_shape=(config.IMAGE_HEIGHT, config.IMAGE_WIDTH, config.CHANNELS)))
    model.add(Convolution2D(3, 1, 1, border_mode='same', name='color_conv'))
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
    model.compile(loss=config.LOSS, optimizer=config.OPTIMIZER)
    return model


def generate_arrays(X_train, y_train, batch_size):
    batch_images = np.zeros((batch_size, config.IMAGE_HEIGHT, config.IMAGE_WIDTH, 3))
    batch_steering = np.zeros(batch_size)
    while 1:
        for i_batch in range(batch_size):
            i_line = np.random.randint(len(X_train))
            img = X_train[i_line]
            x = config.return_image(cv2.imread('data/{0}'.format(img.strip())))
            y = y_train[i_line]
            batch_images[i_batch] = x
            batch_steering[i_batch] = y
        yield batch_images, batch_steering


def train(model, path, checkpoint_path):
    X_train, y_train = load_data.load_data(path=path)
    print(model.summary())
    print('X_train samples: {0}'.format(len(X_train)))
    SAMPLES_PER_EPOCH = 50000 # math.floor((len(X_train) // config.BATCH_SIZE * config.BATCH_SIZE) / 2)
    print('Samples Per Epoch: {0}'.format(SAMPLES_PER_EPOCH))
    checkpoint = ModelCheckpoint(checkpoint_path, verbose=1, save_best_only=False, save_weights_only=True, mode='auto')
    model.fit_generator(generate_arrays(X_train, y_train, config.BATCH_SIZE),
                        samples_per_epoch=SAMPLES_PER_EPOCH,
                        nb_epoch=config.NB_EPOCH, verbose=1, callbacks=[checkpoint])

def load_saved_model(path, model):
    model.load_weights(path)
    return model


def save_model(path, model):
    import json
    model.load_weights(path)
    json_string = model.to_json()
    json.dump(json_string, open('model.json', 'w'))
    model.save_weights('model.h5')
