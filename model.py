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


def steering_net():
    p = .5
    model = Sequential()
    # Vivek, color space conversion layer so the model automatically figures out the best color space
    model.add(Lambda(lambda x: x / 255. - .5,
                     input_shape=(config.IMAGE_HEIGHT, config.IMAGE_WIDTH, 3)))
    # model.add(Convolution2D(3, 1, 1, border_mode='same', name='color_conv'))
    # Subsample == stride
    # keras.layers.convolutional.Convolution2D(nb_filter, nb_row, nb_col, border_mode='valid')
    model.add(Convolution2D(24, 5, 5, init='he_normal', subsample=(2, 2), name='conv1'))
    model.add(ELU())
    model.add(Convolution2D(36, 5, 5, init='he_normal', subsample=(2, 2), name='conv2'))  #
    model.add(ELU())
    model.add(Convolution2D(48, 5, 5, init='he_normal', subsample=(2, 2), name='conv3'))  #
    model.add(ELU())
    model.add(Convolution2D(64, 3, 3, init='he_normal', subsample=(1, 1), name='conv4'))  #
    model.add(ELU())
    model.add(Convolution2D(64, 3, 3, init='he_normal', subsample=(1, 1), name='conv5'))  #
    model.add(ELU())
    model.add(Flatten())
    model.add(Dense(1164, init='he_normal', name="dense_1164"))  #
    model.add(ELU())
    # model.add(Dropout(p))
    model.add(Dense(100, init='he_normal', name="dense_100"))  #
    model.add(ELU())
    # model.add(Dropout(p))
    model.add(Dense(50, init='he_normal', name="dense_50"))  #
    model.add(ELU())
    # model.add(Dropout(p))
    model.add(Dense(10, init='he_normal', name="dense_10"))  #
    model.add(ELU())
    model.add(Dense(1, init='he_normal', name="dense_1"))  #
    return model


def get_model():
    model = steering_net()
    model.compile(loss=config.LOSS, optimizer=config.OPTIMIZER)
    return model


def generate_arrays(X_train, y_train):
    while 1:
        for ix in range(math.floor(len(X_train) / config.BATCH_SIZE)):
            paths = X_train[ix * config.BATCH_SIZE:(ix + 1) * config.BATCH_SIZE]
            imgs = [config.return_image(cv2.imread(f)) for f in paths]
            yield np.array(imgs), np.array(y_train[ix * config.BATCH_SIZE:(ix + 1) * config.BATCH_SIZE])


def train(data=None):
    model = get_model()
    print("Loaded model")
    if data is None:
        X_train, X_validate, y_train, y_validate = load_data.return_validation()
    else:
        X_train, X_validate, y_train, y_validate = data[0], data[1], data[2], data[3]
    X_validate = [config.return_image(cv2.imread(f)) for f in X_validate]
    print(model.summary())
    print("Loaded validation datasetset")
    print("Training..")
    checkpoint_path = "model_1164_no_3x1x1-{epoch:02d}-{val_loss:.3f}.h5"
    checkpoint = ModelCheckpoint(checkpoint_path, verbose=1, save_best_only=False, save_weights_only=False, mode='auto')
    model.fit_generator(generate_arrays(X_train, y_train),
                        validation_data=(np.asarray(X_validate), np.asarray(y_validate)),
                        samples_per_epoch=len(X_validate) * 5,
                        nb_epoch=config.NB_EPOCH, verbose=1, callbacks=[checkpoint])


def load_saved_model(path='full_model.03-0.011.h5'):
    model = steering_net()
    model.load_weights(path)
    model.compile(loss=config.LOSS, optimizer=config.OPTIMIZER)
    return model


def save_model(model):
    import json
    json_string = model.to_json()
    model.save_weights('model.h5')
    json.dump(json_string, open('model.json', 'w'))
