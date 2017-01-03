import numpy as np
import tensorflow as tf
from keras import backend as K
from keras.callbacks import ModelCheckpoint
from keras.layers import Convolution2D, ELU
from keras.layers import Dense, Flatten
from keras.layers.core import Lambda
from keras.models import Sequential
from keras.optimizers import Adam

import load_data


def global_average_pooling(x):
    return tf.reduce_mean(x, (1, 2))


def global_average_pooling_shape(input_shape):
    return input_shape[0], input_shape[3]


def atan_layer(x):
    return tf.mul(tf.atan(x), 2)


def atan_layer_shape(input_shape):
    return input_shape


def normal_init(shape, name=None):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return K.variable(initial)


def steering_net():
    # p = .5
    model = Sequential()
    model.add(Convolution2D(24, 5, 5, init='he_normal', subsample=(2, 2), name='conv1_1', input_shape=(108, 320, 3)))  #
    model.add(ELU())
    model.add(Convolution2D(36, 5, 5, init='he_normal', subsample=(2, 2), name='conv2_1'))  #
    model.add(ELU())
    model.add(Convolution2D(48, 5, 5, init='he_normal', subsample=(2, 2), name='conv3_1'))  #
    model.add(ELU())
    model.add(Convolution2D(64, 3, 3, init='he_normal', subsample=(1, 1), name='conv4_1'))  #
    model.add(ELU())
    model.add(Convolution2D(64, 3, 3, init='he_normal', subsample=(1, 1), name='conv4_2'))  #
    model.add(ELU())
    model.add(Flatten())
    model.add(Dense(1164, init='he_normal', name="dense_0"))  #
    model.add(ELU())
    # model.add(Dropout(p))
    model.add(Dense(100, init='he_normal', name="dense_1"))  #
    model.add(ELU())
    # model.add(Dropout(p))
    model.add(Dense(50, init='he_normal', name="dense_2"))  #
    model.add(ELU())
    # model.add(Dropout(p))
    model.add(Dense(10, init='he_normal', name="dense_3"))  #
    model.add(ELU())
    model.add(Dense(1, init='he_normal', name="dense_4"))  #
    model.add(Lambda(atan_layer, output_shape=atan_layer_shape, name="atan_0"))

    return model


def get_model():
    model = steering_net()
    model.compile(loss='mse', optimizer='Adam')
    return model


def generate_arrays(X_train, y_train, batch_size):
    while 1:
        for ix in range(int(len(X_train) / batch_size)):
            yield np.array(X_train[ix * batch_size:(ix + 1) * batch_size]), np.array(
                y_train[ix * batch_size:(ix + 1) * batch_size])


def train():
    model = get_model()
    print("Loaded model")
    X_train, X_validate, y_train, y_validate = load_data.return_validation()
    print(model.summary())
    print("Loaded validation datasetset")
    print("Training..")
    checkpoint_path = "full_model.{epoch:02d}-{val_loss:.3f}.h5"
    checkpoint = ModelCheckpoint(checkpoint_path, verbose=1, save_best_only=False, save_weights_only=False, mode='auto')
    model.fit_generator(generate_arrays(X_train, y_train, 256),
                        validation_data=(np.asarray(X_validate), np.asarray(y_validate)),
                        samples_per_epoch=len(X_train),
                        nb_epoch=10, verbose=1, callbacks=[checkpoint])


def load_model():
    model = steering_net()
    model.load_weights('full_model.00-9.882.h5')
    model.compile(loss='mse', optimizer=Adam(lr=1e-5))
    return model


def save_model(model):
    import json
    json_string = model.to_json()
    model.save_weights('model.h5')
    json.dump(json_string, open('model.json', 'w'))


def return_saved_model_with_weights():
    model = steering_net()
    model.load_weights('model.h5')
    model.compile(loss='mse', optimizer=Adam(lr=1e-5))
    return model


    #######
    #
    # TODO: Try taking out the override intitilization to train and then try the drive.py again
    #
    #######
