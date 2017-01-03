from keras.optimizers import Adam
IMAGE_HEIGHT = 108
IMAGE_WIDTH = 320
LR = 1e-5
OPTIMIZER = Adam(lr=LR)
LOSS = 'mse'
NB_EPOCH = 10
BATCH_SIZE = 256