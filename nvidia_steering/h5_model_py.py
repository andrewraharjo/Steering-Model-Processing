import pandas as pd
import numpy as np
import os
import keras as k
import keras.layers as l
import cv2
import progressbar
import pickle
import h5py

# Load full data into memory
load_full = True

ch, row, col = 3, 224, 224  # camera format
h5_loc = 'udacity_data_224.h5'
#dataset = 'dataset0/dataset.csv'
#prefix = 'dataset0'
#cache_loc = 'cache/'

def get_model(time_len=1, vgg_weights=False):
    model = k.models.Sequential()

    model.add(l.BatchNormalization(input_shape=(ch, row, col)))

    model.add(l.Convolution2D(24, 5, 5, subsample=(2, 2)))
    model.add(l.Convolution2D(36, 5, 5, subsample=(2, 2)))
    model.add(l.Convolution2D(48, 5, 5, subsample=(2, 2)))

    model.add(l.Convolution2D(64, 3, 3))
    model.add(l.Convolution2D(64, 3, 3))

    model.add(l.Flatten())

    model.add(l.Dense(100))
    model.add(l.Activation('relu'))
    model.add(l.Dense(50))
    model.add(l.Activation('relu'))
    model.add(l.Dense(10))
    model.add(l.Activation('relu'))

    model.add(l.Dense(1))

    #opt = k.optimizers.RMSProp()
    model.compile(optimizer='adam', loss="mse", metrics=['mae'])

    return model

def batch_generator(h5_loc, valid_split, valid, batch_size=64):
    h5 = h5py.File(h5_loc, 'r')
    h5_len = h5['angle'].shape[0]
    if valid:
        choices = range(valid_split, h5_len)
    else:
        choices = range(0, valid_split)
    while True:
        x_batch = np.zeros((batch_size, ch, row, col))
        y_batch = np.zeros((batch_size))
        for i in range(batch_size):
            c = np.random.choice(choices)
            img = h5['main'][c]
            angle = h5['angle'][c]
            x_batch[i, :, :, :] = img
            y_batch[i] = angle
        yield (x_batch, y_batch)

h5 = h5py.File(h5_loc, 'r')
h5_len = h5['angle'].shape[0]
print('Number of images:', h5_len)
valid_split = 10000
len_train = len(range(0, valid_split))
len_valid = len(range(valid_split, h5_len))
print('Train count:', len_train, 'Valid count:', len_valid)

print('Compiling model ...')

model = get_model()

model.fit_generator(batch_generator(h5_loc, valid_split, False), samples_per_epoch=len_train, nb_epoch=10,
                    verbose=1, validation_data=batch_generator(h5_loc, valid_split, True), nb_val_samples=len_valid, nb_worker=1, pickle_safe=False)
#preds = model.predict(x_valid, verbose=1, batch_size=128)
#print(preds.tolist())
