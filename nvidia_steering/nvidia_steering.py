#!/usr/bin/env python
"""
Steering angle prediction model
"""
import os
import argparse
import json
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Lambda, ELU
from keras.layers.convolutional import Convolution2D
from keras.layers import Input, Dropout, Flatten, Convolution2D, MaxPooling2D, Dense, Activation, ZeroPadding2D, BatchNormalization
import cv2
from keras.optimizers import RMSprop, SGD
import numpy as np

from server import client_generator


def gen(hwm, host, port):
  for tup in client_generator(hwm=hwm, host=host, port=port):
    X, Y, _ = tup
    Y = Y[:, -1]
    if X.shape[1] == 1:  # no temporal context
      X = X[:, -1]
    #print(X.shape)
    # X = np.swapaxes(X, 1, 3)
    # X = np.array([cv2.resize(x, (40, 40), interpolation = cv2.INTER_AREA).flatten() for x in X])
    # #X = np.swapaxes(X, 3, 1)
    # #print(X.shape)
    # X = X-128 / 128
    yield X, Y

optimizer = SGD(lr=1e-2, decay=1e-6, momentum=0.9, nesterov=True)

def get_model(time_len=1, vgg_weights=False):
    ch, row, col = 3, 160, 320  # camera format

    model = Sequential()
    # model.add(Lambda(lambda x: x,
    #           input_shape=(ch, row, col),
    #           output_shape=(ch, row, col)))
    #model.add(Input(shape=(ch, row, col)))

    model.add(BatchNormalization(input_shape=(ch, row, col)))

    model.add(Convolution2D(24, 5, 5, subsample=(2, 2)))
    model.add(Convolution2D(36, 5, 5, subsample=(2, 2)))
    model.add(Convolution2D(48, 5, 5, subsample=(2, 2)))

    model.add(Convolution2D(64, 3, 3))
    model.add(Convolution2D(64, 3, 3))

    model.add(Flatten())

    model.add(Dense(100))
    model.add(Activation('relu'))
    model.add(Dense(50))
    model.add(Activation('relu'))
    model.add(Dense(10))
    model.add(Activation('relu'))

    model.add(Dense(1))

    model.compile(optimizer='adam', loss="mae", metrics=['mse'])

    return model


if __name__ == "__main__":
  parser = argparse.ArgumentParser(description='Steering angle model trainer')
  parser.add_argument('--host', type=str, default="localhost", help='Data server ip address.')
  parser.add_argument('--port', type=int, default=5557, help='Port of server.')
  parser.add_argument('--val_port', type=int, default=5556, help='Port of server for validation dataset.')
  parser.add_argument('--batch', type=int, default=64, help='Batch size.')
  parser.add_argument('--epoch', type=int, default=200, help='Number of epochs.')
  parser.add_argument('--epochsize', type=int, default=10000, help='How many frames per epoch.')
  parser.add_argument('--skipvalidate', dest='skipvalidate', action='store_true', help='Multiple path output.')
  parser.set_defaults(skipvalidate=False)
  parser.set_defaults(loadweights=False)
  args = parser.parse_args()

  model = get_model(vgg_weights='vgg16_weights.h5')

  print('Fitting model.')
  model.fit_generator(
    gen(20, args.host, port=args.port),
    samples_per_epoch=10000,
    nb_epoch=args.epoch,
    validation_data=gen(20, args.host, port=args.val_port),
    nb_val_samples=2500,
    verbose=1
  )
  print("Saving model weights and configuration file.")

  if not os.path.exists("./outputs/steering_model"):
      os.makedirs("./outputs/steering_model")

  model.save_weights("./outputs/steering_model/steering_angle.keras", True)
  with open('./outputs/steering_model/steering_angle.json', 'w') as outfile:
    json.dump(model.to_json(), outfile)
