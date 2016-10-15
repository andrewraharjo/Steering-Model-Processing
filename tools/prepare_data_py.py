import pandas as pd
import numpy as np
import os
import cv2
import progressbar
import time
import h5py

ch, rows, cols = 3, 224, 224
create_dataset = True
dataset = 'udacity_data_224.h5'
dataset_id = 1

index = 'dataset0/dataset.csv'
prefix = 'dataset0/'

data = pd.read_csv(index)

def image_preprocessing(fname):
    img = cv2.imread(os.path.join(prefix, fname))
    img = cv2.resize(img, (rows, cols))
    img = img.transpose()
    return img

print('Number of images:', len(data))

if os.path.isfile(dataset):
    print('Appending to existing h5.')
    time.sleep(5)  # In case user wants to cancel.
    print('Running ...')
    h5 = h5py.File(dataset, 'r+')
    ix = h5['id'].shape[0]

    print('Prior shape:', h5['main'].shape)

    h5['main'].resize((len(data) + ix, ch, rows, cols))
    h5['id'].resize((len(data) + ix, 1))
    h5['angle'].resize((len(data) + ix, 1))

    print('Loading images ...')
    bar = progressbar.ProgressBar(max_value=len(data))
    for i, fname in bar(enumerate(data['filename'].values)):
        img = image_preprocessing(fname)
        h5['main'][ix + i] = img
        h5['id'][ix + i] = dataset_id
        h5['angle'][ix + i] = data['angle'][i]

    print('Final shape:', h5['main'].shape)

else:
    print('Creating new h5')
    h5 = h5py.File(dataset, 'w')
    h5.create_dataset("main", (len(data), ch, rows, cols), maxshape=(None, ch, rows, cols))
    h5.create_dataset('id', (len(data), 1), maxshape=(None))
    h5.create_dataset('angle', (len(data), 1), maxshape=(None))

    print('Loading images ...')
    bar = progressbar.ProgressBar(max_value=len(data))
    for i, fname in bar(enumerate(data['filename'].values)):
        img = image_preprocessing(fname)
        h5['main'][i] = img
        h5['id'][i] = dataset_id
        h5['angle'][i] = data['angle'][i]

    print('Final shape:', h5['main'].shape)
