import os
import numpy as np
from videos_to_hdf5 import convert_videos_to_hdf5


def label_cbk(f, return_len=False):
    # define callback that calculates label from file name. If we don't
    # pass a label callback, the next function will only save the frames.
    if return_len:
        # label_callback must know the number of classes in the dataset
        return 51
    dirname = f.split('/')[1]
    names = [name for name in os.listdir('./dataset') if os.path.isdir(os.path.join('./dataset', name))]
    label = np.asarray([w == dirname for w in names])
    return label.astype('float32')

convert_videos_to_hdf5('final_file.hdf5', 'dataset', ext='*.mp4',
                        label_callback=label_cbk)


