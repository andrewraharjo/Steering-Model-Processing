import tensorflow as tf
import os
import numpy as np
import cv2
import math
import augmentation as aug
import pandas as pd

IMAGE_HEIGHT = 66
IMAGE_WIDTH = 200
IMAGE_CHANNELS = 3
ORIGINAL_IMAGE_HEIGHT = 480
ORIGINAL_IMAGE_WIDTH = 640
CROP_X = 10
CROP_Y = 240

def _generate_image_and_label_batch(image, label, min_queue_examples,
                                    batch_size=128, shuffle=False):
    """Construct a queued batch of images and labels.
    Args:
        image: 3-D Tensor of [height, width, 3] of type.float32.
        label: 1-D Tensor of type.int32
        min_queue_examples: int32, minimum number of samples to retain
          in the queue that provides of batches of examples.
        batch_size: Number of images per batch.
        shuffle: boolean indicating whether to use a shuffling queue.
    Returns:
        images: Images. 4D tensor of [batch_size, height, width, 3] size.
        labels: Labels. 1D tensor of [batch_size] size.
    """
    # Create a queue that shuffles the examples, and then
    # read 'batch_size' images + labels from the example queue.
    num_preprocess_threads = 16
    if shuffle:
        images, label_batch = tf.train.shuffle_batch(
            [image, label],
            batch_size=batch_size,
            num_threads=num_preprocess_threads,
            capacity=min_queue_examples + 3 * batch_size,
            min_after_dequeue=min_queue_examples)
    else:
        images, label_batch = tf.train.batch(
            [image, label],
            batch_size=batch_size,
            num_threads=num_preprocess_threads,
            capacity=min_queue_examples + 3 * batch_size)

    # Display the training images in the visualizer.
    tf.image_summary('images', images)

    #return images, tf.reshape(label_batch, [batch_size, NBINS])
    return images, label_batch


def angle_to_vec(angle, nbins, min_angle, max_angle):
    angle_bin = int((nbins-1) * (angle - min_angle) / (max_angle - min_angle))

    if angle_bin < 0:
        angle_bin = 0
    if angle_bin >= nbins:
        angle_bin = nbins - 1

    angle_vec = [0] * nbins
    angle_vec[angle_bin] = 1

    return angle_vec

def vec_to_angle(angle_vec, nbins, max_angle, min_angle=None):

    if min_angle is None:
        min_angle = -max_angle

    max_idx = np.argmax(angle_vec)
    return max_idx * (max_angle - min_angle) / (nbins-1) + min_angle


def normalize(angle, min_angle, max_angle):
    #return 2. * (angle - min_angle) / (max_angle - min_angle) - 1.
    return (angle - min_angle) / (max_angle - min_angle)


def angle_from_normalized(norm, max_angle, min_angle=None):

    if min_angle is None:
        min_angle = -max_angle

    #return (norm + 1.) * (max_angle - min_angle) / 2. + min_angle
    return norm * (max_angle - min_angle) + min_angle


def rgb_yuv(img):
        yuv = cv2.cvtColor(img, cv2.COLOR_RGB2YUV)
        return yuv

def get_augmented(image, steering_wheel_angle, speed, filename):

    if 'left' in str(filename):
        initial_shift = -.5

    elif 'right' in str(filename):
        initial_shift = .5

    else:
        initial_shift = 0


    augmented_image, new_angle, r, s = aug.steer_back_distortion(image,
                                                       steering_wheel_angle,
                                                       speed,
                                                       initial_shift=initial_shift)
    return augmented_image, new_angle


def read_tf(data_dir, shuffle=False):

    files = [os.path.join(data_dir, f) for f in os.listdir(data_dir) if f.endswith('.tfrecords')]

    filename_queue = tf.train.string_input_producer(files, shuffle=shuffle)
    #filename_queue = tf.train.string_input_producer(
    #    tf.train.match_filenames_once(os.path.join(data_dir, file_pattern)), shuffle=False)

    reader = tf.TFRecordReader()
    # One can read a single serialized example from a filename
    # serialized_example is a Tensor of type string.
    _, serialized_example = reader.read(filename_queue)

    # The serialized example is converted back to actual values.
    # One needs to describe the format of the objects to be returned
    features = tf.parse_single_example(
      serialized_example,
      # Defaults are not specified since both keys are required.
      features={
          'image': tf.FixedLenFeature([], tf.string),
          'steering_angle': tf.FixedLenFeature([], tf.float32),
          'speed': tf.FixedLenFeature([], tf.float32),
          'timestamp': tf.FixedLenFeature([], tf.float32),
          'image/height': tf.FixedLenFeature([], tf.int64),
          'image/width': tf.FixedLenFeature([], tf.int64),
          'image/channels': tf.FixedLenFeature([], tf.int64),
      })

    # Convert from a scalar string tensor to a uint8 tensor with shape
    # [IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_CHANNELS].
    image = tf.decode_raw(features['image'], tf.uint8)
    image = tf.reshape(image, tf.pack([IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_CHANNELS]))
    #image = tf.image.decode_png(data['image'], channels=IMAGE_CHANNELS)
    #image.set_shape([IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_CHANNELS])
    features['image'] = image

    return features

def filter_frames(data, cameras, min_angle, max_angle):

    if cameras:
        # filter cameras
        data = data[data.frame_id.isin(cameras)]

    target_angles = (data.angle < max_angle) & (data.angle > min_angle)
    target_speed = data.speed > 3.
    return data[target_angles & target_speed]

def get_steering_angle(frame):

    if 'left' in frame.frame_id:
        initial_shift = -.5

    elif 'right' in frame.frame_id:
        initial_shift = .5

    else:
        initial_shift = 0

    _, _, angle = aug.get_steer_back_angle(frame.angle, frame.speed, 0, initial_shift)
    frame.angle = angle
    return frame

def read_files(data_dirs, cameras, min_angle, max_angle, shuffle=False, data_file='interpolated.csv'):

    image_list = []
    angle_list = []
    speed_list = []

    for data_dir in data_dirs:

        try:

            interpolated = pd.read_csv(os.path.join(data_dir, data_file))
            interpolated = filter_frames(interpolated, cameras, min_angle, max_angle)

            interpolated = interpolated.apply(lambda x: get_steering_angle(x), axis=1)

            filtered = filter_frames(interpolated, cameras, min_angle, max_angle)
            image_list.extend([os.path.join(data_dir, f) for f in filtered.filename.values.tolist()])
            angle_list.extend(filtered.angle.values.tolist())
            speed_list.extend(filtered.speed.values.tolist())

        except Exception as e:
            print(e)
            print('WARNING: dataset file not found: ', os.path.join(data_dir, data_file))

    max_angle = np.max(angle_list)
    min_angle = np.min(angle_list)
    mean_angle = np.mean(angle_list)

    print('Training angle statistics: min={}; max={}; mean={}; std={}'.format(min_angle, max_angle, mean_angle, np.std(angle_list)))

    images = tf.convert_to_tensor(image_list, dtype=tf.string)
    angles = tf.convert_to_tensor(angle_list, dtype=tf.float32)
    speeds = tf.convert_to_tensor(speed_list, dtype=tf.float32)

    # Makes an input queue
    input_queue = tf.train.slice_input_producer([images, angles, speeds],
                                                shuffle=shuffle)

    file_contents = tf.read_file(input_queue[0])

    image = tf.image.decode_png(file_contents, channels=IMAGE_CHANNELS)
    image.set_shape([ORIGINAL_IMAGE_HEIGHT, ORIGINAL_IMAGE_WIDTH, IMAGE_CHANNELS])
    data = {
        'image': image,
        'filename': input_queue[0],
        'angle': input_queue[1],
        'speed': input_queue[2],
        'max_angle': max_angle,
        'min_angle': min_angle,
        'mean_angle': mean_angle
    }

    return data


def inputs(data_dirs, num_examples_per_epoch=10000, batch_size=128,
           shuffle=False, num_classes=1, augment_data=False, raw_labels=False,
           cameras=None, min_angle=None, max_angle=math.pi/8):
    """Construct input using the Reader ops.
    Args:
        batch_size: Number of images per batch.
    Returns:
        images: Images. 4D tensor of [batch_size, IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_CHANNELS] size.
        labels: Labels. 1D tensor of [batch_size] size.
    """

    if min_angle is None:
        min_angle = -max_angle

    # Read examples from files in the filename queue.
    #data = read_tf(data_dir, shuffle=shuffle)
    data = read_files(data_dirs, cameras, min_angle, max_angle, shuffle=shuffle)

    speed = data['speed']
    steering_angle = data['angle']
    image = data['image']
    filename= data['filename']
    data_max_angle = data['max_angle']
    data_min_angle = data['min_angle']
    data_mean_angle = data['mean_angle']

    if augment_data:
        augmented = tf.py_func(get_augmented, [image, steering_angle, speed, filename],
                               [tf.uint8, tf.double])
        image = augmented[0]
        steering_angle = augmented[1]

        image.set_shape([ORIGINAL_IMAGE_HEIGHT - CROP_Y,
                         ORIGINAL_IMAGE_WIDTH - 2 * CROP_X,
                         IMAGE_CHANNELS])
        steering_angle.set_shape([])
    else:
        image = image[CROP_Y:, CROP_X:ORIGINAL_IMAGE_WIDTH-CROP_X, :]

    if num_classes > 1 and not raw_labels:
        label_angle = steering_angle
        label_vec = tf.py_func(angle_to_vec, [label_angle, num_classes, min_angle, max_angle],
                               [tf.int64]*num_classes)
        label = tf.pack(label_vec)
        label.set_shape([num_classes])
    else:
        label = steering_angle
        label = tf.cast(label, tf.float32)

        if not raw_labels:
            # normalize labels
            label = normalize(label, min_angle, max_angle)

            #label = normalize(label)

    resized_image = tf.image.resize_images(image, [IMAGE_HEIGHT, IMAGE_WIDTH])

    image_yuv = tf.py_func(rgb_yuv, [resized_image], [tf.float32])[0]
    image_yuv.set_shape([IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_CHANNELS])

    # Subtract off the mean and divide by the variance of the pixels.
    float_image = tf.image.per_image_whitening(image_yuv)
    #float_image = tf.cast(image, tf.float32)
    #norm_image = (float_image / 127.5) - 1

    # Ensure that the random shuffling has good mixing properties.
    min_fraction_of_examples_in_queue = 0.4
    min_queue_examples = int(num_examples_per_epoch *
                           min_fraction_of_examples_in_queue)

    # Generate a batch of images and labels by building up a queue of examples.
    return _generate_image_and_label_batch(float_image, label,
                                         min_queue_examples, batch_size,
                                         shuffle=shuffle)


def unlabeled_inputs(data_dirs, batch_size=128):
    """Construct input for test.
    Args:
        data_dir: Path to the data directory.
        batch_size: Number of images per batch.
    Returns:
        images: Images. 4D tensor of [batch_size, IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_CHANNELS] size.
    """

    files = []

    for data_dir in data_dirs:

        try:
            files.extend([os.path.join(data_dir, f) for f in os.listdir(data_dir) if f.endswith('.png')])

        except Exception as e:
            print(e)
            print('WARNING: dir file not found: ', data_dir)

    # Create a queue that produces the filenames to read.
    filename_queue = tf.train.string_input_producer(files, shuffle=False)
    # Read examples from files in the filename queue.
    reader = tf.WholeFileReader()
    key, value = reader.read(filename_queue)

    image = tf.image.decode_png(value, channels=IMAGE_CHANNELS)
    image.set_shape([ORIGINAL_IMAGE_HEIGHT, ORIGINAL_IMAGE_WIDTH, IMAGE_CHANNELS])

    image_crop = image[CROP_Y:, CROP_X:ORIGINAL_IMAGE_WIDTH-CROP_X, :]

    resized_image = tf.image.resize_images(image_crop, [IMAGE_HEIGHT, IMAGE_WIDTH])

    image_yuv = tf.py_func(rgb_yuv, [resized_image], [tf.float32])[0]
    image_yuv.set_shape([IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_CHANNELS])

    # Subtract off the mean and divide by the variance of the pixels.
    float_image = tf.image.per_image_whitening(resized_image)

    # Ensure that the random shuffling has good mixing properties.
    #min_fraction_of_examples_in_queue = 0.4
    min_queue_examples = 500

    def fix_label(k):
        k = str(k).split('/')[-1].split(".")[0]
        return float(k)

    label = tf.py_func(fix_label, [key], [tf.double])[0]
    label.set_shape([])

    return _generate_image_and_label_batch(float_image, label,
                                         min_queue_examples, batch_size,
                                         shuffle=False)

