# Repo for Predicting Steering Wheel

TensorFlow implementation based on [End to End Learning for Self-Driving Cars](https://arxiv.org/abs/1604.07316).

# Preprocessing
input.ipynb

# Model training
nvidia_steering.ipynb

### Extracting camera images - Tools

```
./bag_to_images.py dataset.bag right_camera/ /right_camera/image_color
./bag_to_images.py dataset.bag left_camera/ /left_camera/image_color
./bag_to_images.py dataset.bag center_camera/ /center_camera/image_color
```

### Extracting camera timestamps - Tools

```
./camera_timestamps.py dataset.bag timestamps-center.csv /center_camera/image_color
./camera_timestamps.py dataset.bag timestamps-left.csv /left_camera/image_color
./camera_timestamps.py dataset.bag timestamps-right.csv /right_camera/image_color
```

### Extracting steering angles - Tools

This script can extract any topic data to csv.

```
./bag_to_csv.py dataset.bag steering.csv /vehicle/steering_report
```

## Data preprocessing - Tools

Image resizing, pickling and steering interpolation is implemented in steering_input.ipynb

## Data augmentation

Generating random, labeled frames from original frames:
```python
import augmentation as aug

transformed_image, new_steering_wheel_angle, rotation, shift = aug.steer_back_distortion(
                                                                    image, 
                                                                    steering_wheel_angle, 
                                                                    speed)
```
Generating distorted images:
```python
import augmentation as aug
rotation = 0.01 # radians
shift = 0.5 # meters
distorted = aug.apply_distortion(img, rotation, shift)
```

## Model definition and training

NVIDIA_Steering.ipynb