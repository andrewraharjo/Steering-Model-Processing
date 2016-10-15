# SelfSteeringModel_Challenge
Repo for Predicting Steering Wheel

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

TBD

## Model definition and training

NVIDIA_Steering.ipynb