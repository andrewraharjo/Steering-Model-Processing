#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Copyright 2016 Massachusetts Institute of Technology

"""Extract images from a rosbag.
"""

import os
import argparse
import numpy as np

import cv2

import rosbag
from sensor_msgs.msg import Image
from cv_bridge import CvBridge

def main():
    parser = argparse.ArgumentParser(description="Extract images from a ROS bag.")
    parser.add_argument("bag_file", help="Input ROS bag.")
    parser.add_argument("output_dir", help="Output directory.")
    parser.add_argument("image_topic", help="Image topic.")

    args = parser.parse_args()

    print "Extract images from %s on topic %s into %s" % (args.bag_file,
                                                          args.image_topic, args.output_dir)

    bag = rosbag.Bag(args.bag_file, "r")
    bridge = CvBridge()
    count = 0
    img_format = 'png'

    for topic, msg, t in bag.read_messages(topics=[args.image_topic]):

        if hasattr(msg, 'format') and 'compressed' in msg.format:

            buf = np.ndarray(shape=(1, len(msg.data)), dtype=np.uint8, buffer=msg.data)
            cv_img = cv2.imdecode(buf, cv2.IMREAD_ANYCOLOR)

            if cv_img.shape[2] != 3:
                print("Invalid image")
                return

            # Avoid re-encoding if we don't have to
            #if check_image_format(msg.data) == self.write_img_format:
            #    encoded = buf
            #else:
            #_, encoded = cv2.imencode('.' + img_format, cv_img)

        else:
            #cv_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
            cv_img = bridge.imgmsg_to_cv2(msg, desired_encoding="passthrough")
            #_, encoded = cv2.imencode('.' + img_format, cv_img)

        cv2.imwrite(os.path.join(args.output_dir, "frame%06i.png" % count), cv_img)
        print "Wrote image %i" % count

        count += 1

    bag.close()

    return

if __name__ == '__main__':
    main()
