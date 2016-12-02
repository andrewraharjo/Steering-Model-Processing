#!/usr/bin/env python
# -*- coding: utf-8 -*-

import rosbag
import csv
import argparse

def main():
    parser = argparse.ArgumentParser(description="Extract timestamps from a ROS bag.")
    parser.add_argument("bag_file", help="Input ROS bag.")
    parser.add_argument("output_file", help="Output file.")
    parser.add_argument("topic", help="Topic.")

    args = parser.parse_args()

    print "Extract timestamps from %s on topic %s into %s" % (args.bag_file, args.topic, args.output_file)

    bag = rosbag.Bag(args.bag_file, "r")

    with open(args.output_file, 'wb') as fp:
        a = csv.writer(fp, delimiter=',')

        for topic, msg, t in bag.read_messages(topics=[args.topic]):
            #print t
            a.writerow([t])

        bag.close()

	return

if __name__ == '__main__':
    main()
