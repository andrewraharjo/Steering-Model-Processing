#!/usr/bin/env python
# -*- coding: utf-8 -*-

import rosbag
import csv
import string
import argparse

def main():
    parser = argparse.ArgumentParser(description="Extract data from a ROS bag.")
    parser.add_argument("bag_file", help="Input ROS bag.")
    parser.add_argument("output_file", help="Output file.")
    parser.add_argument("topic", help="Topic.")

    args = parser.parse_args()

    print "Extract data from %s on topic %s into %s" % (args.bag_file, args.topic, args.output_file)

    bag = rosbag.Bag(args.bag_file, "r")

    with open(args.output_file, 'wb') as fp:
        filewriter = csv.writer(fp, delimiter=',')

        firstIteration = True	#allows header row

        for topic, msg, t in bag.read_messages(topics=[args.topic]):
            msgString = str(msg)
            msgList = string.split(msgString, '\n')
            instantaneousListOfData = []
            for nameValuePair in msgList:
                splitPair = string.split(nameValuePair, ':')
                for i in range(len(splitPair)):	#should be 0 to 1
                    splitPair[i] = string.strip(splitPair[i])
                instantaneousListOfData.append(splitPair)

            #write the first row from the first element of each pair
            if firstIteration:	# header
                headers = ["rosbagTimestamp"]	#first column header
                for pair in instantaneousListOfData:
                    headers.append(pair[0])
                filewriter.writerow(headers)
                firstIteration = False

            # write the value from each pair to the file
            values = [str(t)]	#first column will have rosbag timestamp
            for pair in instantaneousListOfData:
                if len(pair) > 1:
                    values.append(pair[1])
            filewriter.writerow(values)

        bag.close()

    return

if __name__ == '__main__':
    main()
