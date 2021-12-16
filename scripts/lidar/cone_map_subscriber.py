#! /usr/bin/env python

import rospy
from std_msgs.msg import UInt16MultiArray
import numpy as np
import csv
import cv2
import math

def callback(msg):
    cone_map = np.asarray(msg.data)
    cone_map = cone_map.astype(np.uint8)
    map_size = int(math.sqrt(len(cone_map)))
    cone_map = cone_map.reshape(map_size,map_size)
    cv2.imshow('cone_map',cone_map)
    cv2.waitKey(1)
        
if __name__ == '__main__':
    rospy.init_node('cone_map_visualizer')
    sub = rospy.Subscriber('/cone_map', UInt16MultiArray, callback)
    rospy.spin()
