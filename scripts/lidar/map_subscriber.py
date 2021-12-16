#! /usr/bin/env python

import rospy
from nav_msgs.msg import OccupancyGrid
import numpy as np
import csv

def callback(msg):
    rospy.loginfo("receiving and saving map")
    output = np.reshape(msg.data,(384,384))
    with open(r'/home/jakab/catkin_ws/src/my_program_package/tmp/map.txt', 'w') as f: 
        write = csv.writer(f)
        write.writerows(output)
        

rospy.init_node('map')
sub = rospy.Subscriber('/map', OccupancyGrid, callback)
rospy.spin()
