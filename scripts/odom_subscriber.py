#! /usr/bin/env python

import rospy
from nav_msgs.msg import Odometry
from tf.transformations import euler_from_quaternion, quaternion_from_euler
import time
import csv

prev_frame_time = 0

def callback(msg):

    roll = pitch = yaw = 0.0
    orientation_q = msg.pose.pose.orientation
    orientation_list = [orientation_q.x, orientation_q.y, orientation_q.z, orientation_q.w]
    (roll, pitch, yaw) = euler_from_quaternion (orientation_list)
    
    position_q = msg.pose.pose.position
    
    pos = [position_q.x, position_q.y, yaw]
    
    print(pos)
    
    global prev_frame_time
    new_frame_time = rospy.get_time()
    if (new_frame_time-prev_frame_time > 0):
        fps = str(round( 1/(new_frame_time-prev_frame_time), 2 ))
    else:
        fps = 'inf'
    prev_frame_time = new_frame_time
    logfps = 'odom rate: '+fps+' Hz'
    rospy.loginfo(logfps)
    
rospy.init_node('odom')
sub = rospy.Subscriber('/odom', Odometry, callback)
rospy.spin()
