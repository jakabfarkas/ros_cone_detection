#!/usr/bin/env python3

import rospy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
import numpy as np
import time

prev_frame_time = 0
new_frame_time = 0

n_fps = 0
sum_fps = 0

def callback(img):
    cv_br = CvBridge()
    frame = cv_br.imgmsg_to_cv2(img, "bgr8")
    
    scale_percent = 50
    width = int(frame.shape[1] * scale_percent / 100)
    height = int(frame.shape[0] * scale_percent / 100)
    dim = (width, height)
    frame = cv2.resize(frame, dim, interpolation = cv2.INTER_AREA)
    cv2.imshow("camera", frame)
    cv2.waitKey(1)
    
    global prev_frame_time
    new_frame_time = rospy.get_time()
    if (new_frame_time-prev_frame_time > 0):
        fps = str(round( 1/(new_frame_time-prev_frame_time), 2 ))
    else:
        fps = 'inf'
    prev_frame_time = new_frame_time
    logfps = 'camera fps: ' + fps
    rospy.loginfo(logfps)
    
    global n_fps, sum_fps
    sum_fps += int(float(fps))
    n_fps += 1
    if (n_fps > 100):
        print(sum_fps / n_fps)
    
    

def receive_message():
    rospy.init_node('cone_detector', anonymous=True)
    #rate = rospy.Rate(5)
    rospy.Subscriber('/camera/rgb/image_raw', Image, callback)
    rospy.spin()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    receive_message()

