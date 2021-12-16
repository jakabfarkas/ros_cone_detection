#! /usr/bin/env python

import rospy
from nav_msgs.msg import OccupancyGrid
from std_msgs.msg import UInt16MultiArray
from nav_msgs.msg import Odometry
from sensor_msgs.msg import Joy
import message_filters
from tf.transformations import euler_from_quaternion, quaternion_from_euler
import numpy as np
import cv2
import csv
import math
import time

synchronized = True

#pub = rospy.Publisher('/cone_map', UInt16MultiArray, queue_size=10)
#msg = UInt16MultiArray()

prev_map_time = 0
prev_pos_time = 0
prev_position = np.zeros(3)

cones = []

#first_run = True
#prev_x = 0
#prev_y = 0
#prev_a = 0
#vx = np.zeros(20)
#vy = np.zeros(20)
#va = np.zeros(10)

def draw_line(img, start_x, start_y, angle, length, color, force):
    end_x = int(round( start_x + math.cos(angle) * length ))
    end_y = int(round( start_y + math.sin(angle) * length ))
    if ((start_x >= 0 and start_x < img.shape[1] and start_y >= 0 and start_y < img.shape[0] and
            end_x >= 0 and end_x < img.shape[1] and end_y >= 0 and end_y < img.shape[0]) or force):
        cv2.line(img, (start_x, start_y), (end_x, end_y), color, 1)
        i = True
    else:
        end_x = -1
        end_y = -1
    
    return img, end_x, end_y

def build_map(position, detections): # slam elso bemenet
    global cones
    global synchronized
    global prev_pos_time
    global prev_position
    
    if synchronized:
        pos_time = position.header.stamp
        det_time = detections.header.stamp
        
        #slam = process_slam_input(slam)
        position = process_odometry(position)
        detections = process_detections(detections)

    cam_angle = 1.3439
    max_dist = 4
    max_confidence = 100
    min_cone_distance = 0.6
    im_w_px = 1920
    map_w = 17.2
    cam_dist_A = 115.9
    cam_dist_B = -0.947
    #if (slam is None):
    map_w_px = 384 # elozo sor vegere
    #else: map_w_px = slam.shape[0]
    
    current_map = np.full((map_w_px,map_w_px,3), 255, np.uint8)
    min_cone_distance_px = int(round(min_cone_distance/map_w*map_w_px))
    cone_size_px = int(round(map_w_px/200))
    bot_size_px = int(round(map_w_px/150))
    
    position_copy = np.copy(position)
    if (prev_pos_time != 0):
        pos_time_diff = (pos_time - prev_pos_time).to_sec()
        pos_time_delay = (det_time - pos_time).to_sec()
        position += (position - prev_position) / pos_time_diff * pos_time_delay
    prev_position = position_copy
    prev_pos_time = pos_time
    
    
    
    x = int(round((position[0]/map_w*2)*map_w_px/2+map_w_px/2))
    y = int(round((-position[1]/map_w*2)*map_w_px/2+map_w_px/2))
    a = position[2]
    
    #global prev_x, prev_y, prev_a, vx, vy, va
    #if ('pos_time_diff' in locals()):
    #    vx = np.roll(vx, 1)
    #    vy = np.roll(vy, 1)
    #    va = np.roll(va, 1)
    #    vx[0] = (x - prev_x) / pos_time_diff
    #    vy[0] = (y - prev_y) / pos_time_diff
    #    va[0] = (a - prev_a) / pos_time_diff
        
    #prev_x = x
    #prev_y = y
    #prev_a = a
    
    min_angle = cam_angle/im_w_px*(-im_w_px/2)
    max_angle = cam_angle/im_w_px*(im_w_px/2)
    min_angle -= a
    max_angle -= a
    current_map, xxx1, yyy1 = draw_line(current_map, x, y, min_angle, max_dist/map_w*map_w_px, (169,169,169), True)
    current_map, xxx2, yyy2 = draw_line(current_map, x, y, max_angle, max_dist/map_w*map_w_px, (169,169,169), True)
    #cv2.line(current_map, (xxx1, yyy1), (xxx2, yyy2), (0,0,0), 3)
    axis = int(round( max_dist/map_w*map_w_px))
    cv2.ellipse(current_map, (x,y), (axis,axis), 0, (-a+cam_angle/2)/math.pi*180, (-a-cam_angle/2)/math.pi*180, (169,169,169), 1)
    
    for detection in detections:
        dist = cam_dist_A * math.pow(detection[3], cam_dist_B)
        angle = cam_angle/im_w_px*(detection[1]-im_w_px/2)
        dist /= math.cos(angle)
        angle -= a
        if (dist <= max_dist):
            if (detection[0] == 0): color = (255,102,0)
            if (detection[0] == 1): color = (0,64,255)
            current_map, xc, yc = draw_line(current_map, x, y, angle, dist/map_w*map_w_px, color, False)
            if (xc != -1 and yc != -1):
                #cv2.circle(current_map, (xc,yc), cone_size_px, color, cone_size_px*2)
                new_cone = True
                if (cones):
                    if (isinstance(cones[0], list)):
                        for c in range(len(cones)):
                            if (cones[c][0] == detection[0] and math.sqrt((cones[c][1]-xc)**2 + (cones[c][2]-yc)**2) < min_cone_distance_px):
                                new_cone = False
                                cones[c][1] = int(round( (cones[c][1]*cones[c][3] + xc*2) / (cones[c][3] + 2) ))
                                cones[c][2] = int(round( (cones[c][2]*cones[c][3] + yc*2) / (cones[c][3] + 2) ))
                                if (cones[c][3] < max_confidence): cones[c][3] += 1
                    else:
                        if (cones[0] == detection[0] and math.sqrt((cones[1]-xc)**2 + (cones[2]-yc)**2) < min_cone_distance_px):
                            new_cone = False
                            cones[1] = int(round( (cones[1]*cones[3] + xc*2) / (cones[3] + 2) ))
                            cones[2] = int(round( (cones[2]*cones[3] + yc*2) / (cones[3] + 2) ))
                            if (cones[3] < max_confidence): cones[3] += 1
                if (new_cone):
                    cones.append([detection[0], xc, yc, 1])
    
    #merged_cone = False
    #if (len(cones) > 1): 
    #    for i in range(len(cones)-1):
    #        for j in range(i+1, len(cones)):
    #            if (merged_cone == False):
    #                if (math.sqrt((cones[i][1]-cones[j][1])**2 + (cones[i][2]-cones[j][2])**2) < min_cone_distance_px):
    #                    cones[i][1] = int(round( (cones[i][1]+cones[j][1])/2 ))
    #                    cones[i][2] = int(round( (cones[i][2]+cones[j][2])/2 ))
    #                    cones[i][3] = 5
    #                    #cones = np.delete(cones, j, 0)
    #                    cones = cones.pop(j)
    #                    merged_cone = True
                
    #print(cones)
    if (cones):
        if (isinstance(cones[0], list)):
            for cone in cones:
                if (cone[3] >= 10):
                    if (cone[0] == 0): color = (255,102,0)
                    if (cone[0] == 1): color = (0,64,255)
                    cv2.line(current_map, (cone[1],0), (cone[1],current_map.shape[0]), (0,0,0), 1)
                    cv2.circle(current_map, (cone[1],cone[2]), cone_size_px, color, cone_size_px*3)
        else:
            if (cones[3] >= 10):
                if (cones[0] == 0): color = (255,102,0)
                if (cones[0] == 1): color = (0,64,255)
                cv2.circle(current_map, (cones[1],cones[2]), cone_size_px, color, cone_size_px*2)
    
    #font = cv2.FONT_HERSHEY_SIMPLEX
    #cv2.putText(current_map, 'vx = '+str(round(np.average(vx)/10, 3))+' m/s', (10,25), font, 0.6, (0, 0, 0), 1, cv2.LINE_AA)
    #cv2.putText(current_map, 'vy = '+str(round(np.average(vy)/10, 3))+' m/s', (10,55), font, 0.6, (0, 0, 0), 1, cv2.LINE_AA)
    #cv2.putText(current_map, 'w = '+str(round(np.average(va), 3))+' rad/s', (10,85), font, 0.6, (0, 0, 0), 1, cv2.LINE_AA)
    
    #if (slam is not None):
    #    for i in range(len(current_map)):
    #        for j in range(len(current_map[i])):
    #            if (slam[i][j] == 1): current_map[i][j] = (0,0,0)
                
    cv2.circle(current_map, (x,y), bot_size_px, (0,0,0), bot_size_px*2)
    
    global prev_map_time
    new_map_time = rospy.get_time()
    if (new_map_time-prev_map_time > 0):
        fps = str(round( 1/(new_map_time-prev_map_time), 2 ))
    else:
        fps = 'inf'
    prev_map_time = new_map_time
    log_fps = 'data received and map built, fps:' + fps
    rospy.loginfo(log_fps)
    
    cv2.imshow('map',current_map)
    cv2.waitKey(1)
    
    #cone_map = slam.flatten()
    #msg.data = cone_map
    #pub.publish(msg)
    
def process_slam_input(slam_input):
    slam_input = np.asarray(slam_input.data)
    slam_size = int(math.sqrt(len(slam_input)))
    slam_input = np.reshape(slam_input,(slam_size,slam_size))
    slam_input = np.flip(slam_input, 0)
    slam_input[slam_input < 1] = 0
    slam_input[slam_input == 100] = 1
    slam_input = slam_input.astype(np.uint8)
    
    return slam_input
    
def process_odometry(odom):
    orientation_q = odom.pose.pose.orientation
    orientation_list = [orientation_q.x, orientation_q.y, orientation_q.z, orientation_q.w]
    (roll, pitch, yaw) = euler_from_quaternion (orientation_list)
    position_q = odom.pose.pose.position
    pos = [position_q.x, position_q.y, yaw]
    pos = np.asarray(pos)
    
    return pos
    
def process_detections(det):
    det = np.asarray(det.buttons)
    det = det.reshape(int(len(det)/4),4)
    return det
    
class Server:
    def __init__(self):
        self.slam_input = None
        self.position_input = None
        self.detections_input = None
        
    def slam_callback(self, msg):
        slam_map = np.asarray(msg.data)
        map_size = int(math.sqrt(len(slam_map)))
        slam_map = np.reshape(slam_map,(map_size,map_size))
        slam_map = np.flip(slam_map, 0)
        slam_map[slam_map < 1] = 0
        slam_map[slam_map == 100] = 1
        self.slam_input = slam_map.astype(np.uint8)
        
    def position_callback(self, msg):
        orientation_q = msg.pose.pose.orientation
        orientation_list = [orientation_q.x, orientation_q.y, orientation_q.z, orientation_q.w]
        (roll, pitch, yaw) = euler_from_quaternion (orientation_list)
        position_q = msg.pose.pose.position
        self.position_input = [position_q.x, position_q.y, yaw]
        
    def detections_callback(self, msg):
        self.detections_input = process_detections(msg)
        
        self.computation()
        
        
    def computation(self):
        if (self.position_input is not None and self.detections_input is not None):
            build_map(self.slam_input, self.position_input, self.detections_input)
        
if __name__ == '__main__':
    rospy.init_node('map_builder')
    server = Server()
    
    if synchronized:
        #map_sub = message_filters.Subscriber('/map', OccupancyGrid)
        odom_sub = message_filters.Subscriber('/odom', Odometry)
        det_sub = message_filters.Subscriber('/detections', Joy)
        ts = message_filters.ApproximateTimeSynchronizer([odom_sub, det_sub], 100000, 5) # tombben elso a map_sub
        ts.registerCallback(build_map)
    else:
        rospy.Subscriber('/map', OccupancyGrid, server.slam_callback)
        rospy.Subscriber('/odom', Odometry, server.position_callback)
        rospy.Subscriber('/detections', Joy, server.detections_callback)
    
    rospy.spin()
    cv2.destroyAllWindows()
    
    
    
    
