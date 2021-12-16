#!/usr/bin/env python3

import rospy
from sensor_msgs.msg import Image
from std_msgs.msg import UInt16MultiArray
from sensor_msgs.msg import Joy
from cv_bridge import CvBridge
import cv2
import numpy as np
import time


net = cv2.dnn.readNetFromDarknet('darknet/cfg/yolov4-tiny-custom.cfg', 'darknet/yolov4-tiny-custom_best.weights')
model = cv2.dnn_DetectionModel(net)
model.setInputParams(scale=1/255, size=(416, 416), swapRB=False)

pub = rospy.Publisher('/detections', Joy, queue_size=10)
msg = Joy()
total_published_msgs = 0

prev_frame_time = 0

def thresholds(classId):
    if (classId == 0):
        lower_1 = np.array([110, 80, 60], np.uint8)
        upper_1 = np.array([160, 255, 255], np.uint8)
        lower_2 = np.array([0, 0, 0], np.uint8)
        upper_2 = np.array([100, 100, 100], np.uint8)
    if (classId == 1):
        lower_1 = np.array([10, 30, 40], np.uint8)
        upper_1 = np.array([20, 255, 255], np.uint8)
        lower_2 = np.array([0, 100, 40], np.uint8)
        upper_2 = np.array([10, 255, 255], np.uint8)
        
    return lower_1, upper_1, lower_2, upper_2
    
def cut_points(frame, x, y, w, h):
    if (x-w*0.2 > 0): x1 = int(round(x-w*0.2))
    else: x1 = 0
    if (y-h*0.2 > 0): y1 = int(round(y-h*0.2))
    else: y1 = 0
    if (x+w*1.2 < frame.shape[1]): x2 = int(round(x+w*1.2))
    else: x2 = frame.shape[1]
    if (y+h*1.2 < frame.shape[0]): y2 = int(round(y+h*1.2))
    else: y2 = frame.shape[0]
    
    return x1, y1, x2, y2
    
def mask_function(img, lower, upper, kernel_size):
    hsv = img.copy()
    hsv = cv2.cvtColor(hsv, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, lower, upper)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size,kernel_size))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size,kernel_size))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    
    return mask
    
def max_area(contours):
    max_area = 0
    for c in contours:
        if(cv2.contourArea(c) > max_area):
            max_area = cv2.contourArea(c)
    
    return max_area
    
def image_overlap(img_1, img_2):
    img_overlap = np.where((img_1>0) & (img_2>0), 255, 0)

    sum_x = 0
    sum_y = 0
    px_cnt = 0
    for i in range(img_1.shape[0]):
        for j in range(img_1.shape[1]):
            if (img_1[i][j] > 0 and img_2[i][j] > 0):
                sum_x = sum_x + j
                sum_y = sum_y + i
                px_cnt += 1
    if (px_cnt > 0):
        xc = int(round(sum_x/px_cnt))
        yc = int(round(sum_y/px_cnt))
    else:
        xc = -1
        yc = -1
        
    return img_overlap, xc, yc
    
def get_contour_sizes(contour):
    contour_sizes = []
    for c in contour:
        _,_,w,_ = cv2.boundingRect(c)
        contour_sizes.append(w)
    
    return contour_sizes
    
def filter_lines(line_contours, img_overlap, max_dev):
    line_sizes = get_contour_sizes(line_contours)
    largest_line_index = line_sizes.index(max(line_sizes))
    M = cv2.moments(line_contours[largest_line_index])
    largest_line_center = int(round(M["m10"] / M["m00"]))
    deviation = []
    for i in range(len(line_contours)):
        M = cv2.moments(line_contours[i])
        if (M["m00"] > 0):
            center_x = M["m10"] / M["m00"]
            deviation.append([abs(largest_line_center - center_x), i])
    deviation.sort()
    filtered_lines = []
    sum_x = 0
    sum_y = 0
    sum_area = 0
    if (len(deviation) >= 4):
        for i in range(4):
            if (deviation[i][0] < max_dev):
                M = cv2.moments(line_contours[deviation[i][1]])
                sum_x += M["m10"]
                sum_y += M["m01"]
                sum_area += M["m00"]
                if (img_overlap[int(round(M["m01"]/M["m00"]))][int(round(M["m10"]/M["m00"]))] == 255):
                    filtered_lines.append(line_contours[deviation[i][1]])
        if (sum_area > 0):
            xc = int(round( sum_x / sum_area ))
            yc = int(round( sum_y / sum_area ))
        else:
            xc = -1
            yc = -1
    else:
        xc = -1
        yc = -1
            
    return filtered_lines, xc, yc
    
def add_dots(cut_cone, x1, y1, line_contours, dot_list):
    max_y = 0
    min_y = cut_cone.shape[1]
    new_dot_cnt = 0
    for c in line_contours:
        left = tuple(c[c[:,:,0].argmin()][0])[0]
        right = tuple(c[c[:,:,0].argmax()][0])[0]
        top = tuple(c[c[:,:,1].argmin()][0])[1]
        bottom = tuple(c[c[:,:,1].argmax()][0])[1]
        dot_1_x = int(left + (bottom-top)/2)
        dot_1_y = int(top + (bottom-top)/2)
        dot_list.append([x1+dot_1_x,y1+dot_1_y,-1])
        #cv2.circle(frame, (x1+dot_1_x,y1+dot_1_y), dot_size, (0,255,0), dot_size*2)
        dot_2_x = int(right - (bottom-top)/2)
        dot_2_y = int(bottom - (bottom-top)/2)
        dot_list.append([x1+dot_2_x,y1+dot_2_y,-1])
        #cv2.circle(frame, (x1+dot_2_x,y1+dot_2_y), dot_size, (0,255,0), dot_size*2)
        if ((dot_1_y+dot_2_y)/2 < min_y):
            min_y = (dot_1_y+dot_2_y)/2
        if ((dot_1_y+dot_2_y)/2 > max_y):
            max_y = (dot_1_y+dot_2_y)/2
        new_dot_cnt += 2
    height_in_pixels = int((max_y-min_y)/2)
    height = height_in_pixels
    for i in range(len(dot_list)-new_dot_cnt, len(dot_list)):
        dot_list[i][2] = height
    
    return dot_list, height

def callback(img):
    cv_br = CvBridge()
    frame = cv_br.imgmsg_to_cv2(img, "bgr8")
    
    global total_published_msgs
    msg.header.seq = total_published_msgs
    msg.header.stamp = rospy.Time.now()
    msg.header.frame_id = 'det'
    total_published_msgs += 1
    
    classIds, scores, boxes = model.detect(frame, confThreshold=0.3, nmsThreshold=0.2)
    
    detection_list = []
    dot_list = []
     
    for (classId, score, box) in zip(classIds, scores, boxes):
        
        lower_1, upper_1, lower_2, upper_2 = thresholds(classId)
            
        x = box[0]
        y = box[1]
        w = box[2]
        h = box[3]
        x1, y1, x2, y2 = cut_points(frame, x, y, w, h)
    
        #cv2.rectangle(frame, (x1, y1), (x2, y2), color=rect_color, thickness=5)
        cut_cone = frame[y1:y2, x1:x2]
        
        kernel_size = int(cut_cone.shape[0]/30)
        mask_1 = mask_function(cut_cone, lower_1, upper_1, kernel_size)
        mask_2 = mask_function(cut_cone, lower_2, upper_2, kernel_size)
        contour_areas_1, _ = cv2.findContours(mask_1, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        contour_areas_2, _ = cv2.findContours(mask_2, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        if (len(contour_areas_1) >= 2 and len(contour_areas_2) >= 3):
            max_contour_area = max_area(contour_areas_1)
            cnt_thickness = int(round(max_contour_area/500))
            contour_areas_im_1 = np.zeros((cut_cone.shape[0],cut_cone.shape[1]), np.uint8)
            contour_areas_im_2 = np.zeros((cut_cone.shape[0],cut_cone.shape[1]), np.uint8)
            cv2.drawContours(contour_areas_im_1, contour_areas_1, contourIdx=-1, color=255, thickness=cnt_thickness, lineType=cv2.LINE_AA)
            cv2.drawContours(contour_areas_im_2, contour_areas_2, contourIdx=-1, color=255, thickness=cnt_thickness, lineType=cv2.LINE_AA)
            img_overlap = np.where((contour_areas_im_1>0) & (contour_areas_im_2>0), 255, 0).astype(np.uint8)
            line_contours, _ = cv2.findContours(img_overlap, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            if (len(line_contours) >= 4):
                filtered_line_contours, xc, yc = filter_lines(line_contours, img_overlap, cut_cone.shape[1]/10)
                if (len(filtered_line_contours) == 4):
                    dot_list, height = add_dots(cut_cone, x1, y1, filtered_line_contours, dot_list)
                    detection_list.append([classId, x1+xc, y1+yc, height])
            #else:
                #detection_list.append([classId, x+w/2, y+h/2, h])
    
    for cone in detection_list:
        if (cone[0] == 0):
            rect_color = (255, 0, 0)
        if (cone[0] == 1):
            rect_color = (0, 216, 255)
        x = cone[1]
        y = cone[2]
        x1 = int(round((x)-cone[3]*0.8))
        y1 = int(round((y)-cone[3]*1.5))
        x2 = int(round((x)+cone[3]*0.8))
        y2 = int(round((y)+cone[3]*1.5))
        rect_thickness = int(round(cone[3]/20))
        cv2.rectangle(frame, (x1, y1), (x2, y2), color=rect_color, thickness=rect_thickness)
    
    for dot in dot_list:
        dot_size = int(round(dot[2]/20))
        cv2.circle(frame, (dot[0],dot[1]), dot_size, (0,255,0), dot_size*2)
    
    detection_list = np.array(detection_list)
    detection_list = detection_list.flatten()
    msg.buttons = detection_list
    pub.publish(msg)
    
    scale_percent = 50
    width = int(frame.shape[1] * scale_percent / 100)
    height = int(frame.shape[0] * scale_percent / 100)
    dim = (width, height)
    frame = cv2.resize(frame, dim, interpolation = cv2.INTER_AREA)
    cv2.imshow("camera_detected", frame)
    cv2.waitKey(1)
    
    global prev_frame_time
    new_frame_time = rospy.get_time()
    if (new_frame_time-prev_frame_time > 0):
        fps = str(round( 1/(new_frame_time-prev_frame_time), 2 ))
    else:
        fps = 'inf'
    prev_frame_time = new_frame_time
    log_fps = 'received and detected frame, fps:' + fps
    rospy.loginfo(log_fps)
    

#def receive_message():
rospy.init_node('cone_detector', anonymous=True)
#rate = rospy.Rate(5)
rospy.Subscriber('/camera/rgb/image_raw', Image, callback)
rospy.spin()
cv2.destroyAllWindows()


#if __name__ == '__main__':
    #receive_message()

