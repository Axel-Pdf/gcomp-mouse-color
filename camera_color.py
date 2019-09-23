# -*- coding: utf-8 -*-
"""
Created on Mon Sep  9 19:19:18 2019

@author: apdde
"""

import cv2
import numpy as np

def nothing():
    pass

h = 0
s = 0
v = 0

cv2.cv2.namedWindow("Camera")
cv2.cv2.createTrackbar('h', 'Camera', 0, 255, nothing)
cv2.cv2.createTrackbar('s', 'Camera', 0, 255, nothing)
cv2.cv2.createTrackbar('v', 'Camera', 0, 255, nothing)


camera = cv2.VideoCapture(1)  #Vale url do droidcam



while True:
    _, frame = camera.read(0)
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    h = cv2.getTrackbarPos('h', 'Camera')
    s = cv2.getTrackbarPos('s', 'Camera')
    v = cv2.getTrackbarPos('v', 'Camera')
    
    lower = np.array([h, s, v])
    upper = np.array([255, 255, 255])
    
    mascara = cv2.inRange(hsv, lower, upper)
    resultado = cv2.bitwise_and(frame, frame, mask = mascara)
    
    
    
    
    
    cv2.imshow("Camera", mascara)
    
    
    k = cv2.waitKey(50)
    if k == 27:
        break



cv2.destroyAllWindows()
camera.release()