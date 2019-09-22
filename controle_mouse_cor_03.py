# -*- coding: utf-8 -*-
"""
Created on Sat Sep 14 19:20:43 2019

@author: apdde
"""

from cv2 import cv2 as cv
import pyautogui as pag
import numpy as np
import imutils as imt
from collections import deque



#Previne algumas maluquices do pyautogui, mas da uma dor de cabeca
#pag.FAILSAFE = False

#Upper bound das cores detectadas
#Lower bount das cores detectadas
#corUpper = (24, 100, 100)
#corLower = (44, 255, 255)
#
##clique - magenta
#cliqueUpper = (152, 102, 217)
#cliqueLower = (172, 112, 237)
#
##fecha - anil
#fechaUpper = (87, 91, 183)
#fechaLower = (107, 111, 203)

#boundaries
lower = {'amarelo':(24, 100, 100), 'magenta':(152, 102, 217), 'anil':(87, 91, 183)}
upper = {'amarelo':(44, 255, 255), 'magenta':(172, 255, 255), 'anil':(107, 255, 255)}

#Cor padrao dos circulos
colors = {'amarelo': (0, 0, 255), 'magenta':(0, 255, 0), 'anil':(255, 0, 0)}


#numero máximo de pontos. Pesquisar melhor implementacao
comp = 50
pontos = deque(maxlen = comp) 
#pontosC
compC = 50
pontosC = deque(maxlen = compC)
#pontosF
compF = 50
pontosF = deque(maxlen = compF)


#Inicia a camera
cam = cv.VideoCapture(1)


while True:
    
    #Captura o frame atual
    (grabbed, frame) = cam.read()
    
    #Dimensoes da tela
    screen = pag.size()
    wdt = screen[0]
    hgt = screen[1]   
    
    #Aumenta tamanho da janela
    frame = imt.resize(frame, width = wdt, height=int(hgt / 1.5))
       
    #precisar dar flip na imagem
    frame = cv.flip(frame, 1)
    
    blurred = cv.GaussianBlur(frame, (11, 11), 0)
    hsv = cv.cvtColor(blurred, cv.COLOR_BGR2HSV)
    #for each color in dictionary check object in frame
    for key, value in upper.items():
        # construct a mask for the color from dictionary`1, then perform
        # a series of dilations and erosions to remove any small
        # blobs left in the mask
        kernel = np.ones((9,9),np.uint8)
        mask = cv.inRange(hsv, lower[key], upper[key])
        mask = cv.morphologyEx(mask, cv.MORPH_OPEN, kernel)
        mask = cv.morphologyEx(mask, cv.MORPH_CLOSE, kernel)
                
        # find contours in the mask and initialize the current
        # (x, y) center of the ball
        cnts = cv.findContours(mask.copy(), cv.RETR_EXTERNAL,
            cv.CHAIN_APPROX_SIMPLE)[-2]
        center = None
        
        # only proceed if at least one contour was found
        if len(cnts) > 0:
            # find the largest contour in the mask, then use
            # it to compute the minimum enclosing circle and
            # centroid
            c = max(cnts, key=cv.contourArea)
            ((x, y), radius) = cv.minEnclosingCircle(c)
            M = cv.moments(c)
            center = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))
        
            # only proceed if the radius meets a minimum size. Correct this value for your obect's size
            if radius > 0.5:
                # draw the circle and centroid on the frame,
                # then update the list of tracked points
                cv.circle(frame, (int(x), int(y)), int(radius), colors[key], 2)
                pag.moveTo(x = (center[0] - 5), y=(center[1] - 5))
                
                if key == 'magenta':
                    pag.click()
                elif key == 'anil':
                    break
                    
                #cv.putText(frame,key + " ball", (int(x-radius),int(y-radius)), cv2.FONT_HERSHEY_SIMPLEX, 0.6,colors[key],2)  


    
    #Mostra imagem na linha
    cv.imshow("Camera", frame)
    
    key = cv.waitKey(1) & 0xFF
    
    if key == ord("q"):
        break
    

cam.release()
cv.destroyAllWindows()
    
