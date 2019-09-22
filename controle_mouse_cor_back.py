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

#movimento - amarelo
corUpper = (24, 100, 100)
corLower = (44, 120, 120)

#clique - magenta
cliqueUpper = (152, 102, 217)
cliqueLower = (172, 112, 237)

#fecha - anil
fechaUpper = (87, 91, 183)
fechaLower = (107, 111, 203)

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
#    resScreen = [wdt, hgt]
#    resImage = frame.shape
#    resImage = [640, 480]
#   resScreen = [1920, 1080]
    
    
    
    #Aumenta tamanho da janela
    frame = imt.resize(frame, width = wdt, height=int(hgt / 1.5))
    
    #Não precisar dar flip na imagem, mas fica ai
    #frame = frame.rotate(frame, angle=180)
    
    #precisar dar flip na imagem
    frame = cv.flip(frame, 1)
    
    #blur = cv.GaussianBlurr(frame, (11, 11), 0)
    hsv = cv.cvtColor(frame, cv.COLOR_BGR2HSV)
    
    
    #constroi a marcara com a cor designada e um lower bound
    #remove marcas ou bolhas na deteccao ocm dilataçoes e erosoes de imagem
    mascara = cv.inRange(hsv, corUpper, corLower)
    mascara = cv.erode(mascara, None, iterations=2)
    mascara = cv.dilate(mascara, None, iterations=2)
    
    #Mesmo processo para elementos de clique e fechamento
    mascaraClique = cv.inRange(hsv, cliqueUpper, cliqueLower)
    mascaraClique = cv.erode(mascaraClique, None, iterations=2)
    mascaraClique = cv.dilate(mascaraClique, None, iterations=2)
#    
    mascaraFecha = cv.inRange(hsv, fechaUpper, fechaLower)
    mascaraFecha = cv.erode(mascaraFecha, None, iterations=2)
    mascaraFecha = cv.dilate(mascaraFecha, None, iterations=2)
    
    contornosC = cv.findContours(mascaraClique.copy(), cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)[-2]
    
    contornosF = cv.findContours(mascaraFecha.copy(), cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)[-2]
    
    #encontra contornos na mascara e inicializa
    #(x, y) 
    contornos = cv.findContours(mascara.copy(), cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)[-2]
    centro = None
    
    centroC = None
    
    centroF = None
    
    #continua apenas se foi encontrado um contorno
    if len(contornos) > 0:
        #encontra maior contorno na mascara, e usa para
        #computar circundacao minima e ponto centro
        c = max(contornos, key=cv.contourArea)
        ((x, y), raio) = cv.minEnclosingCircle(c)
        M = cv.moments(c)
        centro = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))
        
        #procede apenas se raio tiver um tamanho minimo
        if raio > 10:
            #Desenha circulo e centroide no frame
            #Atualiza a lista de pontos rastreados
            cv.circle(frame, (int(x), int(y)), int(raio), (0, 255, 255), 2)
            
            cv.circle(frame, centro, 5, (0, 0, 255), -1)
            
            
    #atualiza fila de pontos
    pontos.appendleft(centro)
    
    
    #percorre setn de pontos e, havendo, s=desenha linha
    for i in range(1, len(pontos)):
        
        if pontos[i -1] is None or pontos[i] is None:
            continue
        
        
        grossLinha = int(np.sqrt(comp / float(i + 1)) * 2.5)
        cv.line(frame, pontos[i - 1], pontos[i], (0, 0, 255), grossLinha)



    #continua se o contorno do clique for encontrado
    if len(contornosC) > 0:
        
        cC = max(contornosC, key=cv.contourArea)
        ((w, z), raioC) = cv.minEnclosingCircle(cC)
        Mc = cv.moments(cC)
        centroC = (int(Mc["m10"] / Mc["m00"]), int(Mc["m01"] / Mc["m00"]))
        
        if raioC > 10:
            #Desenha circulo e centroide no frame
            #Atualiza a lista de pontos rastreados
            cv.circle(frame, (int(w), int(z)), int(raioC), (0, 255, 255), 2)
            
            cv.circle(frame, centroC, 5, (0, 0, 255), -1)

    pontosC.appendleft(centroC)       
        #percorre setn de pontos e, havendo, s=desenha linha
    
    for j in range(1, len(pontosC)):
        
        if pontos[j -1] is None or pontosC[j] is None:
            continue
        
        
        grossLinhaC = int(np.sqrt(compC / float(j + 1)) * 2.5)
        cv.line(frame, pontosC[j - 1], pontosC[j], (0, 0, 255), grossLinhaC)
        
        
            
    #continua se o contorno do clique for encontrado
    if len(contornosF) > 0:
        
        cF = max(contornosF, key=cv.contourArea)
        ((a, b), raioF) = cv.minEnclosingCircle(cF)
        Mf = cv.moments(cF)
        centroF = (int(Mf["m10"] / Mf["m00"]), int(Mf["m01"] / Mf["m00"]))
        
        if raioF > 10:
            #Desenha circulo e centroide no frame
            #Atualiza a lista de pontos rastreados
            cv.circle(frame, (int(a), int(b)), int(raioF), (0, 255, 255), 2)
            
            cv.circle(frame, centroF, 5, (0, 0, 255), -1)


    pontosF.appendleft(centroF)
    
    for l in range(1, len(pontosF)):
        
        if pontosF[l -1] is None or pontosF[l] is None:
            continue
        
        grossLinhaF = int(np.sqrt(compF / float(l + 1)) * 2.5)
        cv.line(frame, pontosF[l - 1], pontosF[l], (0, 0, 255), grossLinhaF)





        
    #Move mouse usando centro
    if centro is not None:
        pag.moveTo(x = (centro[0] - 3), y=(centro[1] - 3))
    
    
    #clique se cor aparece para a camera

    if len(contornosC) > 0:
        #pag.click()
        print("COR 1")
        
    #encerra programa se cor aparece para a cameraqq
    #if contornosF is not None:
#        
    if len(contornosF) > 0:
        #break
        print("COR 2")
##    
    #Mostra imagem na linha
    cv.imshow("Camera", frame)
    
    key = cv.waitKey(1) & 0xFF
    
    if key == ord("q"):
        break
    

cam.release()
cv.destroyAllWindows()
    
