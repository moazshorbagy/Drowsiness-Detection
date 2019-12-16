# For more info: http://docs.opencv.org/3.0-beta/doc/py_tutorials/py_gui/py_video_display/py_video_display.html
import cv2
import numpy as np
from skimage.color import rgb2ycbcr
from skimage.morphology.selem import disk
# from commonfunctions import *
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
clahe = cv2.createCLAHE(clipLimit=3.0)

se = disk(5)

cap = cv2.VideoCapture(0)

def dist(x1, y1, x2, y2):
    return np.sqrt((x1-x2)**2 + (y1-y2)**2)

while(True):
    
    _, frame = cap.read()
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)

    faces = face_cascade.detectMultiScale(gray, 1.15, 5)

    for (x, y, w, h) in faces:
        roi_color = frame[y : y + h, x : x + w]
        roi_gray = gray[y : y + h, x : x + w]

        face_ycbcr = rgb2ycbcr(roi_color)
        Y = face_ycbcr[:,:,0]
        Cb = face_ycbcr[:,:,1]
        Cr = face_ycbcr[:,:,2]

        c1 = np.power(Cb, 2)
        c1Min = np.min(c1)
        c1 = (c1 - c1Min) / (np.max(c1) - c1Min)

        c2 = np.power(np.max(Cr) - Cr, 2)
        c2Min = np.min(c2)
        c2 = (c2 - c2Min) / (np.max(c2) - c2Min)
        
        c3 = Cb / Cr
        c3Min = np.min(c3)
        c3 = (c3 - c3Min) / (np.max(c3) - c3Min)
        
        EyeMapC = ((c1 + c2 + c3)/3)
        EyeMapC = clahe.apply((EyeMapC*255).astype(np.uint8))/255

        dilated = cv2.dilate(Y, se)
        eroded = cv2.erode(Y, se)

        EyeMapL = dilated / eroded
        EyeMapLMin = np.min(EyeMapL)
        EyeMapL = (EyeMapL - EyeMapLMin) / (np.max(EyeMapL) - EyeMapLMin)
        cv2.imshow('EyeMapL', EyeMapL)
        
        EyeMap = EyeMapC * EyeMapL
        EyeMap = cv2.erode(EyeMap, se, iterations = 2)
        EyeMap = clahe.apply((EyeMap*255).astype(np.uint8))/255
        EyeMap = cv2.dilate(EyeMap, se, iterations = 2)
        EyeMap = cv2.erode(EyeMap, se, iterations = 2)

        threshold = 0.85 * np.max(EyeMap) + 0.8
        eyeNotFound = True
        for shit in range(1):
            threshold -= 0.8
            EyeMapD = EyeMap.copy()
            EyeMapD[EyeMapD >= threshold] = 1
            EyeMapD[EyeMapD < threshold] = 0
            
            center = ((int)(EyeMapD.shape[0]/2), (int)(EyeMapD.shape[1]/2))
            eye1 = (0, 0)
            eye2 = (0, 0)
            r1 = range(30, int(EyeMapD.shape[0]/2))
            r2 = range(30, int(EyeMapD.shape[1]/2))
            r4 = range(int(EyeMapD.shape[1]/2), EyeMapD.shape[1])
            for i1 in r1:
                if(not eyeNotFound):
                    break
                for j1 in r2:
                    if(not eyeNotFound):
                        break
                    if(EyeMapD[i1, j1] == 1):
                        for i2 in r1:
                            if(not eyeNotFound):
                                break
                            for j2 in range(int(EyeMapD.shape[1]/2), EyeMapD.shape[1], 1):
                                if(EyeMapD[i2, j2] == 1 and not (i1 == i2 and j1 == j2)):
                                    distCandidate1 = dist(i1, j1, center[0], center[1])
                                    distCandidate2 = dist(i2, j2, center[0], center[1])
                                    if(distCandidate1 != 0):
                                        if(np.abs((distCandidate1 - distCandidate2) / distCandidate1) < 0.3):
                                            #eyesDist = dist(i1, j1, i2, j2)
                                            #if(eyesDist > distCandidate1 and eyesDist > distCandidate2):
                                                eye1 = (i1, j1)
                                                eye2 = (i2, j2)
                                                eyeNotFound = False
                                                break
            if(not eyeNotFound):
                break
        
        roi_color = cv2.rectangle(roi_color, (eye1[1]-15, eye1[0]-15), (eye1[1]+15, eye1[0]+15), (0, 255, 0), 2)
        roi_color = cv2.rectangle(roi_color, (eye2[1]-15, eye2[0]-15), (eye2[1]+15, eye2[0]+15), (0, 255, 0), 2)
        cv2.imshow('EyeMap', EyeMap)
        if(eyeNotFound):
            zeros = np.zeros(roi_color.shape)
            cv2.imshow('frame', zeros)
        else:
            cv2.imshow('frame', roi_color)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()