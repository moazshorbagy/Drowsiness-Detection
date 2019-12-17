# For more info: http://docs.opencv.org/3.0-beta/doc/py_tutorials/py_gui/py_video_display/py_video_display.html
import cv2
import numpy as np
from skimage.color import rgb2ycbcr, rgb2gray, rgb2hsv, hsv2rgb
from skimage.exposure import adjust_gamma, is_low_contrast, rescale_intensity

from skimage.morphology.selem import disk
from commonfunctions import *
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

se = disk(5)
se[4, 5] = 0
se[5, [4, 5, 6]] = 0
se[6, 5] = 0
print(se)

frame = io.imread('moaz.png')

# Our operations on the frame come here
gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)

faces = face_cascade.detectMultiScale(gray, 1.15, 5)

def dist(x1, y1, x2, y2):
    return np.sqrt((x1-x2)**2 + (y1-y2)**2)

for (x, y, w, h) in faces:
    roi_color = frame[y : y + h, x : x + w]

    roi_color = adjust_gamma(roi_color, 0.85)

    face_ycbcr = rgb2ycbcr(roi_color)
    Y = face_ycbcr[:,:,0]
    Cb = face_ycbcr[:,:,1]
    Cr = face_ycbcr[:,:,2]
    #show_images([roi_color, Y, Cb, Cr], ['Face', 'Y', 'Cb', 'Cr'])

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
    EyeMapCMin = np.min(EyeMapC)
    EyeMapC = (EyeMapC - EyeMapCMin) / (np.max(EyeMapC) - EyeMapCMin)

    dilated = cv2.dilate(Y, se)
    eroded = cv2.erode(Y, se)

    EyeMapL = dilated / (eroded)
    show_images([EyeMapL], ['EyeMapL'])
    EyeMapLMin = np.min(EyeMapL)
    EyeMapL = (EyeMapL - EyeMapLMin) / (np.max(EyeMapL) - EyeMapLMin)

    
    EyeMap = EyeMapC * EyeMapL
    EyeMap = cv2.erode(EyeMap, se, iterations = 2)
    EyeMap = cv2.dilate(EyeMap, se, iterations = 2)
    EyeMap = cv2.erode(EyeMap, se, iterations = 2)
    EyeMapMin = np.min(EyeMap)
    EyeMap = (EyeMap - EyeMapMin) / (np.max(EyeMap) - EyeMapMin)

    threshold = 0.7 * np.max(EyeMap) + 0.1
    eyeNotFound = True
    while eyeNotFound:
        threshold -= 0.1
        EyeMapD = EyeMap.copy()
        EyeMapD[EyeMapD >= threshold] = 1
        EyeMapD[EyeMapD < threshold] = 0
        
        center = ((int)(EyeMapD.shape[0]/2), (int)(EyeMapD.shape[1]/2))
        eye1 = (0, 0)
        eye2 = (0, 0)
        for i1 in range(int(EyeMapD.shape[0]/2)):
            for j1 in range(int(EyeMapD.shape[1]/2)):
                if(EyeMapD[i1, j1] == 1):
                    for i2 in range(int(EyeMapD.shape[0]/2)):
                        for j2 in range(int(EyeMapD.shape[1]/2), EyeMapD.shape[1]):
                            if(EyeMapD[i2, j2] == 1 and not (i1 == i2 and j1 == j2)):
                                distCandidate1 = dist(i1, j1, center[0], center[1])
                                distCandidate2 = dist(i2, j2, center[0], center[1])
                                if(distCandidate1 != 0):
                                    if(np.abs((distCandidate1 - distCandidate2) / distCandidate1) < 0.3):
                                        eye1 = (i1, j1)
                                        eye2 = (i2, j2)
                                        eyeNotFound = False


    roi_color = cv2.rectangle(roi_color, (eye1[1]-15, eye1[0]-15), (eye1[1]+15, eye1[0]+15), (0, 255, 0), 2)
    roi_color = cv2.rectangle(roi_color, (eye2[1]-15, eye2[0]-15), (eye2[1]+15, eye2[0]+15), (0, 255, 0), 2)

    show_images([roi_color, c1, c2, c3, EyeMapC, EyeMapL, EyeMap, EyeMapD], ['Face', 'c1', 'c2', 'c3', 'EyeMapC', 'EyeMapL', 'EyeMap', 'EyeMapD'])
