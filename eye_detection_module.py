# For more info: http://docs.opencv.org/3.0-beta/doc/py_tutorials/py_gui/py_video_display/py_video_display.html
import cv2
import numpy as np
from skimage.color import rgb2ycbcr, rgb2gray, rgb2hsv, hsv2rgb
from skimage.exposure import adjust_gamma, is_low_contrast, rescale_intensity
from commonfunctions import *
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

se = np.ones((7,7), np.uint8)
se[0,[0, 1, 2, 4, 5, 6]] = 0
se[6,[0, 1, 2, 4, 5, 6]] = 0
se[[0, 1, 2, 4, 5, 6],0] = 0
se[[0, 1, 2, 4, 5, 6],6] = 0

frame = io.imread('moaz2.png')

# Our operations on the frame come here
gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)

faces = face_cascade.detectMultiScale(gray, 1.15, 5)

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

    EyeMapL = dilated / eroded
    EyeMapLMin = np.min(EyeMapL)
    EyeMapL = (EyeMapL - EyeMapLMin) / (np.max(EyeMapL) - EyeMapLMin)
    
    EyeMap = EyeMapC * EyeMapL

    EyeMapD = cv2.dilate(EyeMap, se, iterations = 3)
    EyeMapD = cv2.erode(EyeMapD, se, iterations = 3)
    EyeMapD[EyeMapD >= 0.5] = 1.0
    EyeMapD[EyeMapD < 0.5] = 0.0
    EyeMapD = cv2.dilate(EyeMapD, se, iterations = 2)

    #threshold = 0.7 * np.max(EyeMap)
    #while True:
    #    Thresholded = EyeMap.copy()
    #    Thresholded[Thresholded >= threshold] = 1.0
    #    Thresholded[Thresholded < threshold] = 0.0

    show_images([roi_color, c1, c2, c3, EyeMapC, EyeMapL, EyeMap, EyeMapD], ['Face', 'c1', 'c2', 'c3', 'EyeMapC', 'EyeMapL', 'EyeMap', 'Thresholded'])