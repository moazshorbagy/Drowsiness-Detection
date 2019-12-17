import cv2
import numpy as np
import skimage.io as io
from skimage.filters import threshold_otsu
import time

def dist(p1, p2):
    return np.sqrt((p1[0]-p2[0])**2 + (p1[1]-p2[1])**2)

def getEyes(img, filter):
    x = filter.shape[0]
    y = filter.shape[1]

    height = img.shape[0]
    width = img.shape[1]

    eyes = []

    for i in range(height):
        for j in range(width):
            if i * x + x > height or j * y + y > width:
                break
            roi = img[i * x : i * x + x, j * y : j * y + y]
            difference = np.sum(np.subtract(roi, filter)/255).astype(int)
            if difference < 1:
                eyes.append((i * x, i * x + x, j * y, j * y + y))
    return eyes

def Face_detection(frame):

    kernel = np.ones((7, 7), np.uint8)
    ycbcr = cv2.cvtColor(frame, cv2.COLOR_RGB2YCrCb)
    gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
    hsv=cv2.cvtColor(frame,cv2.COLOR_RGB2HSV)
    rgb=frame

    # Rule A
    red=rgb[:,:,0]
    green=rgb[:,:,1]
    blue=rgb[:,:,2]
    sum=red+green+blue

    red=np.divide(red,sum,out=np.zeros_like(sum,dtype='float'),where=sum!=0)
    green=np.divide(green,sum,out=np.zeros_like(sum,dtype='float'),where=sum!=0)
    blue=np.divide(blue,sum,out=np.zeros_like(sum,dtype='float'),where=sum!=0)
    mask1=np.logical_and.reduce((red>=0.4,red<=0.6,green>=0.22,green<=0.33,red>=green,green>=((1-red)/2)))

    # Rule B
    h=hsv[:,:,0]
    s=hsv[:,:,1]
    v=hsv[:,:,2]
    sum=h+s+v
    h=np.divide(h,sum,out=np.zeros_like(sum,dtype='float'),where=sum!=0)
    s=np.divide(s,sum,out=np.zeros_like(sum,dtype='float'),where=sum!=0)
    v=np.divide(v,sum,out=np.zeros_like(sum,dtype='float'),where=sum!=0)
    mask2 = np.logical_and.reduce((h>=0,h<=0.2,s>=0.3,s<=0.7,v>=0.22,v<=0.8))

    # Rule C
    y=ycbcr[:,:,0]
    cr=ycbcr[:,:,1]
    cb=ycbcr[:,:,2]

    mask3=np.logical_and.reduce((cb>=90,cb<=117,cr>=138,cr<=170))

    # final Mask
    mask=np.logical_and.reduce((mask1,mask2,mask3))

    gray[np.invert(mask)]=0
    gray[mask]=255


    erosionkernel=cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(7,7))
    dielationkernel=cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(13,13))

    gray = cv2.dilate(gray, dielationkernel,iterations=5)
    gray = cv2.erode(gray, erosionkernel)

    contours, _ = cv2.findContours(gray, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    return contours
    #cv2.drawContours(frame, contours, -1, (0,255,0), 1)

def Draw(frame,contours):
    if(len(contours)):
            index = 0
            secondMax = 0
            thirdMax = 0
            max_c = contours[0].shape[0]
            for i in range(len(contours)):
                if(max_c < contours[i].shape[0]):
                    thirdMax = secondMax
                    secondMax = index
                    index = i
                    max_c = contours[i].shape[0]

            meanIndex = np.mean(contours[index][:, 0, :], axis=0)
            meanSecond = np.mean(contours[secondMax][:, 0, :], axis=0)
            meanThird = np.mean(contours[thirdMax][:, 0, :], axis=0)

            x, y, w, h = cv2.boundingRect(contours[index])
            x2, y2, w2, h2 = cv2.boundingRect(contours[secondMax])
            x3, y3, w3, h3 = cv2.boundingRect(contours[thirdMax])
            if(dist(meanIndex, meanSecond) > w):
                x2=x;y2=y;w2=w;h2=h
            if(dist(meanIndex, meanThird) > w):
                x3=x;y3=y;w3=w;h3=h

            h = int(3 * (max(max(x+w, x2+w2), x3+w3) - min(min(x, x2), x3)) / 2)

            p1_x = min(min(x, x2), x3)
            p1_y = min(min(y, y2), y3)
            p2_x = max(max(x+w, x2+w2), x3+w3)
            p2_y = min(min(y, y2), y3) + h
            face=frame[p1_y:p2_y+1,p1_x:p2_x+1,:]
            gray=cv2.cvtColor(face,cv2.COLOR_RGB2GRAY)
            cv2.imshow("f",gray)
            frame = cv2.rectangle(frame, (p1_x, p1_y), (p2_x, p2_y), (0, 0, 255), 2)
            return p1_x, p1_y, p2_x, p2_y
    return None, None, None, None
