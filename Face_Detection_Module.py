import cv2
import numpy as np
import skimage.io as io
from skimage.filters import threshold_otsu
import time

cap = cv2.VideoCapture(0)

time.sleep(1)
kernel = np.ones((7, 7), np.uint8)

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

while True:
    ret, frame = cap.read()
    ycbcr = cv2.cvtColor(frame, cv2.COLOR_BGR2YCrCb)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    hsv=cv2.cvtColor(frame,cv2.COLOR_BGR2HSV)
    rgb=cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)

    # Rule A
    red=rgb[:,:,0]
    green=rgb[:,:,1]
    blue=rgb[:,:,2]
    idx1 = np.logical_and.reduce((red>50, green>40, blue>20, (np.maximum.reduce((red,green,blue))-np.minimum.reduce((red,green,blue)))>10, red>=green+10, red>blue))

    idx2 = np.logical_and.reduce((red>220, green>210, blue>170, red>blue, green>blue, np.abs(red-green)<=15))
    mask1=np.logical_or(idx1,idx2)


    # Rule B
    h=hsv[:,:,0]
    s=hsv[:,:,1]
    mask2 = np.logical_and.reduce((h>=0,h<=50,s>=0.1*np.max(s),s<=0.9*np.max(s)))

    # Rule C
    y=ycbcr[:,:,0]
    cr=ycbcr[:,:,1]
    cb=ycbcr[:,:,2]
    mask3=np.logical_and.reduce((cb>=60,cb<=130,cr>=130,cr<=165))

    # final Mask
    mask=np.logical_and.reduce((mask1,mask2,mask3))
    gray[np.invert(mask)]=0
    gray[mask]=255


    gray = cv2.dilate(gray, kernel, iterations=4)
    gray = cv2.erode(gray, kernel, iterations=5)
    gray = cv2.dilate(gray, kernel, iterations=1)
    gray = cv2.morphologyEx(gray, cv2.MORPH_CLOSE, kernel, iterations=4)
    
    contours, _ = cv2.findContours(gray, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    #cv2.drawContours(frame, contours, -1, (0,255,0), 1)
    
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
            frame = cv2.rectangle(frame, (min(min(x, x2), x3), min(min(y, y2), y3)), (max(max(x+w, x2+w2), x3+w3), min(min(y, y2), y3) + h), (0, 255, 0), 2)

    cv2.imshow("f", frame)

    k = cv2.waitKey(1)
    if k == 27:
        break

cv2.destroyAllWindows()
