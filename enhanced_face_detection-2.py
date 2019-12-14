import cv2
import numpy as np
import skimage.io as io
from skimage.filters import threshold_otsu
import time


def dist(p1, p2):
    return np.sqrt((p1[0]-p2[0])**2 + (p1[1]-p2[1])**2)

cap = cv2.VideoCapture(0)

time.sleep(1)

while True:
    ret, frame = cap.read()
    rgb =cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)/255
    HSV=np.zeros(frame.shape)

    red=rgb[:,:,0]
    green=rgb[:,:,1]
    blue=rgb[:,:,2]

    Max=np.maximum.reduce((red,green,blue))
    Min=np.minimum.reduce((red,green,blue))

    t1=np.zeros(Max.shape)
    t2=np.zeros(Max.shape)
    t3=np.zeros(Max.shape)

    t1=np.divide(green-blue,Max-Min,out=np.zeros_like(Max,dtype='float'),where=(Max-Min)!=0)*60
    t2=2+np.divide(blue-red,Max-Min,out=np.zeros_like(Max,dtype='float'),where=(Max-Min)!=0)*60
    t3=4+np.divide(red-green,Max-Min,out=np.zeros_like(Max,dtype='float'),where=(Max-Min)!=0)*60

    t1[red!=Max]=0
    t2[green!=Max]=0
    t3[blue!=Max]=0

    h=t1+t2+t3
    s=np.divide(Max-Min,Max,out=np.zeros_like(Max,dtype='float'),where=Max!=0)
    v=Max


    HSV[:,:,0]=h
    HSV[:,:,1]=s
    HSV[:,:,2]=v

    skin_color_H_min = 0.10
    skin_color_H_max = 0.90
    skin_color_S_min = 0.30
    skin_color_S_max = 0.99
    lower = np.array([skin_color_H_min, skin_color_S_min, 0])
    upper = np.array([skin_color_H_max, skin_color_S_max, 255])

    roi = cv2.inRange(HSV, lower, upper)
    roi=cv2.dilate(roi, np.ones((7,7)), iterations=5)


    contours, _ = cv2.findContours(roi, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
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

    k = cv2.waitKey(30)
    if k == 27:
        break

cap.release()
cv2.destroyAllWindows()
