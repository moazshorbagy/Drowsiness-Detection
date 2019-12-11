import cv2
import numpy as np
import skimage.io as io
from skimage.filters import threshold_otsu
import time
import imutils

cap = cv2.VideoCapture(0)

time.sleep(1)

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
    hsv=cv2.cvtColor(frame,cv2.COLOR_BGR2HSV)
    rgb=cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)

    # Rule A
    red=rgb[:,:,0]
    green=rgb[:,:,1]
    blue=rgb[:,:,2]
    idx1=np.bitwise_and.reduce((red>50,green>40,blue>20,\
        (np.maximum.reduce((red,green,blue))-np.minimum.reduce((red,green,blue)))>=10\
         ,red-green>=10,red>green,red>blue))

    idx2=np.bitwise_and.reduce((red>220,green>210,blue>170,red>blue,green>blue,np.abs(red-green)<=15))
    mask1=np.bitwise_or(idx1,idx2)


    # Rule B
    h=hsv[:,:,0]
    s=hsv[:,:,1]
    v=hsv[:,:,2]
    mask2 = np.bitwise_and.reduce((h>=0,h<=50,s>=26,s<=230))


    # Rule C
    y=ycbcr[:,:,0]
    cb=ycbcr[:,:,1]
    cr=ycbcr[:,:,2]
    mask3=np.bitwise_and.reduce((cb>=60,cb<=130,cr>=130,cr<=165))


    # final Mask
    mask=np.bitwise_and.reduce((mask1,mask2,mask3))
    ycbcr[np.invert(mask)]=0


    # print(res)

    _, thresh = cv2.threshold(ycbcr, 120, 255, type=cv2.THRESH_BINARY)
    edged = cv2.Canny(thresh[:,:,0], 127, 200)

    contours, hierarchy = cv2.findContours(edged, mode=cv2.RETR_TREE, method=cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(frame, contours, -1, (0,255,0), 1)
    cs = np.array(contours[0])

    # eyes = getEyes(filter, filter)
    # for eye in eyes:
    #     print('got eye')
    #     x, y, z, w = eye
    #     cv2.rectangle(frame, (x, y), (z, w), (255, 0, 0), 1)

    cv2.imshow("f", frame)

    k = cv2.waitKey(1)
    if k == 27:
        break

cv2.destroyAllWindows()
