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

    cr_range = np.array([140, 165]) 
    cb_range = np.array([140, 195])
    lower = np.array([0, 90, 90])
    upper = np.array([255, 165, 195])

    mask = cv2.inRange(ycbcr, lower, upper)

    res = cv2.bitwise_and(ycbcr,ycbcr, mask=mask)
    # print(res)

    _, thresh = cv2.threshold(res, 120, 255, type=cv2.THRESH_BINARY)
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