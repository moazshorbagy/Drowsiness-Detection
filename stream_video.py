# For more info: http://docs.opencv.org/3.0-beta/doc/py_tutorials/py_gui/py_video_display/py_video_display.html
import cv2
import numpy as np

# Playing video from file:
# cap = cv2.VideoCapture('vtest.avi')
# Capturing video from webcam:
cap = cv2.VideoCapture(0)

# Used for erosion and dilation
kernel = np.ones((3, 3), np.uint8)

currentFrame = 0
while(True):
    # Capture frame-by-frame
    ret, frame = cap.read()

    # Handles the mirroring of the current frame
    frame = cv2.flip(frame, 1)

    # Our operations on the frame come here
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Edge detection
    frame = cv2.Canny(frame, 60, 70)

    # Opening
    closed = cv2.morphologyEx(frame, cv2.MORPH_CLOSE, kernel, iterations=4)

    # Display the resulting frame
    cv2.imshow('frame', frame)
    cv2.imshow('opened', closed)
    
    # End when pressing q
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    
    # To stop duplicate images
    currentFrame += 1

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()