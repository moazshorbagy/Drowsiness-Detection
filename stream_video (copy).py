# For more info: http://docs.opencv.org/3.0-beta/doc/py_tutorials/py_gui/py_video_display/py_video_display.html
import cv2
import numpy as np
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

def getFaces(gray):
    return face_cascade.detectMultiScale(gray, 1.3, 5)

def getEyes():
    pass


if __name__ == '__main__':

    # Playing video from file:
    # cap = cv2.VideoCapture('vtest.avi')
    # Capturing video from webcam:
    cap = cv2.VideoCapture(0)

    # Used for erosion and dilation
    kernel = np.ones((3, 3), np.uint8)

    clahe = cv2.createCLAHE(clipLimit=3.0)

    while(True):
        # Capture frame-by-frame
        ret, frame = cap.read()

        #frame4 = frame[50:-50,50:-50,:]

        # Handles the mirroring of the current frame
        colored = cv2.flip(frame, 1)
        #f = frame

        # Our operations on the frame come here
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Global histogram equalization
        # equalized = cv2.equalizeHist(frame)

        # Applying CLAHE (adaptive/local histogram equalization)
        equalized = clahe.apply(gray)

        faces = getFaces(equalized)

        for (x, y, w, h) in faces:
            roi_gray = frame[y : y + h, x : x + w]
            cv2.imshow('framee', roi_gray)
            
        #res = np.hstack((frame, equalized))
        
        # Edge detection
        edged = cv2.Canny(gray, 60, 200)
        contours, _ = cv2.findContours(edged, mode=cv2.RETR_TREE, method=cv2.CHAIN_APPROX_SIMPLE)
        
        if(len(contours)):
            index = 0
            max_c = contours[0].shape[0]
            for i in range(len(contours)):
                if(max_c < contours[i].shape[0]):
                    index = i
                    max_c = contours[i].shape[0]

            contours = [contours[index]]

            x, y, w, h = cv2.boundingRect(contours[0])
            frame = cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        #res = cv2.dilate(res,kernel,iterations = 1)
        #res = cv2.erode(res,kernel,iterations = 2)
        #res = cv2.morphologyEx(res, cv2.MORPH_CLOSE, kernel, iterations=4)
        #res = cv2.morphologyEx(res, cv2.MORPH_OPEN, kernel, iterations=4)

        # Display the resulting frame
        cv2.imshow('frame', frame)
        
        cv2.imshow('edged', edged)
        
        key_pressed = cv2.waitKey(1)
        # End when pressing q
        if key_pressed & 0xFF == ord('q'):
            break
        # Save fram when Space is pressed
        if key_pressed%256 == 32:
            img_name = "opencv_frame.png"
            cv2.imwrite(img_name, colored)

    # When everything done, release the capture
    cap.release()
    cv2.destroyAllWindows()