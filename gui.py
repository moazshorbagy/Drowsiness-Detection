from __future__ import print_function
from PIL import Image
from PIL import ImageTk
import tkinter as tki
import threading
import datetime
import cv2
import os

class App:
    def __init__(self):
        self.root = tki.Tk()
        self.root.configure(background="white", width=800, height=640)

        btn = tki.Button(self.root, text='Start monitoring', command=self.startStream, font=("Times 14", 16, "bold"), bg="white")
        # btn.pack(side="bottom", fill="both", padx=100, pady=100)
        btn.place(relx=0.5, rely=0.9, anchor="center")

        label = tki.Label(self.root, text="Welcome to Eye detection monitoring system", bg="white")
        label.configure(font=("Century Gothic", 24, "bold"))
        # label.pack(side="top", fill="both", padx=40, pady=10)
        label.place(relx=0.5, rely=0.2, anchor="center")

    def startStream(self):
        cap = cv2.VideoCapture(0)

        currentFrame = 0
        while(True):
            # Capture frame-by-frame
            ret, frame = cap.read()

            # Handles the mirroring of the current frame
            frame = cv2.flip(frame,1)

            # Our operations on the frame come here
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            # Saves image of the current frame in jpg file
            # name = 'frame' + str(currentFrame) + '.jpg'
            # cv2.imwrite(name, frame)

            # Display the resulting frame
            cv2.imshow('frame',frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            
            # To stop duplicate images
            currentFrame += 1

        
        # When everything done, release the capture
        cap.release()
        cv2.destroyAllWindows()

app = App()
app.root.mainloop()