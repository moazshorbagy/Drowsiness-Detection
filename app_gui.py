import tkinter
import cv2
import PIL.Image, PIL.ImageTk
import time
from enhanced_face_detection import Face_detection,Draw
from tkinter import filedialog

class App:
    def __init__(self, window, window_title, video_source=0):
        self.window = window
        self.window.title(window_title)
        self.monitoringStarted = False
        self.video_source = video_source
        self.vid = None

        self.window.resizable(0,0)

        self.canvas = tkinter.Canvas(self.window, width = 720, height = 640, bg="white")
        self.canvas.pack()

        ## Toolbar and brightness control buttons
        self.vid_canvas = None
        self.toolBarCanvas = None
        self.brightnessLevel = 0
        self.incBrightnessBtn = None
        self.decBrightnessBtn = None
        self.defaultBrightnessBtn = None
        self.toolBarLabel = None

        label = tkinter.Label(self.window, text="Welcome to Eye detection monitoring system", bg="white")
        label.configure(font=("Century Gothic", 24, "bold"))
        label.place(relx=0.5, rely=0.05, anchor="center")

        self.btn = tkinter.Button(self.window, text='Start monitoring', command=self.startStream, font=("Times 14", 16, "bold"), bg="white")
        self.btn.place(relx=0.5, rely=0.95, anchor="center")

        self.streamFromCamBtn = tkinter.Button(self.window, text="Stream from Computer Camera", command=self.setVideoSourceTo0, font=("Times 14", 16, "bold"), bg="white")
        self.streamFromCamBtn.place(relx=0.39, rely=0.1, anchor="center")

        self.browseFilesBtn = tkinter.Button(self.window, text="Browse videos", command=self.getVidSource, font=("Times 14", 16, "bold"), bg="white")
        self.browseFilesBtn.place(relx=0.66, rely=0.1, anchor="center")

    def setVideoSourceTo0(self):
        self.video_source = 0

    def startApp(self):
        self.window.mainloop()

    def getVidSource(self):
        filename = filedialog.askopenfilename(initialdir = "/",title = "Select file")
        self.video_source = filename

    def endStream(self):
        if self.monitoringStarted:
           self.btn.configure(text="Start monitoring", command=self.startStream)
           self.vid_canvas.pack_forget()
           self.vid.vid.release()
           self.vid_canvas.destroy()
           self.removeToolBar()
           self.monitoringStarted = False

    def startStream(self):
        if not self.monitoringStarted:
            self.vid = MyVideoCapture(self.video_source)
            self.addToolBar()
            self.vid_canvas = tkinter.Canvas(self.window, width=640, height=480, highlightthickness=1, highlightbackground="#616161")
            self.vid_canvas.place(relx=0.565, rely=0.53, anchor=tkinter.CENTER)
            tkinter.Misc.lift(self.vid_canvas)
            self.btn.configure(text="Stop monitoring", command=self.endStream)
            self.monitoringStarted = True
            self.delay = 24
            self.update()

    def addToolBar(self):
        self.toolBarLabel = tkinter.Label(self.window, text="Toolbar",  font=("Times 14", 14, "bold"), bg="white")
        self.toolBarLabel.place(relx=0.055, rely=0.34, anchor=tkinter.CENTER)
        self.btn_snapshot = tkinter.Button(self.window, width=8, height=5, text="Snapshot", command=self.snapshot, bg="white", highlightthickness=1, highlightbackground="#616161", font=("Times 14", 14, "bold"))
        self.btn_snapshot.place(relx=0.06, rely=0.424, anchor=tkinter.CENTER)
        self.incBrightnessBtn = tkinter.Button(self.window, width=8, command=self.setIncreaseBrightness, text="increase brightness", wraplength=80, bg="white", highlightthickness=1, highlightbackground="#616161", font=("Times 14", 14, "bold"))
        self.incBrightnessBtn.place(relx=0.06, rely=0.524, anchor=tkinter.CENTER)
        self.decBrightnessBtn = tkinter.Button(self.window, width=8, command=self.setDecreaseBrightness, text="decrease brightness", wraplength=80, bg="white", highlightthickness=1, highlightbackground="#616161", font=("Times 14", 14, "bold"))
        self.decBrightnessBtn.place(relx=0.06, rely=0.586, anchor=tkinter.CENTER)
        self.defaultBrightnessBtn = tkinter.Button(self.window, width=8, command=self.setDefaultBrightness, text="default brightness", wraplength=80, bg="white", highlightthickness=1, highlightbackground="#616161", font=("Times 14", 14, "bold"))
        self.defaultBrightnessBtn.place(relx=0.06, rely=0.648, anchor=tkinter.CENTER)

    def removeToolBar(self):
        self.toolBarLabel.pack_forget()
        self.toolBarLabel.destroy()
        self.btn_snapshot.pack_forget()
        self.btn_snapshot.destroy()
        self.incBrightnessBtn.pack_forget()
        self.incBrightnessBtn.destroy()
        self.decBrightnessBtn.pack_forget()
        self.decBrightnessBtn.destroy()
        self.defaultBrightnessBtn.pack_forget()
        self.defaultBrightnessBtn.destroy()


    def update(self):
        if self.vid == None or not self.monitoringStarted:
            return
        ret, frame = self.vid.get_frame()
        if ret:
            frame = cv2.resize(frame, (640, 480))
            contours=Face_detection(frame)
            Draw(frame,contours)
            frame = self.set_brightness(frame, self.brightnessLevel)
            image = PIL.Image.fromarray(frame)
            self.photo = PIL.ImageTk.PhotoImage(image=image)
            self.vid_canvas.create_image(0, 0, image = self.photo, anchor = tkinter.NW)
            self.window.after(self.delay, self.update)

    def set_brightness(self, img, value=30):
        hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
        h, s, v = cv2.split(hsv)

        if value < 0:
            lim = 0 - value
            v[v < lim] = 0
            v[v >= lim] = v[v >= lim] + value
        else:
            lim = 255 - value
            v[v > lim] = 255
            v[v <= lim] += value

        final_hsv = cv2.merge((h, s, v))
        img = cv2.cvtColor(final_hsv, cv2.COLOR_HSV2RGB)
        return img

    def setIncreaseBrightness(self):
        self.brightnessLevel = self.brightnessLevel + 20

    def setDecreaseBrightness(self):
        self.brightnessLevel = self.brightnessLevel - 20

    def setDefaultBrightness(self):
        self.brightnessLevel = 0

    def snapshot(self):
        ret, frame = self.vid.get_frame()
        frame = self.set_brightness(frame, self.brightnessLevel)
        if ret:
            cv2.imwrite("frame-" + time.strftime("%d-%m-%Y-%H-%M-%S") + ".jpg", cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))


class MyVideoCapture:
    def __init__(self, video_source=0):
        # Open the video source
        self.vid = cv2.VideoCapture(video_source)
        if not self.vid.isOpened():
            raise ValueError("Unable to open video source", video_source)

        # Get video source width and height
        self.width = self.vid.get(cv2.CAP_PROP_FRAME_WIDTH)
        self.height = self.vid.get(cv2.CAP_PROP_FRAME_HEIGHT)

    def get_frame(self):
        if self.vid.isOpened():
            ret, frame = self.vid.read()
            if ret:
                return ret, cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            else:
                return ret, None


    # Release the video source when the object is destroyed
    def __del__(self):
        if self.vid.isOpened():
            self.vid.release()

app = App(tkinter.Tk(), "Drowsiness Detection System")
app.startApp()
