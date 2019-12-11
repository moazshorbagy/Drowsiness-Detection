import tkinter
import cv2
import PIL.Image, PIL.ImageTk
import time

class App:
    def __init__(self, window, window_title, video_source=0):
        self.window = window
        self.window.title(window_title)
        self.monitoringStarted = False
        self.video_source = video_source
        self.vid = None

        self.window.resizable(0,0)

        self.canvas = tkinter.Canvas(self.window, width = 800, height = 640)
        self.canvas.pack()

        self.vid_canvas = None

        label = tkinter.Label(self.window, text="Welcome to Eye detection monitoring system", bg="white")
        label.configure(font=("Century Gothic", 24, "bold"))
        label.place(relx=0.5, rely=0.05, anchor="center")

        self.btn = tkinter.Button(self.window, text='Start monitoring', command=self.startStream, font=("Times 14", 16, "bold"), bg="white")
        self.btn.place(relx=0.5, rely=0.95, anchor="center")
    
    def startApp(self):
        self.window.mainloop()

    def endStream(self):
        if self.monitoringStarted:
           self.btn.configure(text="Start monitoring", command=self.startStream) 
           self.vid_canvas.pack_forget()
           self.vid.vid.release()
           self.vid_canvas.destroy()
           self.monitoringStarted = False

    def startStream(self):
        if not self.monitoringStarted:
            self.vid = MyVideoCapture(self.video_source)

            self.vid_canvas = tkinter.Canvas(self.window, width=640, height=480)
            self.vid_canvas.place(relx=0.5, rely=0.5, anchor=tkinter.CENTER)
            tkinter.Misc.lift(self.vid_canvas)
            self.btn_snapshot = tkinter.Button(self.window, text="Snapshot", width=50, command=self.snapshot)
            self.btn.configure(text="Stop monitoring", command=self.endStream)
            self.monitoringStarted = True
            self.delay = 10
            self.update()


    def update(self):
        if self.vid == None or not self.monitoringStarted:
            return
        ret, frame = self.vid.get_frame()
        if ret:
            frame = cv2.resize(frame, (640, 480))
            image = PIL.Image.fromarray(frame)
            self.photo = PIL.ImageTk.PhotoImage(image=image)
            self.vid_canvas.create_image(0, 0, image = self.photo, anchor = tkinter.NW)
            self.window.after(self.delay, self.update)

    def snapshot(self):
        ret, frame = self.vid.get_frame()
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
