from enhanced_face_detection import detect_face
from eye_detection_module import detect_eyes
import numpy as np

def doTheWork(frame):
    x, y, w, h = detect_face(frame)
    roi = np.array([(x, y, w - x, h - y)])

    return detect_eyes(roi, frame)

