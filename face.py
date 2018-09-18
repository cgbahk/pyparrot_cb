import numpy as np
import cv2
import inspect
from os.path import join
import pyparrot

# get path
full_path = inspect.getfile(cv2)
short_path_index = full_path.rfind("/")
short_path = full_path[0:short_path_index]
data_path = join(short_path, "data")

full_path = inspect.getfile(pyparrot)
short_path_index = full_path.rfind("/")
short_path = full_path[0:short_path_index]
image_path = join(short_path, "images")

face_cascade = cv2.CascadeClassifier(join(data_path, 'haarcascade_frontalface_alt.xml'))
eye_cascade = cv2.CascadeClassifier(join(data_path, 'haarcascade_eye.xml'))
smile_cascade = cv2.CascadeClassifier(join(data_path, 'haarcascade_smile.xml'))

img_idx = 1
while True:
    img = cv2.imread(join(image_path, "image_%06d.png" % img_idx))
    # img = cv2.imread('image_282.png')
    # img = cv2.imread('lena.png')
    if img is None:
        break

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    if faces.__len__():
        for (x, y, w, h) in faces:
            cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
            roi_gray = gray[y:y + h, x:x + w]
            roi_color = img[y:y + h, x:x + w]
            eyes = eye_cascade.detectMultiScale(roi_gray)
            for (ex, ey, ew, eh) in eyes:
                cv2.rectangle(roi_color, (ex, ey), (ex + ew, ey + eh), (0, 255, 0), 2)
            smiles = smile_cascade.detectMultiScale(roi_gray)
            for (sx, sy, sw, sh) in smiles:
                cv2.rectangle(roi_color, (sx, sy), (sx + sw, sy + sh), (0, 0, 255), 2)
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(img, str(img_idx), (100, 100), font, 3, (255, 255, 255), 5, cv2.LINE_AA)
        cv2.imshow('image', img)
        cv2.waitKey(0)
        # cv2.destroyAllWindows()

    img_idx += 1
