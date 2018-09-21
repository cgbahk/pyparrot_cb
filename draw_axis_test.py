import cv2
import cv2.aruco as aruco
import yaml
import numpy as np
from os.path import join

aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_4X4_50)
arucoParams = aruco.DetectorParameters_create()

img = cv2.imread('/home/truefalse/pyparrot_cb/pyparrot/images/image_000229.bmp')

corners, ids, rejectedImgPoints = aruco.detectMarkers(img, aruco_dict, parameters=arucoParams)

path_cal = "/home/truefalse/pyparrot_cb/aruco_calibration"
with open(join(path_cal, "calibration_.yaml"), 'r') as f:
    loaded_dict = yaml.load(f)
mtx = loaded_dict.get('camera_matrix')
dist = loaded_dict.get('dist_coeff')
mtx = np.array(mtx)
# dist = np.array(dist)
dist = np.array([0.0, 0.0, 0.0, 0.0, 0.0])  # ignore dist and use default

rvecs, tvecs, ret = aruco.estimatePoseSingleMarkers(corners, 0.168, mtx, dist)

for i in range(len(ids)):
    aruco.drawAxis(img, mtx, dist, rvecs[i], tvecs[i], 0.1)

cv2.imshow('img',img)
cv2.waitKey(0)
