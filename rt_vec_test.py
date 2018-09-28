import os
from os.path import join
import cv2
import cv2.aruco as aruco
import yaml
import numpy as np

# path setting
root = os.getcwd()
image_path = join(root, 'rt_vec')

# aruco setting
aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_4X4_50)
arucoParams = aruco.DetectorParameters_create()

# import camera setting
with open(join(root, "aruco_calibration", "calibration_.yaml"), 'r') as f:
    loaded_dict = yaml.load(f)
mtx = np.array(loaded_dict.get('camera_matrix'))
dist = np.array([0.0, 0.0, 0.0, 0.0, 0.0])  # ignore dist and use default

# draw setting
unit = 1
draw_length = unit * 0.5

for fname in os.listdir(image_path):
    # read img
    img = cv2.imread(join(image_path, fname))

    # analysis #
    # get rvec, tvec
    corners, ids, rejectedImgPoints = aruco.detectMarkers(img, aruco_dict, parameters=arucoParams)
    rvecs, tvecs, ret = aruco.estimatePoseSingleMarkers(corners, unit, mtx, dist)

    # draw #
    # draw markers
    aruco.drawDetectedMarkers(img, corners, ids, (0, 0, 255))
    # draw axis
    if ids is None:
        ids = []
    for i in range(len(ids)):
        aruco.drawAxis(img, mtx, dist, rvecs[i], tvecs[i], draw_length)
    print(fname)
    print("  Rotation ", rvecs, "\n  Translation", tvecs, "\n")

    break
    # show img
    # cv2.imshow('image', img)
    # cv2.waitKey(0)

cv2.destroyAllWindows()
