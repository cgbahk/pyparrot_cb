"""
This code assumes that images used for calibration are of the same arUco marker board provided with code

"""

import cv2
from cv2 import aruco
import yaml
import numpy as np
import os

# Set this flsg True for calibrating camera and False for validating results real time
Calibrate_camera = False

# Set number of images taken using data_generation script.
numberOfImages = 1190

# Set path to the images
path = "/home/truefalse/pyparrot_cb/aruco_calibration"

# For validating results, show aruco board to camera.


aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_6X6_1000)

# Provide length of the marker's side
markerLength = 3.75  # Here, measurement unit is centimetre.

# Provide separation between markers
markerSeparation = 0.5  # Here, measurement unit is centimetre.

# create arUco board
board = aruco.GridBoard_create(4, 5, markerLength, markerSeparation, aruco_dict)

'''uncomment following block to draw and show the board'''
# img = board.draw((864,1080))
# cv2.imshow("aruco", img)
# cv2.waitKey(0)
# cv2.destroyAllWindows()


arucoParams = aruco.DetectorParameters_create()

if Calibrate_camera == True:

    img_list = []

    for fname in os.listdir(path):
        img = cv2.imread(os.path.join(path, fname))
        cv2.imshow("image", img)
        key = cv2.waitKey(0) & 0xFF
        if key == ord('s'):
            img_list.append(img)
        elif key == ord('q'):
            break
        h, w, c = img.shape
    cv2.destroyAllWindows()

    # i = 0
    # while i < numberOfImages:
    #     name = "aruco_data/" + str(i) + ".jpg"
    #     img = cv2.imread(name)
    #     img_list.append(img)
    #     h, w, c = img.shape
    #     i += 1

    counter = []
    corners_list = []
    id_list = []
    first = True
    for im in img_list:

        img_gray = cv2.cvtColor(im, cv2.COLOR_RGB2GRAY)
        corners, ids, rejectedImgPoints = aruco.detectMarkers(img_gray, aruco_dict, parameters=arucoParams)
        if ids is None:
            continue
        elif len(ids) != 20:
            continue

        if first == True:
            corners_list = corners
            print(type(corners))
            id_list = ids
            first = False
        else:
            corners_list = np.vstack((corners_list, corners))
            id_list = np.vstack((id_list, ids))
        counter.append(len(ids))

    counter = np.array(counter)
    print("Calibrating camera .... Please wait...")
    # mat = np.zeros((3,3), float)
    ret, mtx, dist, rvecs, tvecs = aruco.calibrateCameraAruco(corners_list, id_list, counter, board, img_gray.shape, None, None)

    print("Camera matrix is \n", mtx,
          "\n And is stored in calibration.yaml file along with distortion coefficients : \n", dist)
    data = {'camera_matrix': np.asarray(mtx).tolist(), 'dist_coeff': np.asarray(dist).tolist()}
    with open(os.path.join(path,"calibration.yaml"), "w") as f:
        yaml.dump(data, f)


else:

    camera = cv2.VideoCapture(0)
    ret, img = camera.read()

    with open(os.path.join(path,"calibration.yaml"), "w") as f:
        loadeddict = yaml.load(f)
    mtx = loadeddict.get('camera_matrix')
    dist = loadeddict.get('dist_coeff')
    mtx = np.array(mtx)
    dist = np.array(dist)

    ret, img = camera.read()
    img_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    h, w = img_gray.shape[:2]
    newcameramtx, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, (w, h), 1, (w, h))

    pose_r = []
    pose_t = []
    count = 0
    # while True:
    #     ret, img = camera.read()
    for fname in os.listdir(os.path.join(path,"data")):
        img = cv2.imread(os.path.join(path+"/data", fname))
        img_aruco = img
        im_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        h, w = im_gray.shape[:2]
        dst = cv2.undistort(im_gray, mtx, dist, None, newcameramtx)
        corners, ids, rejectedImgPoints = aruco.detectMarkers(dst, aruco_dict, parameters=arucoParams)
        # cv2.imshow("original", img_gray)
        if corners == None:
            print("pass")
        else:

            ret, rvec, tvec = aruco.estimatePoseBoard(corners, ids, board, newcameramtx, dist)  # For a board
            print("Rotation ", rvec, "Translation", tvec)
            if ret != 0:
                img_aruco = aruco.drawDetectedMarkers(img, corners, ids, (0, 255, 0))
                img_aruco = aruco.drawAxis(img_aruco, newcameramtx, dist, rvec, tvec,
                                           10)  # axis length 100 can be changed according to your requirement

            if cv2.waitKey(0) & 0xFF == ord('q'):
                break;
        cv2.imshow("World co-ordinate frame axes", img_aruco)

cv2.destroyAllWindows()
