import numpy as np
import cv2
import cv2.aruco as aruco
import os

sample_path = "/home/truefalse/pyparrot_cb/aruco_sample"
ls = os.listdir(sample_path)
ls.sort()

for fname in ls:
    frame = cv2.imread(os.path.join(sample_path, fname))

    # cv2.imshow('frame', frame)
    # if cv2.waitKey(0) & 0xFF == ord('q'):
    #     break

    # print(frame.shape) #480x640
    # Our operations on the frame come here
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    aruco_dict = aruco.Dictionary_get(aruco.DICT_4X4_50)
    parameters = aruco.DetectorParameters_create()

    # print(parameters)

    '''    detectMarkers(...)
        detectMarkers(image, dictionary[, corners[, ids[, parameters[, rejectedI
        mgPoints]]]]) -> corners, ids, rejectedImgPoints
        '''
    # lists of ids and the corners beloning to each id
    corners, ids, rejectedImgPoints = aruco.detectMarkers(gray, aruco_dict, parameters=parameters)
    print(corners)

    # It's working.
    # my problem was that the cellphone put black all around it. The algorithm
    # depends very much upon finding rectangular black blobs

    gray = aruco.drawDetectedMarkers(frame, corners, ids, (0, 0, 255))

    # print(rejectedImgPoints)
    # Display the resulting frame
    cv2.imshow('frame', frame)
    if cv2.waitKey(0) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()
