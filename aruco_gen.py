import numpy as np
import cv2
import cv2.aruco as aruco

'''
    drawMarker(...)
        drawMarker(dictionary, id, sidePixels[, img[, borderBits]]) -> img
'''

aruco_dict = aruco.Dictionary_get(aruco.DICT_4X4_50)
print(aruco_dict)

for i in range(50):
    # second parameter is id number
    # last parameter is total image size
    img = aruco.drawMarker(aruco_dict, i, 300)
    cv2.imwrite('aruco_%02d.jpg' % i, img)
