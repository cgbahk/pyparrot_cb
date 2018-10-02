"""
Demo of the ffmpeg based mambo vision code (basically flies around and saves out photos as it flies)

Author: Amy McGovern
"""
from pyparrot.Minidrone import Mambo
from pyparrot.DroneVision import DroneVision
import threading
import cv2
import cv2.aruco as aruco
import time
import inspect
from os.path import join
import numpy as np
import yaml

from OpenGL.GL import *
from OpenGL.GLUT import *
from OpenGL.GLU import *
from PIL import Image
from myObjloader import *

# set this to true if you want to fly for the demo
testFlying = True


class UserVision:
    def __init__(self, vision):
        self.index = 0
        self.detected = 0
        self.vision = vision
        data_path = self.get_data_path()
        self.face_cascade = cv2.CascadeClassifier(join(data_path, 'haarcascade_frontalface_default.xml'))
        self.eye_cascade = cv2.CascadeClassifier(join(data_path, 'haarcascade_eye.xml'))
        self.smile_cascade = cv2.CascadeClassifier(join(data_path, 'haarcascade_smile.xml'))

        self.path_cal = "/home/truefalse/pyparrot_cb/aruco_calibration"
        with open(join(self.path_cal, "calibration_.yaml"), 'r') as f:
            loaded_dict = yaml.load(f)
        mtx = loaded_dict.get('camera_matrix')
        dist = loaded_dict.get('dist_coeff')
        self.mtx = np.array(mtx)
        # self.dist = np.array(dist)
        self.dist = np.array([0.0, 0.0, 0.0, 0.0, 0.0])  # ignore dist and use default
        self.aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_4X4_50)
        self.arucoParams = aruco.DetectorParameters_create()

        # initialise shapes
        self.wolf = None

        # initialise texture
        self.texture_background = None

    def save_pictures(self, args):
        print("in save pictures on image %d " % self.index)

        img = self.vision.get_latest_valid_picture()

        if img is not None:
            filename = "test_image_%06d.png" % self.index
            cv2.imwrite(filename, img)

            self.index += 1
            # print(self.index)

    def show_pictures(self, args):
        img = self.vision.get_latest_valid_picture()
        # img = self.vision.buffer[self.vision.buffer_index]
        if img is not None:
            # print(self.index)
            self.index += 1
            cv2.imshow('image', img)
            cv2.waitKey(1)
            # alternative
            # plt.imshow(img, interpolation='bicubic')
            # plt.xticks([]), plt.yticks([])  # to hide tick values on X and Y axis
            # plt.show()

    def show_pictures_with_face(self, args):
        img = self.vision.get_latest_valid_picture()
        if img is None:
            return

        # print(self.index)
        self.index += 1

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # gray = img
        faces = self.face_cascade.detectMultiScale(gray, 1.3, 5)
        if faces.__len__():
            for (x, y, w, h) in faces:
                cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
                roi_gray = gray[y:y + h, x:x + w]
                roi_color = img[y:y + h, x:x + w]
                # eyes = self.eye_cascade.detectMultiScale(roi_gray)
                # for (ex, ey, ew, eh) in eyes:
                #     cv2.rectangle(roi_color, (ex, ey), (ex + ew, ey + eh), (0, 255, 0), 2)
                smiles = self.smile_cascade.detectMultiScale(roi_gray)
                if smiles.__len__():
                    self.detected += 1
                    print(self.detected)
                for (sx, sy, sw, sh) in smiles:
                    cv2.rectangle(roi_color, (sx, sy), (sx + sw, sy + sh), (0, 0, 255), 2)
            # todo: display text
            # font = cv2.FONT_HERSHEY_SIMPLEX
            # cv2.putText(img, str(img_idx), (100, 100), font, 3, (255, 255, 255), 5, cv2.LINE_AA)
            cv2.imshow('image', img)
            cv2.waitKey(1)
            # print("detected %d / %.2f percent" % (self.detected, 100 * self.detected / self.index))
        else:
            cv2.imshow('image', img)
            cv2.waitKey(1)

        if testFlying and self.detected > 10:
            self.detected = 0
            print("flip left")
            print("flying state is %s" % mambo.sensors.flying_state)
            _success = mambo.flip(direction="left")
            print("mambo flip result %s" % _success)
            mambo.smart_sleep(5)
            print("landing")
            print("flying state is %s" % mambo.sensors.flying_state)
            mambo.safe_land(5)
            print("Ending the sleep and vision")
            mamboVision.close_video()
            mambo.smart_sleep(5)
            print("disconnecting")
            mambo.disconnect()

    def show_corner(self, args):
        img = self.vision.get_latest_valid_picture()
        if img is None:
            return

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        gray = np.float32(gray)
        dst = cv2.cornerHarris(gray, 2, 3, 0.04)
        # result is dilated for marking the corners, not important
        dst = cv2.dilate(dst, None)
        # Threshold for an optimal value, it may vary depending on the image.
        img[dst > 0.01 * dst.max()] = [0, 0, 255]

        cv2.imshow('image', img)
        cv2.waitKey(1)

    def detect_aruco(self, args):
        img = self.vision.get_latest_valid_picture()
        if img is None:
            return

        # print(frame.shape) #480x640
        # Our operations on the frame come here
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        aruco_dict = aruco.Dictionary_get(aruco.DICT_4X4_50)  # todo: optimize
        parameters = aruco.DetectorParameters_create()

        # print(parameters)

        '''    detectMarkers(...)
            detectMarkers(image, dictionary[, corners[, ids[, parameters[, rejectedI
            mgPoints]]]]) -> corners, ids, rejectedImgPoints
            '''
        # lists of ids and the corners belonging to each id
        corners, ids, rejectedImgPoints = aruco.detectMarkers(gray, aruco_dict, parameters=parameters)
        # print(corners)

        # It's working.
        # my problem was that the cellphone put black all around it. The algorithm
        # depends very much upon finding rectangular black blobs

        aruco.drawDetectedMarkers(img, corners, ids, (0, 0, 255))

        # print(rejectedImgPoints)
        # Display the resulting frame
        cv2.imshow('image', img)
        cv2.waitKey(1)

    def picture_for_calibration(self, args):
        img = self.vision.get_latest_valid_picture()
        if img is not None:
            cv2.imshow('image', img)
            key = cv2.waitKey(1) & 0xFF
            if key == ord('s'):
                cv2.imwrite('aruco_calibration/%02d.bmp' % self.index, img)
                print(self.index)
                self.index += 1

    def draw_axis(self, args):
        img = self.vision.get_latest_valid_picture()
        if img is None:
            return

        corners, ids, rejectedImgPoints = aruco.detectMarkers(img, self.aruco_dict, parameters=self.arucoParams)
        if corners:
            rvecs, tvecs, ret = aruco.estimatePoseSingleMarkers(corners, 0.168, self.mtx, self.dist)

            # print("Rotation ", rvecs, "Translation", tvecs)
            aruco.drawDetectedMarkers(img, corners, ids, (0, 0, 255))
            # print(ids)
            for i in range(len(ids)):
                aruco.drawAxis(img, self.mtx, self.dist, rvecs[i], tvecs[i], 0.1)

        cv2.imshow('image', img)
        cv2.waitKey(1)

    def get_data_path(self):
        full_path = inspect.getfile(cv2)
        short_path_index = full_path.rfind("/")
        short_path = full_path[0:short_path_index]
        data_path = join(short_path, "data")
        return data_path

    def follow_aruco(self, args):
        img = self.vision.get_latest_valid_picture()
        if img is None:
            return

        corners, ids, rejectedImgPoints = aruco.detectMarkers(img, self.aruco_dict, parameters=self.arucoParams)

        if corners:  # if detected at least one
            # find pursuit-marker
            pursuit_idx = -1
            for i in range(len(ids)):
                if ids[i][0] == 0:  # :pursuit-marker id
                    pursuit_idx = i
                    break

            if pursuit_idx > -1:  # if pursuit-marker detected
                rvecs, tvecs, ret = \
                    aruco.estimatePoseSingleMarkers(corners[pursuit_idx:pursuit_idx + 1], 1, self.mtx, self.dist)

                rvec = rvecs[0]
                tvec = tvecs[0]

                dx = tvec[0][0]
                dy = -tvec[0][1]
                dz = tvec[0][2] - 10

                dx = self.saturate(dx, 1.8)
                dy = self.saturate(dy, 1.8)
                dz = self.saturate(dz, 3)

                aruco.drawAxis(img, self.mtx, self.dist, rvec, tvec, 0.5)
                font = cv2.FONT_HERSHEY_SIMPLEX
                str = "dx: %.1f dy: %.1f dz: %.1f" % (dx, dy, dz)
                cv2.putText(img, str, (10, 30), font, 0.5, (0, 0, 255), 2)

                # move
                gain = 6  # 6
                mambo.fly_direct(roll=dx * gain, pitch=dz * gain * 0.3, yaw=0, vertical_movement=dy * gain * 1.5,
                                 duration=0.1)
                # mambo.smart_sleep(0.01)

        cv2.imshow('image', img)
        cv2.waitKey(1)

    def saturate(self, d, saturation):
        if d > saturation:
            return saturation
        if d < -saturation:
            return -saturation
        return d

    # code imported from OpenGLGlyphs

    INVERSE_MATRIX = np.array([[1.0, 1.0, 1.0, 1.0],
                               [-1.0, -1.0, -1.0, -1.0],
                               [-1.0, -1.0, -1.0, -1.0],
                               [1.0, 1.0, 1.0, 1.0]])

    def _init_gl(self, Width, Height):
        glClearColor(0.0, 0.0, 0.0, 0.0)
        glClearDepth(1.0)
        glDepthFunc(GL_LESS)
        glEnable(GL_DEPTH_TEST)
        glShadeModel(GL_SMOOTH)
        glMatrixMode(GL_PROJECTION)
        glLoadIdentity()
        gluPerspective(37, 1.3, 0.1, 100.0)
        glMatrixMode(GL_MODELVIEW)

        glLightfv(GL_LIGHT0, GL_POSITION, (-40, 300, 200, 0.0))
        glLightfv(GL_LIGHT0, GL_AMBIENT, (0.2, 0.2, 0.2, 1.0))
        glLightfv(GL_LIGHT0, GL_DIFFUSE, (0.5, 0.5, 0.5, 1.0))
        glEnable(GL_LIGHT0)
        glEnable(GL_LIGHTING)
        glEnable(GL_COLOR_MATERIAL)

        # assign shapes
        File = 'Sinbad_4_000001.obj'
        # File = 'mount.blend1.obj'
        self.wolf = OBJ(File, swapyz=True)

        # assign texture
        glEnable(GL_TEXTURE_2D)
        self.texture_background = glGenTextures(1)

    def _draw_scene(self):
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        glLoadIdentity()

        # get image from webcam
        # image = self.webcam.read()  # original
        image = self.vision.get_latest_valid_picture()
        # image = imutils.resize(image,width=640)

        # convert image to OpenGL texture format
        bg_image = cv2.flip(image, 0)
        bg_image = Image.fromarray(bg_image)
        ix = bg_image.size[0]
        iy = bg_image.size[1]
        bg_image = bg_image.tobytes("raw", "BGRX", 0, -1)

        # create background texture
        glBindTexture(GL_TEXTURE_2D, self.texture_background)
        glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST)
        glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST)
        glTexImage2D(GL_TEXTURE_2D, 0, 3, ix, iy, 0, GL_RGBA, GL_UNSIGNED_BYTE, bg_image)

        # draw background
        glBindTexture(GL_TEXTURE_2D, self.texture_background)
        glPushMatrix()
        glTranslatef(0.0, 0.0, -10.0)
        self._draw_background()
        glPopMatrix()

        # handle glyphs
        image = self._handle_glyphs(image)

        glutSwapBuffers()

    def _handle_glyphs(self, image):
        # aruco data
        aruco_dict = aruco.Dictionary_get(aruco.DICT_4X4_50)
        parameters = aruco.DetectorParameters_create()

        height, width, channels = image.shape
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        corners, ids, rejectedImgPoints = aruco.detectMarkers(gray, aruco_dict, parameters=parameters)
        if ids is not None and corners is not None:
            rvecs, tvecs, _objpoints = \
                aruco.estimatePoseSingleMarkers(corners[0], 0.6, self.mtx, self.dist)

            # control
            rvec = rvecs[0]
            tvec = tvecs[0]

            dx = tvec[0][0]
            dy = -tvec[0][1]
            dz = tvec[0][2] - 5

            dx = self.saturate(dx, 1.8)
            dy = self.saturate(dy, 1.8)
            dz = self.saturate(dz, 3)

            gain = 6  # 6
            mambo.fly_direct(roll=dx * gain, pitch=dz * gain * 0.3, yaw=0, vertical_movement=dy * gain * 1.5,
                             duration=0.1)
            # mambo.smart_sleep(0.01)
            # build view matrix
            # board = aruco.GridBoard_create(6,8,0.05,0.01,aruco_dict)
            # corners, ids, rejectedImgPoints,rec_idx = aruco.refineDetectedMarkers(gray,board,corners,ids,rejectedImgPoints)
            # ret,rvecs,tvecs = aruco.estimatePoseBoard(corners,ids,board,self.cam_matrix,self.dist_coefs)

            rmtx = cv2.Rodrigues(rvecs)[0]

            rot_vecs = np.array([[[1.54, 0, 0]]])
            rot_mtx = cv2.Rodrigues(rot_vecs)[0]
            rmtx = np.matmul(rmtx, rot_mtx)

            rot_vecs = np.array([[[0, 3.14, 0]]])
            rot_mtx = cv2.Rodrigues(rot_vecs)[0]
            rmtx = np.matmul(rmtx, rot_mtx)

            view_matrix = np.array([[rmtx[0][0], rmtx[0][1], rmtx[0][2], tvecs[0][0][0]],
                                    [rmtx[1][0], rmtx[1][1], rmtx[1][2], tvecs[0][0][1] + 1],
                                    [rmtx[2][0], rmtx[2][1], rmtx[2][2], tvecs[0][0][2]+1],
                                    [0.0, 0.0, 0.0, 1.0]])

            # view_matrix = np.array([[rmtx[0][0],rmtx[0][1],rmtx[0][2],tvecs[0]],
            #                         [rmtx[1][0],rmtx[1][1],rmtx[1][2],tvecs[1]],
            #                         [rmtx[2][0],rmtx[2][1],rmtx[2][2],tvecs[2]],
            #                         [0.0       ,0.0       ,0.0       ,1.0    ]])

            view_matrix = view_matrix * self.INVERSE_MATRIX

            view_matrix = np.transpose(view_matrix)

            # load view matrix and draw shape
            glPushMatrix()
            glLoadMatrixd(view_matrix)

            glCallList(self.wolf.gl_list)

            glPopMatrix()
        # cv2.imshow("cv frame", image)
        # cv2.waitKey(1)

    def _draw_background(self):
        # draw background
        glBegin(GL_QUADS)
        glTexCoord2f(0.0, 1.0)
        glVertex3f(-4.0, -3.0, 0.0)
        glTexCoord2f(1.0, 1.0)
        glVertex3f(4.0, -3.0, 0.0)
        glTexCoord2f(1.0, 0.0)
        glVertex3f(4.0, 3.0, 0.0)
        glTexCoord2f(0.0, 0.0)
        glVertex3f(-4.0, 3.0, 0.0)
        glEnd()

    def main(self):
        # setup and run OpenGL
        glutInit()
        glutInitDisplayMode(GLUT_RGBA | GLUT_DOUBLE | GLUT_DEPTH)
        glutInitWindowSize(640, 360)
        glutInitWindowPosition(500, 400)
        self.window_id = glutCreateWindow("OpenGL Glyphs")
        glutDisplayFunc(self._draw_scene)
        glutIdleFunc(self._draw_scene)
        self._init_gl(640, 360)
        glutMainLoop()


# you will need to change this to the address of YOUR mambo
mamboAddr = "e0:14:d0:63:3d:d0"

# make my mambo object
# remember to set True/False for the wifi depending on if you are using the wifi or the BLE to connect
mambo = Mambo(mamboAddr, use_wifi=True)
print("trying to connect to mambo now")
success = mambo.connect(num_retries=3)
print("connected: %s" % success)

if success:
    # get the state information
    print("sleeping")
    mambo.smart_sleep(1)
    mambo.ask_for_state_update()
    mambo.smart_sleep(1)

    print("Preparing to open vision")
    mamboVision = DroneVision(mambo, is_bebop=False, buffer_size=30)
    userVision = UserVision(mamboVision)
    # mamboVision.set_user_callback_function(userVision.follow_aruco, user_callback_args=None)

    success = mamboVision.open_video()
    print("Success in opening vision is %s" % success)



    if success:
        print("Vision successfully started!")
        # removed the user call to this function (it now happens in open_video())
        # mamboVision.start_video_buffering()

        userVision.main()
        print("out of main")
        if testFlying:
            print("taking off!")
            mambo.safe_takeoff(5)

            if mambo.sensors.flying_state != "emergency":
                print("flying state is %s" % mambo.sensors.flying_state)
                # print("Flying direct: going up")
                # mambo.fly_direct(roll=0, pitch=0, yaw=0, vertical_movement=40, duration=1)

                mambo.smart_sleep(300)

                # print("flip left")
                # print("flying state is %s" % mambo.sensors.flying_state)
                # success = mambo.flip(direction="left")
                # print("mambo flip result %s" % success)
                # mambo.smart_sleep(5)

            print("landing")
            print("flying state is %s" % mambo.sensors.flying_state)
            mambo.safe_land(5)
        else:
            print("Sleeeping for 60 seconds - move the mambo around")
            mambo.smart_sleep(300)

        # done doing vision demo
        print("Ending the sleep and vision")
        mamboVision.close_video()

        mambo.smart_sleep(5)

    print("disconnecting")
    mambo.disconnect()
