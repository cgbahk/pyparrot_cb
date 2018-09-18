"""
Demo of the ffmpeg based mambo vision code (basically flies around and saves out photos as it flies)

Author: Amy McGovern
"""
from pyparrot.Minidrone import Mambo
from pyparrot.DroneVision import DroneVision
import threading
import cv2
import time
from matplotlib import pyplot as plt
import inspect
from os.path import join

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

    def get_data_path(self):
        full_path = inspect.getfile(cv2)
        short_path_index = full_path.rfind("/")
        short_path = full_path[0:short_path_index]
        data_path = join(short_path, "data")
        return data_path


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
    mamboVision = DroneVision(mambo, is_bebop=False, buffer_size=3)
    userVision = UserVision(mamboVision)
    mamboVision.set_user_callback_function(userVision.show_pictures_with_face, user_callback_args=None)
    success = mamboVision.open_video()
    print("Success in opening vision is %s" % success)

    if success:
        print("Vision successfully started!")
        # removed the user call to this function (it now happens in open_video())
        # mamboVision.start_video_buffering()

        if testFlying:
            print("taking off!")
            mambo.safe_takeoff(5)

            if mambo.sensors.flying_state != "emergency":
                print("flying state is %s" % mambo.sensors.flying_state)
                print("Flying direct: going up")
                mambo.fly_direct(roll=0, pitch=0, yaw=0, vertical_movement=30, duration=1)

                mambo.smart_sleep(60)

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
            mambo.smart_sleep(60)

        # done doing vision demo
        print("Ending the sleep and vision")
        mamboVision.close_video()

        mambo.smart_sleep(5)

    print("disconnecting")
    mambo.disconnect()
