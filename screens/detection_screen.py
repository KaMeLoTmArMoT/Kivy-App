import os
import subprocess
import threading
from functools import partial

import cv2
from kivy.clock import Clock
from kivy.graphics.texture import Texture
from kivy.uix.screenmanager import Screen

from screens.additional import BaseScreen


class DetectionScreen(Screen, BaseScreen):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.camara: cv2.VideoCapture = None
        self.labelimg_process = None

        self.app_folder = os.getcwd()
        self.projects_folder = os.path.join(self.app_folder, "projects_detection")

        self.show_frames = False

    def on_enter(self, *args):
        self.ids.header.ids[self.manager.current].background_color = 1, 1, 1, 1

        self.display_start()

    def init_camera(self) -> None:
        if self.camara is not None:
            return

        self.camara = cv2.VideoCapture(0, cv2.CAP_DSHOW)
        self.camara.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        self.camara.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        self.camara.set(cv2.CAP_PROP_FPS, 30)

    def release_camera_and_windows(self) -> None:
        cv2.destroyAllWindows()
        if self.camara is not None:
            self.camara.release()
            self.camara = None

    def display_start(self):
        self.show_frames = True
        print("init")
        self.init_camera()
        print("start thread")
        # TODO: make sure only 1 thread is alive (probably by self.show_frames)
        threading.Thread(target=self.display_thread, daemon=True).start()
        print("after thread")

    def display_thread(self):
        while self.show_frames:
            ret, frame = self.camara.read()
            print("call clock")
            Clock.schedule_once(partial(self.display_frame, frame))

    def display_frame(self, frame, tm, colorfmt="bgr"):
        texture: Texture = Texture.create(
            size=(frame.shape[1], frame.shape[0]), colorfmt=colorfmt
        )
        texture.blit_buffer(
            frame.tobytes(order=None), colorfmt=colorfmt, bufferfmt="ubyte"
        )
        texture.flip_vertical()
        self.ids.image.texture = texture
        print("put texture")

    def display_stop(self):
        self.show_frames = False

    def labelimg_open(self) -> None:
        # TODO: make dynamic path for different projects
        if self.labelimg_process is not None:
            self.labelimg_close()

        pth_images = os.path.join(self.projects_folder, "cup\\dataset\\raw\\images")
        pth_classes = os.path.join(
            self.projects_folder, "cup\\dataset\\raw\\annotations\\classes.txt"
        )
        pth_annotations = os.path.join(
            self.projects_folder, "cup\\dataset\\raw\\annotations"
        )

        self.labelimg_process = subprocess.Popen(
            ["labelImg", pth_images, pth_classes, pth_annotations]
        )

    def labelimg_status(self) -> bool:
        if self.labelimg_process is not None:
            # check if alive (None - running, 1 - terminated)
            code = self.labelimg_process.poll()
            if code != 1:
                return True

            self.labelimg_process = None
        return False

    def labelimg_close(self) -> None:
        if self.labelimg_process is not None:
            self.labelimg_process.terminate()
            self.labelimg_process = None
