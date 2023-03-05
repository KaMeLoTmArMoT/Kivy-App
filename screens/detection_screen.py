import os
import subprocess

import cv2
from kivy.uix.screenmanager import Screen

from screens.additional import BaseScreen


class DetectionScreen(Screen, BaseScreen):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.camara: cv2.VideoCapture = None
        self.labelimg_process = None

        self.app_folder = os.getcwd()
        self.projects_folder = os.path.join(self.app_folder, "projects_detection")

    def on_enter(self, *args):
        self.ids.header.ids[self.manager.current].background_color = 1, 1, 1, 1

    def init_camera(self) -> None:
        if self.camara is not None:
            return

        self.camara = cv2.VideoCapture(0)
        self.camara.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        self.camara.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        self.camara.set(cv2.CAP_PROP_FPS, 30)

    def release_camera_and_windows(self) -> None:
        cv2.destroyAllWindows()
        if self.camara is not None:
            self.camara.release()
            self.camara = None

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
