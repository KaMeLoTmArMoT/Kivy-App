import gc
import os
import subprocess
import threading
from functools import partial

import cv2
import numpy as np
from kivy.clock import Clock
from kivy.graphics.texture import Texture
from kivy.uix.screenmanager import Screen
from ultralytics import YOLO

from screens.additional import BaseScreen

"""
Detection projects structure:
.../app.py
├── projects_detection
│   ├── {project name}
│   │   ├── dataset
│   │   │   ├── raw
│   │   │   │   ├── annotations
│   │   │   │   │   ├── classes.txt
│   │   │   │   │   ├── {annotation}.txt
│   │   │   │   │   └── ...
│   │   │   │   ├── images
│   │   │   │   │   ├── {img}.jpg (TODO: check .png support)
│   │   │   │   │   └── ...
│   │   │   │   └── out <- temporary save train test split
│   │   │   │       ├── test
│   │   │   │       │   ├── {img1}.jpg
│   │   │   │       │   ├── {img1}.txt
│   │   │   │       │   └── ...
│   │   │   │       └── train
│   │   │   │           ├── {img2}.jpg
│   │   │   │           ├── {img2}.txt
│   │   │   │           └── ...
│   │   │   │
│   │   │   ├── train
│   │   │   │   ├── images
│   │   │   │   │   ├── {img1}.jpg
│   │   │   │   │   └── ...
│   │   │   │   ├── labels
│   │   │   │   │   ├── {img1}.txt
│   │   │   │   │   └── ...
│   │   │   │   └── labels.cache
│   │   │   │
│   │   │   ├── val
│   │   │   │   ├── images
│   │   │   │   │   ├── {img2}.jpg
│   │   │   │   │   └── ...
│   │   │   │   ├── labels
│   │   │   │   │   ├── {img2}.txt
│   │   │   │   │   └── ...
│   │   │   │   │
│   │   │   │   └── labels.cache
│   │   │   │
│   │   │   └── custom_dataset.yaml
│   │   │
│   │   └── yolov8{n/s/m/l/x}.pt  <- trained model
│   │
│   ├── {project 2 name}
...
"""


class DetectionScreen(Screen, BaseScreen):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.camara: cv2.VideoCapture = None
        self.labelimg_process = None

        self.app_folder = os.getcwd()
        self.projects_folder = os.path.join(self.app_folder, "projects_detection")

        self.show_frames = False

        self.projects = []
        self.active_project = None

        self.model: YOLO = None
        self.confidence = 0.5

    def on_enter(self, *args):
        self.ids.header.ids[self.manager.current].background_color = 1, 1, 1, 1

        self.projects = self.get_projects()
        self.active_project = self.projects[0]
        print("active project:", self.active_project)

        self.display_camera_paused()

    def get_projects(self) -> list:
        projects = os.listdir(self.projects_folder)
        if len(projects) == 0:
            default_project = "default"
            os.makedirs(default_project)
            projects.append(default_project)
        print(projects)
        return projects

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

    def display_frame(self, frame, tm=None, colorfmt="bgr"):
        if self.model is not None:
            print("use model")
            frame = self.yolo_inference(frame)

        texture: Texture = Texture.create(
            size=(frame.shape[1], frame.shape[0]), colorfmt=colorfmt
        )
        texture.blit_buffer(
            frame.tobytes(order=None), colorfmt=colorfmt, bufferfmt="ubyte"
        )
        texture.flip_vertical()
        self.ids.image.texture = texture
        print("put texture")

    def display_stop(self, msg="Camera paused"):
        self.show_frames = False
        self.display_camera_paused(msg)

    def display_camera_paused(self, msg="Camera paused"):
        frame = np.zeros((720, 1280, 3), dtype=np.float32)
        cv2.putText(
            frame,
            msg,
            (40, 100),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (255, 255, 255),
            thickness=2,
        )
        Clock.schedule_once(partial(self.display_frame, frame), 0.15)

    def labelimg_open(self) -> None:
        # TODO: make dynamic path for different projects
        if self.labelimg_process is not None:
            self.labelimg_close()

        pth_images = os.path.join(
            self.projects_folder, self.active_project, "dataset\\raw\\images"
        )
        pth_classes = os.path.join(
            self.projects_folder,
            self.active_project,
            "dataset\\raw\\annotations\\classes.txt",
        )
        pth_annotations = os.path.join(
            self.projects_folder, self.active_project, "dataset\\raw\\annotations"
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

    def yolo_load(self):
        last_display_mode = self.show_frames
        self.display_stop("Model initialize")
        Clock.schedule_once(partial(self.yolo_init, last_display_mode), 0.25)

    def yolo_init(self, last_display_mode, tm=None):
        self.model = YOLO(
            os.path.join(self.app_folder, "runs\\detect\\train3\\weights\\best.pt")
        )
        self.model.fuse()
        self.model.overrides["verbose"] = False
        print("model initialised")

        # model warmup
        self.model(np.ones((500, 500, 3)))
        print("warmup done")

        if last_display_mode:
            self.display_start()

    def yolo_inference(self, cv2_frame):
        cv2_frame = cv2_frame[:, :, ::-1]

        # TODO: confidence from UI
        results = self.model(cv2_frame, conf=self.confidence)

        if len(results) > 1:
            print("yolo_inference: more results")

        res_plotted = results[0].plot()
        res_plotted = res_plotted[:, :, ::-1]

        return res_plotted

    def yolo_terminate(self):
        self.display_stop()
        self.model = None
        gc.collect()
        self.display_start()

    def update_confidence(self):
        # TODO: check why double call happens
        self.confidence = self.ids.slider.value
        print(self.confidence)
