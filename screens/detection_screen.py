import gc
import os
import shutil
import subprocess
import threading
import time
import webbrowser
from functools import partial

import cv2
import numpy as np
from kivy.clock import Clock
from kivy.graphics.texture import Texture
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.button import Button
from kivy.uix.dropdown import DropDown
from kivy.uix.label import Label
from kivy.uix.popup import Popup
from kivy.uix.screenmanager import Screen
from kivy.uix.textinput import TextInput
from sklearn.model_selection import train_test_split
from tensorboard import program
from ultralytics import YOLO

from screens.additional import BaseScreen, MDLabelBtn
from screens.custom_logging import get_logger
from screens.db import DB
from utils import get_system_type

logger = get_logger(__name__)
chrome_path = DB().get_config_typed("chrome_path")


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
        os.makedirs(self.projects_folder, exist_ok=True)

        self.show_frames = False

        self.projects = []
        self.active_project = None

        self.model: YOLO = None
        self.confidence = 0.5

        self.tensorboard = None
        self.tensorboard_port = 6006
        self.tensorboard_folder = os.path.join(self.app_folder, "runs", "detect")

        self.dropdown = None
        self.main_button = self.ids.project_label
        self.popup = None

        self.active_project_folder = None
        self.selected_model = None

        self.yolo_generation = 11

    def on_enter(self, *args):
        self.ids.header.ids[self.manager.current].background_color = 1, 1, 1, 1

        self.projects = self.get_projects()
        latest_active_project = self.db_get_last_active_project()

        if len(latest_active_project) != 0:
            logger.debug("check latest from db")
            latest_active_project = latest_active_project[0][0]
            if latest_active_project in self.projects:
                logger.debug("use latest from db")
                self.active_project = latest_active_project

        if self.active_project is None:
            self.active_project = self.projects[0]
        self.db_set_last_active_project()
        logger.info(f"active project: {self.active_project}")

        self.update_project_paths()
        self.display_camera_paused()
        self.load_model_names()

    def db_get_last_active_project(self):
        val = self.db.get_latest_detection_project()
        logger.info(f"db get: {val} {type(val)}")
        return val

    def db_set_last_active_project(self):
        self.db.set_latest_detection_project(self.active_project)

    def get_projects(self) -> list:
        projects = []
        for folder in os.listdir(self.projects_folder):
            if os.path.isdir(os.path.join(self.projects_folder, folder)):
                projects.append(folder)

        default_project = "default"
        if len(projects) == 0 or default_project not in projects:
            default_path = os.path.join(self.projects_folder, default_project)
            os.makedirs(default_path, exist_ok=True)
            projects.append(default_project)

        logger.debug(f"projects: {projects}")
        return projects

    def init_camera(self) -> None:
        if self.camara is not None:
            return

        if get_system_type() == "Linux":
            self.camara = cv2.VideoCapture(-1)
        else:
            self.camara = cv2.VideoCapture(0, cv2.CAP_DSHOW)

        if not self.camara.isOpened():
            logger.warning("Error: Unable to open camera")
            return

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
        logger.debug("init")
        self.init_camera()
        logger.debug("start thread")
        # TODO: make sure only 1 thread is alive (probably by self.show_frames)
        threading.Thread(target=self.display_thread, daemon=True).start()
        logger.debug("after thread")

    def display_thread(self):
        while self.show_frames:
            ret, frame = self.camara.read()
            if not ret:
                logger.warning("Warning: Unable to read frame from camera")
                frame = np.zeros((720, 1280, 3), dtype=np.uint8)
                self.release_camera_and_windows()
                time.sleep(0.25)
                self.init_camera()
                time.sleep(0.25)
            Clock.schedule_once(partial(self.display_frame, frame))
        self.display_stop()

    def display_frame(self, frame, tm=None, colorfmt="bgr"):
        if self.model is not None:
            frame = self.yolo_inference(frame)

        texture: Texture = Texture.create(
            size=(frame.shape[1], frame.shape[0]), colorfmt=colorfmt
        )
        texture.blit_buffer(
            frame.tobytes(order=None), colorfmt=colorfmt, bufferfmt="ubyte"
        )
        texture.flip_vertical()
        self.ids.image.texture = texture

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
        if self.selected_model:
            if self.active_project == "default":
                model_path = os.path.join(
                    self.active_project_folder, self.selected_model.text
                )
            else:
                # TODO: parse for models at runs folder
                model_path = os.path.join(
                    self.active_project_folder, self.selected_model.text
                )
        else:
            model_path = os.path.join(
                self.app_folder, "runs\\detect\\train3\\weights\\best.pt"
            )

        logger.info(f"{model_path=}")
        self.model = YOLO(model_path)
        self.model.fuse()
        self.model.overrides["verbose"] = False
        logger.debug("model initialised")

        # model warmup
        self.model(np.ones((500, 500, 3)))
        logger.debug("warmup done")

        if last_display_mode:
            self.display_start()

    def yolo_inference(self, cv2_frame):
        cv2_frame = cv2_frame[:, :, ::-1]

        # TODO: confidence from UI
        results = self.model(cv2_frame, conf=self.confidence)

        if len(results) > 1:
            logger.debug("yolo_inference: more results")

        res_plotted = results[0].plot()
        res_plotted = res_plotted[:, :, ::-1]

        return res_plotted

    def yolo_terminate(self):
        self.display_stop()
        self.model = None
        gc.collect()
        self.unselect_model_btn()
        if self.show_frames:
            self.display_start()

    def update_confidence(self):
        # TODO: check why double call happens
        self.confidence = self.ids.slider.value
        logger.info(f"{self.confidence}")

    def select_project_button(self):
        projects = self.get_projects()

        # If projects folders changed or dropdown was not created
        if projects != self.projects or self.dropdown is None:
            if self.dropdown is not None:
                logger.debug("clear bind")
                self.main_button.unbind(on_release=self.dropdown.open)

            logger.debug("create bind")
            self.dropdown = DropDown()
            for folder in projects:
                btn = Button(text=f"{folder}", size_hint_y=None, height=44)
                btn.bind(on_release=lambda b: self.dropdown.select(b.text))
                self.dropdown.add_widget(btn)

            btn_new = Button(text="New project", size_hint_y=None, height=44)
            btn_new.bind(on_release=lambda b: self.dropdown.select(b.text))
            btn_new.background_color = 0.5, 0.9, 0.5, 1
            self.dropdown.add_widget(btn_new)

            self.main_button.bind(on_release=self.dropdown.open)
            self.dropdown.bind(
                on_select=lambda instance, project: self.open_project_folder(project)
            )
            self.projects = projects
        else:
            logger.debug("use bind")

    def open_project_folder(self, project_name):
        logger.info(f"project_name {project_name}")
        if project_name == "":
            return

        if self.popup is not None:
            self.popup.dismiss()

        # trigger popup and then call this method again with correct name
        if project_name == "New project":
            self.create_project_name_input_popup()
            return

        self.main_button.text = project_name
        cur_project_path = os.path.join(self.projects_folder, project_name)
        os.makedirs(cur_project_path, exist_ok=True)

        self.active_project = project_name
        self.restore_project_params(project_name, cur_project_path)
        self.db_set_last_active_project()

    def restore_project_params(self, project_name, cur_project_path):
        self.update_project_paths()
        self.yolo_terminate()
        self.load_model_names()
        self.unselect_model_btn()

    def create_project_name_input_popup(self):
        self.popup = Popup(
            title="New project creation", size_hint=(None, None), size=(400, 150)
        )
        box = BoxLayout(orientation="vertical")

        lbl = Label(text="Please enter new name", size_hint_y=0.3)

        name_input = TextInput(
            text="",
            hint_text="Project name",
            size_hint_y=0.4,
            multiline=False,
            font_size=16,
        )
        name_input.bind(
            on_text_validate=lambda x: self.open_project_folder(
                name_input.text,
            ),
        )

        submit_btn = MDLabelBtn(text="Create", size_hint_y=0.3)
        submit_btn.bind(
            on_release=lambda x: self.open_project_folder(
                name_input.text,
            )
        )
        submit_btn.allow_hover = True

        box.add_widget(lbl)
        box.add_widget(name_input)
        box.add_widget(submit_btn)

        self.popup.content = box
        self.popup.open()

    def launch_tensorboard(self):
        # TODO: move to base and define different ports for projects

        if not os.path.isdir(self.tensorboard_folder):
            logger.warning("No tensorboard folder")
            return

        if len(os.listdir(self.tensorboard_folder)) == 0:
            self.error_popup_clock("No data to show TB!")
            return
        # TODO: check freeze issue here.
        if self.tensorboard is None:
            self.tensorboard = program.TensorBoard()
            self.tensorboard.configure(argv=[None, "--logdir", self.tensorboard_folder])
            url = self.tensorboard.launch()
            logger.info(f"{url=}")

        webbrowser.get(chrome_path).open(url)

    def update_project_paths(self):
        self.active_project_folder = os.path.join(
            self.projects_folder, self.active_project
        )

    def load_model_names(self):
        self.ids.model_grid.clear_widgets()

        if self.active_project == "default":
            name = "yolov"
            if self.yolo_generation == 8:
                models = ["n", "s", "m", "l", "x"]
            elif self.yolo_generation == 9:
                models = ["t", "s", "m", "c", "e"]
            elif self.yolo_generation == 10:
                models = ["n", "s", "m", "b", "l", "x"]
            else:  # 11 gen default
                models = ["n", "s", "m", "l", "x"]
                name = "yolo"

            for model in models:
                btn = MDLabelBtn(text=f"{name}{self.yolo_generation}{model}.pt")
                btn.bind(on_press=self.select_model_btn)
                btn.allow_hover = True
                self.ids.model_grid.add_widget(btn)
        else:
            pass  # TODO parse models at runs folder or exported ones

    def select_model_btn(self, instance):
        logger.debug(f"The model button <{instance.text}> is being pressed")
        if self.selected_model:
            if instance.uid == self.selected_model.uid:
                # custom double touch event
                self.unselect_model_btn()
                return

        # reset selection
        for btn in self.ids.model_grid.children:
            btn.md_bg_color = (1.0, 1.0, 1.0, 0.0)

        instance.md_bg_color = (1.0, 1.0, 1.0, 0.1)
        instance.radius = (20, 20, 20, 20)
        self.selected_model = instance

    def unselect_model_btn(self):
        self.selected_model = None
        for btn in self.ids.model_grid.children:
            btn.md_bg_color = (1.0, 1.0, 1.0, 0.0)

    def update_value(self, increment):
        current_value = int(self.ids.label_spinner.text)
        new_value = current_value + increment

        if 8 <= new_value <= 11:
            self.ids.label_spinner.text = str(new_value)
            self.yolo_generation = new_value
            self.load_model_names()

    def split(self):
        pth_annotations = os.path.join(
            self.projects_folder, self.active_project, "dataset\\raw\\annotations"
        )
        pth_images = os.path.join(
            self.projects_folder, self.active_project, "dataset\\raw\\images"
        )
        logger.info(f"{pth_annotations=}, {pth_images=}")

        annotations = os.listdir(pth_annotations)
        annotations.remove("classes.txt")
        images = os.listdir(pth_images)
        logger.info(f"all: {len(annotations)=}, {len(images)=}")

        # select images only with annotations
        selected_images = []

        for annotation in annotations:
            name = annotation.replace(
                ".txt", ".png"
            )  # TODO: check image type png or jpg
            if name in images:
                selected_images.append(name)

        logger.info(f"clear: {len(annotations)=}, {len(selected_images)=}")

        X_train, X_test, y_train, y_test = train_test_split(
            selected_images, annotations, test_size=0.2
        )
        logger.info(f"{len(X_train)=} {len(y_train)=}\n{len(X_test)=} {len(y_test)=}")

        # TODO: create target dirs

        out_train = os.path.join(
            self.projects_folder, self.active_project, "dataset\\raw\\out\\train"
        )
        out_test = os.path.join(
            self.projects_folder, self.active_project, "dataset\\raw\\out\\val"
        )

        os.makedirs(out_train, exist_ok=True)
        os.makedirs(out_test, exist_ok=True)
        os.makedirs(os.path.join(out_train, "labels"), exist_ok=True)
        os.makedirs(os.path.join(out_train, "images"), exist_ok=True)
        os.makedirs(os.path.join(out_test, "labels"), exist_ok=True)
        os.makedirs(os.path.join(out_test, "images"), exist_ok=True)

        for img, ann in zip(X_train, y_train):
            shutil.copy(
                os.path.join(pth_annotations, ann),
                os.path.join(out_train, "labels", ann),
            )
            shutil.copy(
                os.path.join(pth_images, img), os.path.join(out_train, "images", img)
            )

        for img, ann in zip(X_test, y_test):
            shutil.copy(
                os.path.join(pth_annotations, ann),
                os.path.join(out_test, "labels", ann),
            )
            shutil.copy(
                os.path.join(pth_images, img), os.path.join(out_test, "images", img)
            )

        class_file = os.path.join(pth_annotations, "classes.txt")
        logger.debug(f"{class_file}")
        with open(class_file, "r") as f:
            classes = f.read().split("\n")
            classes.remove("")
            logger.debug(f"{classes=}, {len(classes)=}")

        yaml_file = os.path.join(
            self.projects_folder, self.active_project, "dataset\\custom_dataset.yaml"
        )
        with open(yaml_file, "w") as f:
            f.write("train: ./train\n")
            f.write("val: ./val\n")
            f.write("\n")
            f.write(f"nc: {len(classes)}\n")
            f.write("\n")
            f.write(f"names: {classes}")

        # move from out to dataset
        shutil.move(
            out_train,
            os.path.join(self.projects_folder, self.active_project, "dataset"),
        )
        shutil.move(
            out_test, os.path.join(self.projects_folder, self.active_project, "dataset")
        )

    def train(self):
        yaml_file = os.path.join(
            self.projects_folder, self.active_project, "dataset\\custom_dataset.yaml"
        )

        # TODO: use selected model
        cmd = f"yolo detect train data={yaml_file} model=yolov8m.pt epochs=30 imgsz=640"
        train_process = subprocess.Popen(cmd.split(" "))
        logger.debug(f"{train_process}")
