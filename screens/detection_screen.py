import gc
import os
import shutil
import subprocess
import threading
import webbrowser
from functools import partial

import cv2
import numpy as np
import torch
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
from screens.configs import chrome_path
from utils import call_db

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

        self.tensorboard = None
        self.tensorboard_port = 6006
        self.tensorboard_folder = os.path.join(self.app_folder, "runs", "detect")

        self.dropdown = None
        self.main_button = self.ids.project_label
        self.popup = None

        self.active_project_folder = None
        self.selected_model = None

    def on_enter(self, *args):
        self.ids.header.ids[self.manager.current].background_color = 1, 1, 1, 1
        self.create_db_and_check()

        self.projects = self.get_projects()
        latest_active_project = self.db_get_last_active_project()

        if len(latest_active_project) != 0:
            print("check latest from db")
            latest_active_project = latest_active_project[0][0]
            if latest_active_project in self.projects:
                print("use latest from db")
                self.active_project = latest_active_project

        if self.active_project is None:
            self.active_project = self.projects[0]
        self.db_set_last_active_project()
        print("active project:", self.active_project)

        self.update_project_paths()
        self.display_camera_paused()
        self.load_model_names()

        print(f"{torch.__version__=}")
        print(f"{torch.cuda.is_available()=}")

    def create_db_and_check(self):
        # Create a table
        call_db(
            """
        CREATE TABLE IF NOT EXISTS configs (
            name text unique,
            value text
        ) """
        )

    def db_get_last_active_project(self):
        val = call_db("SELECT value FROM configs WHERE name='latest_detection_project'")
        print("db get:", val, type(val))
        return val

    def db_set_last_active_project(self):
        call_db(
            f"INSERT OR REPLACE INTO configs VALUES "
            f"('latest_detection_project', '{self.active_project}')"
        )

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

        print(f"projects: {projects}")
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

        print(f"{model_path=}")
        self.model = YOLO(model_path)
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

    def yolo_terminate(self, display=True):
        self.display_stop()
        self.model = None
        gc.collect()
        self.unselect_model_btn()
        if display:
            self.display_start()

    def update_confidence(self):
        # TODO: check why double call happens
        self.confidence = self.ids.slider.value
        print(self.confidence)

    def select_project_button(self):
        projects = self.get_projects()

        # If projects folders changed or dropdown was not created
        if projects != self.projects or self.dropdown is None:
            if self.dropdown is not None:
                print("clear bind")
                self.main_button.unbind(on_release=self.dropdown.open)

            print("create bind")
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
            print("use bind")

    def open_project_folder(self, project_name):
        print("project_name", project_name)
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
        self.yolo_terminate(display=self.show_frames)
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
            print("No tensorboard folder")
            return

        if len(os.listdir(self.tensorboard_folder)) == 0:
            self.error_popup_clock("No data to show TB!")
            return
        # TODO: check freeze issue here.
        if self.tensorboard is None:
            self.tensorboard = program.TensorBoard()
            self.tensorboard.configure(argv=[None, "--logdir", self.tensorboard_folder])
            url = self.tensorboard.launch()
            print(f"{url=}")

        webbrowser.get(chrome_path).open(url)

    def update_project_paths(self):
        self.active_project_folder = os.path.join(
            self.projects_folder, self.active_project
        )

    def load_model_names(self):
        self.ids.model_grid.clear_widgets()

        if self.active_project == "default":
            models = ["n", "s", "m", "l", "x"]

            for model in models:
                btn = MDLabelBtn(text=f"yolov8{model}.pt")
                btn.bind(on_press=self.select_model_btn)
                self.ids.model_grid.add_widget(btn)
        else:
            pass  # TODO parse models at runs folder or exported ones

    def select_model_btn(self, instance):
        print(f"The model button <{instance.text}> is being pressed")
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

    def split(self):
        pth_annotations = os.path.join(
            self.projects_folder, self.active_project, "dataset\\raw\\annotations"
        )
        pth_images = os.path.join(
            self.projects_folder, self.active_project, "dataset\\raw\\images"
        )
        print(f"{pth_annotations=}, {pth_images=}")

        annotations = os.listdir(pth_annotations)
        annotations.remove("classes.txt")
        images = os.listdir(pth_images)
        print(f"all: {len(annotations)=}, {len(images)=}")

        # select images only with annotations
        selected_images = []

        for annotation in annotations:
            name = annotation.replace(".txt", ".png")  # TODO: check image type png or jpg
            if name in images:
                selected_images.append(name)

        print(f"clear: {len(annotations)=}, {len(selected_images)=}")

        X_train, X_test, y_train, y_test = train_test_split(selected_images, annotations, test_size=.2)
        print(f"{len(X_train)=} {len(y_train)=}\n{len(X_test)=} {len(y_test)=}")

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
            shutil.copy(os.path.join(pth_annotations, ann), os.path.join(out_train, "labels", ann))
            shutil.copy(os.path.join(pth_images, img), os.path.join(out_train, "images", img))

        for img, ann in zip(X_test, y_test):
            shutil.copy(os.path.join(pth_annotations, ann), os.path.join(out_test, "labels", ann))
            shutil.copy(os.path.join(pth_images, img), os.path.join(out_test, "images", img))

        class_file = os.path.join(pth_annotations, "classes.txt")
        print(class_file)
        with open(class_file, "r") as f:
            classes = f.read().split("\n")
            classes.remove("")
            print(f"{classes=}, {len(classes)=}")

        yaml_file = os.path.join(self.projects_folder, self.active_project, "dataset\\custom_dataset.yaml")
        with open(yaml_file, "w") as f:
            f.write("train: ./train\n")
            f.write("val: ./val\n")
            f.write("\n")
            f.write(f"nc: {len(classes)}\n")
            f.write("\n")
            f.write(f"names: {classes}")

        # move from out to dataset
        shutil.move(out_train, os.path.join(self.projects_folder, self.active_project, "dataset"))
        shutil.move(out_test, os.path.join(self.projects_folder, self.active_project, "dataset"))

    def train(self):
        yaml_file = os.path.join(self.projects_folder, self.active_project, "dataset\\custom_dataset.yaml")

        # TODO: use selected model
        cmd = f"yolo detect train data={yaml_file} model=yolov8m.pt epochs=30 imgsz=640"
        train_process = subprocess.Popen(
            cmd.split(" ")
        )
