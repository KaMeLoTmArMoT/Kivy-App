import gc
import os
import shutil
import subprocess
import threading
import time
from functools import partial

import cv2
import numpy as np
import torch
from kivy.clock import Clock
from kivy.factory import Factory
from kivy.graphics.texture import Texture
from kivy.uix.screenmanager import Screen
from kivymd.uix.slider import MDSlider
from sklearn.model_selection import train_test_split
from ultralytics import YOLO

from app.screens.utils.additional import BaseScreen, MDLabelBtn, MlUiHelper
from app.screens.utils.custom_logging import LazyLogger, get_logger
from app.screens.utils.db import DB
from app.screens.utils.detection_utils import PerformanceMonitor
from app.screens.utils.ml import export_to_best_available, get_best_model_paths
from app.screens.utils.tensorboard_utils import TBServer
from app.screens.utils.utils import get_system_type

logger = get_logger(__name__)
lazy_logger = LazyLogger(logger, 2.0)


class SafeMDSlider(MDSlider):
    def __init__(self, **kwargs):
        kwargs.setdefault("value_track_width", 1)  # must be > 0
        super().__init__(**kwargs)


Factory.register("SafeMDSlider", cls=SafeMDSlider)


class DetectionScreen(Screen, BaseScreen, MlUiHelper):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.camara: cv2.VideoCapture = None
        self.labelimg_process = None

        self.app_folder = os.getcwd()
        self.projects_folder = os.path.join(self.app_folder, "app/training/detection")
        os.makedirs(self.projects_folder, exist_ok=True)

        self.show_frames = False

        self.projects = []
        self.active_project = None

        self.model: YOLO = None
        self.model_name = None
        self.confidence = 0.5

        self.tb_folder = os.path.join(
            self.app_folder, "app/training/detection/tensorboard"
        )
        self.tb_server = TBServer()

        self.dropdown = None
        self.main_button = self.ids.project_label
        self.popup = None

        self.active_project_folder = None
        self.selected_model = None

        self.yolo_generation = 11

        self.video_source = None
        self.is_optimizing = False

        self.display_stats = True
        self.window_size = 100

        self.processing_flag = False
        self.frame_time = None
        self.monitor = PerformanceMonitor()

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Using device: {self.device}")

    def on_enter(self, *args):
        self.ids.header.ids[self.manager.current].background_color = 1, 1, 1, 1

        self.projects = self.get_all_projects()
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

        self.video_source = DB().get_config_typed("video_source")

        self.update_project_paths()
        self.display_camera_paused()
        self.load_model_names()

    def db_get_last_active_project(self):
        val = self.db.get_latest_detection_project()
        logger.info(f"db get: {val} {type(val)}")
        return val

    def db_set_last_active_project(self):
        self.db.set_latest_detection_project(self.active_project)

    def get_all_projects(self) -> list:
        projects = self.get_projects()
        projects = self.setup_default_project(projects)
        logger.debug(f"projects: {projects}")
        return projects

    def setup_default_project(self, projects):
        default_project = "default"
        if len(projects) == 0 or default_project not in projects:
            default_path = os.path.join(self.projects_folder, default_project)
            os.makedirs(default_path, exist_ok=True)
            projects.append(default_project)

        return projects

    def init_camera(self) -> None:
        if self.camara is not None:
            return

        if self.video_source == "file":
            video_source_path = DB().get_config_typed("video_source_path")
            logger.warning(
                f"Custom video source: {self.video_source} | {video_source_path}"
            )
            # TODO: handle errors
            self.camara = cv2.VideoCapture(video_source_path)

        else:
            if get_system_type() == "Linux":
                self.camara = cv2.VideoCapture(-1)
            else:
                self.camara = cv2.VideoCapture(0, cv2.CAP_DSHOW)

        if not self.camara.isOpened():
            logger.error("Unable to open camera")
            return

        self.camara.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        self.camara.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

        if self.video_source == "file":
            fps = self.camara.get(cv2.CAP_PROP_FPS)
            self.frame_time = 1.0 / fps
            frame_time_ms = self.frame_time / 1000.0

            logger.error(
                f"Original camera FPS: {fps} and frame time: "
                f"{self.frame_time:.3f}s, "
                f"{frame_time_ms:.3f}ms"
            )

        else:
            self.camara.set(cv2.CAP_PROP_FPS, 30)
            logger.error("Set camera FPS to 30")

    def release_camera_and_windows(self) -> None:
        cv2.destroyAllWindows()
        if self.camara is not None:
            self.camara.release()
            self.camara = None

    def display_start(self):
        if self.show_frames and self.camara is not None:
            logger.warning("Display already started.")
            return

        self.show_frames = True
        logger.debug("init")
        self.init_camera()
        logger.debug("start thread")
        threading.Thread(target=self.display_thread, daemon=True).start()
        logger.debug("after thread")

    def display_thread(self):
        while self.show_frames:
            loop_start_time = time.perf_counter()
            frame_wait_start = time.perf_counter()
            (
                ret,
                frame,
            ) = self.camara.read()  # TODO: free main thread while no input frame
            self.monitor.record("camera", frame_wait_start)

            if not ret:
                if self.video_source == "file":
                    self.camara.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    logger.info("Starting video from the beginning")
                    continue

                logger.warning("Warning: Unable to read frame from camera")
                frame = np.zeros((720, 1280, 3), dtype=np.uint8)
                self.release_camera_and_windows()
                time.sleep(0.2)
                self.init_camera()
                time.sleep(0.2)

            self.processing_flag = True
            Clock.schedule_once(partial(self.display_frame, frame))

            lock_start_time = time.perf_counter()
            while self.processing_flag:
                time.sleep(0.001)
            # TODO: manage processing_flag and fps_limiter

            elapsed = time.perf_counter() - loop_start_time
            time_to_wait = self.frame_time - elapsed
            if time_to_wait > 0:  # fps limiter
                time.sleep(time_to_wait)

            self.monitor.record("lock", lock_start_time)
            self.monitor.record("global", loop_start_time)
            lazy_logger.debug("\n" + self.monitor.report(), key="perf")

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
        self.processing_flag = False

    def display_stop(self, msg="Camera paused"):
        self.show_frames = False
        self.display_camera_paused(msg)

    def draw_text_on_frame(
        self,
        frame,
        text: str,
        position,
    ):
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 1
        color = (255, 255, 255)
        thickness = 2

        if frame.shape[2] == 3:
            cv2.putText(
                frame,
                text,
                position,
                font,
                font_scale,
                color,
                thickness,
            )
        return frame

    def display_camera_paused(self, msg="Camera paused"):
        frame = np.zeros((720, 1280, 3), dtype=np.float32)
        self.draw_text_on_frame(frame, "Camera paused", position=(40, 100))
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

        env = None
        if get_system_type() == "Linux":
            env = os.environ.copy()
            env.pop("QT_PLUGIN_PATH", None)
            env.pop("QT_QPA_FONTDIR", None)
            env.pop("QT_QPA_PLATFORM_PLUGIN_PATH", None)

        self.labelimg_process = subprocess.Popen(
            ["labelImg", pth_images, pth_classes, pth_annotations],
            env=env,
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
        if not self.selected_model:
            logger.warning("No model to load")
            return

        model_name = self.selected_model.text

        # TODO: parse for models at runs folder
        model_path = os.path.join(self.active_project_folder, model_name)
        if not os.path.exists(model_path):
            logger.warning(f"Model not found at {model_path}, downloading...")
            YOLO(model_path)

        logger.info(f"Original {model_path=}")

        available_models = get_best_model_paths(
            self.active_project_folder, model_name.split(".")[0]
        )

        for model_path, model_type in available_models:
            logger.info(f"Trying to load {model_path=}")

            try:
                self.model = YOLO(model_path, task="detect")
                self.model.overrides["verbose"] = False

                if model_type == "PyTorch":
                    try:
                        self.model.fuse()
                        logger.debug("Fuse ok")
                    except Exception as e:
                        logger.error(f"Failed to fuse model {model_path}\n{e}")

                warmup_image = np.random.randint(
                    0, 255, size=(640, 640, 3), dtype=np.uint8
                )
                self.model(warmup_image)
                logger.debug("Warmup done successfully")
                logger.warning(f"Model {model_path} initialised")
                break

            except Exception as e:
                logger.error(f"Failed to load model {model_path}\n{e}")

        self.model_name = model_name
        self.update_all_button_states()
        self.monitor.clear_timings()
        if last_display_mode:
            self.display_start()

    def yolo_inference(self, cv2_frame):
        cv2_frame = cv2_frame[:, :, ::-1]

        model_start_time = time.perf_counter()
        results = self.model(cv2_frame, conf=self.confidence)

        self.monitor.record("model", model_start_time)

        if len(results) > 1:
            logger.debug("yolo_inference: more results")

        res_plotted = results[0].plot()
        res_plotted = res_plotted[:, :, ::-1]

        return res_plotted

    def yolo_terminate(self):
        self.display_stop()
        self.model = None
        self.model_name = None
        gc.collect()
        self.unselect_model_btn()
        self.update_all_button_states()
        self.monitor.clear_timings()
        if self.show_frames:
            self.display_start()

    def update_confidence(self):
        # TODO: check why double call happens
        self.confidence = self.ids.slider.value
        logger.info(f"{self.confidence}")

    def select_project_button(self):
        projects = self.get_all_projects()
        self.setup_project_dropdown(projects)

    def after_project_selection_hook(self, project_name, path):
        self.db_set_last_active_project()
        logger.debug(f"Database updated for {project_name}")

    def restore_project_params(self, project_name, cur_project_path):
        self.update_project_paths()
        self.yolo_terminate()
        self.load_model_names()
        self.unselect_model_btn()

    def launch_tensorboard(self):
        status = self.tb_server.launch_tensorboard(self.tb_folder)
        logger.warning(f"{status}")

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
                btn = MDLabelBtn(
                    text=f"{name}{self.yolo_generation}{model}.pt",
                    theme_text_color="Custom",
                    text_color="white",
                )
                btn.bind(on_press=self.select_model_btn)
                # btn.allow_hover = True
                self.ids.model_grid.add_widget(btn)
        else:
            pass  # TODO parse models at runs folder or exported ones

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

        self.update_all_button_states()

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

    def update_all_button_states(self):
        # Highlight active model
        for btn in self.ids.model_grid.children:
            btn.text_color = "red" if btn.text == self.model_name else "white"

    def yolo_optimize(self):
        if not self.selected_model:
            logger.error("No model or name")
            return

        if self.is_optimizing:
            logger.warning("Optimization is already in progress.")
            return

        self.is_optimizing = True

        model_path = os.path.join(self.active_project_folder, self.selected_model.text)
        if not os.path.exists(model_path):
            logger.warning(f"Model not found at {model_path}, downloading...")
            YOLO(model_path)

        threading.Thread(
            target=self._threaded_export_wrapper, args=(model_path,), daemon=True
        ).start()

    def _threaded_export_wrapper(self, model_path):
        try:
            export_to_best_available(model_path, force_export=[])
            Clock.schedule_once(partial(self._on_export_complete, success=True))
        except Exception as e:
            logger.error(f"Failed to export model {self.model_name}: {e}")
            Clock.schedule_once(partial(self._on_export_complete, success=False))

    def _on_export_complete(self, dt=None, success=True):
        if success:
            logger.info("Model optimization finished successfully.")
        else:
            logger.error("Model optimization failed.")
        self.is_optimizing = False
