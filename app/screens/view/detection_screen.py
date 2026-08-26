import time
from functools import partial

import cv2
import numpy as np
from kivy.clock import Clock
from kivy.factory import Factory
from kivy.graphics.texture import Texture
from kivy.uix.screenmanager import Screen
from kivymd.uix.slider import MDSlider

from app.screens.services.camera_stream import CameraStream
from app.screens.services.detection_workspace import DetectionWorkspaceService
from app.screens.services.model_export import export_to_best_available
from app.screens.services.yolo_pipeline import YoloInferencePipeline
from app.screens.utils.additional import BaseScreen, MDLabelBtn
from app.screens.utils.custom_logging import get_logger
from app.screens.utils.detection_utils import PerformanceMonitor
from app.screens.utils.model_selection import clear_model, select_model
from app.screens.utils.project_picker import ProjectPicker
from app.screens.utils.tensorboard_utils import TBServer

logger = get_logger(__name__)


class SafeMDSlider(MDSlider):
    def __init__(self, **kwargs):
        kwargs.setdefault("value_track_width", 1)  # must be > 0
        super().__init__(**kwargs)


Factory.register("SafeMDSlider", cls=SafeMDSlider)


class DetectionScreen(Screen, BaseScreen):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.camara: cv2.VideoCapture = None
        self.camera_stream = CameraStream(self._dispatch_frame)
        self.workspace = DetectionWorkspaceService()
        self.projects_folder = str(self.workspace.projects_root)

        self.show_frames = False

        self.projects = []
        self.active_project = None

        self.pipeline = YoloInferencePipeline(confidence=0.5)

        self.tb_folder = self.workspace.tensorboard_folder
        self.tb_server = TBServer(db=self.db)

        self.dropdown = None
        self.main_button = self.ids.project_label
        self.project_picker = ProjectPicker(
            self.main_button,
            self.projects_folder,
            self.open_project_folder,
        )

        self.active_project_folder = self.workspace.active_project_folder
        self.selected_model = None

        self.yolo_generation = 11

        self.video_source = None
        self.is_optimizing = False
        self.display_stats = True
        self.window_size = 100
        self.processing_flag = False
        self.frame_time = 1 / 30
        self.monitor = PerformanceMonitor()

    @property
    def model(self):
        return self.pipeline.model

    @model.setter
    def model(self, val):
        self.pipeline.model = val

    @property
    def model_name(self):
        return self.pipeline.model_name

    @model_name.setter
    def model_name(self, val):
        self.pipeline.model_name = val

    @property
    def confidence(self):
        return self.pipeline.confidence

    @confidence.setter
    def confidence(self, val):
        self.pipeline.confidence = val

    def on_enter(self, *args):
        self.setup_header()

        self.projects = self.get_all_projects()
        db_project = self.db.get_latest_detection_project()

        if db_project and db_project[0][0] in self.projects:
            self.active_project = db_project[0][0]

        if self.active_project is None:
            self.active_project = self.projects[0]
        self.db.set_latest_detection_project(self.active_project)
        logger.info(f"active project: {self.active_project}")

        self.video_source = self.db.get_config_typed("video_source")

        self.update_project_paths()
        self.display_camera_paused()
        self.load_model_names()

    def get_all_projects(self) -> list:
        projects = self.setup_default_project(self.workspace.list_projects())
        logger.debug(f"projects: {projects}")
        return projects

    def select_project_button(self):
        self.project_picker.open()
        self.dropdown = self.project_picker.dropdown
        self.projects = self.project_picker.projects

    def open_project_folder(self, project_name: str):
        project_path = self.workspace.ensure_project(project_name)
        self.main_button.text = project_name
        self.after_project_selection_hook(project_name, project_path)
        self.active_project = project_name
        self.restore_project_params(project_name, project_path)

    def select_model_btn(self, instance):
        select_model(self, instance)

    def unselect_model_btn(self):
        clear_model(self)

    def setup_default_project(self, projects):
        default_project = "default"
        if len(projects) == 0 or default_project not in projects:
            self.workspace.ensure_project(default_project)
            projects.append(default_project)

        return projects

    def init_camera(self) -> bool:
        video_path = None
        if self.video_source == "file":
            video_path = self.db.get_config_typed("video_source_path")
            logger.warning(f"Custom video source: {self.video_source} | {video_path}")

        opened = self.camera_stream.open(self.video_source or "webcam", video_path)
        self.camara = self.camera_stream.capture
        self.frame_time = self.camera_stream.frame_time
        return opened

    def release_camera_and_windows(self) -> None:
        self.camera_stream.stop()
        self.camara = None

    def display_start(self):
        if self.show_frames and self.camera_stream.running.is_set():
            logger.warning("Display already started.")
            return

        self.show_frames = True
        if not self.init_camera():
            self.show_frames = False
            return
        self.camera_stream.start()

    def _dispatch_frame(self, frame, frame_done):
        if self.show_frames:
            Clock.schedule_once(lambda _: self.display_frame(frame, frame_done))
        else:
            frame_done.set()

    def display_frame(self, frame, frame_done=None, colorfmt="bgr"):
        if self.model is not None:
            frame = self.yolo_inference(frame)

        texture: Texture = Texture.create(size=(frame.shape[1], frame.shape[0]), colorfmt=colorfmt)
        texture.blit_buffer(frame.tobytes(order=None), colorfmt=colorfmt, bufferfmt="ubyte")
        texture.flip_vertical()
        self.ids.image.texture = texture
        self.processing_flag = False
        if frame_done is not None:
            frame_done.set()

    def display_stop(self, msg="Camera paused"):
        self.show_frames = False
        self.release_camera_and_windows()
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
        Clock.schedule_once(lambda _: self.display_frame(frame), 0.15)

    def labelimg_open(self) -> None:
        self.workspace.open_labelimg()

    def labelimg_status(self) -> bool:
        return self.workspace.labelimg_running()

    def labelimg_close(self) -> None:
        self.workspace.close_labelimg()

    def yolo_load(self):
        last_display_mode = self.show_frames
        self.display_stop("Model initialize")
        Clock.schedule_once(partial(self.yolo_init, last_display_mode), 0.25)

    def yolo_init(self, last_display_mode, tm=None):
        if not self.pipeline.is_available:
            logger.error("ultralytics not installed. Install ultralytics to use detection.")
            return

        if not self.selected_model:
            logger.warning("No model to load")
            return

        model_name = self.selected_model.text
        model_path = self.workspace.model_path(model_name)
        if not self.pipeline.ensure_model(model_path):
            return

        self.pipeline.load_model(self.active_project_folder, model_name)
        self.update_all_button_states()
        self.monitor.clear_timings()
        if last_display_mode:
            self.display_start()

    def yolo_inference(self, cv2_frame):
        model_start_time = time.perf_counter()
        res_plotted = self.pipeline.infer_frame(cv2_frame)
        self.monitor.record("model", model_start_time)
        return res_plotted

    def yolo_terminate(self):
        was_showing = self.show_frames
        self.display_stop()
        self.pipeline.unload()
        self.unselect_model_btn()
        self.update_all_button_states()
        self.monitor.clear_timings()
        if was_showing:
            self.display_start()

    def update_confidence(self):
        # TODO: check why double call happens
        self.confidence = self.ids.slider.value
        logger.info(f"{self.confidence}")

    def after_project_selection_hook(self, project_name, path):
        self.db.set_latest_detection_project(project_name)
        logger.debug(f"Database updated for {project_name}")

    def restore_project_params(self, project_name, cur_project_path):
        self.update_project_paths()
        self.yolo_terminate()
        self.load_model_names()
        self.unselect_model_btn()

    def update_project_paths(self):
        self.workspace.set_active_project(str(self.active_project))
        self.active_project_folder = self.workspace.active_project_folder

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

    def update_value(self, increment):
        current_value = int(self.ids.label_spinner.text)
        new_value = current_value + increment

        if 8 <= new_value <= 11:
            self.ids.label_spinner.text = str(new_value)
            self.yolo_generation = new_value
            self.load_model_names()

        self.update_all_button_states()

    def split(self):
        self.workspace.split_dataset()

    def train(self):
        self.workspace.start_training()

    def update_all_button_states(self):
        # Highlight active model
        for btn in self.ids.model_grid.children:
            btn.text_color = "red" if btn.text == self.model_name else "white"

    def yolo_optimize(self):
        if not self.pipeline.is_available:
            logger.error("ultralytics not installed. Install ultralytics to use detection.")
            return

        if not self.selected_model:
            logger.error("No model or name")
            return

        if self.is_optimizing:
            logger.warning("Optimization is already in progress.")
            return

        self.is_optimizing = True

        model_path = self.workspace.model_path(self.selected_model.text)
        if not self.pipeline.ensure_model(model_path):
            self.is_optimizing = False
            return

        def _worker():
            export_to_best_available(model_path, force_export=[])
            return True

        def _on_done(success):
            if success:
                logger.info("Model optimization finished successfully.")
            else:
                logger.error("Model optimization failed.")
            self.is_optimizing = False

        self.run_async(_worker, _on_done)
