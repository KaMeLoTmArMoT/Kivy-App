import datetime
import io
import os
import shutil
import time
from math import ceil

from kivy.clock import Clock
from kivy.core.image import Image as CoreImage
from kivy.metrics import dp
from kivy.properties import ListProperty
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.progressbar import ProgressBar
from kivy.uix.screenmanager import Screen
from kivymd.uix.label import MDLabel
from PIL import Image

from app.screens.services.classification import KModel, prepare_dataset
from app.screens.services.image_library_service import ImageLibraryService
from app.screens.services.ml_model_service import MLModelService
from app.screens.services.ml_workspace import MLWorkspaceManager
from app.screens.utils.additional import (
    BaseScreen,
    ImageMDButton,
    MDLabelBtn,
    SelectableImage,
)
from app.screens.utils.custom_logging import get_logger
from app.screens.utils.image_loader import ImageLoadController
from app.screens.utils.model_selection import clear_model, select_model
from app.screens.utils.model_type_dialog import build_model_type_popup
from app.screens.utils.project_picker import ProjectPicker
from app.screens.utils.tensorboard_utils import TBServer
from app.screens.utils.utils import extend_key

logger = get_logger(__name__)


class MLViewScreen(Screen, BaseScreen):
    rgba = ListProperty([1, 1, 0, 0])  # error message popup color

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.key = ""
        self.selected_dir = None
        self.selected_dir_full = None
        self.selected_images = []
        self.images_to_load = []
        self.progress_bar: ProgressBar = self.ids.progress_bar
        self.touch_time = time.time()
        self.train_active = False
        self.load_event = None
        self.image_loader = None
        self.page = 1
        self.total_pages = None

        self.selected_model = None
        self.model_name = None
        self.total_steps = None
        self.model_type = "MobileNetV2"
        self.model_type_popup = None
        self.tmp_model_type = None
        self.tb_server = TBServer(db=self.db)

        self.k_model = KModel(db=self.db)
        self.image_service = ImageLibraryService(db=self.db)

        self.eval_event = None

        self.loaded_hash = ""

        self.cur_dir = ""

        self.workspace = MLWorkspaceManager(active_project="Kivy")
        self.model_service = MLModelService(self.workspace, self.k_model, db=self.db)

        self.main_button = self.ids.project_label
        self.project_picker = ProjectPicker(
            self.main_button,
            str(self.projects_folder),
            self.open_project_folder,
        )

        self.max_images_per_page = 20

        self.num_predictions = 0
        self._inputs_bound = False
        self.dropdown = None
        self.projects = []

    @property
    def active_project(self) -> str:
        return self.workspace.active_project

    @active_project.setter
    def active_project(self, val: str) -> None:
        self.workspace.set_active_project(val)

    projects_folder = property(lambda self: self.workspace.projects_root)
    active_project_folder = property(lambda self: self.workspace.active_project_folder)
    images_path = property(lambda self: self.workspace.images_path)
    ml_train_folder = property(lambda self: self.workspace.ml_train_folder)
    ml_configs_folder = property(lambda self: self.workspace.ml_configs_folder)
    ml_models_folder = property(lambda self: self.workspace.ml_models_folder)
    tb_folder = property(lambda self: self.workspace.tb_folder)

    def on_enter(self, *args):
        self.setup_header()
        login_key = getattr(self.manager.get_screen("login"), "key", None)
        if not login_key:
            logger.warning("MLVIEW: on_enter skipped due to empty login key")
            return
        self.key = extend_key(login_key)
        self.load_classes()
        self.load_model_names()

        dir_hash = self.workspace.get_images_hash()

        self.max_images_per_page = self.db.get_config_typed("MAX_IMAGES_PER_PAGE")
        if self.loaded_hash != dir_hash:
            self.show_folder_images(path=self.images_path)

        self.exit_screen = False
        self.main_button.text = self.active_project

        if not getattr(self, "_inputs_bound", False):
            self.ids.class_input.bind(text=self.on_text_input_class)
            self.ids.model_input.bind(text=self.on_text_input_model)
            self._inputs_bound = True

        try:
            self.k_model.update_params()
        except RuntimeError as e:
            if "PyTorch is not installed" in str(e):
                self.label_out("ML features disabled (torch not installed).")
                return
            raise

    def update_project_paths(self):
        self.workspace.ensure_workspace()

    def select_project_button(self):
        self.project_picker.open()
        self.dropdown = self.project_picker.dropdown
        self.projects = self.project_picker.projects

    def open_project_folder(self, project_name: str):
        project_path = os.path.join(str(self.projects_folder), project_name)
        os.makedirs(project_path, exist_ok=True)
        self.main_button.text = project_name
        self.after_project_selection_hook(project_name, project_path)
        self.active_project = project_name
        self.restore_project_params(project_name, project_path)

    def get_all_projects(self) -> list[str]:
        return self.workspace.list_projects()

    def select_model_btn(self, instance):
        select_model(self, instance)

    def unselect_model_btn(self):
        clear_model(self)

    def load_classes(self):
        self.ids.class_grid.clear_widgets()

        btn = MDLabelBtn(
            text="all",
            theme_text_color="Custom",
            text_color="red",
        )
        btn.bind(on_press=self.select_label_btn)
        self.ids.class_grid.add_widget(btn)

        classes = self.workspace.list_classes()
        if not classes:
            logger.warning("No classes folder or empty classes")

        for class_name in classes:
            btn = MDLabelBtn(
                text="train/" + class_name,
                theme_text_color="Custom",
                text_color="white",
            )
            btn.bind(on_press=self.select_label_btn)
            self.ids.class_grid.add_widget(btn)

    def select_label_btn(self, instance):
        logger.info(f"The label button <{instance.text}> is being pressed")

        current_time = time.time()
        double_click_threshold = 0.3
        is_same_button = self.selected_dir and instance.uid == self.selected_dir.uid
        time_diff = current_time - getattr(self, "touch_time", 0)

        # TODO: fix if already selected
        if is_same_button:
            if time_diff < double_click_threshold:
                if not self.ids.open_class.disabled:
                    path = os.path.join(self.active_project_folder, instance.text)
                    logger.info(f"Double-click: opening {path}")
                    self.show_folder_images(path, new=True)
                return
            else:
                logger.info("Single-click: deselect")
                self.unselect_label_btn()
                return

        # reset selection
        for btn in self.ids.class_grid.children:
            btn.md_bg_color = (1.0, 1.0, 1.0, 0.0)

        instance.md_bg_color = (1.0, 1.0, 1.0, 0.1)
        instance.radius = (20, 20, 20, 20)
        self.selected_dir = instance
        self.selected_dir_full = (
            os.path.join(self.active_project_folder, self.selected_dir.text)
            if self.selected_dir
            else None
        )
        self.touch_time = current_time

        self.ids.delete_class.disabled = instance.text == "all"
        self.ids.open_class.disabled = self.selected_dir_full == self.cur_dir

        self.update_all_button_states()

    def unselect_label_btn(self):
        self.selected_dir = None
        self.selected_dir_full = None
        for btn in self.ids.class_grid.children:
            btn.md_bg_color = (1.0, 1.0, 1.0, 0.0)

        self.ids.delete_class.disabled = True
        self.ids.open_class.disabled = True
        self.update_all_button_states()

    def add_class(self):
        name = self.ids.class_input.text
        if not name:
            self.error_popup_clock("Enter name!")
            return

        try:
            self.workspace.add_class(name)
        except FileExistsError:
            self.error_popup_clock("Class exists!")
            return

        self.load_classes()
        self.ids.class_input.text = ""

    def error_popup_clock(self, text="Error", show_time=1):
        self.toggle_error_popup("on", text)
        eff_time = 0.01 if os.getenv("APP_ENV") == "test" else show_time
        Clock.schedule_once(lambda tm: self.toggle_error_popup("off"), eff_time)

    def toggle_error_popup(self, mode, text="Error"):
        if mode == "on":
            self.ids.error_popup_text.text = text
            self.ids.error_popup.size_hint_y = 0.1
            self.rgba = [1, 1, 0, 1]
        else:  # off
            self.ids.error_popup_text.text = ""
            self.ids.error_popup.size_hint_y = 0.0
            self.rgba = [1, 1, 0, 0]

    def delete_class(self):
        if self.selected_dir is None:
            self.error_popup_clock("Select class dir!")
            return

        if self.selected_dir.text == "all":
            self.error_popup_clock("Can`t delete main dir!")
            return

        self.workspace.delete_class_folder(str(self.selected_dir_full))

        self.unselect_label_btn()
        self.load_classes()

        # TODO: if delete cur dataset - switch to all tab

    def disable_switch_buttons(self):
        self.ids.prev_page.disabled = True
        self.ids.next_page.disabled = True

    def enable_switch_buttons(self):
        self.ids.prev_page.disabled = False
        self.ids.next_page.disabled = False

    def toggle_switch_buttons(self):
        if self.total_pages == 0:
            self.ids.page_selector.disabled = True
            self.ids.page_selector.opacity = 0
        else:
            self.ids.page_selector.disabled = False
            self.ids.page_selector.opacity = 1

    def show_folder_images(self, path=None, new=False):
        if self.selected_dir is None and path is None:
            self.error_popup_clock("Select dir!")
            return

        self.toggle_load_label("on")
        if path is None:  # TODO: re-check if we call without path
            path = self.selected_dir_full

        if not os.path.isdir(path):
            self.toggle_load_label("no_dir")
            return

        if self.image_loader is not None:
            self.image_loader.stop()
            self.load_event = None

        files = self.image_service.get_supported_images_in_dir(path)

        self.ids.image_grid.clear_widgets()
        self.unselect_all_images()
        self.images_to_load.clear()

        if new:
            self.page = 1
            logger.debug("reset page")

            for btn in self.ids.class_grid.children:
                btn.text_color = "white"

            if self.selected_dir is not None:
                self.selected_dir.text_color = "red"

        self.disable_switch_buttons()  # disable load button
        self.cur_dir = path

        n_images = len(files)
        self.total_pages = ceil(n_images / int(self.max_images_per_page))
        self.toggle_switch_buttons()
        self.ids.open_class.disabled = True

        self.update_page_counter()
        page_size = int(self.max_images_per_page)
        start = (self.page - 1) * page_size
        self.images_to_load = files[start : start + page_size]

        if not self.images_to_load:
            self.toggle_load_label("no_dir")
            self.enable_switch_buttons()
            return

        self.image_loader = ImageLoadController(
            self.ids.image_grid,
            self.progress_bar,
            self._create_image_widget,
            self._image_load_finished,
            lambda: self.exit_screen,
        )
        self.load_event = self.image_loader.start(self.images_to_load)
        self.images_to_load = self.image_loader.pending

    def update_page_counter(self):  # TODO: reset page when open new folder
        self.ids.page_label.text = f"{self.page}/{self.total_pages}"

    def _create_image_widget(self, im_path):
        selectable_img = SelectableImage(source=im_path)
        img = selectable_img.ids.img
        img.nocache = True
        img.bind(on_press=self.image_click)

        # 3) A fixed-height label container at the very bottom
        label_container = BoxLayout(
            size_hint=(1, None),
            height=dp(30),
            pos_hint={"x": 0, "y": 0},
            padding=[dp(4), 0],
            spacing=dp(4),
        )
        # Store it for later:
        img.label_container = label_container

        selectable_img.add_widget(label_container)
        return selectable_img

    def _image_load_finished(self):
        self.images_to_load = self.image_loader.pending
        self.load_event = None
        self.loaded_hash = self.workspace.get_images_hash()
        self.toggle_load_label("success")
        self.enable_switch_buttons()

    def unselect_all_images(self):
        # Work on a copy since we'll mutate the original list
        for instance in list(self.selected_images):
            instance.parent.selected = False
            self.selected_images.remove(instance)

        self.update_all_button_states()

    def select_all_images(self):
        # Nothing to select
        if not self.ids.image_grid.children:
            self.update_all_button_states()
            return

        # Each tile is SelectableImage containing
        # ImageMDButton + MDCheckbox + labelcontainer
        for tile in list(self.ids.image_grid.children):
            if isinstance(tile, SelectableImage):
                img = tile.ids.img
                if img not in self.selected_images:
                    self.image_click(img)

        self.update_all_button_states()

    def clear_predictions(self):
        for tile in self.ids.image_grid.children:
            for child in tile.children:
                if isinstance(child, ImageMDButton) and hasattr(child, "label_container"):
                    child.label_container.clear_widgets()
        self.num_predictions = 0
        self.unselect_all_images()

    def image_click(self, instance):
        # path = instance.source
        selectable_img = instance.parent

        if instance in self.selected_images:
            selectable_img.selected = False
            self.selected_images.remove(instance)
        else:
            selectable_img.selected = True
            self.selected_images.append(instance)

        self.update_all_button_states()

    def transfer_images(self):
        if len(self.selected_images) == 0 or self.selected_dir is None:
            self.error_popup_clock("Select images and dir!")
            return

        logger.debug(f"{self.cur_dir}")
        logger.debug(f"{self.selected_dir_full}")

        if self.cur_dir == self.selected_dir_full:
            self.error_popup_clock("Can`t paste to same dir!")
            return

        for image in self.selected_images:
            out_img = image.source.replace(self.cur_dir, self.selected_dir_full)
            shutil.move(image.source, out_img)

        self.unselect_all_images()
        self.unselect_label_btn()
        self.show_folder_images(self.cur_dir)

    def trigger_training(self):
        if self.train_active:
            logger.warning("Training termination requested")
            self.k_model.terminate_training = True
            self.ids.train_btn.text = "Train"
            self.ids.train_btn.disabled = True  # Temporarily disable until cleanup
            return

        if self.k_model.model is None:
            if self.selected_model:
                self.load_model()
            else:
                self.error_popup_clock("Select or load model!")
                return

        self.train_active = True
        self.k_model.terminate_training = False
        self.ids.train_btn.text = "Stop"
        self.ids.train_btn.disabled = False
        self.update_all_button_states()
        self.error_popup_clock("Open tensorboard to get status.", 5)
        self.run_async(self.train_model, self._on_train_done)

    def _on_train_done(self, _res=None):
        self.train_active = False
        self.k_model.terminate_training = False
        self.ids.train_btn.text = "Train"
        self.ids.train_btn.disabled = False
        self.update_all_button_states()
        self.save_model()

    def prev_page(self):
        if self.page > 1:
            if self.image_loader:
                self.image_loader.stop()

            self.page -= 1
            self.show_folder_images(self.cur_dir)

    def next_page(self):
        if self.page < (self.total_pages or 0):
            if self.image_loader:
                self.image_loader.stop()

            self.page += 1
            self.show_folder_images(self.cur_dir)

    def train_model(self):
        data = prepare_dataset(
            self.ml_train_folder,
            self.k_model.transform,
            batch_size=self.db.get_config_typed("train_batch_size"),
            shuffle=True,
        )

        log_dir = os.path.join(
            self.tb_folder,
            datetime.datetime.now().strftime("%Y_%m_%d-%H_%M") + f"_{self.model_name}",
        )

        self.k_model.train_model(data, log_dir)

    def select_model_type(self):
        self.model_type_popup = build_model_type_popup(
            self.model_type,
            self.select_model_type_btn,
            self.submit_model_type_btn,
            self.dismiss_model_select_popup,
        )
        self.model_type_popup.open()

    def dismiss_model_select_popup(self, instance):
        self.tmp_model_type = None

    def select_model_type_btn(self, instance):
        grid = instance.parent

        if (
            self.tmp_model_type == instance.text.split(" ")[0]
            and time.time() - self.touch_time < 0.2
        ):
            self.submit_model_type_btn("instance")  # may cause error

        for btn_name in grid.ids:
            grid.ids[btn_name].background_color = (1.0, 1.0, 1.0, 1.0)

        instance.background_color = (1.0, 1.0, 1.0, 0.5)
        self.tmp_model_type = instance.text.split(" ")[0]
        self.touch_time = time.time()

    def submit_model_type_btn(self, instance):
        if not self.tmp_model_type:
            self.error_popup_clock("Type not selected!")
            return

        self.model_type = self.tmp_model_type
        self.ids.model_label.text = self.model_type
        self.model_type_popup.dismiss()
        self.unload_model()

    def load_model(self):
        if self.selected_model is None:
            self.error_popup_clock("Select model!")
            return

        self.model_name = self.selected_model.text
        logger.info(f"load model: {self.workspace.model_path(self.model_name)}")
        self.model_type = self.model_service.load_model(self.model_name)

        self.ids.model_label.text = self.model_type

        logger.debug("load complete")
        # logger.debug(f"{self.k_model.model}")
        self.update_all_button_states()

    def unload_model(self):
        if self.k_model.model is None:
            return

        self.k_model.unload_model()
        self.model_name = None
        self.unselect_model_btn()

    def save_model(self):
        if self.k_model.model is None:
            logger.warning("No model to save.")
            return

        # TODO: if change ach and save - have wrong name, test it.
        if not self.model_name:
            logger.warning("No model name to save.")
            return
        self.model_service.save_model(self.model_name)

        self.update_all_button_states()

    def evaluate_model(self, data=None):
        if self.eval_event is not None:
            Clock.unschedule(self.eval_event)
            self.eval_event = None
            self.toggle_error_popup("off")
            self.ids.evaluate_btn.text = "Evaluate"
            logger.warning("Cancel eval process")
            return

        self.ids.evaluate_btn.text = "Stop eval"
        if self.k_model.model is None:
            if self.selected_model is None:
                self.error_popup_clock("Select/Load model first!")
                return
            else:
                # in case model selected but not loaded
                self.load_model()

        if data is None:
            data = prepare_dataset(
                self.ml_train_folder,
                self.k_model.transform,
                batch_size=self.db.get_config_typed("inference_batch_size"),
                shuffle=False,
            )
        self.total_steps = len(data)
        self.k_model.evaluate_model(
            data,
        )
        self.toggle_error_popup("on", "Start eval...")
        self.eval_event = Clock.schedule_interval(lambda tm: self.async_eval_cycle(), 0.0001)

    def async_eval_cycle(self):
        (state, processed_steps, loss, acc) = self.k_model.async_eval_cycle()

        if not state:
            Clock.unschedule(self.eval_event)
            self.eval_event = None
            self.toggle_error_popup("off")
            self.ids.evaluate_btn.text = "Evaluate"
            return

        self.toggle_error_popup(
            "on",
            f"[{processed_steps}/{self.total_steps}] Loss: {round(loss, 4)} | Acc: {round(acc, 4)}",
        )

    def get_classes(self):
        return sorted(
            [btn.text.replace("\\", "/").split("/")[-1] for btn in self.ids.class_grid.children if btn.text != "all"]
        )

    def create_model(self, name):
        if not name:
            self.error_popup_clock("No model name.")
            return

        classes = self.get_classes()
        num_classes = len(classes)

        if num_classes < 2:
            self.error_popup_clock("Model can`t have 0 or 1 class.")
            return

        self.model_name = self.model_service.create_model(name, self.model_type, classes)

        self.load_model_names()
        self.ids.model_input.text = ""
        self.update_all_button_states()

    def delete_model(self):
        if self.selected_model is None:
            self.error_popup_clock("Select model!")
            return

        if self.selected_model.text == self.model_name:
            self.error_popup_clock("Can`t delete while loaded!")
            return

        logger.info(f"delete model from: {self.workspace.model_folder(self.selected_model.text)}")
        self.model_service.delete_model(self.selected_model.text)

        self.unselect_model_btn()
        self.load_model_names()

    def model_predict(self):
        if not self.selected_images or self.k_model.model is None:
            self.error_popup_clock("Select model and images!")
            return

        for selected in self.selected_images:
            path = selected.source
            image = Image.open(path).convert("RGB")
            cls_name = self.k_model.model_predict(image)

            lc = selected.label_container
            lc.clear_widgets()
            lbl = MDLabel(
                text=cls_name,
                halign="center",
                valign="middle",
                theme_text_color="Custom",
                text_color=(1, 1, 1, 1),
                font_size="18sp",
                size_hint=(1, 1),
                md_bg_color=(0, 0, 0, 0.6),
            )
            lc.add_widget(lbl)
            self.num_predictions += 1

        logger.debug("")
        self.update_all_button_states()

    def load_model_names(self):
        self.ids.model_grid.clear_widgets()

        if not os.path.isdir(self.ml_models_folder):
            logger.warning("No models folder")
            return

        for model_name in self.model_service.list_models():
            btn = MDLabelBtn(
                text=model_name,
                theme_text_color="Custom",
                text_color="white",
            )
            btn.bind(on_press=self.select_model_btn)
            self.ids.model_grid.add_widget(btn)

    def update_all_button_states(self):
        has_selection = bool(self.selected_images)
        can_transfer = (
            has_selection and self.selected_dir and self.cur_dir != self.selected_dir_full
        )
        is_model_selected = bool(self.selected_model)
        is_model_loaded = bool(self.k_model.model)
        is_model_named = bool(self.model_name)
        model_name_differs = is_model_selected and self.selected_model.text != self.model_name

        # Transfer button
        self.ids.transfer_image.disabled = not can_transfer

        # Rotate buttons
        self.ids.rotate_right.disabled = not has_selection
        self.ids.rotate_left.disabled = not has_selection

        # Model control buttons
        self.ids.model_load.disabled = not (is_model_selected and not is_model_named)
        self.ids.model_delete.disabled = not model_name_differs

        # Highlight active model
        for btn in self.ids.model_grid.children:
            btn.text_color = "red" if btn.text == self.model_name else "white"

        self.ids.model_unload.disabled = not (
            is_model_loaded and is_model_named and not self.train_active
        )
        self.ids.evaluate_btn.disabled = not (
            is_model_loaded and is_model_named and not self.train_active
        )
        self.ids.save_btn.disabled = not (
            is_model_loaded and is_model_named and not self.train_active
        )

        # Predict
        self.ids.predict_btn.disabled = not (
            is_model_loaded and is_model_named and has_selection and not self.train_active
        )

        # Train
        self.ids.train_btn.disabled = not (is_model_loaded and is_model_named)

        self.ids.clear_predictions.disabled = not self.num_predictions

        # Image selection info
        self.ids.unselect_all_images.disabled = not has_selection
        self.ids.num_selected_images.text = f"{len(self.selected_images)}" if has_selection else ""

        tb_folder_exists = os.path.isdir(self.tb_folder)
        empty_tb_folder = len(os.listdir(self.tb_folder)) != 0
        self.ids.tensorboard_btn.disabled = not (tb_folder_exists and empty_tb_folder)

    def rotate(self, side):
        import cv2

        if len(self.selected_images) == 0:
            self.error_popup_clock("Select image(s)!")
            return

        rot = cv2.ROTATE_90_COUNTERCLOCKWISE if side == "left" else cv2.ROTATE_90_CLOCKWISE

        for image in self.selected_images:
            path: str = image.source

            img = cv2.imread(path)
            img = cv2.rotate(img, rot)
            os.remove(path)
            cv2.imwrite(path, img)

            with open(path, "rb") as f:
                blob_data = f.read()
                data = io.BytesIO(blob_data)
                texture = CoreImage(data, ext="png").texture

                image.texture = texture

        self.unselect_all_images()

    def after_project_selection_hook(self, project_name, path):
        folders = [
            "all",
            "configs",
            "models",
            "tensorboard",
            "train",
        ]  # TODO: rename train to dataset(s)
        for folder in folders:
            os.makedirs(os.path.join(path, folder), exist_ok=True)
        logger.debug(f"Subfolders verified/created in {path}")

    def restore_project_params(self, project_name, cur_project_path):
        self.loaded_hash = ""
        self.update_project_paths()
        self.unload_model()
        self.unselect_model_btn()
        self.unselect_label_btn()
        self.unselect_all_images()
        self.load_classes()
        self.load_model_names()
        self.show_folder_images(path=os.path.join(cur_project_path, "all"))

    def on_text_input_class(self, instance, value):
        text = self.ids.class_input.text

        if len(text) > 0:
            self.ids.add_class.disabled = False
        else:
            self.ids.add_class.disabled = True

    def on_text_input_model(self, instance, value):
        text = self.ids.model_input.text

        if len(text) > 0:
            self.ids.create_model.disabled = False
        else:
            self.ids.create_model.disabled = True
