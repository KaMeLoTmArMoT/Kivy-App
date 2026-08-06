import datetime
import io
import os
import shutil
import time
from math import ceil
from threading import Thread

from checksumdir import dirhash
from kivy.clock import Clock
from kivy.core.image import Image as CoreImage
from kivy.metrics import dp
from kivy.properties import ListProperty
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.button import Button
from kivy.uix.gridlayout import GridLayout
from kivy.uix.label import Label
from kivy.uix.popup import Popup
from kivy.uix.progressbar import ProgressBar
from kivy.uix.screenmanager import Screen
from kivymd.uix.label import MDLabel
from PIL import Image

from app.screens.utils.additional import (
    BaseScreen,
    ImageMDButton,
    MDLabelBtn,
    MlUiHelper,
    SelectableImage,
)
from app.screens.utils.custom_logging import get_logger
from app.screens.utils.db import DB
from app.screens.utils.ml import KModel, create_config_file, prepare_dataset
from app.screens.utils.tensorboard_utils import TBServer
from app.screens.utils.utils import extend_key

logger = get_logger(__name__)


class MLViewScreen(Screen, BaseScreen, MlUiHelper):
    rgba = ListProperty([1, 1, 0, 0])  # error message popup color

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        BaseScreen.__init__(self)
        self.key = ""
        self.selected_dir = None
        self.selected_dir_full = None
        self.selected_images = []
        self.images_to_load = []
        self.progress_bar: ProgressBar = self.ids.progress_bar
        self.touch_time = time.time()
        self.train_active = False
        self.load_event = None
        self.page = 1
        self.total_pages = None

        self.selected_model = None
        self.model_name = None
        self.total_steps = None
        self.model_type = "MobileNetV2"
        self.model_type_popup = None
        self.tmp_model_type = None
        self.tb_server = TBServer()

        self.k_model = KModel()

        self.eval_event = None

        self.loaded_hash = ""

        self.cur_dir = ""

        self.app_folder = os.getcwd()
        self.projects_folder = os.path.join(self.app_folder, "app/training/classification")
        os.makedirs(self.projects_folder, exist_ok=True)

        self.active_project = "Kivy"
        self.active_project_folder = os.path.join(self.projects_folder, self.active_project)

        self.images_path = os.path.join(self.active_project_folder, "all")
        self.ml_train_folder = os.path.join(self.active_project_folder, "train")
        self.ml_configs_folder = os.path.join(self.active_project_folder, "configs")
        self.ml_models_folder = os.path.join(self.active_project_folder, "models")
        self.tb_folder = os.path.join(self.active_project_folder, "tensorboard")
        os.makedirs(self.tb_folder, exist_ok=True)
        os.makedirs(self.images_path, exist_ok=True)

        self.dropdown = None
        self.projects = []
        self.popup = None
        self.main_button = self.ids.project_label

        self.max_images_per_page = 20

        self.num_predictions = 0

    def on_enter(self, *args):
        self.setup_header()
        self.key = extend_key(self.manager.get_screen("login").key)
        self.load_classes()
        self.load_model_names()

        dir_hash = dirhash(self.images_path, "sha1")

        self.max_images_per_page = DB().get_config_typed("MAX_IMAGES_PER_PAGE")
        if self.loaded_hash != dir_hash:
            self.show_folder_images(path=self.images_path)

        self.exit_screen = False
        self.main_button.text = self.active_project

        self.ids.class_input.bind(text=self.on_text_input_class)
        self.ids.model_input.bind(text=self.on_text_input_model)

        try:
            self.k_model.update_params()
        except RuntimeError as e:
            if "PyTorch is not installed" in str(e):
                self.label_out("ML features disabled (torch not installed).")
                return
            raise

    def update_project_paths(self):
        os.makedirs(self.projects_folder, exist_ok=True)
        self.active_project_folder = os.path.join(self.projects_folder, self.active_project)

        self.images_path = os.path.join(self.active_project_folder, "all")
        self.ml_train_folder = os.path.join(self.active_project_folder, "train")
        self.ml_configs_folder = os.path.join(self.active_project_folder, "configs")
        self.ml_models_folder = os.path.join(self.active_project_folder, "models")
        self.tb_folder = os.path.join(self.active_project_folder, "tensorboard")

    def load_classes(self):
        self.ids.class_grid.clear_widgets()

        btn = MDLabelBtn(
            text="all",
            theme_text_color="Custom",
            text_color="red",
        )
        btn.bind(on_press=self.select_label_btn)
        self.ids.class_grid.add_widget(btn)

        if not os.path.isdir(self.ml_train_folder):
            logger.warning("No classes folder")
            return

        for file in os.listdir(self.ml_train_folder):
            path = os.path.join(self.ml_train_folder, file)
            if os.path.isdir(path):
                btn = MDLabelBtn(
                    text="train/" + file,
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
        if name == "":
            self.error_popup_clock("Enter name!")
            return

        path = os.path.join(self.ml_train_folder, name)
        if os.path.exists(path):
            self.error_popup_clock("Class exists!")
            return

        os.makedirs(path)
        self.load_classes()
        self.ids.class_input.text = ""

    def error_popup_clock(self, text="Error", show_time=1):
        self.toggle_error_popup("on", text)
        Clock.schedule_once(lambda tm: self.toggle_error_popup("off"), show_time)

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

        try:
            shutil.rmtree(self.selected_dir_full)
        except FileNotFoundError as e:
            logger.warning(f"No such file or directory, skipping {e}")

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

        files = os.listdir(path) if os.path.isdir(path) else None

        self.ids.image_grid.clear_widgets()
        self.unselect_all_images()

        if files is None:
            self.toggle_load_label("no_dir")
            return

        if new:
            self.page = 1
            logger.debug("reset page")

            for btn in self.ids.class_grid.children:
                btn.text_color = "white"

            if self.selected_dir is not None:
                self.selected_dir.text_color = "red"

        self.disable_switch_buttons()  # disable load button
        self.cur_dir = path

        for name in files:
            if ".jpg" in name or ".png" in name:
                im_path = os.path.join(path, name)
                self.images_to_load.append(im_path)

        n_images = len(self.images_to_load)
        self.total_pages = ceil(n_images / self.max_images_per_page)
        self.toggle_switch_buttons()
        self.ids.open_class.disabled = True

        self.update_page_counter()
        if n_images > self.max_images_per_page:
            self.images_to_load = self.images_to_load[
                self.page * self.max_images_per_page : (self.page + 1) * self.max_images_per_page
            ]

        self.progress_bar.value = 1
        self.progress_bar.max = len(self.images_to_load)
        self.load_event = Clock.schedule_interval(lambda tm: self.async_image_load(), 0.001)

    def update_page_counter(self):  # TODO: reset page when open new folder
        self.ids.page_label.text = f"{self.page}/{self.total_pages}"

    def async_image_load(self):
        stop = False
        if len(self.images_to_load) == 0:
            self.loaded_hash = dirhash(self.images_path, "sha1")
            stop = True

        if self.exit_screen:
            logger.warning("terminate loading")
            stop = True

        if stop:
            Clock.unschedule(self.load_event)
            self.toggle_load_label("success")
            self.enable_switch_buttons()
            return

        self.progress_bar.value += 1
        im_path = self.images_to_load.pop(0)

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
        self.ids.image_grid.add_widget(selectable_img)

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
        Thread(target=self.train_model).start()

    def prev_page(self):
        if self.page > 1:
            if self.load_event:
                Clock.unschedule(self.load_event)

            self.page -= 1
            self.show_folder_images(self.cur_dir)

    def next_page(self):
        if self.page < self.total_pages:
            if self.load_event:
                Clock.unschedule(self.load_event)

            self.page += 1
            self.show_folder_images(self.cur_dir)

    def train_model(self):
        data = prepare_dataset(
            self.ml_train_folder,
            self.k_model.transform,
            batch_size=DB().get_config_typed("train_batch_size"),
            shuffle=True,
        )

        log_dir = os.path.join(
            self.tb_folder,
            datetime.datetime.now().strftime("%Y_%m_%d-%H_%M") + f"_{self.model_name}",
        )

        self.k_model.train_model(data, log_dir)

        # self.evaluate_model(data)
        self.train_active = False
        self.k_model.terminate_training = False
        self.ids.train_btn.text = "Train"
        self.ids.train_btn.disabled = False
        self.update_all_button_states()
        self.save_model()

    def select_model_type(self):
        popup = Popup(
            title="Please select model type:",
            title_align="center",
            title_size=20,
            size_hint=(None, None),
            size=(500, 400),
        )

        lbl2_1 = Label(text="Current:", font_size=18)
        lbl2_2 = Label(text=self.model_type, font_size=18)

        box_inner = BoxLayout(orientation="horizontal", size_hint_y=0.2)
        box_inner.add_widget(lbl2_1)
        box_inner.add_widget(lbl2_2)

        model_types = [
            ["MobileNetV2", 3.5, 72.15],  # TODO v2
            ["MobileNetV3", 5.5, 75.27],  # TODO large one, v2
            ["ResNet", 11.7, 69.76],  # TODO 18
            ["ResNeXt", 25.0, 81.20],  # TODO 50_32x4d, v2
            ["EfficientNet", 5.3, 77.69],  # TODO b0
            ["EfficientNetV2", 21.5, 84.23],  # TODO s
            ["AlexNet", 61.1, 56.52],  # TODO
            ["VGG", 132.9, 69.02],  # TODO 11
        ]

        grid = GridLayout(cols=2)
        for name, size, acc in model_types:
            text = f"{name:<18} | {size}M | {acc}%"
            btn = Button(text=text)
            btn.bind(on_press=self.select_model_type_btn)
            grid.add_widget(btn)
            grid.ids[name] = btn

        btn_submit = MDLabelBtn(text="Submit", size_hint_y=0.15)
        btn_submit.allow_hover = True
        btn_submit.bind(on_press=self.submit_model_type_btn)

        box = BoxLayout(orientation="vertical")

        box.add_widget(box_inner)
        box.add_widget(grid)
        box.add_widget(btn_submit)
        box.ids["box_inner"] = box_inner
        box.ids["grid"] = grid
        box.ids["btn_submit"] = btn_submit

        popup.content = box
        popup.bind(on_dismiss=self.dismiss_model_select_popup)
        self.model_type_popup = popup
        popup.open()

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
        model_dir = os.path.join(self.active_project_folder, "models", self.model_name)
        save_path = os.path.join(model_dir, self.model_name + ".pth")
        logger.info(f"load model: {save_path}")

        config_path = os.path.join(self.ml_configs_folder, self.model_name + ".conf")
        self.k_model.load_model(save_path, config_path)

        self.model_type = self.model_name.split("_")[-2]

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
        model_dir = os.path.join(self.active_project_folder, "models", self.model_name)
        save_path = os.path.join(model_dir, self.model_name + ".pth")
        self.k_model.save_model(model_dir, save_path)

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
                batch_size=DB().get_config_typed("inference_batch_size"),
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
            [btn.text.split("\\")[-1] for btn in self.ids.class_grid.children if btn.text != "all"]
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

        self.model_name = f"{name}_{self.model_type}_{num_classes}"
        model_dir = os.path.join(self.active_project_folder, "models", self.model_name)
        save_path = os.path.join(model_dir, self.model_name + ".pth")

        classes = self.get_classes()

        self.k_model.create_model(
            self.model_name,
            classes,
            self.model_type,
            model_dir,
            save_path,
        )

        create_config_file(
            self.model_name,
            self.model_type,
            num_classes,
            classes,
            self.ml_configs_folder,
        )

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

        path = os.path.join(self.ml_models_folder, self.selected_model.text)
        logger.info(f"delete model from: {path}")
        shutil.rmtree(path)
        config_path = os.path.join(self.ml_configs_folder, self.selected_model.text + ".conf")
        os.remove(config_path)

        self.unselect_model_btn()
        self.load_model_names()

    def model_predict(self):
        if self.selected_images is None or self.k_model.model is None:
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

        for file in os.listdir(self.ml_models_folder):
            path = os.path.join(self.ml_models_folder, file)
            logger.debug(f"------ {path}")
            if os.path.isdir(path):
                btn = MDLabelBtn(
                    text=file,
                    theme_text_color="Custom",
                    text_color="white",
                )
                btn.bind(on_press=self.select_model_btn)
                # btn.allow_hover = True
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
