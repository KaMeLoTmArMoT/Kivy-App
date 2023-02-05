import datetime
import io
import os
import shutil
import time
import webbrowser
from math import ceil
from threading import Thread

import numpy as np
import tensorflow as tf
from checksumdir import dirhash
from kivy.clock import Clock
from kivy.core.image import Image as CoreImage
from kivy.properties import ListProperty
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.button import Button
from kivy.uix.gridlayout import GridLayout
from kivy.uix.label import Label
from kivy.uix.popup import Popup
from kivy.uix.progressbar import ProgressBar
from kivy.uix.screenmanager import Screen
from kivymd.uix.floatlayout import MDFloatLayout
from kivymd.uix.label import MDLabel
from kivymd.uix.selectioncontrol import MDCheckbox
from tensorboard import program

from screens.additional import BaseScreen, ImageMDButton, MDLabelBtn
from screens.configs import (
    IMG_SHAPE,
    MAX_IMAGES_PER_PAGE,
    ML_CONFIGS_FOLDER,
    ML_FOLDER,
    ML_TRAIN_FOLDER,
    TENSORBOARD_PATH,
    chrome_path,
)
from screens.ml import (
    create_config_file,
    get_base_model,
    get_model_preprocess,
    read_config_file,
)
from utils import call_db, extend_key


class MLViewScreen(Screen, BaseScreen):
    rgba = ListProperty([1, 1, 0, 0])  # error message popup color

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        BaseScreen.__init__(self)
        self.key = ""
        self.selected_dir = None
        self.selected_images = []
        self.images_to_load = []
        self.progress_bar: ProgressBar = self.ids.progress_bar
        self.cur_dir = "all"
        self.touch_time = time.time()
        self.train_active = False
        self.load_event = None
        self.page = 1
        self.total_pages = None

        self.selected_model = None
        self.model = None
        self.base_model = None
        self.model_preprocess = None
        self.model_name = None
        self.model_type = "MobileNetV2"
        self.model_type_popup = None
        self.tmp_model_type = None
        self.num_classes = 0
        self.classes = None
        self.tensorboard = None

        self.data = None
        self.eval_event = None
        self.acc = None
        self.loss = None

        self.loaded_hash = ""
        self.path = ML_FOLDER + "all"  # TODO: remove static path

    def on_enter(self, *args):
        self.key = extend_key(self.manager.get_screen("login").key)
        self.create_db_and_check()
        self.load_classes()
        self.load_model_names()

        dir_hash = dirhash(self.path, "sha1")

        if self.loaded_hash != dir_hash:
            self.show_folder_images(path=self.path)

        self.exit_screen = False

    def load_classes(self):
        self.ids.class_grid.clear_widgets()

        btn = MDLabelBtn(text="all")
        btn.bind(on_press=self.select_label_btn)
        self.ids.class_grid.add_widget(btn)

        if not os.path.isdir(ML_TRAIN_FOLDER):
            print("No classes folder")
            return

        for file in os.listdir(ML_TRAIN_FOLDER):
            path = os.path.join(ML_TRAIN_FOLDER, file)
            if os.path.isdir(path):
                btn = MDLabelBtn(text="train\\" + file)
                btn.bind(on_press=self.select_label_btn)
                self.ids.class_grid.add_widget(btn)

    def select_label_btn(self, instance):
        print(f"The button <{instance.text}> is being pressed")
        if self.selected_dir:
            if instance.uid == self.selected_dir.uid:
                # custom double touch event
                self.unselect_label_btn()

                if time.time() - self.touch_time < 0.2:
                    if self.ids.open.disabled:
                        return

                    self.show_folder_images(ML_FOLDER + instance.text, new=True)
                return

        # reset selection
        for btn in self.ids.class_grid.children:
            btn.md_bg_color = (1.0, 1.0, 1.0, 0.0)

        instance.md_bg_color = (1.0, 1.0, 1.0, 0.1)
        instance.radius = (20, 20, 20, 20)
        self.selected_dir = instance
        self.touch_time = time.time()

    def unselect_label_btn(self):
        self.selected_dir = None
        for btn in self.ids.class_grid.children:
            btn.md_bg_color = (1.0, 1.0, 1.0, 0.0)

    def add_class(self):
        name = self.ids.class_input.text
        if name == "":
            self.error_popup_clock("Enter name!")
            return

        path = os.path.join(ML_TRAIN_FOLDER, name)
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

        path = ML_FOLDER + self.selected_dir.text
        shutil.rmtree(path)

        self.unselect_label_btn()
        self.load_classes()

    def disable_switch_buttons(self):
        self.ids.open.disabled = True
        self.ids.prev_page.disabled = True
        self.ids.next_page.disabled = True

    def enable_switch_buttons(self):
        self.ids.open.disabled = False
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
        if path is None:
            path = ML_FOLDER + self.selected_dir.text

        if os.path.isdir(path):
            files = os.listdir(path)
        else:
            files = None

        self.ids.image_grid.clear_widgets()
        self.unselect_all_images()

        if files is None:
            self.toggle_load_label("no_dir")
            return

        if new:
            self.page = 1
            print("reset page")

        self.disable_switch_buttons()  # disable load button
        self.cur_dir = path

        for name in files:
            if ".jpg" in name or ".png" in name:
                im_path = os.path.join(path, name)
                self.images_to_load.append(im_path)

        n_images = len(self.images_to_load)
        self.total_pages = ceil(n_images / MAX_IMAGES_PER_PAGE)
        self.toggle_switch_buttons()

        self.update_page_counter()
        if n_images > MAX_IMAGES_PER_PAGE:
            self.images_to_load = self.images_to_load[
                self.page * MAX_IMAGES_PER_PAGE : (self.page + 1) * MAX_IMAGES_PER_PAGE
            ]

        self.progress_bar.value = 1
        self.progress_bar.max = len(self.images_to_load)
        self.load_event = Clock.schedule_interval(
            lambda tm: self.async_image_load(), 0.001
        )

    def update_page_counter(self):  # TODO: reset page when open new folder
        self.ids.page_label.text = f"{self.page}/{self.total_pages}"

    def async_image_load(self):
        stop = False
        if len(self.images_to_load) == 0:
            self.loaded_hash = dirhash(self.path, "sha1")
            stop = True

        if self.exit_screen:
            print("terminate loading")
            stop = True

        if stop:
            Clock.unschedule(self.load_event)
            self.toggle_load_label("success")
            self.enable_switch_buttons()
            return

        self.progress_bar.value += 1
        im_path = self.images_to_load.pop(0)
        img = ImageMDButton(
            source=im_path,
            allow_stretch=True,
            keep_ratio=True,
            pos_hint={"center_x": 0.5, "center_y": 0.5},
            nocache=True,
        )

        checkbox = MDCheckbox(
            size_hint=(None, None),
            size=("48dp", "48dp"),
            pos_hint={"center_x": 0.96, "center_y": 0.96},
        )

        fl = MDFloatLayout()
        fl.add_widget(img)
        fl.add_widget(checkbox)

        img.line_color = (1.0, 1.0, 1.0, 0.2)
        img.bind(on_press=self.image_click)
        self.ids.image_grid.add_widget(fl)

    def unselect_all_images(self):
        instances = self.selected_images.copy()
        for instance in instances:
            instance.md_bg_color = (1.0, 1.0, 1.0, 0.0)
            instance.line_color = (1.0, 1.0, 1.0, 0.2)
            instance.parent.children[0].active = False
            self.selected_images.remove(instance)

    def image_click(self, instance):
        path = instance.source

        if instance in self.selected_images:
            instance.md_bg_color = (1.0, 1.0, 1.0, 0.0)
            instance.line_color = (1.0, 1.0, 1.0, 0.2)
            self.selected_images.remove(instance)

            instance.parent.children[0].active = False
        else:
            instance.line_color = (1.0, 1.0, 1.0, 0.6)
            instance.md_bg_color = (1.0, 1.0, 1.0, 0.1)
            self.selected_images.append(instance)

            instance.parent.children[0].active = True

    def transfer_images(self):
        if len(self.selected_images) == 0 or self.selected_dir is None:
            self.error_popup_clock("Select images and dir!")
            return

        in_dir = self.cur_dir
        out_dir = ML_FOLDER + self.selected_dir.text

        if in_dir == out_dir:
            self.error_popup_clock("Can`t paste to same dir!")
            return

        for image in self.selected_images:
            out_img = image.source.replace(in_dir, out_dir)
            shutil.move(image.source, out_img)

        self.unselect_all_images()
        self.unselect_label_btn()
        self.show_folder_images(in_dir)

    def trigger_training(self):
        if self.model is None:
            if self.selected_model:
                self.load_model()
            else:
                self.error_popup_clock("Select or load model!")
                return

        self.train_active = True
        self.ids.train.disabled = True
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

    def prepare_dataset(self):
        img_height, img_width = 224, 224

        train_ds = tf.keras.utils.image_dataset_from_directory(
            ML_TRAIN_FOLDER,
            image_size=(img_height, img_width),
            batch_size=8,
        )

        # class_names = train_ds.class_names
        # num_classes = len(class_names)

        normalized_ds = train_ds.map(lambda x, y: (self.model_preprocess(x), y))
        return normalized_ds

    def train_model(self):
        normalized_ds = self.prepare_dataset()

        self.model.compile(
            optimizer="adam",
            loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
            metrics=["accuracy"],
        )

        log_dir = (
            ML_FOLDER
            + "tensorboard\\"
            + datetime.datetime.now().strftime("%Y_%m_%d-%H_%M")
            + f"_{self.model_name}"
        )
        tensorboard_callback = tf.keras.callbacks.TensorBoard(
            log_dir=log_dir, histogram_freq=1
        )

        self.model.fit(normalized_ds, epochs=5, callbacks=[tensorboard_callback])

        # self.evaluate_model(normalized_ds)
        self.train_active = False
        self.ids.train.disabled = False
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
            ["MobileNet", 4.3, 70.4],
            ["MobileNetV2", 3.5, 71.3],
            ["DenseNet121", 8.1, 75.0],
            ["NASNetMobile", 5.3, 74.4],
            ["EfficientNetB0", 5.3, 77.1],
            ["EfficientNetB1", 7.9, 79.1],
            ["EfficientNetV2B0", 7.2, 78.7],
            ["EfficientNetV2B1", 8.2, 79.8],
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

        if self.tmp_model_type == instance.text.split(" ")[0]:
            if time.time() - self.touch_time < 0.2:
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
        self.unload_model()

        if self.selected_model is None:
            self.error_popup_clock("Select model!")
            return

        self.model_name = self.selected_model.text
        self.model = tf.keras.models.load_model(
            ML_FOLDER + "models\\" + self.model_name
        )

        config_path = ML_CONFIGS_FOLDER + self.model_name + ".conf"
        if os.path.exists(config_path):
            model_type, num_classes, img_shape, classes = read_config_file(
                self.model_name
            )
            self.classes = classes
            print(model_type, num_classes, img_shape)
            # TODO: use config, not just load

        self.model_type = self.model_name.split("_")[-2]
        self.model_preprocess = get_model_preprocess(self.model_type)

        self.ids.model_label.text = self.model_type

        print("load complete")
        self.model.summary()

    def unload_model(self):
        if self.model is None:
            return

        self.model_name = None
        self.model = None
        self.base_model = None
        self.model_preprocess = None

        tf.keras.backend.clear_session()

        self.unselect_model_btn()
        print("unload complete")

    def save_model(self):
        if self.model is None:
            return

        # TODO: if change ach and save - have wrong name, test it.
        self.model.save(ML_FOLDER + "models\\" + self.model_name)
        print("save complete")

    def evaluate_model(self, data=None):
        if self.model is None:
            if self.selected_model is None:
                self.error_popup_clock("Select/Load model first!")
                return
            else:
                self.load_model()

        if data is None:
            data = self.prepare_dataset()

        self.data = list(data)
        self.toggle_error_popup("on", "Start eval...")
        self.eval_event = Clock.schedule_interval(
            lambda tm: self.async_eval_cycle(), 0.0001
        )

    def async_eval_cycle(self):
        iters = len(self.data)
        if iters % 100 == 0:
            print(len(self.data))
            self.toggle_error_popup(
                "on",
                f"iters: {iters} "
                f"loss: {round(self.loss, 3)} "
                f"acc: {round(self.acc, 3)}",
            )

        if iters == 0:
            Clock.unschedule(self.eval_event)
            self.toggle_error_popup("off")
            print(f"Eval loss: {self.loss}, acc: {self.acc}")
            return

        batch = self.data.pop()
        loss, acc = self.model.evaluate(batch[0], batch[1], verbose=0)

        if self.loss is None and self.acc is None:
            self.loss, self.acc = loss, acc
        else:
            self.loss, self.acc = (self.loss + loss) / 2, (self.acc + acc) / 2

    def create_model(self, name):
        if name == "":
            self.error_popup_clock("No model name.")
            return
        self.unload_model()

        self.num_classes = len(self.ids.class_grid.children) - 1
        self.model_name = f"{name}_{self.model_type}_{self.num_classes}"

        if self.num_classes < 2:
            self.error_popup_clock("Model can`t have 0 or 1 class.")
            return

        print("creating", self.model_name)

        self.base_model = get_base_model(self.model_type)
        self.model_preprocess = get_model_preprocess(self.model_type)
        self.base_model.trainable = False

        inputs = tf.keras.Input(shape=IMG_SHAPE)
        x = self.base_model(inputs, training=False)
        x = tf.keras.layers.GlobalAveragePooling2D()(x)
        outputs = tf.keras.layers.Dense(self.num_classes)(x)
        self.model = tf.keras.Model(inputs, outputs)
        print("create complete")
        print(self.model.summary())

        self.model.compile(
            optimizer="adam",
            loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
            metrics=["accuracy"],
        )

        self.classes = sorted(
            [
                btn.text.split("\\")[-1]
                for btn in self.ids.class_grid.children
                if btn.text != "all"
            ]
        )
        create_config_file(
            self.model_name, self.model_type, self.num_classes, self.classes
        )
        self.save_model()
        self.load_model_names()

    def delete_model(self):
        if self.selected_model is None:
            self.error_popup_clock("Select model!")
            return

        if self.selected_model.text == self.model_name:
            self.error_popup_clock("Can`t delete while loaded!")
            return

        path = ML_FOLDER + "models\\" + self.selected_model.text
        shutil.rmtree(path)
        config_path = ML_CONFIGS_FOLDER + self.selected_model.text + ".conf"
        os.remove(config_path)

        self.unselect_model_btn()
        self.load_model_names()

    def model_predict(self):
        # TODO: check if model is loaded
        if self.selected_images is None or self.model is None:
            self.error_popup_clock("Select model and images!")
            return

        for selected in self.selected_images:
            path = selected.source
            image = tf.keras.preprocessing.image.load_img(path, target_size=IMG_SHAPE)
            image = self.model_preprocess(image)
            image = np.expand_dims(image, axis=0)
            pred = self.model.predict(image)
            pred = np.argmax(pred, axis=1)[0]
            print(self.classes[pred], pred, end=", ")
        print()

    def load_model_names(self):
        self.ids.model_grid.clear_widgets()

        models_path = ML_FOLDER + "models"
        if not os.path.isdir(models_path):
            print("No models folder")
            return

        for file in os.listdir(models_path):
            path = os.path.join(models_path, file)
            if os.path.isdir(path):
                btn = MDLabelBtn(text=file)
                btn.bind(on_press=self.select_model_btn)
                self.ids.model_grid.add_widget(btn)

    def select_model_btn(self, instance):
        print(f"The button <{instance.text}> is being pressed")
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

    def launch_tensorboard(self):
        if not os.path.isdir(TENSORBOARD_PATH):
            print("No tensorboard folder")
            return

        if len(os.listdir(TENSORBOARD_PATH)) == 0:
            self.error_popup_clock("No data to show TB!")
            return
        # TODO: check freeze issue here.
        if self.tensorboard is None:
            self.tensorboard = program.TensorBoard()
            self.tensorboard.configure(argv=[None, "--logdir", TENSORBOARD_PATH])
            url = self.tensorboard.launch()
            print(f"{url=}")

        webbrowser.get(chrome_path).open("http://localhost:6006/")

    def rotate(self, side):
        import cv2

        if len(self.selected_images) == 0:
            self.error_popup_clock("Select image(s)!")
            return

        if side == "left":
            rot = cv2.ROTATE_90_COUNTERCLOCKWISE
        else:
            rot = cv2.ROTATE_90_CLOCKWISE

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
