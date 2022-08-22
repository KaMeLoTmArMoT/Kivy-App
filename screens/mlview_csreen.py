import os
import shutil
import threading
import time
from math import ceil

import keras
import tensorflow as tf
from kivy.clock import Clock
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

from screens.additional import ML_FOLDER, BaseScreen, ImageMDButton, MDLabelBtn
from utils import call_db

MAX_IMAGES_PER_PAGE = 100
ML_TRAIN_FOLDER = ML_FOLDER + "train\\"


class MLViewScreen(Screen, BaseScreen):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
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
        self.model = None

    def on_enter(self, *args):
        self.key = self.manager.get_screen("main").key
        self.create_db_and_check()
        self.load_classes()
        self.show_folder_images(path=ML_FOLDER + "all")

    def create_db_and_check(self):
        # Create a table
        call_db(
            """
        CREATE TABLE IF NOT EXISTS images (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            image blob
        ) """
        )

    def load_classes(self):
        self.ids.class_grid.clear_widgets()

        btn = MDLabelBtn(text="all")
        btn.bind(on_press=self.select_label_btn)
        self.ids.class_grid.add_widget(btn)

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
            print("no input")
            return

        path = os.path.join(ML_TRAIN_FOLDER, name)
        if os.path.exists(path):
            print("we have such class!")
            return

        os.makedirs(path)
        self.load_classes()
        self.ids.class_input.text = ""

    def delete_class(self):
        if self.selected_dir is None:
            return

        if self.selected_dir.text == "all":
            print("can`t delete main folder")
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

    def show_folder_images(self, path=None, new=False):
        if self.selected_dir is None and path is None:
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
        if len(self.images_to_load) == 0:
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

    def toggle_load_label(self, mode):
        lbl: MDLabel = self.ids.load_label

        def lbl_prop(text="", lbl_hint_y=0.2, color=(1, 1, 1, 1), pbar_hint_y=0.1):
            lbl.text = text
            lbl.size_hint_y = lbl_hint_y
            lbl.color = color
            self.progress_bar.size_hint_y = pbar_hint_y

        if mode == "on":
            lbl_prop("Loading, please wait...")

        elif mode == "no_dir":
            lbl_prop("No images, please select folder.", pbar_hint_y=0)

        elif mode == "success":
            lbl_prop("Success!", color=(0, 1, 0, 1))
            Clock.schedule_once(lambda tm: self.toggle_load_label("off"), 1)

        elif mode == "off":
            lbl_prop(lbl_hint_y=0, pbar_hint_y=0)

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
        if self.selected_images is None or self.selected_dir is None:
            return

        in_dir = self.cur_dir
        out_dir = ML_FOLDER + self.selected_dir.text

        if in_dir == out_dir:
            return

        for image in self.selected_images:
            out_img = image.source.replace(in_dir, out_dir)
            shutil.move(image.source, out_img)

        self.unselect_all_images()
        self.show_folder_images(in_dir)

    def trigger_training(self):
        threading.Thread(target=self.train_model).start()
        self.train_active = True
        self.ids.train.disabled = True

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
        img_height, img_width = 224, 224

        train_ds = tf.keras.utils.image_dataset_from_directory(
            ML_TRAIN_FOLDER,
            image_size=(img_height, img_width),
            batch_size=8,
        )

        class_names = train_ds.class_names
        num_classes = len(class_names)

        normalization_layer = keras.layers.Rescaling(1.0 / 127.5, offset=-1)
        normalized_ds = train_ds.map(lambda x, y: (normalization_layer(x), y))

        IMG_SHAPE = (img_height, img_width) + (3,)
        base_model = tf.keras.applications.MobileNetV2(
            input_shape=IMG_SHAPE, include_top=False, weights="imagenet"
        )
        base_model.trainable = False

        inputs = keras.Input(shape=(IMG_SHAPE))
        x = base_model(inputs, training=False)
        x = keras.layers.GlobalAveragePooling2D()(x)
        outputs = keras.layers.Dense(num_classes)(x)
        model = keras.Model(inputs, outputs)

        model.compile(
            optimizer="adam",
            loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
            metrics=["accuracy"],
        )

        model.fit(normalized_ds, epochs=5)

        print(model.evaluate(normalized_ds))
        self.train_active = False
        self.ids.train.disabled = False

    def select_model_type(self):
        popup = Popup(
            title="Please select model type:",
            title_align="center",
            title_size=20,
            size_hint=(None, None),
            size=(400, 400),
        )

        lbl2_1 = Label(text="Current:", font_size=18)
        lbl2_2 = Label(text="MobileNet_v2", font_size=18)

        box_inner = BoxLayout(orientation="horizontal", size_hint_y=0.2)
        box_inner.add_widget(lbl2_1)
        box_inner.add_widget(lbl2_2)

        model_types = [
            "MobileNet",
            "MobileNetV2",
            "DenseNet121",
            "NASNetMobile",
            "EfficientNetB0",
            "EfficientNetB1",
            "EfficientNetV2B0",
            "EfficientNetV2B1",
        ]

        grid = GridLayout(cols=2)
        for model in model_types:
            btn = Button(text=model)
            grid.add_widget(btn)

        btn_submit = MDLabelBtn(text="Submit", size_hint_y=0.1)
        btn_submit.allow_hover = True

        box = BoxLayout(orientation="vertical")
        box.add_widget(box_inner)
        box.add_widget(grid)
        box.add_widget(btn_submit)

        popup.content = box
        popup.open()

        pass

    def load_model(self):
        pass

    def unload_model(self):
        pass

    def save_model(self):
        pass

    def evaluate_model(self):
        pass

    def create_model(self):
        pass

    def delete_model(self):
        pass

    def model_predict(self):
        pass
