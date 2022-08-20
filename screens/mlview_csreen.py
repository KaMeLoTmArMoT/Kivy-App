import os
import shutil

from kivy.clock import Clock
from kivy.uix.button import Button
from kivy.uix.gridlayout import GridLayout
from kivy.uix.label import Label
from kivy.uix.progressbar import ProgressBar
from kivy.uix.screenmanager import Screen
from kivymd.uix.floatlayout import MDFloatLayout
from kivymd.uix.label import MDLabel
from kivymd.uix.selectioncontrol import MDCheckbox

from screens.additional import ML_FOLDER, BaseScreen, ImageMDButton, MDLabelBtn
from utils import call_db


class MLViewScreen(Screen, BaseScreen):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.key = ''
        self.selected = None
        self.selected_images = []
        self.images_to_load = []
        self.progress_bar: ProgressBar = self.ids.progress_bar

    def on_enter(self, *args):
        self.key = self.manager.get_screen('main').key
        self.create_db_and_check()
        self.load_classes()
        self.show_folder_images(path=ML_FOLDER + 'all')

    def create_db_and_check(self):
        # Create a table
        call_db("""
        CREATE TABLE IF NOT EXISTS images (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            image blob
        ) """)

    def load_classes(self):
        self.ids.grid.clear_widgets()

        for file in os.listdir(ML_FOLDER):
            path = os.path.join(ML_FOLDER, file)
            if os.path.isdir(path):
                btn = MDLabelBtn(text=file)
                btn.bind(on_press=self.select_label_btn)
                self.ids.grid.add_widget(btn)

    def select_label_btn(self, instance):
        print(f'The button <{instance.text}> is being pressed')
        if self.selected:
            if instance.uid == self.selected.uid:
                self.unselect_label_btn()
                return

        # reset selection
        for btn in self.ids.grid.children:
            btn.md_bg_color = (1.0, 1.0, 1.0, 0.0)

        instance.md_bg_color = (1.0, 1.0, 1.0, 0.1)
        instance.radius = (20, 20, 20, 20)
        self.selected = instance

    def unselect_label_btn(self):
        self.selected = None
        for btn in self.ids.grid.children:
            btn.md_bg_color = (1.0, 1.0, 1.0, 0.0)

    def add_class(self):
        name = self.ids.class_input.text
        if name == "":
            print('no input')
            return

        path = os.path.join(ML_FOLDER, name)
        if os.path.exists(path):
            print('we have such class!')
            return

        os.makedirs(path)
        self.load_classes()
        self.ids.class_input.text = ''

    def delete_class(self):
        if self.selected is None:
            return

        if self.selected.text == 'all':
            print('can`t delete main folder')

        path = ML_FOLDER + self.selected.text
        shutil.rmtree(path)

        self.unselect_label_btn()
        self.load_classes()

    def show_folder_images(self, path=None):
        if self.selected is None and path is None:
            return

        self.toggle_load_label('on')
        if path is None:
            path = ML_FOLDER + self.selected.text

        if os.path.isdir(path):
            files = os.listdir(path)
        else:
            files = None

        self.ids.image_grid.clear_widgets()
        self.unselect_all_images()

        if files is None:
            self.toggle_load_label('no_dir')
            return

        self.ids.open.disabled = True  # disable load button

        for name in files:
            if '.jpg' in name or '.png' in name:
                im_path = os.path.join(path, name)
                self.images_to_load.append(im_path)

        self.progress_bar.value = 1
        self.progress_bar.max = len(self.images_to_load)
        self.load_event = Clock.schedule_interval(
            lambda tm: self.async_image_load(), .001
        )

    def async_image_load(self):
        if len(self.images_to_load) == 0:
            Clock.unschedule(self.load_event)
            self.toggle_load_label('success')
            self.ids.open.disabled = False
            return

        self.progress_bar.value += 1
        im_path = self.images_to_load.pop(0)
        img = ImageMDButton(
            source=im_path,
            allow_stretch=True,
            keep_ratio=True,
            pos_hint={'center_x': .5, 'center_y': .5},
        )

        checkbox = MDCheckbox(
            size_hint=(None, None),
            size=("48dp", "48dp"),
            pos_hint={'center_x': 0.96, 'center_y': 0.96},
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

        if mode == 'on':
            lbl_prop("Loading, please wait...")

        elif mode == 'no_dir':
            lbl_prop("No images, please select folder.", pbar_hint_y=0)

        elif mode == 'success':
            lbl_prop("Success!", color=(0, 1, 0, 1))
            Clock.schedule_once(lambda tm: self.toggle_load_label('off'), 1)

        elif mode == 'off':
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
