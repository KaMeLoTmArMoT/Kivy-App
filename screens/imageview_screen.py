import os
import shutil

from Cryptodome.Cipher import AES
from kivy.clock import Clock
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.filechooser import FileChooserIconView, FileChooserListView
from kivy.uix.label import Label
from kivy.uix.popup import Popup
from kivy.uix.progressbar import ProgressBar
from kivy.uix.screenmanager import Screen
from kivymd.uix.floatlayout import MDFloatLayout
from kivymd.uix.label import MDLabel
from kivymd.uix.selectioncontrol import MDCheckbox

from screens.additional import BaseScreen, ImageMDButton, MDLabelBtn
from utils import call_db
from screens.additional import ML_FOLDER


class ImageViewScreen(Screen, BaseScreen):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.lock_schedule = False
        self.grid = None
        self.key = ''
        self.loaded = False
        self.selected_images = []
        self.images_to_load = []
        self.progress_bar: ProgressBar = self.ids.progress_bar
        self.load_event = None

    def on_enter(self, *args):
        self.key = self.manager.get_screen('main').key
        self.grid = self.ids.grid
        self.selected_counter_update()
        self.create_db_and_check()
        if not self.loaded:
            self.show_folder_images('G://Downloads//photo')   # TODO: remove this

    def create_db_and_check(self):
        # Create a table
        call_db("""
        CREATE TABLE IF NOT EXISTS images (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            image blob
        ) """)

    def file_chooser_popup(self):
        popup = Popup(
            title='Filechooser',
            size_hint=(None, None), size=(400, 400)
        )

        box = BoxLayout(orientation='vertical')
        lbl = Label(text='Please select folder', size_hint_y=0.1)
        # chooser = FileChooserListView()
        chooser = FileChooserIconView()

        btn = MDLabelBtn(text='Submit', size_hint_y=0.1)
        btn.bind(
            on_press=lambda x: self.show_folder_images(
                chooser.path,
                chooser.selection,
                popup,
            )
        )
        btn.allow_hover = True

        box.add_widget(lbl)
        box.add_widget(chooser)
        box.add_widget(btn)

        popup.content = box
        popup.open()

    def show_folder_images(self, path, selection=None, popup=None):
        self.toggle_load_label('on')

        if os.path.isdir(path):
            files = os.listdir(path)
        else:
            files = None

        self.loaded = True
        self.grid.clear_widgets()
        self.unselect_all_images()
        self.selected_counter_update()

        if files is None:
            self.toggle_load_label('no_dir')
            return

        self.ids.choose_image.disabled = True  # disable load button

        for name in files:
            if '.jpg' in name or '.png' in name:
                im_path = os.path.join(path, name)
                self.images_to_load.append(im_path)

        self.progress_bar.value = 1
        self.progress_bar.max = len(self.images_to_load)
        self.load_event = Clock.schedule_interval(
            lambda tm: self.async_image_load(), .001
        )

        if popup is not None:
            popup.dismiss()

    def async_image_load(self):
        if len(self.images_to_load) == 0:
            Clock.unschedule(self.load_event)
            self.toggle_load_label('success')
            self.ids.choose_image.disabled = False
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
        self.grid.add_widget(fl)

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

        self.selected_counter_update()

    def save_img_to_db(self, enc):
        num_images = len(self.selected_images)
        self.schedule_counter_update()
        if num_images == 0:
            self.ids.selected_images.text = 'Choose 1+'
            return

        for path in self.selected_images:
            with open(path.source, 'rb') as f:
                blob_data = f.read()

                if enc:
                    cipher = AES.new(self.key, AES.MODE_EAX, nonce=b'TODO')
                    blob_data = cipher.encrypt(blob_data)

                call_db(
                    f"INSERT INTO images (image) VALUES (?)",
                    [blob_data]
                )
        self.unselect_all_images()
        self.ids.selected_images.text = f'Added {num_images}'

    def save_img_to_ml(self):
        num_images = len(self.selected_images)
        self.schedule_counter_update()
        if num_images == 0:
            self.ids.selected_images.text = 'Choose 1+'
            return

        if not os.path.isdir(ML_FOLDER + "all"):
            os.makedirs(ML_FOLDER + "all")
        for path in self.selected_images:
            shutil.copy(path.source, ML_FOLDER + "all")

        self.unselect_all_images()
        self.ids.selected_images.text = f'Copied {num_images}'

    def schedule_counter_update(self):
        if not self.lock_schedule:  # to trigger schedule only once at a time
            print('lock')
            self.lock_schedule = True
            Clock.schedule_once(lambda dt: self.selected_counter_update(schedule=True), 1)

    def unselect_all_images(self):
        instances = self.selected_images.copy()
        for instance in instances:
            instance.md_bg_color = (1.0, 1.0, 1.0, 0.0)
            instance.line_color = (1.0, 1.0, 1.0, 0.2)
            instance.parent.children[0].active = False
            self.selected_images.remove(instance)
        self.selected_counter_update()

    def selected_counter_update(self, schedule=False):
        self.ids.selected_images.text = f'Selected: {len(self.selected_images)}'

        if schedule:
            self.lock_schedule = False
            print('release')
