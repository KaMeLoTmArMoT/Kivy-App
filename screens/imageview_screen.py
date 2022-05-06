import os

from Cryptodome.Cipher import AES
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.filechooser import FileChooserIconView, FileChooserListView
from kivy.uix.label import Label
from kivy.uix.popup import Popup
from kivy.uix.screenmanager import Screen

from screens.additional import BaseScreen, ImageMDButton, MDLabelBtn
from utils import call_db


class ImageViewScreen(Screen, BaseScreen):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.grid = None
        self.key = ''
        self.loaded = False
        self.selected_images = []

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
        files = os.listdir(path)
        self.loaded = True
        self.grid.clear_widgets()
        self.unselect_all_images()
        self.selected_counter_update()

        for name in files:
            if '.jpg' in name or '.png' in name:
                im_path = os.path.join(path, name)
                img = ImageMDButton(
                    source=im_path,
                    allow_stretch=True,
                    keep_ratio=True
                )
                img.line_color = (1.0, 1.0, 1.0, 0.2)
                img.bind(on_press=self.image_click)
                self.grid.add_widget(img)

        if popup is not None:
            popup.dismiss()
        self.toggle_load_label('off')

    def toggle_load_label(self, mode):
        lbl = self.ids.load_label

        if mode == 'on':
            lbl.text = "Loading, please wait..."
            lbl.size_hint_y = 0.2
        else:
            lbl.text = ""
            lbl.size_hint_y = 0

    def image_click(self, instance):
        path = instance.source

        if instance in self.selected_images:
            instance.md_bg_color = (1.0, 1.0, 1.0, 0.0)
            instance.line_color = (1.0, 1.0, 1.0, 0.2)
            self.selected_images.remove(instance)
        else:
            instance.line_color = (1.0, 1.0, 1.0, 0.6)
            instance.md_bg_color = (1.0, 1.0, 1.0, 0.1)
            self.selected_images.append(instance)

        self.selected_counter_update()

    def save_img_to_db(self, enc):
        num_images = len(self.selected_images)
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

    def unselect_all_images(self):
        instances = self.selected_images.copy()
        for instance in instances:
            instance.md_bg_color = (1.0, 1.0, 1.0, 0.0)
            instance.line_color = (1.0, 1.0, 1.0, 0.2)
            self.selected_images.remove(instance)
        self.selected_counter_update()

    def selected_counter_update(self):
        self.ids.selected_images.text = f'Selected: {len(self.selected_images)}'
