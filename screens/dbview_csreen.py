import io

import cv2
import numpy as np
from Cryptodome.Cipher import AES
from kivy.core.image import Image as CoreImage
from kivy.graphics.texture import Texture
from kivy.uix.popup import Popup
from kivy.uix.screenmanager import Screen

from screens.additional import BaseScreen, ImageMDButton
from utils import call_db


class DbViewScreen(Screen, BaseScreen):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.grid_1 = None
        self.grid_2 = None
        self.key = ''
        self.loaded = False
        self.selected_image = None
        self.prev_line_color = None

    def on_enter(self, *args):
        self.key = self.manager.get_screen('main').key
        self.grid_1 = self.ids.grid_1
        self.grid_2 = self.ids.grid_2
        self.create_db_and_check()
        self.show_db_images()
        if not self.loaded:     # TODO: probably better to load ecah time
            self.show_db_images()

    def create_db_and_check(self):
        # Create a table
        call_db("""
        CREATE TABLE IF NOT EXISTS images (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            image blob
        ) """)

    def show_db_images(self):
        db_images = call_db("SELECT * FROM images")
        self.grid_1.clear_widgets()
        self.grid_2.clear_widgets()
        self.unselect_image()

        if len(db_images) == 0:
            self.toggle_load_label('on', text='No images in DB.')
            self.loaded = False
            return
        else:
            self.toggle_load_label('on')
            self.loaded = True

        for pk, b_image in db_images:
            img_button = ImageMDButton(
                allow_stretch=True,
                keep_ratio=True,
            )

            success = False
            try:
                data = io.BytesIO(b_image)
                texture = CoreImage(data, ext="png").texture
                success = True
                img_button.line_color = (1.0, 0.6, 0.0, 0.5)
                self.grid_1.add_widget(img_button)
            except Exception as e:
                print(e)

            if not success:     # try to decrypt
                try:
                    cipher = AES.new(self.key, AES.MODE_EAX, nonce=b'TODO')
                    data = io.BytesIO(cipher.decrypt(b_image))
                    texture = CoreImage(data, ext="png").texture
                    success = True
                    img_button.line_color = (0.0, 1.0, 0.0, 0.5)
                    self.grid_2.add_widget(img_button)
                except Exception as e:
                    print(e)

            if not success:     # show cross instead of image
                img = np.zeros((600, 800, 1), dtype=np.float32)  # make multiple crosses
                # img = np.zeros((600, 800, 3), dtype=np.float32)
                h, w, *_ = img.shape
                red = (255, 0, 0)
                img = cv2.line(img, (0, 0), (w, h), red, thickness=6)
                img = cv2.line(img, (w, 0), (0, h), red, thickness=6)
                buff = bytes(img.flatten())

                texture = Texture.create(size=(w, h))
                texture.blit_buffer(buff, bufferfmt='ubyte', colorfmt='bgr')
                img_button.line_color = (1.0, 0.0, 0.0, 0.5)
                self.grid_1.add_widget(img_button)

            img_button.source = str(pk)
            img_button.texture = texture
            img_button.bind(on_press=self.image_click)
        self.toggle_load_label('off')

    def toggle_load_label(self, mode, text="Loading, please wait..."):
        lbl = self.ids.load_label

        if mode == 'on':
            lbl.text = text
            lbl.size_hint_y = 0.2
        else:
            lbl.text = ""
            lbl.size_hint_y = 0

    def image_click(self, instance):
        path = instance.source

        if instance is self.selected_image:
            self.unselect_image()
            return

        self.unselect_image()
        self.selected_image = instance
        self.prev_line_color = self.selected_image.line_color
        self.selected_image.line_color = (1.0, 1.0, 1.0, 0.6)
        self.selected_image.md_bg_color = (1.0, 1.0, 1.0, 0.1)

    def unselect_image(self):
        if self.selected_image is not None:
            self.selected_image.line_color = self.prev_line_color
            self.selected_image.md_bg_color = (1.0, 1.0, 1.0, 0.0)
            self.selected_image = None

    def preview_img(self):
        if not self.selected_image:
            return

        popup = Popup(
            title='Preview',
            size_hint=(None, None), size=(700, 500)
        )

        img = ImageMDButton()
        img.texture = self.selected_image.texture
        img.bind(
            on_press=lambda x: popup.dismiss()
        )

        popup.content = img
        popup.open()

    def delete_from_db(self):
        if not self.selected_image:
            return

        key = int(self.selected_image.source)
        call_db(f"DELETE FROM images WHERE id={key}")

        self.unselect_image()
        self.show_db_images()
