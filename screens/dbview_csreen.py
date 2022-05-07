import io

import cv2
import numpy as np
from Cryptodome.Cipher import AES
from kivy.core.image import Image as CoreImage
from kivy.graphics.texture import Texture
from kivy.uix.popup import Popup
from kivy.uix.screenmanager import Screen
from kivymd.uix.floatlayout import MDFloatLayout
from kivymd.uix.selectioncontrol import MDCheckbox

from screens.additional import BaseScreen, ImageMDButton
from utils import call_db


class DbViewScreen(Screen, BaseScreen):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.grid_1 = None
        self.grid_2 = None
        self.key = ''
        self.loaded = False
        self.selected_images = []
        self.prev_line_color = None
        self.checkbox_first = None

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
        self.unselect_all_images()

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
                pos_hint={'center_x': .5, 'center_y': .5},
            )

            success = False
            grid, texture = None, None
            try:
                data = io.BytesIO(b_image)
                texture = CoreImage(data, ext="png").texture
                success = True
                img_button.line_color = (1.0, 0.6, 0.0, 0.5)
                grid = self.grid_1
            except Exception as e:
                print(e)

            if not success:     # try to decrypt
                try:
                    cipher = AES.new(self.key, AES.MODE_EAX, nonce=b'TODO')
                    data = io.BytesIO(cipher.decrypt(b_image))
                    texture = CoreImage(data, ext="png").texture
                    success = True
                    img_button.line_color = (0.0, 1.0, 0.0, 0.5)
                    grid = self.grid_2
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
                grid = self.grid_1

            img_button.source = str(pk)
            img_button.texture = texture
            img_button.bind(on_press=self.image_click)

            checkbox = MDCheckbox(
                size_hint=(None, None),
                size=("48dp", "48dp"),
                pos_hint={'center_x': 0.96, 'center_y': 0.96},
            )
            checkbox.bind(on_press=self.checkbox_click)

            fl = MDFloatLayout()
            fl.add_widget(img_button)
            fl.add_widget(checkbox)

            grid.add_widget(fl)

        self.toggle_load_label('off')

    def toggle_load_label(self, mode, text="Loading, please wait..."):
        lbl = self.ids.load_label

        if mode == 'on':
            lbl.text = text
            lbl.size_hint_y = 0.2
        else:
            lbl.text = ""
            lbl.size_hint_y = 0

    def checkbox_click(self, instance):
        print(instance)
        self.checkbox_first = True

    def image_click(self, instance):
        path = instance.source

        if instance in self.selected_images:
            self.unselect_image(instance)
            return

        if not self.checkbox_first:
            self.unselect_all_images()  # remove all
        else:
            self.checkbox_first = False

        self.select_image(instance)     # choose new

    def select_image(self, instance):
        self.selected_images.append(instance)

        self.prev_line_color = instance.line_color
        instance.line_color = (1.0, 1.0, 1.0, 0.6)
        instance.md_bg_color = (1.0, 1.0, 1.0, 0.1)
        instance.parent.children[0].active = True

    def unselect_image(self, instance):
        if len(self.selected_images) > 0:
            instance.line_color = self.prev_line_color
            instance.md_bg_color = (1.0, 1.0, 1.0, 0.0)
            instance.parent.children[0].active = False   # disable checkbox
            self.selected_images.remove(instance)

    def unselect_all_images(self):
        images = self.selected_images.copy()
        for image in images:
            self.unselect_image(image)

    def preview_img(self):
        if len(self.selected_images) != 1:
            return

        popup = Popup(
            title='Preview',
            size_hint=(None, None), size=(700, 500)
        )

        img = ImageMDButton()
        img.texture = self.selected_images[0].texture
        img.bind(
            on_press=lambda x: popup.dismiss()
        )

        popup.content = img
        popup.open()

    def delete_from_db(self):
        if len(self.selected_images) < 1:
            return

        for image in self.selected_images:
            key = int(image.source)
            call_db(f"DELETE FROM images WHERE id={key}")

        self.unselect_all_images()
        self.show_db_images()
