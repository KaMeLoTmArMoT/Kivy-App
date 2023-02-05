from base64 import b64encode

from kivy.uix.behaviors.button import ButtonBehavior
from kivy.uix.image import Image
from kivymd.uix import SpecificBackgroundColorBehavior
from kivymd.uix.behaviors import HoverBehavior
from kivymd.uix.button import ButtonBehavior as MDButtonBehavior
from kivymd.uix.label import MDLabel

from utils import call_db


class MDLabelBtn(ButtonBehavior, MDLabel, HoverBehavior):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.allow_hover = False
        self.saved_color = None

    def on_enter(self):
        if self.allow_hover:
            self.saved_color = self.md_bg_color.copy()
            self.md_bg_color = (1, 1, 1, 0.1)

    def on_leave(self):
        if self.allow_hover:
            self.md_bg_color = self.saved_color


class ImageMDButton(
    MDButtonBehavior, Image, SpecificBackgroundColorBehavior, HoverBehavior
):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.allow_hover = False
        self.saved_color = None

    def on_enter(self):
        if self.allow_hover:
            self.saved_color = self.md_bg_color.copy()
            self.md_bg_color = (1, 1, 1, 0.1)

    def on_leave(self):
        if self.allow_hover:
            self.md_bg_color = self.saved_color


class BaseScreen:
    def __init__(self):
        self.exit_screen = False

    def label_out(self, text: str):
        """Put string message to the label"""
        self.ids.word_label.text = text

    def get_input(self):
        return self.ids.word_input.text

    def encrypt(self, text):
        from Cryptodome.Cipher import AES

        cipher = AES.new(self.key, AES.MODE_EAX, nonce=b"TODO")
        encoded_text = cipher.encrypt(text.encode("utf-8"))
        b_encoded_text = b64encode(encoded_text).decode("utf-8")
        return b_encoded_text

    def select_direction(self, screen_name):
        self.exit_screen = True
        translations = {"main": 0, "imageview": 1, "dbview": 2, "mlview": 3}

        old = translations[self.manager.current]
        new = translations[screen_name]

        if old < new:
            self.manager.transition.direction = "left"
        else:
            self.manager.transition.direction = "right"

        self.manager.current = screen_name

    def create_db_and_check(self):
        # Create a table
        call_db(
            """
        CREATE TABLE IF NOT EXISTS images (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            image blob
        ) """
        )

    def goto_images(self):
        self.select_direction("imageview")

    def goto_main(self):
        self.select_direction("main")

    def goto_db(self):
        self.select_direction("dbview")

    def goto_ml(self):
        self.select_direction("mlview")
