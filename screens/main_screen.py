import base64
import os
import webbrowser

from Cryptodome.Cipher import AES
from kivy.clock import Clock
from kivy.uix.gridlayout import GridLayout
from kivy.uix.screenmanager import Screen

from screens.additional import BaseScreen, MDLabelBtn
from screens.db import DB
from utils import extend_key

chrome_path = DB().get_config_typed("chrome_path")


class MainScreen(Screen, BaseScreen):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.key = None
        self.selected = None

        self.delete_btn = self.ids.text_delete
        self.update_btn = self.ids.text_update
        self.url_btn = self.ids.url_open
        self.submit_btn = self.ids.text_submit

    def on_enter(self, *args):
        self.ids.header.ids[self.manager.current].background_color = 1, 1, 1, 1
        self.ids.word_input.focus = True
        self.ids.word_input.bind(text=self.on_text_input)

        self.key = extend_key(self.manager.get_screen("login").key)

        self.reload_records()

    def submit(self):
        text = self.get_input()
        self.ids.word_input.text_validate_unfocus = False
        self.unselect_label_btn()

        if len(text) <= 2:
            self.label_out("Text should be longer than 2 letters")
            return

        b_encoded_text = self.encrypt(text)
        self.db.insert_customer(b_encoded_text)

        self.reload_records()

        # show message
        self.label_out(f"{text} added")
        Clock.schedule_once(lambda x: self.label_out("Enter new text:"), 1)

        # clear input box
        self.ids.word_input.text = ""

    def reload_records(self):
        records = self.db.get_customers()

        layout = GridLayout(cols=1, spacing=10, size_hint_y=None)
        layout.bind(minimum_height=layout.setter("height"))

        for word in records:
            cipher = AES.new(self.key, AES.MODE_EAX, nonce=b"TODO")

            tm = word[0]
            tm = base64.b64decode(tm.encode("utf-8"))
            tm = cipher.decrypt(tm).decode("utf-8")

            btn = MDLabelBtn(text=tm)
            btn.bind(on_press=self.select_label_btn)
            layout.add_widget(btn)

            self.ids[f"{word}"] = btn

        self.ids.scroll.clear_widgets()
        self.ids.scroll.add_widget(layout)
        self.unselect_label_btn()
        if len(records) == 0:
            self.label_out("No records. Add any items.")
        else:
            self.label_out("DB instances:")

    def select_label_btn(self, instance):
        print(f"The button <{instance.text}> is being pressed")
        if self.selected:
            if instance.uid == self.selected.uid:
                self.unselect_label_btn()
                return

        # reset selection
        grid = self.ids.scroll.children[0]  # TODO check correct index
        for btn in grid.children:
            btn.md_bg_color = (1.0, 1.0, 1.0, 0.0)

        instance.md_bg_color = (1.0, 1.0, 1.0, 0.1)
        instance.radius = (20, 20, 20, 20)
        self.selected = instance

        self.delete_btn.disabled = False

        if len(self.get_input()) > 2:
            self.update_btn.disabled = False

        if "http" in self.selected.text:
            self.url_btn.disabled = False
        else:
            self.url_btn.disabled = True

    def unselect_label_btn(self):
        self.selected = None
        grid = self.ids.scroll.children[0]
        for btn in grid.children:
            btn.md_bg_color = (1.0, 1.0, 1.0, 0.0)

        self.delete_btn.disabled = True
        self.update_btn.disabled = True
        self.url_btn.disabled = True

    def delete_record(self):
        # TODO: fix - sometimes deleted 2-3 records instead of 1
        if self.selected is None:
            self.reload_records()
            self.label_out("First select any element")
            return

        text = self.selected.text
        b_encoded_text = self.encrypt(text)
        self.db.delete_customer(b_encoded_text)

        self.selected = None
        self.reload_records()
        self.label_out(f"Deleted: {text}")

    def update_record(self):
        if self.selected is None:
            self.reload_records()
            self.label_out("First select any element")
            return

        old_text = self.selected.text
        new_text = self.get_input()

        if len(new_text) <= 2:
            self.label_out("New text should be longer than 2 letters")
            return

        old_encrypted = self.encrypt(old_text)
        new_encrypted = self.encrypt(new_text)

        self.db.update_customer(new_encrypted, old_encrypted)

        self.ids.word_input.text = ""
        self.reload_records()
        self.label_out("Successfully updated.")

    def open_url(self):
        if self.selected is None:
            self.label_out("Select link to open")
            return

        url = self.selected.text
        if "http" not in url:  # TODO: check for other link types
            self.label_out("This is not a link probably")
            return

        if os.name == "posix":
            webbrowser.open(url)
        else:
            webbrowser.get(chrome_path + " --incognito").open(url)

    def on_text_input(self, instance, value):
        text = self.get_input()

        self.update_buttons_state()
        if len(text) > 2:
            self.submit_btn.disabled = False

            if self.selected:
                self.update_btn.disabled = False

        else:
            self.submit_btn.disabled = True
            self.update_btn.disabled = True

    def update_buttons_state(self):
        """TODO: implement and use"""
        pass
