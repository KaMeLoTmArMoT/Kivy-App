import os

from dotenv import load_dotenv
from kivy.clock import Clock
from kivy.uix.screenmanager import Screen

from app.screens.utils.additional import BaseScreen
from app.screens.utils.utils import get_sha

load_dotenv()


class LoginScreen(Screen, BaseScreen):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.passwords = ""
        self.key = ""
        self.source_name = ""

    def on_enter(self, *args):
        self.create_db_and_check()

        self.source_name = (
            "login_test" if os.environ.get("APP_ENV") == "test" else "login"
        )

        self.ids.word_input.focus = True

        if os.environ.get("APP_ENV") in ["dev", "test"]:
            dev_pass = os.environ.get("APP_DEV_PASSWORD", "")
            if dev_pass:
                self.key = dev_pass
            if dev_pass and os.environ.get("APP_AUTOLOGIN") == "1":
                self.ids.word_input.text = dev_pass
                Clock.schedule_once(lambda dt: self.submit(), 0.1)

    def create_db_and_check(self):
        self.passwords = self.db.get_login_password(self.source_name)

        if len(self.passwords) == 0:
            self.label_out("Enter a new password")
            self.ids.login.text = "Register"
            self.ids.word_input.password = False

    def submit(self):
        self.key = self.get_input()
        self.ids.word_input.text_validate_unfocus = False

        if len(self.key) <= 5:
            self.label_out("Password should be longer than 5 letters.")
            return

        if len(self.passwords) == 0:
            self.submit_new_password(self.key)
        else:
            self.validate_password(self.key)

    def validate_password(self, inp_pass):
        real_value = self.passwords[0][1]
        input_value = get_sha(inp_pass)

        if real_value == input_value:
            self.next_screen()
        else:
            self.label_out("Wrong password. Try again.")

    def submit_new_password(self, inp_pass):
        enc_pass = get_sha(inp_pass)

        self.db.set_login_password(enc_pass, origin=self.source_name)
        self.next_screen()

    def next_screen(self):
        self.manager.transition.direction = "left"
        self.manager.current = "main"
