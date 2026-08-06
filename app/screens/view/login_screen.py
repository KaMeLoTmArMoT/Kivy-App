import os

from dotenv import load_dotenv
from kivy.clock import Clock
from kivy.uix.screenmanager import Screen

from app.screens.utils.additional import BaseScreen
from app.screens.utils.custom_logging import get_logger
from app.screens.utils.utils import get_sha

load_dotenv()
logger = get_logger(__name__)


class LoginScreen(Screen, BaseScreen):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.passwords = ""
        self.key = ""
        self.source_name = ""

    def on_enter(self, *args):
        self.source_name = "login_test" if os.environ.get("APP_ENV") == "test" else "login"

        self.create_db_and_check()

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
        logger.debug(f"Submit pressed with key {self.key} length {len(self.key)}")
        self.ids.word_input.text_validate_unfocus = False

        if len(self.key) <= 5:
            self.label_out("Password should be longer than 5 letters.")
            return

        if len(self.passwords) == 0:
            logger.debug("Create new password")
            self.submit_new_password(self.key)
        else:
            logger.debug("Validate password")
            self.validate_password(self.key)

    def validate_password(self, inp_pass):
        real_value = self.passwords[0][1]
        input_value = get_sha(inp_pass)
        logger.debug(f"Validating password: input hash {input_value}, real hash {real_value}")

        if real_value == input_value:
            logger.debug("Password is valid, transitioning to next screen")
            self.next_screen()
        else:
            logger.debug("Password is not valid, showing error message")
            self.label_out("Wrong password. Try again.")

    def submit_new_password(self, inp_pass):
        enc_pass = get_sha(inp_pass)
        logger.debug(f"Encrypting password to {enc_pass}")

        self.db.set_login_password(enc_pass, origin=self.source_name)
        self.next_screen()

    def next_screen(self):
        logger.debug(
            f"LOGIN: before switch "
            f"current={self.manager.current} "
            f"has_main={self.manager.has_screen('main')}"
        )
        self.manager.transition.direction = "left"
        self.manager.current = "main"
        logger.debug(f"LOGIN: after switch current={self.manager.current} (immediate)")
        Clock.schedule_once(
            lambda dt: logger.debug(f"LOGIN: +0 tick current={self.manager.current}"), 0
        )
        Clock.schedule_once(
            lambda dt: logger.debug(f"LOGIN: +0.2s current={self.manager.current}"), 0.2
        )
