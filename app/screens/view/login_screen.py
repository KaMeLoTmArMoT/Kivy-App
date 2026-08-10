import os

from dotenv import load_dotenv
from kivy.clock import Clock
from kivy.uix.screenmanager import Screen

from app.screens.services.auth_service import AuthService
from app.screens.utils.additional import BaseScreen
from app.screens.utils.custom_logging import get_logger

load_dotenv()
logger = get_logger(__name__)


class LoginScreen(Screen, BaseScreen):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.auth = AuthService(db=self.db)
        self.passwords = ""
        self.key = ""
        self.source_name = ""

    def on_enter(self, *args):
        self.source_name = self.auth.get_source_name()

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
        self.passwords = self.auth.get_stored_passwords(self.source_name)

        if not self.auth.is_registered(self.source_name):
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

        if not self.auth.is_registered(self.source_name):
            logger.debug("Create new password")
            self.submit_new_password(self.key)
        else:
            logger.debug("Validate password")
            self.validate_password(self.key)

    def validate_password(self, inp_pass):
        valid, msg = self.auth.validate_password(inp_pass, self.source_name)
        if valid:
            logger.debug("Password is valid, transitioning to next screen")
            self.next_screen()
        else:
            logger.debug("Password is not valid, showing error message")
            self.label_out(msg)

    def submit_new_password(self, inp_pass):
        self.auth.register_password(inp_pass, self.source_name)
        self.next_screen()

    def next_screen(self):
        logger.debug(
            f"LOGIN: before switch "
            f"current={self.manager.current} "
            f"has_main={self.manager.has_screen('main')}"
        )
        self.goto_main()
        logger.debug(f"LOGIN: after switch current={self.manager.current} (immediate)")
        Clock.schedule_once(
            lambda dt: logger.debug(f"LOGIN: +0 tick current={self.manager.current}"), 0
        )
        Clock.schedule_once(
            lambda dt: logger.debug(f"LOGIN: +0.2s current={self.manager.current}"), 0.2
        )
