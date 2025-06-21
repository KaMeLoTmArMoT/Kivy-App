from kivy.uix.screenmanager import Screen

from screens.additional import BaseScreen
from utils import get_sha


class LoginScreen(Screen, BaseScreen):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.passwords = ""
        self.key = ""

    def on_enter(self, *args):
        self.create_db_and_check()

        self.ids.word_input.focus = True

    def create_db_and_check(self):
        self.db.create_passwords_table()
        self.passwords = self.db.get_login_password()

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

        self.db.set_login_password(enc_pass)
        self.next_screen()

    def next_screen(self):
        self.manager.transition.direction = "left"
        self.manager.current = "main"
