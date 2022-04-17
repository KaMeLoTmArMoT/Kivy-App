from Cryptodome.Cipher import AES
from kivy.lang import Builder
from kivymd.app import MDApp
from kivy.uix.screenmanager import Screen, ScreenManager
import sqlite3
import hashlib
import base64


def call_db(call):
    # Create db
    conn = sqlite3.connect('app.db')

    # Create cursor
    c = conn.cursor()

    # Execute SQL command
    c.execute(call)
    records = c.fetchall()

    # Commit changes
    conn.commit()

    # Close connection
    conn.close()

    return records


def get_sha(text):
    enc = hashlib.sha256()
    enc.update(text.encode('utf-8'))
    return enc.hexdigest()


def extend_key(text):
    if len(text) < 16:
        while len(text) < 16:
            text += text
    return text[:16].encode('utf-8')


class BaseScreen:
    def label_out(self, text: str):
        """Put string message to the label"""
        self.ids.word_label.text = text

    def get_input(self):
        return self.ids.word_input.text


class LoginScreen(Screen, BaseScreen):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.passwords = ''
        self.key = ''

    def on_enter(self, *args):
        self.create_db_and_check()

        self.ids.word_input.focus = True

    def create_db_and_check(self):
        # Create a password table
        call_db("""
        CREATE TABLE IF NOT EXISTS passwords (
            destination text,
            password text
        ) """)

        # Check password
        self.passwords = call_db("SELECT * FROM passwords WHERE destination='login'")

        if len(self.passwords) == 0:
            self.label_out('Enter a new password')
            self.ids.login.text = 'Register'

    def submit(self):
        self.key = self.get_input()
        self.ids.word_input.text_validate_unfocus = False

        if len(self.key) <= 5:
            self.label_out('Password should be longer than 5 letters.')
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
            self.label_out('Wrong password. Try again.')

    def submit_new_password(self, inp_pass):
        enc_pass = get_sha(inp_pass)

        call_db(f"INSERT INTO passwords VALUES ('login', '{enc_pass}')")
        self.next_screen()

    def next_screen(self):
        self.manager.transition.direction = 'left'
        self.manager.current = 'main'


class MainScreen(Screen, BaseScreen):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.key = None

    def on_enter(self, *args):
        self.create_db_and_check()

        key = self.manager.get_screen('login').key
        self.key = extend_key(key)

    def create_db_and_check(self):
        # Create a table
        call_db("""
        CREATE TABLE IF NOT EXISTS customers (
            name text
        ) """)

    def submit(self):
        text = self.get_input()
        self.ids.word_input.text_validate_unfocus = False

        if len(text) <= 2:
            self.label_out('Text should be longer than 2 letters')
            return

        cipher = AES.new(self.key, AES.MODE_EAX, nonce=b'TODO')
        encoded_text = cipher.encrypt(text.encode('utf-8'))
        b_encoded_text = base64.b64encode(encoded_text).decode('utf-8')

        call_db(f"INSERT INTO customers VALUES ('{b_encoded_text}')")

        # show message
        self.label_out(f'{text} Added')

        # clear input box
        self.ids.word_input.text = ''

    def show_records(self):
        records = call_db("SELECT * FROM customers")

        words = ''
        for word in records:
            cipher = AES.new(self.key, AES.MODE_EAX, nonce=b'TODO')

            tm = word[0]
            tm = base64.b64decode(tm.encode('utf-8'))
            tm = cipher.decrypt(tm).decode('utf-8')
            words += f'{tm}\n'

        self.label_out(words)


class MainApp(MDApp):
    def build(self):
        sm = ScreenManager()
        sm.add_widget(LoginScreen(name='login'))
        sm.add_widget(MainScreen(name='main'))
        sm.current = 'login'

        self.theme_cls.theme_style = "Dark"
        self.theme_cls.primary_palette = 'BlueGray'

        return sm


if __name__ == '__main__':
    Builder.load_file('app.kv')
    MainApp().run()
