from kivy.lang import Builder
from kivymd.app import MDApp
from kivy.uix.screenmanager import Screen, ScreenManager
import sqlite3
import hashlib


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


class BaseScreen:
    def label_out(self, text: str):
        """Put string message to the label"""
        self.ids.word_label.text = text


class LoginScreen(Screen, BaseScreen):
    def on_enter(self, *args):
        self.check_password()

    def check_password(self):
        # Create a password table
        call_db("""
        CREATE TABLE IF NOT EXISTS passwords (
            section text
            password blob
        ) """)

        # Check password
        passwords = call_db("SELECT * FROM passwords WHERE section='login'")

        if len(passwords) == 0:
            self.label_out('Enter new password')
            self.ids.login.text = 'Register'

    def submit_new_password(self):
        # todo password

        self.manager.transition.direction = 'left'
        self.manager.current = 'main'


class MainScreen(Screen, BaseScreen):
    def __int__(self, **kwargs):
        super().__init__(**kwargs)

    def on_enter(self, *args):
        # Create a table
        call_db("""
        CREATE TABLE IF NOT EXISTS customers (
            name text
        ) """)

    def submit(self):
        text = self.ids.word_input.text

        if len(text) <= 2:
            self.label_out('Text should be longer than 2 letters')
            return

        call_db(f"INSERT INTO customers VALUES ('{text}')")

        # show message
        self.label_out(f'{text} Added')

        # clear input box
        self.ids.word_input.text = ''

    def show_records(self):
        records = call_db("SELECT * FROM customers")

        words = ''
        for word in records:
            words += f'{word[0]}\n'

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
