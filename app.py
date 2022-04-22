import os
import base64
from Cryptodome.Cipher import AES

from kivy.uix.gridlayout import GridLayout
from kivy.uix.behaviors.button import ButtonBehavior
from kivy.uix.screenmanager import Screen, ScreenManager
from kivy.uix.popup import Popup
from kivy.uix.label import Label
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.filechooser import FileChooserListView, FileChooserIconView
from kivy.uix.image import Image

from kivy.lang import Builder
from kivymd.uix.label import MDLabel
from kivymd.app import MDApp

from utils import *


class MDLabelBtn(ButtonBehavior, MDLabel):
    pass


class BaseScreen:
    def label_out(self, text: str):
        """Put string message to the label"""
        self.ids.word_label.text = text

    def get_input(self):
        return self.ids.word_input.text

    def encrypt(self, text):
        cipher = AES.new(self.key, AES.MODE_EAX, nonce=b'TODO')
        encoded_text = cipher.encrypt(text.encode('utf-8'))
        b_encoded_text = base64.b64encode(encoded_text).decode('utf-8')
        return b_encoded_text

    def goto_images(self):
        self.manager.transition.direction = 'left'
        self.manager.current = 'imageview'

    def goto_main(self):
        self.manager.transition.direction = 'right'
        self.manager.current = 'main'


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
        self.selected = None

    def on_enter(self, *args):
        self.ids.word_input.focus = True
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

        b_encoded_text = self.encrypt(text)
        call_db(f"INSERT INTO customers VALUES ('{b_encoded_text}')")

        self.show_records()

        # show message
        self.label_out(f'{text} added')

        # clear input box
        self.ids.word_input.text = ''

    def show_records(self):
        records = call_db("SELECT * FROM customers")

        layout = GridLayout(cols=1, spacing=10, size_hint_y=None)
        layout.bind(minimum_height=layout.setter('height'))

        for word in records:
            cipher = AES.new(self.key, AES.MODE_EAX, nonce=b'TODO')

            tm = word[0]
            tm = base64.b64decode(tm.encode('utf-8'))
            tm = cipher.decrypt(tm).decode('utf-8')

            btn = MDLabelBtn(text=tm)
            btn.bind(on_press=self.select_label_btn)
            layout.add_widget(btn)

            self.ids[f'{word}'] = btn

        self.ids.scroll.clear_widgets()
        self.ids.scroll.add_widget(layout)
        self.label_out('DB instances:')

    def select_label_btn(self, instance):
        print(f'The button <{instance.text}> is being pressed')

        # reset selection
        grid = self.ids.scroll.children[0]  # TODO check correct index
        for btn in grid.children:
            btn.md_bg_color = (1.0, 1.0, 1.0, 0.0)

        instance.md_bg_color = (1.0, 1.0, 1.0, 0.1)
        self.selected = instance

    def delete_name(self):
        if self.selected is None:
            self.show_records()
            self.label_out('First select any element')
            return

        text = self.selected.text
        b_encoded_text = self.encrypt(text)

        call_db(f"DELETE FROM customers WHERE name='{b_encoded_text}'")

        self.selected = None
        self.show_records()
        self.label_out(f'Deleted: {text}')


class ImageViewScreen(Screen, BaseScreen):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.grid = None

    def on_enter(self, *args):
        self.grid = self.ids.grid

    def file_chooser_popup(self):
        popup = Popup(
            title='Filechooser',
            size_hint=(None, None), size=(400, 400)
        )

        box = BoxLayout(orientation='vertical')
        lbl = Label(text='Please select folder', size_hint_y=0.1)
        # chooser = FileChooserListView()
        chooser = FileChooserIconView()

        btn = MDLabelBtn(text='Submit', size_hint_y=0.1)
        btn.bind(
            on_press=lambda x: self.show_folder_images(
                chooser.path,
                chooser.selection,
                popup,
            )
        )

        box.add_widget(lbl)
        box.add_widget(chooser)
        box.add_widget(btn)

        popup.content = box
        popup.open()

    def show_folder_images(self, path, selection, popup):
        files = os.listdir(path)
        self.grid.clear_widgets()

        for name in files:
            if '.jpg' in name or '.png' in name:
                im_path = os.path.join(path, name)
                img = Image(
                    source=im_path,
                    allow_stretch=True,
                    keep_ratio=True
                )
                self.grid.add_widget(img)

        popup.dismiss()


class MainApp(MDApp):
    def build(self):
        sm = ScreenManager()
        sm.add_widget(LoginScreen(name='login'))
        sm.add_widget(MainScreen(name='main'))
        sm.add_widget(ImageViewScreen(name='imageview'))
        sm.current = 'login'

        self.theme_cls.theme_style = "Dark"
        self.theme_cls.primary_palette = 'BlueGray'

        return sm


if __name__ == '__main__':
    Builder.load_file('app.kv')
    MainApp().run()
