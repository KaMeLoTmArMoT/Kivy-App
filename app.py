import os
import base64
import webbrowser
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

from kivymd.uix import SpecificBackgroundColorBehavior
from kivymd.uix.behaviors import HoverBehavior
from kivymd.uix.button import ButtonBehavior as MDButtonBehavior
from kivymd.uix.label import MDLabel
from kivymd.app import MDApp

from utils import *


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


class ImageMDButton(MDButtonBehavior, Image, SpecificBackgroundColorBehavior):
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

    def goto_db(self):
        self.manager.transition.direction = 'left'
        self.manager.current = 'dbview'


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

        self.show_records()

    def create_db_and_check(self):
        # Create a table
        call_db("""
        CREATE TABLE IF NOT EXISTS customers (
            name text
        ) """)

    def submit(self):
        text = self.get_input()
        self.ids.word_input.text_validate_unfocus = False
        self.unselect_label_btn()

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
        self.unselect_label_btn()
        if len(records) == 0:
            self.label_out('No records. Add any items.')
        else:
            self.label_out('DB instances:')

    def select_label_btn(self, instance):
        print(f'The button <{instance.text}> is being pressed')

        # reset selection
        grid = self.ids.scroll.children[0]  # TODO check correct index
        for btn in grid.children:
            btn.md_bg_color = (1.0, 1.0, 1.0, 0.0)

        instance.md_bg_color = (1.0, 1.0, 1.0, 0.1)
        instance.radius = (20, 20, 20, 20)
        self.selected = instance

    def unselect_label_btn(self):
        self.selected = None
        grid = self.ids.scroll.children[0]
        for btn in grid.children:
            btn.md_bg_color = (1.0, 1.0, 1.0, 0.0)

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

    def update_name(self):
        if self.selected is None:
            self.show_records()
            self.label_out('First select any element')
            return

        old_text = self.selected.text
        new_text = self.get_input()

        if len(new_text) <= 2:
            self.label_out('New text should be longer than 2 letters')
            return

        old_encrypted = self.encrypt(old_text)
        new_encrypted = self.encrypt(new_text)

        call_db(f"UPDATE customers SET name='{new_encrypted}' WHERE name='{old_encrypted}'")

        self.ids.word_input.text = ''
        self.show_records()
        self.label_out(f'Successfully updated.')

    def open_url(self):
        if self.selected is None:
            self.label_out("Select link to open")
            return

        url = self.selected.text
        if "http" not in url:   # TODO: check for other link types
            self.label_out("This is not a link probably")
            return

        chrome_path = 'C:/Program Files/Google/Chrome/Application/chrome.exe %s --incognito'
        webbrowser.get(chrome_path).open(url)


class ImageViewScreen(Screen, BaseScreen):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.grid = None
        self.selected_images = []

    def on_enter(self, *args):
        self.grid = self.ids.grid
        self.selected_counter_update()
        self.create_db_and_check()
        self.show_folder_images('G://Downloads//photo')   # TODO: remove this

    def create_db_and_check(self):
        # Create a table
        call_db("""
        CREATE TABLE IF NOT EXISTS images (
            image blob
        ) """)

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
        btn.allow_hover = True

        box.add_widget(lbl)
        box.add_widget(chooser)
        box.add_widget(btn)

        popup.content = box
        popup.open()

    def show_folder_images(self, path, selection=None, popup=None):
        files = os.listdir(path)
        self.grid.clear_widgets()
        self.unselect_all_images()
        self.selected_counter_update()

        for name in files:
            if '.jpg' in name or '.png' in name:
                im_path = os.path.join(path, name)
                img = ImageMDButton(
                    source=im_path,
                    allow_stretch=True,
                    keep_ratio=True
                )
                img.line_color = (1.0, 1.0, 1.0, 0.2)
                img.bind(on_press=self.image_click)
                self.grid.add_widget(img)

        if popup is not None:
            popup.dismiss()

    def image_click(self, instance):
        path = instance.source

        if instance in self.selected_images:
            instance.md_bg_color = (1.0, 1.0, 1.0, 0.0)
            instance.line_color = (1.0, 1.0, 1.0, 0.2)
            self.selected_images.remove(instance)
        else:
            instance.line_color = (1.0, 1.0, 1.0, 0.6)
            instance.md_bg_color = (1.0, 1.0, 1.0, 0.1)
            self.selected_images.append(instance)

        self.selected_counter_update()

    def save_img_to_db(self):
        num_images = len(self.selected_images)
        if num_images == 0:
            self.ids.selected_images.text = 'Choose 1+'
            return

        for path in self.selected_images:
            with open(path.source, 'rb') as f:
                blob_data = f.read()
                call_db(
                    f"INSERT INTO images VALUES (?)",
                    [blob_data]
                )
        self.unselect_all_images()
        self.ids.selected_images.text = f'Added {num_images}'

    def unselect_all_images(self):
        instances = self.selected_images.copy()
        for instance in instances:
            instance.md_bg_color = (1.0, 1.0, 1.0, 0.0)
            instance.line_color = (1.0, 1.0, 1.0, 0.2)
            self.selected_images.remove(instance)

    def selected_counter_update(self):
        self.ids.selected_images.text = f'Selected: {len(self.selected_images)}'


class DbViewScreen(Screen, BaseScreen):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def on_enter(self, *args):
        pass

    def preview_img(self):
        pass

    def delete_from_db(self):
        pass


class MainApp(MDApp):
    def build(self):
        sm = ScreenManager()
        sm.add_widget(LoginScreen(name='login'))
        sm.add_widget(MainScreen(name='main'))
        sm.add_widget(ImageViewScreen(name='imageview'))
        sm.add_widget(DbViewScreen(name='dbview'))
        sm.current = 'login'

        self.theme_cls.theme_style = "Dark"
        self.theme_cls.primary_palette = 'BlueGray'

        return sm


if __name__ == '__main__':
    Builder.load_file('app.kv')
    MainApp().run()
