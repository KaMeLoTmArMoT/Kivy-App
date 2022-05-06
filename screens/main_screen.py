import base64
import webbrowser

from Cryptodome.Cipher import AES
from kivy.uix.gridlayout import GridLayout
from kivy.uix.screenmanager import Screen

from screens.additional import BaseScreen, MDLabelBtn
from utils import call_db, extend_key


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
