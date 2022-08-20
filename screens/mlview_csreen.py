import os
import shutil

from kivy.uix.button import Button
from kivy.uix.gridlayout import GridLayout
from kivy.uix.label import Label
from kivy.uix.screenmanager import Screen

from screens.additional import ML_FOLDER, BaseScreen, MDLabelBtn
from utils import call_db


class MLViewScreen(Screen, BaseScreen):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.key = ''
        self.selected = None

    def on_enter(self, *args):
        self.key = self.manager.get_screen('main').key
        self.create_db_and_check()
        self.load_classes()

    def create_db_and_check(self):
        # Create a table
        call_db("""
        CREATE TABLE IF NOT EXISTS images (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            image blob
        ) """)

    def load_classes(self):
        self.ids.grid.clear_widgets()

        for file in os.listdir(ML_FOLDER):
            path = os.path.join(ML_FOLDER, file)
            if os.path.isdir(path):
                btn = MDLabelBtn(text=file)
                btn.bind(on_press=self.select_label_btn)
                self.ids.grid.add_widget(btn)

    def select_label_btn(self, instance):
        print(f'The button <{instance.text}> is being pressed')
        if self.selected:
            if instance.uid == self.selected.uid:
                self.unselect_label_btn()
                return

        # reset selection
        for btn in self.ids.grid.children:
            btn.md_bg_color = (1.0, 1.0, 1.0, 0.0)

        instance.md_bg_color = (1.0, 1.0, 1.0, 0.1)
        instance.radius = (20, 20, 20, 20)
        self.selected = instance

    def unselect_label_btn(self):
        self.selected = None
        for btn in self.ids.grid.children:
            btn.md_bg_color = (1.0, 1.0, 1.0, 0.0)

    def add_class(self):
        name = self.ids.class_input.text
        if name == "":
            print('no input')
            return

        path = os.path.join(ML_FOLDER, name)
        if os.path.exists(path):
            print('we have such class!')
            return

        os.makedirs(path)
        self.load_classes()

    def delete_class(self):
        if self.selected is None:
            return

        path = ML_FOLDER + self.selected.text
        shutil.rmtree(path)

        self.unselect_label_btn()
        self.load_classes()

    def confirm(self):
        pass
