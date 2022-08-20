from kivy.uix.screenmanager import Screen
from screens.additional import BaseScreen
from utils import call_db


class MLViewScreen(Screen, BaseScreen):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.key = ''

    def on_enter(self, *args):
        self.key = self.manager.get_screen('main').key
        self.create_db_and_check()

    def create_db_and_check(self):
        # Create a table
        call_db("""
        CREATE TABLE IF NOT EXISTS images (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            image blob
        ) """)
