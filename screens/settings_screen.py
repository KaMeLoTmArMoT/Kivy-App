import ast

from kivy.uix.screenmanager import Screen
from kivy.uix.label import Label
from kivymd.uix.textfield import MDTextField

from screens.additional import BaseScreen, MDLabelBtn
from screens.db import DB


class SettingsViewScreen(Screen, BaseScreen):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.db = DB()
        self.grid = self.ids.grid
        self.show_settings()

    def show_settings(self):
        # self.grid.clear_widgets()

        a = self.db.get_config("*")
        print("* values from db", a)

        for config, value in a:
            try:
                val = ast.literal_eval(value)

            except (ValueError, SyntaxError) as e:
                print(f"[get_config_typed] Error: {value} {e}")
                val = value

            print("-----", config, val, type(val))

            lbl = Label(text=config)
            self.grid.add_widget(lbl)
            txt = MDTextField(text=str(val))
            self.grid.add_widget(txt)
