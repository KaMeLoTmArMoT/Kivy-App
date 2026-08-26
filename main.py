import os
import sys

from kivy.config import Config
from kivy.core.window import Window
from kivy.lang import Builder
from kivy.resources import resource_add_path
from kivy.uix.screenmanager import ScreenManager
from kivymd.app import MDApp

from app.screens.view.loading_screen import LoadingScreen

Config.set("input", "mouse", "mouse,multitouch_on_demand")


class MainApp(MDApp):
    def build(self):
        self.theme_cls.theme_style = "Dark"
        self.theme_cls.primary_palette = "#607D8B"

        Builder.load_file("app/ui/loading.kv")

        sm = ScreenManager()
        sm.add_widget(LoadingScreen(name="loading"))
        sm.current = "loading"

        return sm


if __name__ == "__main__":
    if hasattr(sys, "_MEIPASS"):
        resource_add_path(os.path.join(sys._MEIPASS))

    Window.size = (960, 720)
    Window.top = 190
    Window.left = 450

    path = os.path.abspath(__file__)
    dir_path = os.path.dirname(path)
    resource_add_path(os.path.join(dir_path, "icons"))

    MainApp().run()
