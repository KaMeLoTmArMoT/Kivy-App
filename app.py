import os
import sys

from kivy.core.window import Window
from kivy.lang import Builder
from kivy.resources import resource_add_path
from kivy.uix.screenmanager import ScreenManager
from kivymd.app import MDApp

from screens.loading_screen import LoadingScreen


class MainApp(MDApp):
    def build(self):
        sm = ScreenManager()

        sm.add_widget(LoadingScreen(name="loading"))
        sm.current = "loading"

        self.theme_cls.theme_style = "Dark"
        self.theme_cls.primary_palette = "BlueGray"

        return sm


if __name__ == "__main__":
    if hasattr(sys, "_MEIPASS"):
        resource_add_path(os.path.join(sys._MEIPASS))

    Window.size = (960, 720)

    path = os.path.abspath(__file__)
    dir_path = os.path.dirname(path)
    resource_add_path(os.path.join(dir_path, "icons"))

    Builder.load_file("ui/loading.kv")
    MainApp().run()
