import os
import sys

from kivy.config import Config

Config.set("graphics", "width", "960")
Config.set("graphics", "height", "720")
Config.set("graphics", "position", "custom")
Config.set("graphics", "top", "190")
Config.set("graphics", "left", "450")
Config.set("input", "mouse", "mouse,multitouch_on_demand")

from kivy.lang import Builder  # noqa: E402
from kivy.resources import resource_add_path  # noqa: E402
from kivy.uix.screenmanager import ScreenManager  # noqa: E402
from kivymd.app import MDApp  # noqa: E402

from app.screens.view.loading_screen import LoadingScreen  # noqa: E402


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

    path = os.path.abspath(__file__)
    dir_path = os.path.dirname(path)
    resource_add_path(os.path.join(dir_path, "icons"))

    MainApp().run()
