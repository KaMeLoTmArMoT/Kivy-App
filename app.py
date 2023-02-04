import os
import sys

from kivy.lang import Builder
from kivy.resources import resource_add_path
from kivy.uix.screenmanager import ScreenManager
from kivymd.app import MDApp

from screens.dbview_csreen import DbViewScreen
from screens.imageview_screen import ImageViewScreen
from screens.login_screen import LoginScreen
from screens.main_screen import MainScreen
from screens.mlview_csreen import MLViewScreen


class MainApp(MDApp):
    def build(self):
        sm = ScreenManager()
        sm.add_widget(LoginScreen(name="login"))
        sm.add_widget(MainScreen(name="main"))
        sm.add_widget(ImageViewScreen(name="imageview"))
        sm.add_widget(DbViewScreen(name="dbview"))
        sm.add_widget(MLViewScreen(name="mlview"))
        sm.current = "login"

        self.theme_cls.theme_style = "Dark"
        self.theme_cls.primary_palette = "BlueGray"

        return sm


if __name__ == "__main__":
    if hasattr(sys, "_MEIPASS"):
        resource_add_path(os.path.join(sys._MEIPASS))

    path = os.path.abspath(__file__)
    dir_path = os.path.dirname(path)
    resource_add_path(os.path.join(dir_path, "icons"))

    Builder.load_file("ui/app.kv")
    Builder.load_file("ui/login.kv")
    Builder.load_file("ui/main.kv")
    Builder.load_file("ui/imageview.kv")
    Builder.load_file("ui/dbview.kv")
    Builder.load_file("ui/mlview.kv")
    MainApp().run()
