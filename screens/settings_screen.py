from kivy.uix.screenmanager import Screen

from screens.additional import BaseScreen


class SettingsViewScreen(Screen, BaseScreen):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        BaseScreen.__init__(self)
