from kivy.uix.screenmanager import Screen
from screens.additional import BaseScreen, ImageMDButton, MDLabelBtn


class SettingsViewScreen(Screen, BaseScreen):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        BaseScreen.__init__(self)
