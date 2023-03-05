from kivy.uix.screenmanager import Screen

from screens.additional import BaseScreen


class DetectionScreen(Screen, BaseScreen):
    def __int__(self, **kwargs):
        super().__init__(**kwargs)
