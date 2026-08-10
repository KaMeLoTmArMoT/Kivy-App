from kivy.properties import BooleanProperty, ListProperty, ObjectProperty, StringProperty
from kivy.uix.behaviors.button import ButtonBehavior
from kivy.uix.image import Image
from kivymd.uix import SpecificBackgroundColorBehavior
from kivymd.uix.behaviors import HoverBehavior
from kivymd.uix.button import ButtonBehavior as MDButtonBehavior
from kivymd.uix.floatlayout import MDFloatLayout
from kivymd.uix.label import MDLabel


class _HoverColorMixin:
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.allow_hover = False
        self.saved_color = None

    def on_enter(self):
        if self.allow_hover:
            self.saved_color = self.md_bg_color.copy()
            self.md_bg_color = (1, 1, 1, 0.1)

    def on_leave(self):
        if self.allow_hover:
            self.md_bg_color = self.saved_color


class MDLabelBtn(_HoverColorMixin, ButtonBehavior, MDLabel, HoverBehavior):
    pass


class ImageMDButton(
    _HoverColorMixin,
    MDButtonBehavior,
    Image,
    SpecificBackgroundColorBehavior,
    HoverBehavior,
):
    pass


class SelectableImage(MDFloatLayout):
    selected = BooleanProperty(False)
    source = StringProperty("")
    texture = ObjectProperty(None)
    line_color = ListProperty([1.0, 1.0, 1.0, 0.2])
