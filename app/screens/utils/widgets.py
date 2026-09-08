import kivy.lang.builder as _builder
from kivy.properties import BooleanProperty, ListProperty, ObjectProperty, StringProperty
from kivy.uix.behaviors.button import ButtonBehavior
from kivy.uix.image import Image
from kivymd.uix.behaviors import BackgroundColorBehavior, HoverBehavior
from kivymd.uix.button import MDButton, MDButtonText
from kivymd.uix.floatlayout import MDFloatLayout
from kivymd.uix.label import MDLabel

_orig_build_canvas = _builder.BuilderBase._build_canvas


def _safe_build_canvas(self, canvas, widget, rule, rootrule):
    try:
        return _orig_build_canvas(self, canvas, widget, rule, rootrule)
    except Exception as e:
        if "Invalid width value" in str(e):
            return None
        raise e


_builder.BuilderBase._build_canvas = _safe_build_canvas


def _mdbutton_get_text(self):
    for child in self.children:
        if isinstance(child, MDButtonText):
            return child.text
    return getattr(self, "_text_compat", "")


def _mdbutton_set_text(self, val):
    self._text_compat = val
    found = False
    for child in self.children:
        if isinstance(child, MDButtonText):
            child.text = str(val)
            found = True
    if not found and not self.children:
        self.add_widget(MDButtonText(text=str(val)))


MDButton.text = property(_mdbutton_get_text, _mdbutton_set_text)


class _HoverColorMixin:
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.allow_hover = False
        self.saved_color = None

    def on_enter(self):
        if self.allow_hover:
            current_color = getattr(self, "md_bg_color", None)
            if current_color is not None:
                self.saved_color = list(current_color)
            self.md_bg_color = (1, 1, 1, 0.1)

    def on_leave(self):
        if self.allow_hover and self.saved_color is not None:
            self.md_bg_color = self.saved_color


class MDLabelBtn(_HoverColorMixin, ButtonBehavior, MDLabel, HoverBehavior):
    pass


class ImageMDButton(
    _HoverColorMixin,
    ButtonBehavior,
    Image,
    BackgroundColorBehavior,
    HoverBehavior,
):
    def __init__(self, **kwargs):
        kwargs.setdefault("md_bg_color", (0, 0, 0, 0))
        super().__init__(**kwargs)


class SelectableImage(MDFloatLayout):
    selected = BooleanProperty(False)
    source = StringProperty("")
    texture = ObjectProperty(None, allownone=True)
    line_color = ListProperty([1.0, 1.0, 1.0, 0.2])

    def on_source(self, instance, value):
        if hasattr(self, "ids") and "img" in self.ids and value:
            self.ids.img.source = value

    def on_texture(self, instance, value):
        if hasattr(self, "ids") and "img" in self.ids and value is not None:
            self.ids.img.texture = value

    def toggle_select(self) -> bool:
        """Toggle current selection state and return new value."""
        self.selected = not self.selected
        return self.selected
