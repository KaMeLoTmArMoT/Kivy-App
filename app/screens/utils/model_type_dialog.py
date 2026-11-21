from collections.abc import Callable

from kivy.uix.boxlayout import BoxLayout
from kivy.uix.button import Button
from kivy.uix.gridlayout import GridLayout
from kivy.uix.label import Label
from kivy.uix.popup import Popup

from app.screens.utils.additional import MDLabelBtn

MODEL_TYPES = (
    ("MobileNetV2", 3.5, 72.15),
    ("MobileNetV3", 5.5, 75.27),
    ("ResNet", 11.7, 69.76),
    ("ResNeXt", 25.0, 81.20),
    ("EfficientNet", 5.3, 77.69),
    ("EfficientNetV2", 21.5, 84.23),
    ("AlexNet", 61.1, 56.52),
    ("VGG", 132.9, 69.02),
)


def build_model_type_popup(
    current_type: str,
    on_select: Callable,
    on_submit: Callable,
    on_dismiss: Callable,
) -> Popup:
    popup = Popup(
        title="Please select model type:",
        title_align="center",
        title_size=20,
        size_hint=(None, None),
        size=(500, 400),
    )

    current = BoxLayout(orientation="horizontal", size_hint_y=0.2)
    current.add_widget(Label(text="Current:", font_size=18))
    current.add_widget(Label(text=current_type, font_size=18))

    grid = GridLayout(cols=2)
    for name, size, accuracy in MODEL_TYPES:
        button = Button(text=f"{name:<18} | {size}M | {accuracy}%")
        button.bind(on_press=on_select)
        grid.add_widget(button)
        grid.ids[name] = button

    submit = MDLabelBtn(text="Submit", size_hint_y=0.15)
    submit.allow_hover = True
    submit.bind(on_press=on_submit)

    content = BoxLayout(orientation="vertical")
    content.add_widget(current)
    content.add_widget(grid)
    content.add_widget(submit)
    popup.content = content
    popup.bind(on_dismiss=on_dismiss)
    return popup
