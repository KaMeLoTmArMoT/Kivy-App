import ast
from typing import Any

from kivy.clock import Clock
from kivy.uix.screenmanager import Screen
from kivymd.uix.boxlayout import MDBoxLayout
from kivymd.uix.label import MDLabel
from kivymd.uix.textfield import MDTextField, MDTextFieldHintText

from app.screens.utils.additional import BaseScreen
from app.screens.utils.custom_logging import get_logger

logger = get_logger(__name__)


class SettingItemRow(MDBoxLayout):
    """Encapsulated OOP row widget for a configuration key-value pair."""

    def __init__(self, key: str, value: Any, **kwargs):
        super().__init__(**kwargs)
        self.orientation = "horizontal"
        self.size_hint_y = None
        self.height = 42
        self.spacing = 16
        self.padding = [8, 2]

        self.key = key
        self.initial_value = value

        self.label = MDLabel(
            text=str(key),
            size_hint_x=0.45,
            font_size=15,
            valign="center",
            halign="left",
        )
        self.add_widget(self.label)

        self.text_field = MDTextField(
            text=str(value),
            multiline=False,
            size_hint_x=0.55,
            mode="filled",
            font_size=14,
        )
        self.text_field.add_widget(MDTextFieldHintText(text=f"Value for {key}"))
        self.add_widget(self.text_field)

    def get_data(self) -> tuple[str, str]:
        """Return the (key, current_value) tuple."""
        return self.key, self.text_field.text


class SettingsViewScreen(Screen, BaseScreen):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.grid = self.ids.grid
        Clock.schedule_once(self._delayed_init, 0)

    def on_enter(self, *args):
        self.setup_header()

    def _delayed_init(self, dt):
        self.show_settings()

    def show_settings(self):
        self.grid.clear_widgets()
        configs = self.db.get_config("*")

        for config, value in configs:
            try:
                val = ast.literal_eval(value)
            except (ValueError, SyntaxError):
                val = value

            row = SettingItemRow(key=config, value=val)
            self.grid.add_widget(row)

    def apply_changes(self):
        # Clean OOP traversal through SettingItemRow widgets
        for child in reversed(self.grid.children):
            if isinstance(child, SettingItemRow):
                key, value = child.get_data()
                logger.info(f"apply: key={key}, value={value}")
                self.db.set_config(key, value)

        logger.debug("Applying changes")
        self.show_settings()

    def reload_records(self):
        self.show_settings()

    def reset_settings(self):
        self.db.init_default_configs(force=True)
        self.show_settings()
