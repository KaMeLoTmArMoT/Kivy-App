import ast

from kivy.uix.label import Label
from kivy.uix.screenmanager import Screen
from kivymd.uix.textfield import MDTextField

from screens.additional import BaseScreen
from screens.custom_logging import get_logger
from screens.db import DB

logger = get_logger(__name__)


class SettingsViewScreen(Screen, BaseScreen):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.db = DB()
        self.grid = self.ids.grid
        self.show_settings()

    def show_settings(self):
        self.grid.clear_widgets()
        a = self.db.get_config("*")

        for config, value in a:
            try:
                val = ast.literal_eval(value)

            except (ValueError, SyntaxError) as e:
                logger.warning(f"[get_config_typed] Error: {value} {e}")
                val = value

            lbl = Label(text=config)
            self.grid.add_widget(lbl)
            txt = MDTextField(text=str(val))
            self.grid.add_widget(txt)

    def apply_changes(self):
        children = self.grid.children[::-1]
        for i in range(0, len(children), 2):
            key_widget = (
                children[i] if isinstance(children[i], Label) else children[i + 1]
            )
            val_widget = (
                children[i + 1]
                if isinstance(children[i + 1], MDTextField)
                else children[i]
            )

            if isinstance(key_widget, Label) and isinstance(val_widget, MDTextField):
                key = key_widget.text
                value = val_widget.text
                logger.info(f"apply: key={key}, value={value}")
                self.db.set_config(key, value)

        logger.debug("Applying changes")
        self.show_settings()

    def reload_records(self):
        self.show_settings()

    def reset_settings(self):
        self.db.init_default_configs(force=True)
        self.show_settings()
