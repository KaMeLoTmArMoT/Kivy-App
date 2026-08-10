import threading
from collections.abc import Callable
from typing import Any

from kivy.clock import Clock
from kivymd.uix.boxlayout import MDBoxLayout

from app.screens.utils.custom_logging import get_logger
from app.screens.utils.db import DB
from app.screens.utils.widgets import ImageMDButton, MDLabelBtn, SelectableImage

logger = get_logger(__name__)

__all__ = ["BaseScreen", "Header", "ImageMDButton", "MDLabelBtn", "SelectableImage"]


class BaseScreen:
    manager = None
    progress_bar = None
    ids = None
    key = None

    def __init__(self):
        self.exit_screen = False
        self.db = DB()

    def run_async(
        self, worker_fn: Callable[[], Any], callback: Callable[[Any], None] | None = None
    ) -> None:
        """Run worker_fn in a background thread and trigger callback(result) on main Kivy thread."""

        def _target():
            res = None
            try:
                res = worker_fn()
            except Exception as e:
                logger.error(f"[async worker error] {e}", exc_info=True)
            if callback:
                Clock.schedule_once(lambda dt: callback(res))

        threading.Thread(target=_target, daemon=True).start()

    def label_out(self, text: str):
        """Put string message to the label"""
        lbl = self.ids.get("word_label")
        if lbl is None:
            logger.error("[label_out] No label found in ids")
            return
        self.ids.word_label.text = text

    def get_input(self) -> str:
        return self.ids.word_input.text

    def setup_header(self):
        if "header" in self.ids and self.manager:
            current_screen = self.manager.current
            if current_screen in self.ids.header.ids:
                self.ids.header.ids[current_screen].background_color = (1, 1, 1, 1)

    def select_direction(self, screen_name: str):
        self.exit_screen = True
        translations = {
            "loading": -2,
            "login": -1,
            "main": 0,
            "imageview": 1,
            "dbview": 2,
            "mlview": 3,
            "detectionview": 4,
            "settingsview": 5,
        }

        old = translations[self.manager.current]
        new = translations[screen_name]

        if old < new:
            self.manager.transition.direction = "left"
        else:
            self.manager.transition.direction = "right"

        self.manager.current = screen_name

    def toggle_load_label(self, mode: str, text: str = "Loading, please wait..."):
        lbl = self.ids.get("load_label")
        if lbl is None:
            return

        def lbl_prop(
            text: str = "",
            lbl_hint_y: float = 0.1,
            color: tuple[float, float, float, float] = (1, 1, 1, 1),
            pbar_hint_y: float = 0.1,
            opacity: float = 1,
        ):
            lbl.text = text
            lbl.size_hint_y = lbl_hint_y
            lbl.color = color
            if self.progress_bar is not None:
                self.progress_bar.size_hint_y = pbar_hint_y
                self.progress_bar.opacity = opacity

        if mode == "on":
            lbl_prop(text)

        elif mode == "no_dir":
            lbl_prop("No images, please select folder.", opacity=0)

        elif mode == "success":
            lbl_prop("Success!", color=(0, 1, 0, 1))
            Clock.schedule_once(lambda tm: self.toggle_load_label("off"), 1)

        elif mode == "off":
            lbl_prop(lbl_hint_y=0, pbar_hint_y=0, opacity=0)

    def goto_images(self):
        self.select_direction("imageview")

    def goto_main(self):
        self.select_direction("main")

    def goto_db(self):
        self.select_direction("dbview")

    def goto_ml(self):
        self.select_direction("mlview")

    def goto_detection(self):
        self.select_direction("detectionview")

    def goto_settings(self):
        self.select_direction("settingsview")

    def launch_tensorboard(self):
        if getattr(self, "tb_server", None) and getattr(self, "tb_folder", None):
            status = self.tb_server.launch_tensorboard(self.tb_folder)
            logger.warning(status)
            self.update_all_button_states()


class Header(MDBoxLayout, BaseScreen):
    def __init__(self, **kwargs):  # TODO: check
        super().__init__(**kwargs)
