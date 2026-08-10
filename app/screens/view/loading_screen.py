import time
from importlib import import_module

from kivy.clock import Clock
from kivy.lang import Builder
from kivy.uix.screenmanager import Screen

from app.screens.utils.additional import BaseScreen
from app.screens.utils.custom_logging import get_logger

logger = get_logger(__name__)


class LoadingScreen(Screen, BaseScreen):
    SCREEN_SPECS = (
        ("load_login", "app/ui/login.kv", "app.screens.view.login_screen", "LoginScreen", "login"),
        ("load_main", "app/ui/main.kv", "app.screens.view.main_screen", "MainScreen", "main"),
        (
            "load_imageview",
            "app/ui/imageview.kv",
            "app.screens.view.image_screen",
            "ImageViewScreen",
            "imageview",
        ),
        ("load_dbview", "app/ui/dbview.kv", "app.screens.view.db_screen", "DbViewScreen", "dbview"),
        ("load_mlview", "app/ui/mlview.kv", "app.screens.view.ml_screen", "MLViewScreen", "mlview"),
        (
            "load_settings",
            "app/ui/settingsview.kv",
            "app.screens.view.settings_screen",
            "SettingsViewScreen",
            "settingsview",
        ),
        (
            "load_detection",
            "app/ui/detectionview.kv",
            "app.screens.view.detection_screen",
            "DetectionScreen",
            "detectionview",
        ),
    )

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.modules = [getattr(self, spec[0]) for spec in self.SCREEN_SPECS]

        self.steps = 20
        self.ids.pbar.value = 0
        self.ids.pbar.max = len(self.modules) * self.steps
        self.init_time = time.time()

        self.fake_progress = 0
        self.progress_event = None
        self.current_module_idx = 0

        self.module_loaded = False

    def on_enter(self, *args):
        Builder.load_file("app/ui/app.kv")
        self.start_next_module()

    def start_next_module(self, *_):
        if self.current_module_idx >= len(self.modules):
            total_time = time.time() - self.init_time
            logger.info(f"All modules loaded in {total_time:.2f} seconds.")
            self.ids.status.text = "Loading complete"
            Clock.schedule_once(self.next_screen, 0.1)
            return

        self.fake_progress = 0
        self.module_loaded = False

        self.progress_event = Clock.schedule_interval(self.smooth_fake_progress, 0.1)

        # Launch the module loader after a tiny delay (to let UI update)
        Clock.schedule_once(self.modules[self.current_module_idx], 0.1)

    def smooth_fake_progress(self, _):
        if self.fake_progress < 15:
            self.ids.pbar.value += 1
            self.fake_progress += 1

        if self.fake_progress >= 15 and self.module_loaded:
            self.finish_progress()

    def increment_pbar(self):
        self.module_loaded = True
        if self.fake_progress < 15:
            self.ids.pbar.value += 15 - self.fake_progress
            self.fake_progress = 15
        self.finish_progress()

    def finish_progress(self):
        if self.progress_event:
            self.progress_event.cancel()
        self.ids.pbar.value += self.steps - self.fake_progress
        self.current_module_idx += 1
        Clock.schedule_once(self.start_next_module, 0.1)

    def next_screen(self, *_):
        logger.debug("Loading complete, preparing to switch to login screen")
        if self.manager.current != self.name:  # not on LoadingScreen anymore
            logger.debug("Already switched screens, aborting")
            return
        logger.debug("Transitioning to login screen")
        self.manager.transition.direction = "left"
        self.manager.current = "login"

    def _load_screen(self, spec):
        _, kv_path, module_path, class_name, screen_name = spec
        screen_class = getattr(import_module(module_path), class_name)
        Builder.load_file(kv_path)
        self.manager.add_widget(screen_class(name=screen_name))
        self.ids.status.text = f"{screen_name} loaded"
        self.increment_pbar()

    def load_login(self, _):
        self._load_screen(self.SCREEN_SPECS[0])

    def load_main(self, _):
        self._load_screen(self.SCREEN_SPECS[1])

    def load_imageview(self, _):
        self._load_screen(self.SCREEN_SPECS[2])

    def load_dbview(self, _):
        self._load_screen(self.SCREEN_SPECS[3])

    def load_mlview(self, _):
        self._load_screen(self.SCREEN_SPECS[4])

    def load_settings(self, _):
        self._load_screen(self.SCREEN_SPECS[5])

    def load_detection(self, _):
        self._load_screen(self.SCREEN_SPECS[6])
