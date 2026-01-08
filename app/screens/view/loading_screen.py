import time
from functools import wraps

from kivy.clock import Clock
from kivy.lang import Builder
from kivy.uix.screenmanager import Screen

from app.screens.utils.additional import BaseScreen
from app.screens.utils.custom_logging import get_logger

logger = get_logger(__name__)


def log_exec_time(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        exec_time = end_time - start_time
        logger.debug(f"Function {func.__name__} executed in {exec_time:.2f} seconds")
        return result

    return wrapper


class LoadingScreen(Screen, BaseScreen):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.modules = [
            self.load_login,
            self.load_main,
            self.load_imageview,
            self.load_dbview,
            self.load_mlview,
            self.load_settings,
            self.load_detection,
        ]

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

    @log_exec_time
    def load_login(self, _):
        from app.screens.view.login_screen import LoginScreen

        Builder.load_file("app/ui/login.kv")
        self.manager.add_widget(LoginScreen(name="login"))

        self.ids.status.text = "login loaded"
        self.increment_pbar()

    @log_exec_time
    def load_main(self, _):
        from app.screens.view.main_screen import MainScreen

        Builder.load_file("app/ui/main.kv")
        self.manager.add_widget(MainScreen(name="main"))

        self.ids.status.text = "main loaded"
        self.increment_pbar()

    @log_exec_time
    def load_imageview(self, _):
        from app.screens.view.image_screen import ImageViewScreen

        Builder.load_file("app/ui/imageview.kv")
        self.manager.add_widget(ImageViewScreen(name="imageview"))

        self.ids.status.text = "imageview loaded"
        self.increment_pbar()

    @log_exec_time
    def load_dbview(self, _):
        from app.screens.view.db_screen import DbViewScreen

        Builder.load_file("app/ui/dbview.kv")
        self.manager.add_widget(DbViewScreen(name="dbview"))

        self.ids.status.text = "dbview loaded"
        self.increment_pbar()

    @log_exec_time
    def load_mlview(self, _):
        from app.screens.view.ml_screen import MLViewScreen

        Builder.load_file("app/ui/mlview.kv")
        self.manager.add_widget(MLViewScreen(name="mlview"))

        self.ids.status.text = "mlview loaded"
        self.increment_pbar()

    @log_exec_time
    def load_settings(self, _):
        from app.screens.view.settings_screen import SettingsViewScreen

        Builder.load_file("app/ui/settingsview.kv")
        self.manager.add_widget(SettingsViewScreen(name="settingsview"))

        self.ids.status.text = "settingsview loaded"
        self.increment_pbar()

    @log_exec_time
    def load_detection(self, _):
        from app.screens.view.detection_screen import DetectionScreen

        Builder.load_file("app/ui/detectionview.kv")
        self.manager.add_widget(DetectionScreen(name="detectionview"))

        self.ids.status.text = "detectionview loaded"
        self.increment_pbar()
