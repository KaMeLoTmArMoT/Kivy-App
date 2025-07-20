import time
from functools import wraps

from kivy.clock import Clock
from kivy.lang import Builder
from kivy.uix.screenmanager import Screen

from screens.additional import BaseScreen
from screens.custom_logging import get_logger

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
        self.ids.pbar.value = 0
        self.ids.pbar.max = 7 * 20
        self.init_time = time.time()

        self.fake_progress = 0
        self.progress_event = None
        self.current_module_idx = 0
        self.modules = []
        self.module_loaded = False

    def on_enter(self, *args):
        Builder.load_file("ui/app.kv")

        self.modules = [
            self.load_login,
            self.load_main,
            self.load_imageview,
            self.load_dbview,
            self.load_mlview,
            self.load_settings,
            self.load_detection,
        ]

        self.start_next_module()

    def start_next_module(self, *_):
        if self.current_module_idx >= len(self.modules):
            logger.info("All modules loaded.")
            self.ids.status.text = "Loading complete"
            Clock.schedule_once(self.next_screen, 0.5)
            return

        self.fake_progress = 0
        self.module_loaded = False

        self.progress_event = Clock.schedule_interval(self.smooth_fake_progress, 0.2)

        # Launch the module loader after a tiny delay (to let UI update)
        Clock.schedule_once(self.modules[self.current_module_idx], 0.2)

    def smooth_fake_progress(self, _):
        if self.fake_progress < 15:
            self.ids.pbar.value += 1
            self.fake_progress += 1
        elif self.module_loaded:
            self.finish_progress()

    def increment_pbar(self):
        self.module_loaded = True
        if self.fake_progress >= 15:
            self.finish_progress()

    def finish_progress(self):
        if self.progress_event:
            self.progress_event.cancel()
        self.ids.pbar.value += (20 - self.fake_progress)
        self.current_module_idx += 1
        Clock.schedule_once(self.start_next_module, 0.2)

    def next_screen(self, *_):
        self.manager.transition.direction = "left"
        self.manager.current = "login"

    @log_exec_time
    def load_login(self, _):
        from screens.login_screen import LoginScreen

        Builder.load_file("ui/login.kv")
        self.manager.add_widget(LoginScreen(name="login"))

        self.ids.status.text = "login loaded"
        self.increment_pbar()

    @log_exec_time
    def load_main(self, _):
        from screens.main_screen import MainScreen

        Builder.load_file("ui/main.kv")
        self.manager.add_widget(MainScreen(name="main"))

        self.ids.status.text = "main loaded"
        self.increment_pbar()

    @log_exec_time
    def load_imageview(self, _):
        from screens.imageview_screen import ImageViewScreen

        Builder.load_file("ui/imageview.kv")
        self.manager.add_widget(ImageViewScreen(name="imageview"))

        self.ids.status.text = "imageview loaded"
        self.increment_pbar()

    @log_exec_time
    def load_dbview(self, _):
        from screens.dbview_csreen import DbViewScreen

        Builder.load_file("ui/dbview.kv")
        self.manager.add_widget(DbViewScreen(name="dbview"))

        self.ids.status.text = "dbview loaded"
        self.increment_pbar()

    @log_exec_time
    def load_mlview(self, _):
        from screens.mlview_csreen import MLViewScreen

        Builder.load_file("ui/mlview.kv")
        self.manager.add_widget(MLViewScreen(name="mlview"))

        self.ids.status.text = "mlview loaded"
        self.increment_pbar()

    @log_exec_time
    def load_settings(self, _):
        from screens.settings_screen import SettingsViewScreen

        Builder.load_file("ui/settingsview.kv")
        self.manager.add_widget(SettingsViewScreen(name="settingsview"))

        self.ids.status.text = "settingsview loaded"
        self.increment_pbar()

    @log_exec_time
    def load_detection(self, _):
        from screens.detection_screen import DetectionScreen

        Builder.load_file("ui/detectionview.kv")
        self.manager.add_widget(DetectionScreen(name="detectionview"))

        self.ids.status.text = "detectionview loaded"
        self.increment_pbar()
