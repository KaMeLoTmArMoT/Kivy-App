import time
from functools import wraps
from threading import Thread

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
        logger.debug(f"Function {func.__name__} executed in {exec_time:.4f} seconds")
        return result

    return wrapper


class LoadingScreen(Screen, BaseScreen):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.loader: Thread = Thread(target=self.load_all)
        self.ids.pbar.value = 0
        self.ids.pbar.max = 7
        self.init_time = time.time()

    def on_enter(self, *args):
        self.loader.start()

    def increment_pbar(self):
        self.ids.pbar.value += 1

    def load_all(self):
        Builder.load_file("ui/app.kv")

        modules = [
            self.load_login,
            self.load_main,
            self.load_imageview,
            self.load_dbview,
            self.load_mlview,
            self.load_settings,
            self.load_detection,
        ]

        for module in modules:
            time.sleep(0.01)
            Clock.schedule_once(module)

    @log_exec_time
    def load_login(self, tm):
        from screens.login_screen import LoginScreen

        Builder.load_file("ui/login.kv")
        self.manager.add_widget(LoginScreen(name="login"))

        self.ids.status.text = "login loaded"
        logger.debug("login done")
        self.increment_pbar()

    @log_exec_time
    def load_main(self, tm):
        from screens.main_screen import MainScreen

        Builder.load_file("ui/main.kv")
        self.manager.add_widget(MainScreen(name="main"))

        self.ids.status.text = "main loaded"
        logger.debug("main done")
        self.increment_pbar()

    @log_exec_time
    def load_imageview(self, tm):
        from screens.imageview_screen import ImageViewScreen

        Builder.load_file("ui/imageview.kv")
        self.manager.add_widget(ImageViewScreen(name="imageview"))

        self.ids.status.text = "imageview loaded"
        logger.debug("imageview done")
        self.increment_pbar()

    @log_exec_time
    def load_dbview(self, tm):
        from screens.dbview_csreen import DbViewScreen

        Builder.load_file("ui/dbview.kv")
        self.manager.add_widget(DbViewScreen(name="dbview"))

        self.ids.status.text = "dbview loaded"
        logger.debug("dbview done")
        self.increment_pbar()

    @log_exec_time
    def load_mlview(self, tm):
        from screens.mlview_csreen import MLViewScreen

        Builder.load_file("ui/mlview.kv")
        self.manager.add_widget(MLViewScreen(name="mlview"))

        self.ids.status.text = "mlview loaded"
        logger.debug("mlview done")
        self.increment_pbar()

    @log_exec_time
    def load_settings(self, tm):
        from screens.settings_screen import SettingsViewScreen

        Builder.load_file("ui/settingsview.kv")
        self.manager.add_widget(SettingsViewScreen(name="settingsview"))

        self.ids.status.text = "settingsview loaded"
        logger.debug("settingsview done")
        self.increment_pbar()

    @log_exec_time
    def load_detection(self, tm):
        from screens.detection_screen import DetectionScreen

        Builder.load_file("ui/detectionview.kv")
        self.manager.add_widget(DetectionScreen(name="detectionview"))

        self.ids.status.text = "detectionview loaded"
        logger.debug("detectionview done")
        self.increment_pbar()

        Clock.schedule_once(self.next_screen, 0.2)

    def next_screen(self, tm):
        logger.info(f"Loading complete in {round(time.time() - self.init_time, 2)}")
        self.manager.transition.direction = "left"
        self.manager.current = "login"
