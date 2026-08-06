import os
import platform
import shutil
import webbrowser

from app.screens.utils.custom_logging import get_logger
from app.screens.utils.db import DB

logger = get_logger(__name__)


class TBServer:
    def __init__(self):
        self.tb = None
        self.url = None

    def launch_tensorboard(self, tb_folder):
        from tensorboard import program

        if not os.path.isdir(tb_folder):
            return "No tensorboard folder"

        if len(os.listdir(tb_folder)) == 0:
            return "No data to show TB!"

        if self.tb is None:
            self.tb = program.TensorBoard()
            self.tb.configure(argv=[None, "--logdir", tb_folder])
            self.url = self.tb.launch()
            logger.debug(f"{self.url=}")

        system_platform = platform.system()
        chrome_path = DB().get_config_typed("chrome_path")

        if system_platform == "Windows":
            webbrowser.get(chrome_path).open(self.url)
        elif system_platform == "Linux":
            for browser in ["google-chrome", "chromium", "xdg-open"]:
                if shutil.which(browser):
                    webbrowser.get(browser).open(self.url)
                    break
            else:
                logger.error("No known browser found. Please install chrome or use xdg-open.")
        else:
            logger.error("Unknown operating system.")
            webbrowser.open(self.url)

        return "Tensorboard OK"
