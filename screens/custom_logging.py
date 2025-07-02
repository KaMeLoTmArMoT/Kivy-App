import datetime
import logging


class CustomFormatter(logging.Formatter):
    """Colored log formatter for different log levels."""

    grey = "\x1b[37m"
    white = "\x1b[97m"
    blue = "\x1b[38;5;39m"
    yellow = "\x1b[38;5;226m"
    red = "\x1b[38;5;196m"
    bold_red = "\x1b[31;1m"
    reset = "\x1b[0m"

    def __init__(self, fmt):
        super().__init__()
        self.fmt = fmt
        self.FORMATS = {
            logging.DEBUG: self.reset + self.grey + self.fmt + self.reset,
            logging.INFO: self.reset + self.blue + self.fmt + self.reset,
            logging.WARNING: self.reset + self.yellow + self.fmt + self.reset,
            logging.ERROR: self.reset + self.red + self.fmt + self.reset,
            logging.CRITICAL: self.reset + self.bold_red + self.fmt + self.reset,
            logging.NOTSET: self.white + self.fmt + self.reset,
        }

    def format(self, record: logging.LogRecord) -> str:
        log_fmt = self.FORMATS.get(record.levelno)
        formatter = logging.Formatter(log_fmt)
        return formatter.format(record)


def get_logger(name: str = None) -> logging.Logger:
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)

    if logger.hasHandlers():
        return logger  # Prevent duplicate handlers on re-import

    fmt = "%(asctime)s | %(levelname)8s | %(name)s | %(message)s"

    # Console handler with colors
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.DEBUG)
    console_handler.setFormatter(CustomFormatter(fmt))
    logger.addHandler(console_handler)

    # File handler
    today = datetime.date.today()
    file_handler = logging.FileHandler(f"my_app_{today:%Y_%m_%d}.log")
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(logging.Formatter(fmt))
    logger.addHandler(file_handler)

    return logger
