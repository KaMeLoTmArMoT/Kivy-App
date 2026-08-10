import os
import sys
from hashlib import sha256
from os.path import join
from sqlite3 import connect

from app.screens.utils.custom_logging import get_logger

logger = get_logger(__name__)


def get_db_path() -> str:
    env_path = os.getenv("APP_DB_PATH")
    if env_path:
        return env_path

    if hasattr(sys, "_MEIPASS"):
        return join(sys._MEIPASS, "app.db")

    return "app.db"


def call_db(call, data=None):
    with connect(get_db_path()) as conn:
        cursor = conn.cursor()
        cursor.execute(call, data or ())
        return cursor.fetchall()


def call_db_many(call, rows):
    with connect(get_db_path()) as conn:
        conn.executemany(call, rows)


def get_sha(text):
    enc = sha256()
    enc.update(text.encode("utf-8"))
    return enc.hexdigest()


def extend_key(text) -> bytes:
    text = "" if text is None else str(text)

    if not text:
        raise ValueError("Empty key cannot be extended")

    padded = (text * ((16 // len(text)) + 1))[:16]
    return padded.encode("utf-8")


def get_system_type():
    if sys.platform == "win32":
        return "Windows"
    elif sys.platform == "darwin":
        return "MacOS"
    elif sys.platform == "linux":
        return "Linux"
    else:
        return "Unknown"
