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
    conn = connect(get_db_path())

    # Create cursor
    c = conn.cursor()

    # Execute SQL command
    if data is None:
        c.execute(call)
    else:
        c.execute(call, data)
    records = c.fetchall()

    # Commit changes
    conn.commit()

    # Close connection
    conn.close()

    return records


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
