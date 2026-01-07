import faulthandler
import os
import sys
from pathlib import Path

import pytest

from app.screens.utils.db import DB

os.environ.setdefault("KCFG_KIVY_LOG_LEVEL", "warning")

from kivy.base import EventLoop  # noqa: E402
from kivy.clock import Clock  # noqa: E402
from kivy.config import Config  # noqa: E402
from kivy.lang import Builder  # noqa: E402

from app.screens.utils.utils import call_db, get_sha  # noqa: E402

Config.set("graphics", "width", "400")
Config.set("graphics", "height", "600")
Config.set("graphics", "window_state", "hidden")
Config.set("kivy", "exit_on_escape", "0")

DB_PATH = Path("app_test.db").resolve()


@pytest.fixture(scope="session", autouse=True)
def setup_kivy():
    EventLoop.ensure_window()

    base_path = Path(__file__).parent.parent / "ui"
    Builder.load_file(str(base_path / "loading.kv"))

    yield

    EventLoop.close()


@pytest.fixture(scope="module")
def kivy_app(setup_kivy):
    from main import MainApp

    app = MainApp()
    app.root = app.build()

    Clock.tick()

    yield app

    app.stop()


@pytest.fixture
def advance_clock():
    def _advance(frames=1):
        for _ in range(frames):
            Clock.tick()

    return _advance


@pytest.fixture(autouse=True)
def test_env(monkeypatch):
    monkeypatch.setenv("APP_ENV", "test")
    monkeypatch.setenv("APP_AUTOLOGIN", "0")


@pytest.fixture(autouse=True)
def dump_stacks_on_hang():
    out = sys.__stderr__
    faulthandler.dump_traceback_later(20, repeat=False, file=out)
    yield
    faulthandler.cancel_dump_traceback_later()


@pytest.fixture
def reset_login_state(kivy_app):
    """Reset login screen to clean state before each test"""
    # print(f"11111111111111 --- reset_login_state {kivy_app.root.current=}")
    sm = kivy_app.root

    if sm.current != "login":
        sm.current = "login"
        Clock.tick()

    login_screen = sm.get_screen("login")
    Clock.tick()

    login_screen.ids.word_input.text = ""
    login_screen.key = ""

    call_db("DELETE FROM passwords WHERE destination='login_test'")

    test_password = "test_dev_pass_123"
    test_password_hash = get_sha(test_password)
    call_db(f"INSERT INTO passwords VALUES ('login_test', '{test_password_hash}')")

    login_screen.passwords = login_screen.db.get_login_password("login_test")

    Clock.tick()

    yield login_screen

    call_db("DELETE FROM passwords WHERE destination='login_test'")


@pytest.fixture(scope="session", autouse=True)
def test_db_session():
    os.environ["APP_DB_PATH"] = str(DB_PATH)

    DB_PATH.unlink(missing_ok=True)
    DB().create_db_and_check()

    yield

    DB_PATH.unlink(missing_ok=True)
