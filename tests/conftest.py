import faulthandler
import sys
from pathlib import Path

import pytest
from kivy.base import EventLoop
from kivy.clock import Clock
from kivy.config import Config
from kivy.lang import Builder

Config.set("graphics", "width", "400")
Config.set("graphics", "height", "600")
Config.set("graphics", "window_state", "hidden")
Config.set("kivy", "exit_on_escape", "0")

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


@pytest.fixture(scope="session", autouse=True)
def setup_kivy():
    EventLoop.ensure_window()

    base_path = Path(__file__).parent.parent / "app" / "ui"
    Builder.load_file(str(base_path / "loading.kv"))

    yield

    EventLoop.close()


@pytest.fixture(scope="session")
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
