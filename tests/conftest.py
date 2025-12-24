import sys
import pytest
from pathlib import Path
from kivy.config import Config

Config.set('graphics', 'width', '400')
Config.set('graphics', 'height', '600')
Config.set('graphics', 'window_state', 'hidden')
Config.set('kivy', 'exit_on_escape', '0')

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from kivy.lang import Builder
from kivy.base import EventLoop
from kivy.clock import Clock


@pytest.fixture(scope='session', autouse=True)
def setup_kivy():
    EventLoop.ensure_window()

    base_path = Path(__file__).parent.parent / "app" / "ui"
    Builder.load_file(str(base_path / "loading.kv"))

    yield

    EventLoop.close()


@pytest.fixture(scope='session')
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

@pytest.fixture(scope="session", autouse=True)
def test_env(monkeypatch):
    monkeypatch.setenv("APP_ENV", "test")
    monkeypatch.setenv("APP_AUTOLOGIN", "0")
