import pytest
from kivy.clock import Clock


class BaseScreenTest:
    """Base class for common test functionality"""

    @staticmethod
    def wait_for_loading(kivy_app=None, screen_name=None, timeout=15.0):
        if kivy_app is None:
            for _ in range(2):
                Clock.tick()
            return
        start = Clock.time()
        while Clock.time() - start < timeout:
            if screen_name and kivy_app.root.has_screen(screen_name):
                return
            if not screen_name and len(kivy_app.root.screens) >= 7:
                return
            Clock.tick()


class TestAppInitialization:
    def test_app_builds(self, kivy_app):
        assert kivy_app is not None
        assert kivy_app.root is not None

    def test_screen_manager_type(self, kivy_app):
        from kivy.uix.screenmanager import ScreenManager

        assert isinstance(kivy_app.root, ScreenManager)

    def test_theme_configured(self, kivy_app):
        assert kivy_app.theme_cls.theme_style == "Dark"
        assert kivy_app.theme_cls.primary_palette == "#607D8B"


class TestLoadingScreenComponents:
    REQUIRED_IDS = ["pbar", "status"]
    MODULE_METHODS = [
        "load_login",
        "load_main",
        "load_imageview",
        "load_dbview",
        "load_mlview",
        "load_settings",
        "load_detection",
    ]

    def test_loading_screen_exists(self, kivy_app):
        from app.screens.view.loading_screen import LoadingScreen

        loading_screen = kivy_app.root.get_screen("loading")
        assert isinstance(loading_screen, LoadingScreen)

    @pytest.mark.parametrize("widget_id", REQUIRED_IDS)
    def test_loading_screen_has_required_widgets(self, kivy_app, widget_id):
        loading_screen = kivy_app.root.get_screen("loading")

        assert widget_id in loading_screen.ids
        assert loading_screen.ids[widget_id] is not None

    def test_progress_bar_initial_state(self, kivy_app):
        loading_screen = kivy_app.root.get_screen("loading")

        assert isinstance(loading_screen.ids.pbar.value, (int, float))
        expected_max = len(loading_screen.modules) * loading_screen.steps
        assert loading_screen.ids.pbar.max == expected_max

    def test_loading_screen_has_correct_module_count(self, kivy_app):
        loading_screen = kivy_app.root.get_screen("loading")
        assert len(loading_screen.modules) == len(self.MODULE_METHODS)

    @pytest.mark.parametrize("method_name", MODULE_METHODS)
    def test_module_loader_exists_and_callable(self, kivy_app, method_name):
        loading_screen = kivy_app.root.get_screen("loading")

        assert hasattr(loading_screen, method_name)
        assert callable(getattr(loading_screen, method_name))


class TestScreenLoading(BaseScreenTest):
    SCREEN_CONFIGS = [
        ("main", "app.screens.view.main_screen", "MainScreen"),
        ("imageview", "app.screens.view.image_screen", "ImageViewScreen"),
        ("dbview", "app.screens.view.db_screen", "DbViewScreen"),
        ("mlview", "app.screens.view.ml_screen", "MLViewScreen"),
        ("settingsview", "app.screens.view.settings_screen", "SettingsViewScreen"),
        ("detectionview", "app.screens.view.detection_screen", "DetectionScreen"),
    ]

    @pytest.mark.parametrize("screen_name,module_path,class_name", SCREEN_CONFIGS)
    def test_screen_loads(self, kivy_app, screen_name, module_path, class_name):
        self.wait_for_loading(kivy_app, screen_name)

        sm = kivy_app.root

        if not sm.has_screen(screen_name):
            loaded = [s.name for s in sm.screens]
            pytest.fail(
                f"Screen '{screen_name}' not loaded. "
                f"Loaded screens: {loaded}. "
                f"Check if loading crashed."
            )

    @pytest.mark.parametrize("screen_name,module_path,class_name", SCREEN_CONFIGS)
    def test_screen_instance_type(self, kivy_app, screen_name, module_path, class_name):
        from importlib import import_module

        self.wait_for_loading(kivy_app, screen_name)

        sm = kivy_app.root

        if not sm.has_screen(screen_name):
            pytest.skip(f"Screen '{screen_name}' was not loaded (previous error)")

        module = import_module(module_path)
        expected_class = getattr(module, class_name)

        screen = sm.get_screen(screen_name)
        assert isinstance(screen, expected_class)

    def test_all_screens_loaded(self, kivy_app):
        self.wait_for_loading(kivy_app)

        sm = kivy_app.root
        expected_screens = ["loading"] + [cfg[0] for cfg in self.SCREEN_CONFIGS]

        loaded_screens = [s.name for s in sm.screens]
        missing = [s for s in expected_screens if s not in loaded_screens]

        if missing:
            pytest.fail(
                f"Missing screens: {missing}. "
                f"Loaded: {loaded_screens}. "
                f"Check LoadingScreen for errors."
            )


class TestScreenTransitions(BaseScreenTest):
    NAVIGABLE_SCREENS = [
        "main",
        "imageview",
        "dbview",
        "mlview",
        "settingsview",
        "detectionview",
    ]

    @pytest.mark.parametrize("screen_name", NAVIGABLE_SCREENS)
    def test_can_navigate_to_screen(self, kivy_app, screen_name):
        self.wait_for_loading(kivy_app, screen_name)

        sm = kivy_app.root
        sm.current = screen_name
        Clock.tick()

        assert sm.current == screen_name

    def test_navigation_round_trip(self, kivy_app):
        self.wait_for_loading(kivy_app)

        sm = kivy_app.root

        sm.current = "detectionview"
        Clock.tick()
        assert sm.current == "detectionview"

        sm.current = "main"
        Clock.tick()
        assert sm.current == "main"

    def test_transition_direction(self, kivy_app):
        self.wait_for_loading(kivy_app)

        sm = kivy_app.root
        assert sm.transition.direction == "left"


class TestScreenNavigationFlow(BaseScreenTest):
    """Test complete navigation flows"""

    @pytest.mark.parametrize(
        "path",
        [
            ["main", "imageview", "main"],
            ["main", "dbview", "settingsview", "main"],
            ["main", "mlview", "detectionview", "main"],
        ],
    )
    def test_navigation_path(self, kivy_app, path):
        self.wait_for_loading(kivy_app)

        sm = kivy_app.root

        for screen_name in path:
            sm.current = screen_name
            Clock.tick()
            assert sm.current == screen_name, f"Failed to navigate to {screen_name}"
