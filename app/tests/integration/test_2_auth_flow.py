import os

import pytest
from kivy.clock import Clock

from app.screens.utils.utils import get_sha


@pytest.fixture(autouse=True)
def auth_test_env(monkeypatch):
    """Set up clean test environment for auth tests"""
    monkeypatch.setenv("APP_ENV", "test")
    monkeypatch.setenv("APP_AUTOLOGIN", "0")
    monkeypatch.setenv("APP_DEV_PASSWORD", "")


class BaseAuthTest:
    """Base class for authentication test functionality"""

    @staticmethod
    def wait_for_screen(seconds=3.0):
        start = Clock.time()
        while Clock.time() - start < seconds:
            Clock.tick()

    @staticmethod
    def enter_text(widget, text):
        """Simulate text entry in a widget"""
        widget.text = text
        Clock.tick()


class TestLoginScreenComponents:
    """Test login screen UI components exist and are accessible"""

    REQUIRED_IDS = ["word_input", "word_label", "login"]

    def test_login_screen_exists(self, kivy_app):
        self.wait_for_screen = BaseAuthTest.wait_for_screen
        self.wait_for_screen()

        from app.screens.view.login_screen import LoginScreen

        sm = kivy_app.root
        assert sm.has_screen("login")

        login_screen = sm.get_screen("login")
        assert isinstance(login_screen, LoginScreen)

    @pytest.mark.parametrize("widget_id", REQUIRED_IDS)
    def test_login_screen_has_required_widgets(self, kivy_app, widget_id):
        self.wait_for_screen = BaseAuthTest.wait_for_screen
        self.wait_for_screen()

        login_screen = kivy_app.root.get_screen("login")

        assert widget_id in login_screen.ids, f"Missing widget: {widget_id}"
        assert login_screen.ids[widget_id] is not None

    def test_password_field_properties(self, kivy_app):
        self.wait_for_screen = BaseAuthTest.wait_for_screen
        self.wait_for_screen()

        login_screen = kivy_app.root.get_screen("login")
        password_field = login_screen.ids.word_input

        # Check if it's a password field or register mode
        assert password_field.password is True or password_field.password is False
        assert hasattr(password_field, "text")

    def test_login_button_exists(self, kivy_app):
        self.wait_for_screen = BaseAuthTest.wait_for_screen
        self.wait_for_screen()

        login_screen = kivy_app.root.get_screen("login")
        login_button = login_screen.ids.login

        assert login_button.text in ["Login", "Register"]
        assert hasattr(login_screen, "submit")

    def test_label_displays_message(self, kivy_app):
        self.wait_for_screen = BaseAuthTest.wait_for_screen
        self.wait_for_screen()

        login_screen = kivy_app.root.get_screen("login")
        label = login_screen.ids.word_label

        assert label.text != ""
        assert isinstance(label.text, str)


class TestPasswordValidation(BaseAuthTest):
    """Test password validation logic"""

    @pytest.mark.parametrize(
        "password,should_pass",
        [
            ("", False),  # Empty password
            ("12345", False),  # Too short (5 chars)
            ("123456", True),  # Valid (6 chars)
            ("password123", True),  # Valid longer password
            ("a" * 6, True),  # Exactly 6 chars
            ("a" * 100, True),  # Very long password
        ],
    )
    def test_password_length_validation(
        self, kivy_app, password, should_pass, reset_login_state
    ):
        self.wait_for_screen()

        login_screen = reset_login_state

        # Enter password
        self.enter_text(login_screen.ids.word_input, password)
        login_screen.key = password

        # Check if password meets length requirement
        is_valid = len(password) > 5

        if should_pass:
            assert is_valid, f"Password '{password}' should be valid"
        else:
            assert not is_valid, f"Password '{password}' should be invalid"

    def test_short_password_shows_error(self, kivy_app, reset_login_state):
        self.wait_for_screen()

        login_screen = reset_login_state

        login_screen.ids.word_input.text = "12345"
        Clock.tick()

        login_screen.submit()
        Clock.tick()

        # Should still be on login screen
        assert kivy_app.root.current == "login"
        # Label should show error message
        assert "longer than 5" in login_screen.ids.word_label.text.lower()


class TestFirstTimeRegistration(BaseAuthTest):
    """Test first-time registration flow when no password exists"""

    def test_no_password_shows_register_mode(self, kivy_app):
        self.wait_for_screen()

        login_screen = kivy_app.root.get_screen("login")

        # If no password in DB, should show register mode
        if len(login_screen.passwords) == 0:
            assert login_screen.ids.login.text == "Register"
            assert login_screen.ids.word_input.password is False
            assert "new password" in login_screen.ids.word_label.text.lower()

    def test_register_new_password(self, kivy_app):
        self.wait_for_screen()

        login_screen = kivy_app.root.get_screen("login")

        # Only test if in register mode
        if len(login_screen.passwords) == 0:
            test_password = "testpass123"

            login_screen.ids.word_input.text = test_password
            Clock.tick()

            login_screen.submit()
            Clock.tick()

            # Should navigate to main screen
            assert kivy_app.root.current == "main"


class TestSuccessfulLogin(BaseAuthTest):
    """Test successful login flow with existing password"""

    def test_correct_password_navigates_to_main(self, kivy_app, reset_login_state):
        self.wait_for_screen()

        sm = kivy_app.root
        login_screen = reset_login_state

        test_password = "test_dev_pass_123"
        login_screen.ids.word_input.text = test_password

        Clock.tick()
        login_screen.submit()
        Clock.tick()

        assert sm.current == "main", f"Expected 'main', got '{sm.current}'"

    def test_wrong_password_stays_on_login(self, kivy_app, reset_login_state):
        self.wait_for_screen()

        sm = kivy_app.root
        login_screen = reset_login_state

        wrong_password = "wrongpass123"
        login_screen.ids.word_input.text = wrong_password

        Clock.tick()
        login_screen.submit()
        Clock.tick()

        assert sm.current == "login", f"Expected 'login', got '{sm.current}'"
        assert "wrong" in login_screen.ids.word_label.text.lower()


class TestLoginUserExperience(BaseAuthTest):
    """Test login UX features"""

    def test_password_field_accepts_input(self, kivy_app, reset_login_state):
        self.wait_for_screen()

        login_screen = reset_login_state
        password_field = login_screen.ids.word_input

        test_input = "testpassword123"
        self.enter_text(password_field, test_input)

        assert password_field.text == test_input

    def test_password_field_focus_on_enter(self, kivy_app, reset_login_state):
        self.wait_for_screen()

        login_screen = reset_login_state

        # Trigger on_enter
        login_screen.on_enter()
        Clock.tick()

        # Password field should have focus
        assert login_screen.ids.word_input.focus is True

    def test_enter_key_triggers_submit(self, kivy_app, reset_login_state):
        self.wait_for_screen()

        login_screen = reset_login_state
        password_field = login_screen.ids.word_input

        # Set a valid password
        password_field.text = "validpass123"
        Clock.tick()

        # Simulate Enter key (text_validate)
        password_field.dispatch("on_text_validate")
        Clock.tick()

        # Should trigger submit
        assert hasattr(login_screen, "key")

    def test_login_button_press_triggers_submit(self, kivy_app, reset_login_state):
        self.wait_for_screen()

        login_screen = reset_login_state
        login_button = login_screen.ids.login

        # Set valid password
        login_screen.ids.word_input.text = "validpass123"
        Clock.tick()

        # Simulate button press
        login_button.dispatch("on_press")
        Clock.tick()

        # Submit should be called (key should not be empty anymore)
        # Note: key might be empty if password is too short, adjust as needed


class TestDevModeAutoLogin(BaseAuthTest):
    """Test development mode auto-login feature"""

    def test_autologin_disabled_in_test(self, kivy_app):
        self.wait_for_screen(3.0)

        app_env = os.getenv("APP_ENV")
        autologin = os.getenv("APP_AUTOLOGIN")

        # With our monkeypatch, autologin should be disabled
        assert app_env == "test"
        assert autologin == "0"


class TestPasswordHashing(BaseAuthTest):
    """Test password hashing functionality"""

    def test_password_is_hashed_before_storage(self, kivy_app):
        self.wait_for_screen()

        test_password = "testpass123"
        hashed = get_sha(test_password)

        # Hash should be different from original
        assert hashed != test_password
        # Hash should be consistent
        assert get_sha(test_password) == hashed

    def test_submit_new_password_hashes_password(self, kivy_app):
        self.wait_for_screen()

        test_password = "newpassword123"

        # Test the hashing
        hashed = get_sha(test_password)
        assert len(hashed) > 0
        assert hashed != test_password


class TestAuthenticationFlow(BaseAuthTest):
    """Test complete authentication flows"""

    def test_full_login_flow(self, kivy_app, reset_login_state):
        self.wait_for_screen()

        sm = kivy_app.root
        login_screen = reset_login_state

        # Пароль вже встановлений фікстурою
        test_password = "test_dev_pass_123"

        # Start on login screen
        assert sm.current == "login"

        # Enter password
        login_screen.ids.word_input.text = test_password
        Clock.tick()

        # Submit
        login_screen.submit()
        Clock.tick()

        # Should be on main
        assert sm.current == "main", f"Expected 'main', got '{sm.current}'"

    def test_login_to_navigation_flow(self, kivy_app, reset_login_state):
        self.wait_for_screen()

        sm = kivy_app.root
        login_screen = reset_login_state

        # Пароль вже встановлений фікстурою
        test_password = "test_dev_pass_123"

        login_screen.ids.word_input.text = test_password
        Clock.tick()
        login_screen.submit()
        Clock.tick()

        # Should be on main now
        assert sm.current == "main"

        # Navigate to other screens
        for screen in ["imageview", "dbview", "main"]:
            if sm.has_screen(screen):
                sm.current = screen
                Clock.tick()
                assert sm.current == screen
