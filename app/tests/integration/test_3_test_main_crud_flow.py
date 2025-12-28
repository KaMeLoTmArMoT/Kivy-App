import base64
import time

import pytest
from Cryptodome.Cipher import AES
from kivy.clock import Clock


def drain(frames: int = 5) -> None:
    for _ in range(frames):
        Clock.tick()


def wait_until(
    predicate,
    *,
    timeout: float = 5.0,
    step_frames: int = 2,
    msg: str = "Condition not met",
) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        drain(step_frames)
        if predicate():
            return
    raise AssertionError(msg)


def decrypt_main_record(main_screen, row) -> str:
    cipher = AES.new(main_screen.key, AES.MODE_EAX, nonce=b"TODO")
    raw = base64.b64decode(row[0].encode("utf-8"))
    return cipher.decrypt(raw).decode("utf-8")


def ui_texts(main_screen) -> list[str]:
    grid = main_screen.ids.scroll.children[0]
    return [btn.text for btn in grid.children]


def select_by_text(main_screen, text: str) -> None:
    grid = main_screen.ids.scroll.children[0]
    for btn in grid.children:
        if btn.text == text:
            main_screen.select_label_btn(btn)
            drain()
            return
    raise AssertionError(f"UI element with text={text!r} not found")


@pytest.mark.integration
class TestMainCrudFlow:
    def test_create_update_delete_flow(self, kivy_app, reset_login_state):
        sm = kivy_app.root
        login = reset_login_state

        # Let loading screen finish its module-loading work (your current behavior).
        # Keep it, but remove fixed sleeps where possible.
        wait_until(
            lambda: sm.has_screen("main"),
            timeout=15,
            msg="Main screen was not loaded/added",
        )

        # Login (as you already do).
        test_password = "test_dev_pass_123"
        login.ids.word_input.text = test_password
        drain()
        login.submit()
        drain()

        wait_until(
            lambda: sm.current == "main",
            timeout=5,
            msg="Did not navigate to main after login",
        )

        main = sm.get_screen("main")
        sm.current = "main"
        drain()

        # Critical sync point: wait for on_enter to finish (your on_enter_done flag).
        wait_until(
            lambda: getattr(main, "on_enter_done", False),
            timeout=10,
            msg="MainScreen.on_enter not finished",
        )

        text1 = "hello integration"
        text2 = "hello updated"

        # --- CREATE ---
        main.ids.word_input.text = text1
        drain()
        main.submit()
        drain()

        assert text1 in ui_texts(main)
        decrypted = [decrypt_main_record(main, r) for r in main.db.get_customers()]
        assert text1 in decrypted

        # --- UPDATE ---
        select_by_text(main, text1)
        main.ids.word_input.text = text2
        drain()
        main.update_record()
        drain()

        assert text1 not in ui_texts(main)
        assert text2 in ui_texts(main)
        decrypted = [decrypt_main_record(main, r) for r in main.db.get_customers()]
        assert text1 not in decrypted
        assert text2 in decrypted

        # --- DELETE ---
        select_by_text(main, text2)
        main.delete_record()
        drain()

        assert text2 not in ui_texts(main)
        decrypted = [decrypt_main_record(main, r) for r in main.db.get_customers()]
        assert text2 not in decrypted
