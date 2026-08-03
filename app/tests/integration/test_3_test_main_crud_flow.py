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
    def test_create_update_delete_flow(self, kivy_app, request):
        sm = kivy_app.root

        wait_until(
            lambda: sm.has_screen("main"),
            timeout=15,
            msg="Main screen was not loaded/added",
        )

        main = sm.get_screen("main")

        need_login = (sm.current != "main") or (not main.key)
        # print(f"11111111111111 {sm.current=}, {need_login=}")

        if need_login:
            if not sm.has_screen("login"):
                wait_until(
                    lambda: sm.has_screen("login"),
                    timeout=10,
                    msg="Login screen missing",
                )

            login = request.getfixturevalue("reset_login_state")

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

        sm.current = "main"
        drain()

        wait_until(
            lambda: getattr(main, "on_enter_done", False) and getattr(main, "key", None),
            timeout=10,
            msg="MainScreen not ready (on_enter_done/key missing)",
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
