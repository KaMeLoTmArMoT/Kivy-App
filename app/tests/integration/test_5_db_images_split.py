import hashlib
import time
from pathlib import Path

import pytest
from Cryptodome.Cipher import AES
from kivy.clock import Clock

from app.screens.utils.utils import call_db, extend_key

TEST_IMAGES_DIR = Path(
    r"G:\programming\Kivy-App\app\tests\test_data\example_images"
).resolve()


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


def sha256(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def clear_images_table():
    call_db("DELETE FROM images;")
    call_db("DELETE FROM sqlite_sequence WHERE name='images';")


def ensure_logged_in(sm, request):
    if sm.has_screen("main") and getattr(sm.get_screen("main"), "key", None):
        return
    login = request.getfixturevalue("reset_login_state")
    login.ids.word_input.text = "test_dev_pass_123"
    drain()
    login.submit()
    drain()
    wait_until(
        lambda: sm.current == "main", timeout=5, msg="Login did not navigate to main"
    )


def init_dbview_without_on_enter(db, login_key: str):
    db.grid_1 = db.ids.grid_1
    db.grid_2 = db.ids.grid_2
    db.key = extend_key(login_key)


def get_any_image_files(n: int = 4):
    files = sorted(
        [
            p
            for p in TEST_IMAGES_DIR.iterdir()
            if p.suffix.lower() in {".png", ".jpg", ".jpeg"}
        ]
    )
    assert len(files) >= n, f"Need at least {n} images in {TEST_IMAGES_DIR}"
    return files[:n]


def seed_db_2_plain_2_secure(db, key: bytes):
    clear_images_table()

    files = get_any_image_files(4)
    plain_bytes = [files[0].read_bytes(), files[1].read_bytes()]
    secure_plain_bytes = [files[2].read_bytes(), files[3].read_bytes()]

    # pack format: nonce(16) + tag(16) + ciphertext [EAX]
    secure_packed = []
    for b in secure_plain_bytes:
        cipher = AES.new(key, AES.MODE_EAX)
        ciphertext, tag = cipher.encrypt_and_digest(b)
        secure_packed.append(cipher.nonce + tag + ciphertext)

    for b in plain_bytes:
        db.db.insert_image(b)
    for b in secure_packed:
        db.db.insert_image(b)

    return plain_bytes, secure_plain_bytes


@pytest.mark.integration
class TestDbImagesSplit:
    @staticmethod
    def init_test(kivy_app, request):
        sm = kivy_app.root
        wait_until(lambda: sm.has_screen("dbview"), timeout=15, msg="dbview not loaded")
        wait_until(lambda: sm.has_screen("login"), timeout=15, msg="login not loaded")

        ensure_logged_in(sm, request)

        sm.current = "dbview"
        drain()
        db = sm.get_screen("dbview")

        db.autoload_on_enter = False
        if getattr(db, "_autoload_ev", None) is not None:
            db._autoload_ev.cancel()
            db._autoload_ev = None

        init_dbview_without_on_enter(db, sm.get_screen("login").key)

        plain_bytes, secure_plain_bytes = seed_db_2_plain_2_secure(db, db.key)

        expected_plain = {sha256(b) for b in plain_bytes}
        expected_secure = {sha256(b) for b in secure_plain_bytes}
        expected_all = expected_plain | expected_secure

        return db, expected_all, expected_plain, expected_secure

    def test_correct_key_zero_left(self, kivy_app, request):
        db, expected_all, _, _ = self.init_test(kivy_app, request)

        db.show_db_images()
        drain()

        wait_until(
            lambda: (len(db.ids.grid_1.children) + len(db.ids.grid_2.children)) == 4,
            timeout=10,
            msg="Not rendered",
        )

        assert len(db.ids.grid_1.children) == 2
        assert len(db.ids.grid_2.children) == 2
        assert db.ids.simple.text == "Simple images [2]"
        assert db.ids.secure.text == "Secure images [2]"

        matched = db.last_match["matched_simple"] | db.last_match["matched_secure"]
        remaining = expected_all - matched
        assert remaining == set()

    def test_wrong_key_only_secure_left(self, kivy_app, request):
        db, expected_all, expected_plain, expected_secure = self.init_test(
            kivy_app, request
        )

        # wrong key same length
        db.key = b"X" * len(db.key)

        db.show_db_images()
        drain()

        wait_until(
            lambda: (len(db.ids.grid_1.children) + len(db.ids.grid_2.children)) == 4,
            timeout=10,
            msg="Not rendered (wrong key)",
        )

        # with tag verification, secure must be 0
        assert len(db.ids.grid_2.children) == 0
        assert db.ids.secure.text == "Secure images [0]"

        matched = db.last_match["matched_simple"] | db.last_match["matched_secure"]
        remaining = expected_all - matched

        # only secure originals should remain unmatched
        assert remaining == expected_secure
        assert db.last_match["matched_simple"] == expected_plain
        assert db.last_match["matched_secure"] == set()
