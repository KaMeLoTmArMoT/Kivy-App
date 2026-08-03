import os
import time
from pathlib import Path

import pytest
from kivy.clock import Clock

TEST_IMAGES_DIR = Path(__file__).resolve().parent.parent / "test_data" / "example_images"
assert TEST_IMAGES_DIR.exists(), f"Missing test folder: {TEST_IMAGES_DIR}"


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


def grid_floatlayouts(image_screen):
    # image_screen.grid is GridLayout; children are MDFloatLayout
    # Kivy stores children in reverse add order
    return list(reversed(image_screen.grid.children))


def grid_images(image_screen):
    # each MDFloatLayout: children = [checkbox, img] because you add img then checkbox
    # (and Kivy reverses children list)
    imgs = []
    for fl in grid_floatlayouts(image_screen):
        img = fl.children[1]
        imgs.append(img)
    return imgs


def click_images(images):
    for img in images:
        img.dispatch("on_press")  # bound to image_click [via bind(on_press=...)]
        drain(2)


def ensure_imageview_ready(sm, request):
    # wait module loading / screen creation
    wait_until(
        lambda: sm.has_screen("imageview"),
        timeout=15,
        msg="ImageView screen was not loaded/added",
    )

    # enter imageview (triggers on_enter which needs login.key)
    sm.current = "imageview"
    drain()

    scr = sm.get_screen("imageview")

    # If on_enter couldn't set key, login first and re-enter imageview
    if not getattr(scr, "key", ""):
        login = request.getfixturevalue("reset_login_state")
        login.ids.word_input.text = "test_dev_pass_123"
        drain()
        login.submit()
        drain()

        wait_until(
            lambda: sm.current == "main",
            timeout=5,
            msg="Did not navigate to main after login",
        )

        sm.current = "imageview"
        drain()

    # Now wait until on_enter effects are visible
    wait_until(
        lambda: getattr(scr, "grid", None) is not None,
        timeout=5,
        msg="imageview.grid not initialized",
    )
    wait_until(
        lambda: bool(getattr(scr, "key", "")),
        timeout=5,
        msg="imageview.key not initialized",
    )

    return scr


@pytest.mark.integration
class TestImagesFlow:
    def test_images_folder_select_and_actions(self, kivy_app, request):
        sm = kivy_app.root
        img = ensure_imageview_ready(sm, request)

        # ---------- Load folder (bypass popup; test your core logic) ----------
        img.show_folder_images(str(TEST_IMAGES_DIR))
        drain()

        num_images = len(list(TEST_IMAGES_DIR.glob("*.[jp][pn]g")))

        wait_until(
            lambda: len(img.grid.children) == num_images,
            timeout=20,
            msg=f"Expected {num_images} images loaded, got {len(img.grid.children)}",
        )

        wait_until(
            lambda: len(img.grid.children) == num_images,
            timeout=20,
            msg=f"Expected {num_images} images loaded, got {len(img.grid.children)}",
        )

        # ---------- 0 selected -> buttons disabled ----------
        assert len(img.selected_images) == 0
        assert img.ids.to_db_simple_btn.disabled is True
        assert img.ids.to_db_protect_btn.disabled is True
        assert img.ids.to_ml_btn.disabled is True
        assert img.ids.selected_images.text == "Selected: 0"

        # ---------- Select all ----------
        img.select_or_unselect_button_action()
        drain()

        assert len(img.selected_images) == num_images
        assert img.ids.to_db_simple_btn.disabled is False
        assert img.ids.to_db_protect_btn.disabled is False
        assert img.ids.to_ml_btn.disabled is False
        assert img.ids.select_unselect_action_button.text == "Unselect All"
        assert img.ids.selected_images.text == f"Selected: {num_images}"

        # ---------- Unselect all ----------
        img.select_or_unselect_button_action()
        drain()

        assert len(img.selected_images) == 0
        assert img.ids.to_db_simple_btn.disabled is True
        assert img.ids.to_db_protect_btn.disabled is True
        assert img.ids.to_ml_btn.disabled is True
        assert img.ids.select_unselect_action_button.text == "Select All"
        assert img.ids.selected_images.text == "Selected: 0"

        # We'll use a stable order: first-added .. last-added
        images = grid_images(img)
        assert len(images) == num_images

        # ---------- First 2 -> To DB (simple) ----------
        click_images(images[:2])
        assert len(img.selected_images) == 2
        assert img.ids.to_db_simple_btn.disabled is False

        img.save_img_to_db(enc=False)
        drain()

        assert len(img.selected_images) == 0
        assert img.ids.selected_images.text.startswith("Added ")
        # optional DB assert if you want:
        # assert len(img.db.get_images()) == 2

        # ---------- Next 2 -> To DB (protect) ----------
        click_images(images[2:4])
        assert len(img.selected_images) == 2

        img.save_img_to_db(enc=True)
        drain()

        assert len(img.selected_images) == 0
        assert img.ids.selected_images.text.startswith("Added ")
        # optional:
        # assert len(img.db.get_images()) == 4

        # ---------- Next 2 -> To ML (choose project from dropdown) ----------
        click_images(images[4:6])
        assert len(img.selected_images) == 2
        assert img.ids.to_ml_btn.disabled is False

        # save_img_to_ml sets up dropdown + binds
        # actual copy happens when dropdown.select(project)
        img.save_img_to_ml()
        drain()

        assert img.dropdown is not None, "Dropdown was not created"
        assert img.projects, "No ML projects found (app/training/classification/*)"

        project = img.projects[0]

        projects_folder = Path(os.getcwd()) / "app" / "training" / "classification"
        target = projects_folder / project / "all"
        target.mkdir(parents=True, exist_ok=True)

        before_set = set(target.glob("*"))

        try:
            img.dropdown.select(project)
            drain()

            assert target.exists(), f"Target folder not created: {target}"
            after_set = set(target.glob("*"))

            created = {
                p for p in (after_set - before_set) if p.suffix.lower() in {".jpg", ".png"}
            }
            assert len(created) >= 2, (
                f"Expected at least 2 files copied, got {len(created)}"
            )

            assert len(img.selected_images) == 0
            assert img.ids.selected_images.text.startswith("Copied ")

        finally:
            # delete only files created by this test
            for p in set(target.glob("*")) - before_set:
                if p.is_file():
                    p.unlink(missing_ok=True)
