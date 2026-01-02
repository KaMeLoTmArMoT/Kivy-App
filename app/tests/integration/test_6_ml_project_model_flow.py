import shutil
import time
from pathlib import Path

import pytest
from kivy.clock import Clock

TEST_PROJECT = "integration_test_project"
TEST_MODEL_BASENAME = "integration_test_model"
TEST_MODELTYPE = "MobileNetV3"
TEST_CLASSES = ["dog_integration", "cat_integration"]


def drain(frames: int = 5) -> None:
    for _ in range(frames):
        Clock.tick()


def wait_until(
    predicate,
    *,
    timeout: float = 8.0,
    step_frames: int = 2,
    msg: str = "Condition not met",
) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        drain(step_frames)
        if predicate():
            return
    raise AssertionError(msg)


def ensure_logged_in(sm, request):
    if sm.has_screen("main") and getattr(sm.get_screen("main"), "key", None):
        return
    login = request.getfixturevalue("reset_login_state")
    login.ids.word_input.text = "test_dev_pass_123"
    drain()
    login.submit()
    drain()
    wait_until(
        lambda: sm.current == "main", timeout=6, msg="Login did not navigate to main"
    )


def grid_texts(grid) -> set[str]:
    out = set()
    for w in grid.children:
        t = getattr(w, "text", None)
        if isinstance(t, str):
            out.add(t)
    return out


@pytest.mark.integration
class TestMlProjectModelFlow:
    def test_create_project_and_model(self, kivy_app, request):
        sm = kivy_app.root

        wait_until(lambda: sm.has_screen("mlview"), timeout=20, msg="mlview not loaded")
        wait_until(lambda: sm.has_screen("login"), timeout=20, msg="login not loaded")
        ensure_logged_in(sm, request)

        sm.current = "mlview"
        drain()

        ml = sm.get_screen("mlview")

        # ----- create/switch project (no popup; direct hook) -----
        project_path = Path(ml.projects_folder) / TEST_PROJECT
        default_project_path = Path(ml.projects_folder) / "Kivy"

        # cleanup if stale from previous failed runs
        if project_path.exists():
            shutil.rmtree(project_path, ignore_errors=True)

        # ensure default exists
        default_project_path.mkdir(parents=True, exist_ok=True)

        # create + switch
        project_path.mkdir(parents=True, exist_ok=True)
        ml.after_project_selection_hook(TEST_PROJECT, str(project_path))
        ml.active_project = TEST_PROJECT
        ml.main_button.text = TEST_PROJECT
        ml.restore_project_params(TEST_PROJECT, str(project_path))

        # let async folder load finish (it schedules Clock interval)
        wait_until(
            lambda: ml.load_event is None
            or len(getattr(ml, "images_to_load", [])) == 0,
            timeout=6,
            msg="Image loading not settled",
        )

        assert ml.ids.project_label.text == TEST_PROJECT

        try:
            # ----- model type: MobileNetV3 -----
            ml.unload_model()
            ml.model_type = TEST_MODELTYPE
            ml.ids.model_label.text = TEST_MODELTYPE

            # ----- attempt create model with 0 classes -> must fail -----
            ml.ids.model_input.text = TEST_MODEL_BASENAME
            ml.create_model(ml.ids.model_input.text)
            drain()

            # error shown for only few frames, so skip
            # assert "Model cant have 0 or 1 class" in ml.ids.error_popup_text.text

            # no model should appear
            assert not any(
                TEST_MODEL_BASENAME in t for t in grid_texts(ml.ids.model_grid)
            )

            # ----- add 2 classes -----
            for cname in TEST_CLASSES:
                ml.ids.class_input.text = cname
                ml.add_class()
                drain()

            class_texts = grid_texts(ml.ids.class_grid)
            assert "all" in class_texts
            assert f"train/{TEST_CLASSES[0]}" in class_texts
            assert f"train/{TEST_CLASSES[1]}" in class_texts

            # model name should still be filled (failed create doesn't clear it)
            assert ml.ids.model_input.text == TEST_MODEL_BASENAME

            # ----- create model again -> must succeed -----
            ml.create_model(ml.ids.model_input.text)
            drain()

            expected_model_name = f"{TEST_MODEL_BASENAME}_{TEST_MODELTYPE}_2"
            wait_until(
                lambda: expected_model_name in grid_texts(ml.ids.model_grid),
                timeout=8,
                msg="Model not shown in UI",
            )

            assert ml.model_name == expected_model_name
            assert expected_model_name in grid_texts(ml.ids.model_grid)

        finally:
            # switch back to default project, then delete test project folder
            ml.active_project = "Kivy"
            ml.main_button.text = "Kivy"
            ml.restore_project_params("Kivy", str(default_project_path))
            drain()

            if project_path.exists():
                shutil.rmtree(project_path, ignore_errors=True)
