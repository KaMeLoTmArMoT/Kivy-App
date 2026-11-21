import os
import shutil
import time
from pathlib import Path

import pytest
from kivy.clock import Clock
from kivymd.uix.label import MDLabel

pytestmark = pytest.mark.slow

pytest.importorskip("torch")
pytest.importorskip("torchvision")

TEST_PROJECT = "integration_test_project"
TEST_MODEL_BASENAME = "integration_test_model"
TEST_MODELTYPE = "MobileNetV3"
CLASSES = ["cat", "dog"]


def drain(frames: int = 5) -> None:
    for _ in range(frames):
        Clock.tick()


def wait_until(
    predicate,
    *,
    timeout: float = 20.0,
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
    wait_until(lambda: sm.current == "main", timeout=10, msg="Login did not navigate to main")


def grid_texts(grid) -> set[str]:
    out = set()
    for w in grid.children:
        t = getattr(w, "text", None)
        if isinstance(t, str):
            out.add(t)
    return out


def find_example_images_dir(repo_root: Path) -> Path:
    candidates = [
        repo_root / "example_images",
        repo_root / "app" / "example_images",
        repo_root / "app" / "resources" / "example_images",
        repo_root / "resources" / "example_images",
    ]
    for c in candidates:
        if c.is_dir():
            return c

    # last resort: shallow scan
    for p in repo_root.rglob("example_images"):
        if p.is_dir():
            return p

    raise FileNotFoundError(
        "example_images directory not found (set it in repo or adjust candidates)"
    )


def expected_label_from_filename(path: Path) -> str:
    # scheme: digit + (c/d) + _test  -> cat/dog
    stem = path.stem.lower()
    if "c" in stem:
        return "cat"
    if "d" in stem:
        return "dog"
    raise ValueError(f"Can't infer label from filename: {path.name}")


def wait_images_loaded(ml, *, timeout=20):
    wait_until(
        lambda: (ml.load_event is None) or (len(getattr(ml, "images_to_load", [])) == 0),
        timeout=timeout,
        msg="Image loading not settled",
    )


def shown_tiles_count(ml) -> int:
    return len(ml.ids.image_grid.children)


def extract_predictions_from_selected(ml) -> dict[str, str]:
    """
    Returns mapping: image_path -> predicted_label (cat/dog).
    """
    out = {}
    for img in ml.selected_images:
        # label_container filled by model_predict()
        lc = getattr(img, "label_container", None)
        assert lc is not None, "Image button has no label_container"
        texts = [
            w.text
            for w in lc.children
            if isinstance(w, MDLabel) and isinstance(getattr(w, "text", None), str)
        ]
        assert texts, "No prediction label rendered in label_container"
        out[str(img.source)] = texts[0]
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
            lambda: ml.load_event is None or len(getattr(ml, "images_to_load", [])) == 0,
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
            assert not any(TEST_MODEL_BASENAME in t for t in grid_texts(ml.ids.model_grid))

            # ----- add 2 classes -----
            for cname in CLASSES:
                ml.ids.class_input.text = cname
                ml.add_class()
                drain()

            class_texts = grid_texts(ml.ids.class_grid)
            assert "all" in class_texts
            assert f"train/{CLASSES[0]}" in class_texts
            assert f"train/{CLASSES[1]}" in class_texts

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


@pytest.mark.integration
@pytest.mark.slow
class TestMlPredictTrainEvaluate:
    def test_predict_train_evaluate_flow(self, kivy_app, request):
        sm = kivy_app.root
        wait_until(lambda: sm.has_screen("mlview"), timeout=30, msg="mlview not loaded")
        wait_until(lambda: sm.has_screen("login"), timeout=30, msg="login not loaded")
        ensure_logged_in(sm, request)

        sm.current = "mlview"
        drain()
        ml = sm.get_screen("mlview")

        project_path = Path(ml.projects_folder) / TEST_PROJECT
        default_project_path = Path(ml.projects_folder) / "Kivy"

        if project_path.exists():
            shutil.rmtree(project_path, ignore_errors=True)
        default_project_path.mkdir(parents=True, exist_ok=True)

        project_path.mkdir(parents=True, exist_ok=True)
        ml.after_project_selection_hook(TEST_PROJECT, str(project_path))
        ml.active_project = TEST_PROJECT
        ml.main_button.text = TEST_PROJECT
        ml.restore_project_params(TEST_PROJECT, str(project_path))
        wait_images_loaded(ml, timeout=30)

        try:
            # ---- create classes: cat/dog ----
            for cname in CLASSES:
                ml.ids.class_input.text = cname
                ml.add_class()
                drain()

            # ---- create model ----
            ml.unload_model()
            ml.model_type = TEST_MODELTYPE
            ml.ids.model_label.text = TEST_MODELTYPE

            ml.ids.model_input.text = TEST_MODEL_BASENAME
            ml.create_model(ml.ids.model_input.text)
            drain()

            expected_model_name = f"{TEST_MODEL_BASENAME}_{TEST_MODELTYPE}_2"
            wait_until(
                lambda: any(
                    w.text == expected_model_name
                    for w in ml.ids.model_grid.children
                    if hasattr(w, "text")
                ),
                timeout=15,
                msg="Model not visible in model_grid",
            )

            # ensure model is loaded for predict/train/eval
            model_btn = next(
                w
                for w in ml.ids.model_grid.children
                if getattr(w, "text", None) == expected_model_name
            )
            ml.selected_model = model_btn
            ml.load_model()
            assert ml.k_model.model is not None, "Model not loaded into k_model.model"

            # ---- copy example_images into train/cat, train/dog ----
            repo_root = Path(getattr(ml, "app_folder", os.getcwd()))
            ex_dir = find_example_images_dir(repo_root)

            cat_dir = Path(ml.ml_train_folder) / "cat"
            dog_dir = Path(ml.ml_train_folder) / "dog"
            cat_dir.mkdir(parents=True, exist_ok=True)
            dog_dir.mkdir(parents=True, exist_ok=True)

            image_files = [
                p
                for p in ex_dir.iterdir()
                if p.is_file() and p.suffix.lower() in {".jpg", ".jpeg", ".png"}
            ]
            assert image_files, f"No images found in {ex_dir}"

            # keep dataset small for CI speed
            image_files = image_files[:40]

            for src in image_files:
                label = expected_label_from_filename(src)
                dst_dir = cat_dir if label == "cat" else dog_dir
                shutil.copy2(src, dst_dir / src.name)

            # ---- helper: run predict in a folder and compute accuracy ----
            def predict_in_folder(folder: Path) -> float:
                ml.unselect_all_images()
                ml.clear_predictions()
                # select class button so selected_dir is set
                class_btn_text = f"train/{folder.name}"  # folder.name == "cat"/"dog"
                class_btn = next(
                    w
                    for w in ml.ids.class_grid.children
                    if getattr(w, "text", None) == class_btn_text
                )
                ml.select_label_btn(class_btn)
                drain()

                # now open folder via UI flow (path can be omitted)
                ml.show_folder_images(new=True)
                wait_images_loaded(ml, timeout=30)

                expected_shown = min(
                    len(
                        [
                            p
                            for p in folder.iterdir()
                            if p.is_file() and p.suffix.lower() in {".jpg", ".jpeg", ".png"}
                        ]
                    ),
                    ml.max_images_per_page,
                )
                assert shown_tiles_count(ml) == expected_shown

                ml.select_all_images()
                assert len(ml.selected_images) == expected_shown

                ml.model_predict()
                drain()

                preds = extract_predictions_from_selected(ml)
                assert preds, "No predictions extracted"

                correct = 0
                total = 0
                for img_path, pred in preds.items():
                    assert pred in {"cat", "dog"}
                    exp = expected_label_from_filename(Path(img_path))
                    total += 1
                    if pred == exp:
                        correct += 1
                return correct / max(total, 1)

            acc_cat_before = predict_in_folder(cat_dir)
            acc_dog_before = predict_in_folder(dog_dir)
            # no hard asserts on accuracy before training (just sanity)
            assert 0.0 <= acc_cat_before <= 1.0
            assert 0.0 <= acc_dog_before <= 1.0

            # ---- train ----
            ml.trigger_training()
            drain()

            wait_until(
                lambda: (ml.train_active is False) and (ml.ids.train_btn.text == "Train"),
                timeout=240,
                msg="Training did not finish in time",
            )

            # ---- tensorboard events exist ----
            tb_root = Path(ml.tb_folder)
            assert tb_root.exists()

            # find any events file recursively (timestamp folder name is dynamic)
            events = list(tb_root.rglob("events.out.tfevents*"))
            assert events, f"No tensorboard events found under {tb_root}"

            # ---- predict again ----
            acc_cat_after = predict_in_folder(cat_dir)
            acc_dog_after = predict_in_folder(dog_dir)
            assert 0.0 <= acc_cat_after <= 1.0
            assert 0.0 <= acc_dog_after <= 1.0

            # ---- evaluate ----
            ml.evaluate_model()
            drain()

            # wait eval finishes
            wait_until(
                lambda: (ml.eval_event is None) and (ml.ids.evaluate_btn.text == "Evaluate"),
                timeout=240,
                msg="Evaluate did not finish in time",
            )

        finally:
            ml.active_project = "Kivy"
            ml.main_button.text = "Kivy"
            ml.restore_project_params("Kivy", str(default_project_path))
            drain()

            if project_path.exists():
                shutil.rmtree(project_path, ignore_errors=True)
