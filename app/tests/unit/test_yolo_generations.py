import gc
import sqlite3

from app.screens.view.detection_screen import SUPPORTED_YOLO_GENERATIONS, DetectionScreen


def test_supported_yolo_generations_includes_26():
    assert 26 in SUPPORTED_YOLO_GENERATIONS
    assert SUPPORTED_YOLO_GENERATIONS == (8, 9, 10, 11, 12, 26)


def test_yolo_update_value_cycles_through_generations(monkeypatch):
    monkeypatch.setattr(
        "app.screens.view.detection_screen.MDLabelBtn",
        lambda **kwargs: type("Btn", (), {**kwargs, "bind": lambda *args, **kwargs: None})(),
    )

    screen = DetectionScreen.__new__(DetectionScreen)
    label_spinner = type("LabelSpinner", (), {"text": "11"})()
    model_grid = type(
        "ModelGrid",
        (),
        {
            "clear_widgets": lambda *args: None,
            "children": [],
            "add_widget": lambda *args: None,
        },
    )()
    screen.ids = {"label_spinner": label_spinner, "model_grid": model_grid}
    screen.pipeline = type("Pipeline", (), {"model_name": None})()
    screen.yolo_generation = 11
    screen.active_project = "default"

    # Step forward from 11 -> 12 -> 26
    screen.update_value(1)
    assert screen.yolo_generation == 12
    assert screen.ids["label_spinner"].text == "12"

    screen.update_value(1)
    assert screen.yolo_generation == 26
    assert screen.ids["label_spinner"].text == "26"

    # Beyond upper bound
    screen.update_value(1)
    assert screen.yolo_generation == 26

    # Step back 26 -> 12 -> 11 -> 10 -> 9 -> 8
    screen.update_value(-1)
    assert screen.yolo_generation == 12

    screen.update_value(-1)
    assert screen.yolo_generation == 11

    screen.update_value(-1)
    assert screen.yolo_generation == 10

    screen.update_value(-1)
    assert screen.yolo_generation == 9

    screen.update_value(-1)
    assert screen.yolo_generation == 8

    # Beyond lower bound
    screen.update_value(-1)
    assert screen.yolo_generation == 8


def test_load_model_names_for_generation_26(monkeypatch):
    added_widgets = []
    monkeypatch.setattr(
        "app.screens.view.detection_screen.MDLabelBtn",
        lambda **kwargs: type("Btn", (), {**kwargs, "bind": lambda *args, **kwargs: None})(),
    )

    model_grid = type(
        "ModelGrid",
        (),
        {
            "clear_widgets": lambda *args: added_widgets.clear(),
            "add_widget": lambda _self, widget, *args, **kwargs: added_widgets.append(widget),
            "children": [],
        },
    )()

    screen = DetectionScreen.__new__(DetectionScreen)
    screen.ids = {"model_grid": model_grid}
    screen.yolo_generation = 26
    screen.active_project = "default"
    screen.load_model_names()

    model_names = [w.text for w in added_widgets]
    assert model_names == [
        "yolo26n.pt",
        "yolo26s.pt",
        "yolo26m.pt",
        "yolo26l.pt",
        "yolo26x.pt",
    ]

    # Clean up SQLite handles if any to prevent Windows lock during fixture teardown
    gc.collect()
    sqlite3.connect(":memory:").close()
