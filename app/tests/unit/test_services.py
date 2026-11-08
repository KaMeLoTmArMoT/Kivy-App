from base64 import b64encode
from pathlib import Path

from Cryptodome.Cipher import AES

from app.screens.services import camera_stream
from app.screens.services.customer_service import CustomerService
from app.screens.services.detection_workspace import DetectionWorkspaceService
from app.screens.services.image_library_service import ImageLibraryService
from app.screens.services.ml_workspace import MLWorkspaceManager
from app.screens.services.model_export import get_best_model_paths


class MemoryCustomerDb:
    def __init__(self):
        self.records = []

    def insert_customer(self, value):
        self.records.append(value)

    def get_customers(self):
        return [(value,) for value in self.records]

    def delete_customer(self, value):
        self.records.remove(value)

    def update_customer(self, new_value, old_value):
        self.records[self.records.index(old_value)] = new_value


def test_customer_records_use_authenticated_random_nonces():
    db = MemoryCustomerDb()
    service = CustomerService(db=db)
    key = b"0123456789abcdef"

    service.add_customer("one", key)
    service.add_customer("two", key)

    assert len(db.records) == 2
    assert db.records[0] != db.records[1]
    assert [name for name, _ in service.get_decrypted_customers(key)] == ["one", "two"]

    service.update_customer_name("one", "updated", key)
    service.delete_customer_by_name("two", key)
    assert [name for name, _ in service.get_decrypted_customers(key)] == ["updated"]


def test_customer_service_reads_legacy_fixed_nonce_records():
    key = b"0123456789abcdef"
    cipher = AES.new(key, AES.MODE_EAX, nonce=b"TODO")
    legacy = b64encode(cipher.encrypt(b"legacy")).decode("utf-8")

    assert CustomerService(db=MemoryCustomerDb()).decrypt_text(legacy, key) == "legacy"


def test_image_service_filters_supported_extensions(tmp_path: Path):
    for name in ("a.jpg", "b.PNG", "c.txt", "d.mp4"):
        (tmp_path / name).touch()

    service = ImageLibraryService(db=object())
    assert service.get_supported_images_in_dir(str(tmp_path)) == [
        str(tmp_path / "a.jpg"),
        str(tmp_path / "b.PNG"),
    ]


def test_model_paths_are_returned_in_runtime_priority(tmp_path: Path):
    (tmp_path / "model.pt").touch()
    (tmp_path / "model.onnx").touch()
    (tmp_path / "model_openvino_model").mkdir()

    assert get_best_model_paths(str(tmp_path), "model") == [
        (str(tmp_path / "model_openvino_model"), "OpenVINO"),
        (str(tmp_path / "model.onnx"), "ONNX"),
        (str(tmp_path / "model.pt"), "PyTorch"),
    ]


def test_classification_workspace_manages_projects_and_models(tmp_path: Path):
    workspace = MLWorkspaceManager(tmp_path)

    project_path = workspace.create_project("unit_project")
    assert Path(project_path).is_dir()
    assert workspace.add_class("cat") == str(Path(project_path) / "train" / "cat")
    assert workspace.list_classes() == ["cat"]
    assert workspace.model_path("model") == Path(project_path) / "models" / "model" / "model.pth"

    workspace.delete_project("unit_project")
    assert not Path(project_path).exists()


def test_camera_stream_opens_and_releases_capture(monkeypatch):
    class FakeCapture:
        def __init__(self):
            self.released = False

        def isOpened(self):
            return True

        def set(self, *_args):
            return True

        def get(self, *_args):
            return 30

        def release(self):
            self.released = True

    capture = FakeCapture()
    monkeypatch.setattr(camera_stream.cv2, "VideoCapture", lambda *_args: capture)
    monkeypatch.setattr(camera_stream, "get_system_type", lambda: "Windows")

    stream = camera_stream.CameraStream(lambda _frame, _done: None)
    assert stream.open("webcam") is True
    stream.stop()
    assert capture.released is True


def test_detection_workspace_builds_process_commands(tmp_path: Path, monkeypatch):
    workspace = DetectionWorkspaceService(tmp_path, active_project="unit_project")
    calls = []

    class FakeProcess:
        def __init__(self, command, env=None):
            calls.append((command, env))
            self.returncode = None

        def poll(self):
            return self.returncode

        def terminate(self):
            self.returncode = 0

    monkeypatch.setattr("app.screens.services.detection_workspace.subprocess.Popen", FakeProcess)

    workspace.open_labelimg()
    assert calls[0][0][0] == "labelImg"
    assert workspace.labelimg_running() is True
    workspace.close_labelimg()
    assert workspace.labelimg_running() is False

    workspace.start_training()
    assert calls[1][0][0:3] == ["yolo", "detect", "train"]
    assert f"data={workspace.dataset_yaml_path()}" in calls[1][0]
