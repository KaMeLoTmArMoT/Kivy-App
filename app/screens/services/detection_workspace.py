import os
import subprocess
from pathlib import Path

from app.screens.utils.custom_logging import get_logger
from app.screens.utils.detection_utils import split_detection_dataset
from app.screens.utils.utils import get_system_type

logger = get_logger(__name__)


class DetectionWorkspaceService:
    """Own detection project paths, dataset commands, and external processes."""

    def __init__(self, app_root: str | Path | None = None, active_project: str = "default"):
        root = Path(app_root) if app_root else Path.cwd()
        self.projects_root = root / "app" / "training" / "detection"
        self.active_project = active_project
        self.labelimg_process = None
        self.ensure_projects_root()
        self.ensure_project(active_project)

    @property
    def active_project_folder(self) -> str:
        return str(self.projects_root / self.active_project)

    @property
    def tensorboard_folder(self) -> str:
        return str(self.projects_root / "tensorboard")

    def ensure_projects_root(self) -> None:
        self.projects_root.mkdir(parents=True, exist_ok=True)

    def ensure_project(self, project_name: str) -> str:
        path = self.projects_root / project_name
        path.mkdir(parents=True, exist_ok=True)
        return str(path)

    def set_active_project(self, project_name: str) -> None:
        self.active_project = project_name
        self.ensure_project(project_name)

    def list_projects(self) -> list[str]:
        self.ensure_projects_root()
        projects = sorted(path.name for path in self.projects_root.iterdir() if path.is_dir())
        if "default" not in projects:
            self.ensure_project("default")
            projects.append("default")
        return sorted(projects)

    def model_path(self, model_name: str) -> str:
        return str(Path(self.active_project_folder) / model_name)

    def dataset_yaml_path(self) -> str:
        return str(Path(self.active_project_folder) / "dataset" / "custom_dataset.yaml")

    def labelimg_paths(self) -> tuple[str, str, str]:
        raw = Path(self.active_project_folder) / "dataset" / "raw"
        return (
            str(raw / "images"),
            str(raw / "annotations" / "classes.txt"),
            str(raw / "annotations"),
        )

    def open_labelimg(self) -> None:
        self.close_labelimg()
        env = None
        if get_system_type() == "Linux":
            env = os.environ.copy()
            for key in ("QT_PLUGIN_PATH", "QT_QPA_FONTDIR", "QT_QPA_PLATFORM_PLUGIN_PATH"):
                env.pop(key, None)
        self.labelimg_process = subprocess.Popen(
            ["labelImg", *self.labelimg_paths()],
            env=env,
        )

    def labelimg_running(self) -> bool:
        if self.labelimg_process is None:
            return False
        if self.labelimg_process.poll() is None:
            return True
        self.labelimg_process = None
        return False

    def close_labelimg(self) -> None:
        if self.labelimg_process is not None:
            self.labelimg_process.terminate()
            self.labelimg_process = None

    def split_dataset(self) -> None:
        split_detection_dataset(str(self.projects_root), self.active_project)

    def start_training(self):
        command = [
            "yolo",
            "detect",
            "train",
            f"data={self.dataset_yaml_path()}",
            "model=yolov8m.pt",
            "epochs=30",
            "imgsz=640",
        ]
        process = subprocess.Popen(command)
        logger.debug(f"{process}")
        return process
