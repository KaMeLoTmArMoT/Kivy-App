import shutil
from pathlib import Path

from checksumdir import dirhash

from app.screens.utils.custom_logging import get_logger

logger = get_logger(__name__)


class MLWorkspaceManager:
    """Domain service managing ML project workspaces, directory structures, and datasets."""

    def __init__(self, app_root: str | Path | None = None, active_project: str = "Kivy"):
        self.app_root = Path(app_root) if app_root else Path.cwd()
        self.projects_root = self.app_root / "app" / "training" / "classification"
        self.active_project = active_project
        self.ensure_projects_root()
        self.ensure_workspace()

    def ensure_projects_root(self) -> None:
        """Ensure the root classification projects directory exists."""
        self.projects_root.mkdir(parents=True, exist_ok=True)

    @property
    def active_project_folder(self) -> str:
        return str(self.projects_root / self.active_project)

    @property
    def images_path(self) -> str:
        return str(self.projects_root / self.active_project / "all")

    @property
    def ml_train_folder(self) -> str:
        return str(self.projects_root / self.active_project / "train")

    @property
    def ml_configs_folder(self) -> str:
        return str(self.projects_root / self.active_project / "configs")

    @property
    def ml_models_folder(self) -> str:
        return str(self.projects_root / self.active_project / "models")

    @property
    def tb_folder(self) -> str:
        return str(self.projects_root / self.active_project / "tensorboard")

    def ensure_workspace(self) -> None:
        """Ensure all required subdirectories exist for the active project."""
        for p in (
            self.active_project_folder,
            self.images_path,
            self.ml_train_folder,
            self.ml_configs_folder,
            self.ml_models_folder,
            self.tb_folder,
        ):
            Path(p).mkdir(parents=True, exist_ok=True)
        logger.debug(f"Subfolders verified/created in {self.active_project_folder}")

    def set_active_project(self, project_name: str) -> None:
        """Switch active project and ensure its workspace subdirectories exist."""
        self.active_project = project_name
        self.ensure_workspace()

    def list_projects(self) -> list[str]:
        """Return list of existing project directory names."""
        self.ensure_projects_root()
        return [p.name for p in self.projects_root.iterdir() if p.is_dir()]

    def create_project(self, project_name: str) -> str:
        """Create a new project workspace directory structure."""
        target_path = self.projects_root / project_name
        if target_path.exists():
            raise FileExistsError(f"Project '{project_name}' already exists.")
        self.set_active_project(project_name)
        return self.active_project_folder

    def delete_project(self, project_name: str) -> None:
        """Safely remove a project workspace directory."""
        target_path = self.projects_root / project_name
        if target_path.is_dir():
            shutil.rmtree(target_path)
            logger.info(f"Deleted project directory: {target_path}")

    def list_classes(self) -> list[str]:
        """Return list of class directory names inside train folder."""
        train_p = Path(self.ml_train_folder)
        if not train_p.is_dir():
            return []
        return [p.name for p in train_p.iterdir() if p.is_dir()]

    def add_class(self, name: str) -> str:
        """Create a new dataset class folder."""
        path = Path(self.ml_train_folder) / name
        if path.exists():
            raise FileExistsError(f"Class '{name}' already exists.")
        path.mkdir(parents=True, exist_ok=True)
        return str(path)

    def delete_class_folder(self, folder_path: str) -> None:
        """Remove a dataset class directory."""
        p = Path(folder_path)
        if p.is_dir():
            shutil.rmtree(p)
            logger.info(f"Deleted class folder: {p}")

    def get_images_hash(self) -> str:
        """Calculate directory hash of images_path."""
        p = Path(self.images_path)
        if not p.exists():
            return ""
        return dirhash(str(p), "sha1")
