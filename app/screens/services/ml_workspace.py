import os
import shutil

from checksumdir import dirhash

from app.screens.utils.custom_logging import get_logger

logger = get_logger(__name__)


class MLWorkspaceManager:
    """Domain service managing ML project workspaces, directory structures, and datasets."""

    def __init__(self, app_root: str | None = None, active_project: str = "Kivy"):
        self.app_root = app_root or os.getcwd()
        self.projects_root = os.path.join(self.app_root, "app", "training", "classification")
        self.active_project = active_project
        self.ensure_projects_root()
        self.ensure_workspace()

    def ensure_projects_root(self) -> None:
        """Ensure the root classification projects directory exists."""
        os.makedirs(self.projects_root, exist_ok=True)

    @property
    def active_project_folder(self) -> str:
        return os.path.join(self.projects_root, self.active_project)

    @property
    def images_path(self) -> str:
        return os.path.join(self.active_project_folder, "all")

    @property
    def ml_train_folder(self) -> str:
        return os.path.join(self.active_project_folder, "train")

    @property
    def ml_configs_folder(self) -> str:
        return os.path.join(self.active_project_folder, "configs")

    @property
    def ml_models_folder(self) -> str:
        return os.path.join(self.active_project_folder, "models")

    @property
    def tb_folder(self) -> str:
        return os.path.join(self.active_project_folder, "tensorboard")

    def ensure_workspace(self) -> None:
        """Ensure all required subdirectories exist for the active project."""
        os.makedirs(self.active_project_folder, exist_ok=True)
        os.makedirs(self.images_path, exist_ok=True)
        os.makedirs(self.ml_train_folder, exist_ok=True)
        os.makedirs(self.ml_configs_folder, exist_ok=True)
        os.makedirs(self.ml_models_folder, exist_ok=True)
        os.makedirs(self.tb_folder, exist_ok=True)
        logger.debug(f"Subfolders verified/created in {self.active_project_folder}")

    def set_active_project(self, project_name: str) -> None:
        """Switch active project and ensure its workspace subdirectories exist."""
        self.active_project = project_name
        self.ensure_workspace()

    def list_projects(self) -> list[str]:
        """Return list of existing project directory names."""
        self.ensure_projects_root()
        return [
            d
            for d in os.listdir(self.projects_root)
            if os.path.isdir(os.path.join(self.projects_root, d))
        ]

    def create_project(self, project_name: str) -> str:
        """Create a new project workspace directory structure."""
        target_path = os.path.join(self.projects_root, project_name)
        if os.path.exists(target_path):
            raise FileExistsError(f"Project '{project_name}' already exists.")
        self.set_active_project(project_name)
        return self.active_project_folder

    def delete_project(self, project_name: str) -> None:
        """Safely remove a project workspace directory."""
        target_path = os.path.join(self.projects_root, project_name)
        if os.path.exists(target_path) and os.path.isdir(target_path):
            shutil.rmtree(target_path)
            logger.info(f"Deleted project directory: {target_path}")

    def list_classes(self) -> list[str]:
        """Return list of class directory names inside train folder."""
        if not os.path.isdir(self.ml_train_folder):
            return []
        return [
            f
            for f in os.listdir(self.ml_train_folder)
            if os.path.isdir(os.path.join(self.ml_train_folder, f))
        ]

    def add_class(self, name: str) -> str:
        """Create a new dataset class folder."""
        path = os.path.join(self.ml_train_folder, name)
        if os.path.exists(path):
            raise FileExistsError(f"Class '{name}' already exists.")
        os.makedirs(path, exist_ok=True)
        return path

    def delete_class_folder(self, folder_path: str) -> None:
        """Remove a dataset class directory."""
        if os.path.exists(folder_path) and os.path.isdir(folder_path):
            shutil.rmtree(folder_path)
            logger.info(f"Deleted class folder: {folder_path}")

    def get_images_hash(self) -> str:
        """Calculate directory hash of images_path."""
        if not os.path.exists(self.images_path):
            return ""
        return dirhash(self.images_path, "sha1")
