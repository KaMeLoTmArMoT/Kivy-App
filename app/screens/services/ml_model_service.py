from app.screens.services.classification import KModel, create_config_file
from app.screens.services.ml_workspace import MLWorkspaceManager
from app.screens.utils.db import DB


class MLModelService:
    """Coordinate classification model files, configs, and model lifecycle."""

    def __init__(
        self,
        workspace: MLWorkspaceManager,
        model: KModel | None = None,
        db: DB | None = None,
    ):
        self.workspace = workspace
        self.model = model or KModel(db=db)
        self.db = db or self.model.db

    def list_models(self) -> list[str]:
        return self.workspace.list_models()

    def create_model(self, base_name: str, model_type: str, classes: list[str]) -> str:
        model_name = f"{base_name}_{model_type}_{len(classes)}"
        self.model.create_model(
            model_name,
            classes,
            model_type,
            str(self.workspace.model_folder(model_name)),
            str(self.workspace.model_path(model_name)),
        )
        create_config_file(
            model_name,
            model_type,
            len(classes),
            classes,
            str(self.workspace.ml_configs_folder),
            db=self.db,
        )
        return model_name

    def load_model(self, model_name: str) -> str:
        self.model.load_model(
            str(self.workspace.model_path(model_name)),
            str(self.workspace.config_path(model_name)),
        )
        return model_name.split("_")[-2]

    def save_model(self, model_name: str) -> None:
        self.model.save_model(
            str(self.workspace.model_folder(model_name)),
            str(self.workspace.model_path(model_name)),
        )

    def delete_model(self, model_name: str) -> None:
        self.workspace.delete_model(model_name)
