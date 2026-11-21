from pathlib import Path

from src.helpers.config import PROJECT_ROOT, Settings, get_settings


class BaseController:
    def __init__(self):
        self.settings: Settings = get_settings()
        self.project_root: Path = PROJECT_ROOT
        self.database_path: Path = PROJECT_ROOT / "src" / "assets" / "database"

    def get_database_path(self, name: str) -> str:
        path = self.database_path / name
        path.mkdir(parents=True, exist_ok=True)
        return str(path)
