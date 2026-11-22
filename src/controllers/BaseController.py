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

    def get_vectordb_path(self) -> str:
        """Chroma lives in src/assets/database (created if missing); the read-only numpy export
        is a folder in the project root and is never created (the disk may be read-only)."""
        config = self.settings.vectordb
        if config.VECTOR_DB_PROVIDER == "numpy":
            return str(self.project_root / config.VECTOR_DB_PATH)
        return self.get_database_path(config.VECTOR_DB_PATH)
