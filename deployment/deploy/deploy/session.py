"""Session: the folder for one `deploy run` (STRUCTURE.md section 3.2)."""
from __future__ import annotations

import shutil
from datetime import datetime
from pathlib import Path

from deploy.errors import LaunchError


class Session:
    def __init__(self, path: Path) -> None:
        self.path = path

    @classmethod
    def create(cls, root: Path, mock: bool, station_file: Path) -> "Session":
        """Make <root>/<YYYY-MM-DD_HH-MM-SS>[_mock]/ with logs/ and ready/, and copy the station file in."""
        name = datetime.now().strftime("%Y-%m-%d_%H-%M-%S") + ("_mock" if mock else "")
        path = Path(root) / name
        try:
            path.mkdir(parents=True, exist_ok=False)
        except FileExistsError:
            raise LaunchError(f"session folder already exists: {path} (wait a second and run again)") from None
        session = cls(path)
        (path / "logs").mkdir()
        (path / "ready").mkdir()
        shutil.copyfile(station_file, path / "station.yaml")
        return session

    def log_path(self, name: str) -> Path:
        return self.path / "logs" / f"{name}.log"

    def ready_file(self, name: str) -> Path:
        return self.path / "ready" / f"{name}.ready"

    def process_dir(self, name: str) -> Path:
        """Where that process writes its own outputs (the process creates it)."""
        return self.path / name
