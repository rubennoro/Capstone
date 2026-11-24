# adapted from rnoro/deployment:deployment/deploy/deploy/session.py @ 91b178b
# changes: also copies the three contract files (deployment/contracts/link.yaml, frames.yaml, limits.yaml) into the
# session's contracts/ folder, never next to the frames/ folder, so frames.yaml is never mistaken for the frames
# process's settings.
"""Session: the folder of one `policy_deploy run` (STRUCTURE.md 9.3)."""
from __future__ import annotations

import shutil
from datetime import datetime
from pathlib import Path

from deploy_contracts import paths as contract_paths

from policy_deploy.launcher.errors import LaunchError

CONTRACT_FILES = ("link_file", "frames_file", "limits_file")


class Session:
    def __init__(self, path: Path) -> None:
        self.path = path

    @classmethod
    def create(cls, root: Path, mock: bool, station_file: Path) -> Session:
        """Make <root>/<YYYY-MM-DD_HH-MM-SS>[_mock]/ with logs/, ready/ and contracts/; copy the station file and the
        three contract files in."""
        name = datetime.now().strftime("%Y-%m-%d_%H-%M-%S") + ("_mock" if mock else "")
        path = Path(root) / name
        try:
            path.mkdir(parents=True, exist_ok=False)
        except FileExistsError:
            raise LaunchError(f"session folder already exists: {path} (wait a second and run again)") from None
        session = cls(path)
        (path / "logs").mkdir()
        (path / "ready").mkdir()
        (path / "contracts").mkdir()
        shutil.copyfile(station_file, path / "station.yaml")
        for getter in CONTRACT_FILES:
            source = getattr(contract_paths, getter)()
            shutil.copyfile(source, path / "contracts" / source.name)
        return session

    def log_path(self, name: str) -> Path:
        return self.path / "logs" / f"{name}.log"

    def ready_file(self, name: str) -> Path:
        return self.path / "ready" / f"{name}.ready"

    def process_dir(self, name: str) -> Path:
        """Where that process writes its own outputs (the process creates it)."""
        return self.path / name
