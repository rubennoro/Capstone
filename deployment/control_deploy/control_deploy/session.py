"""Session: the folder of one `control_deploy run` or console session (control_deploy/STRUCTURE.md 6.4, 6.5)."""
from __future__ import annotations

import shutil
import time
from pathlib import Path
from typing import Any

import yaml


class Session:
    def __init__(self, folder: Path) -> None:
        self.dir = folder
        self.logs = folder / "logs"

    @classmethod
    def create(cls, root: Path, kind: str, mock: bool = False) -> Session:
        """<root>/<YYYY-MM-DD_HH-MM-SS>_<kind>[_mock]/ with logs/ (a suffix _2, _3, ... if the name is taken)."""
        base = time.strftime("%Y-%m-%d_%H-%M-%S") + f"_{kind}" + ("_mock" if mock else "")
        root = Path(root).expanduser()
        root.mkdir(parents=True, exist_ok=True)
        folder = root / base
        n = 2
        while True:
            try:
                folder.mkdir()
                break
            except FileExistsError:
                folder = root / f"{base}_{n}"
                n += 1
        (folder / "logs").mkdir()
        return cls(folder)

    @property
    def ready_file(self) -> Path:
        return self.dir / "ready"

    @property
    def franka_control_log(self) -> Path:
        return self.logs / "franka_control.log"

    @property
    def wrapper_log(self) -> Path:
        return self.logs / "control_deploy.log"

    def write_yaml(self, name: str, data: dict[str, Any]) -> Path:
        path = self.dir / name
        with open(path, "w", encoding="utf-8") as f:
            yaml.safe_dump(data, f, sort_keys=False)
        return path

    def copy_inputs(self, station: Path, contracts_dir: Path) -> None:
        """station.yaml and the three contract files, as they were for this run."""
        shutil.copy2(station, self.dir / "station.yaml")
        (self.dir / "contracts").mkdir(exist_ok=True)
        for name in ("limits.yaml", "frames.yaml", "link.yaml"):
            shutil.copy2(contracts_dir / name, self.dir / "contracts" / name)
