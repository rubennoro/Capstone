"""`python -m policy.self_check --station FILE [--mock]`: the checks `deploy check` runs (STRUCTURE.md 8.2).

Prints one JSON object per line, {"name", "status", "detail"}, and exits 0 unless it crashes. It never loads the
checkpoint and never binds a socket.
"""

from __future__ import annotations

import argparse
import contextlib
import datetime
import importlib
import json
import sys
from pathlib import Path
from typing import Sequence, TextIO

from policy.errors import SettingsError
from policy.settings import Settings

MODEL_MODULES = ("torch", "dill", "hydra", "timm", "diffusers", "behavior_prompting")
ENV_HINT = "run it with python.policy, the policy environment with torch (pip install -e policy there)"


class SelfCheck:
    def __init__(self, station: Path, mock: bool, out: TextIO) -> None:
        self._station = station
        self._mock = mock
        self._out = out

    def emit(self, name: str, status: str, detail: str) -> None:
        self._out.write(json.dumps({"name": name, "status": status, "detail": detail}) + "\n")
        self._out.flush()

    def run(self) -> None:
        try:
            settings = Settings.load(self._station, None, self._mock)
        except SettingsError as e:
            self.emit("settings", "fail", "; ".join(e.problems))
            return
        self.emit("settings", "pass", f"{settings.station_file} (policy section valid)")
        self.emit("kind", "info", f"policy kind: {settings.active_kind}" + (" (--mock)" if settings.mock else ""))
        if settings.active_kind == "diffusion":
            self._check_checkpoint(settings)
            torch = self._check_imports()
            if torch is not None:
                self._check_device(settings, torch)
            if settings.diffusion_steps is not None:
                self.emit("diffusion_steps", "info", f"diffusion steps: {settings.diffusion_steps}, PLAN §9 backup")

    def _check_checkpoint(self, settings: Settings) -> None:
        path = settings.checkpoint
        if path is None:
            self.emit("checkpoint", "fail", "policy.checkpoint is not set: set it in station.yaml")
        elif not path.exists():
            self.emit("checkpoint", "fail", f"checkpoint not found: {path}: check policy.checkpoint in station.yaml")
        elif not path.is_file() or path.suffix != ".ckpt":
            self.emit("checkpoint", "fail", f"not a .ckpt file: {path}: check policy.checkpoint in station.yaml")
        else:
            try:
                with open(path, "rb") as f:
                    f.read(1)
            except OSError as e:
                self.emit("checkpoint", "fail", f"checkpoint not readable: {path}: {e}")
                return
            st = path.stat()
            mtime = datetime.datetime.fromtimestamp(st.st_mtime).strftime("%Y-%m-%d %H:%M")
            self.emit("checkpoint", "pass", f"{path} ({st.st_size / 1e9:.2f} GB, modified {mtime})")

    def _check_imports(self):
        missing: list[str] = []
        modules = {}
        with contextlib.redirect_stdout(sys.stderr):     # keep stdout for check lines only
            for name in MODEL_MODULES:
                try:
                    modules[name] = importlib.import_module(name)
                except Exception as e:                  # ImportError, or a broken install
                    missing.append(f"{name} ({type(e).__name__}: {e})")
        if missing:
            self.emit("imports", "fail", f"not importable: {', '.join(missing)}: {ENV_HINT}")
            return modules.get("torch")
        self.emit("imports", "pass", f"torch {modules['torch'].__version__}, " + ", ".join(MODEL_MODULES[1:]))
        return modules["torch"]

    def _check_device(self, settings: Settings, torch) -> None:
        if settings.device == "cpu":
            self.emit("device", "info", "device: cpu (PLAN §9 backup)")
        elif torch.cuda.is_available():
            props = torch.cuda.get_device_properties(0)
            self.emit("device", "pass", f"cuda: {props.name}, {props.total_memory / 1e9:.1f} GB")
        else:
            self.emit("device", "fail", "CUDA not available: check the GPU driver, or set policy.device: cpu "
                                        "(PLAN §9 backup)")


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m policy.self_check", description=__doc__.split("\n\n")[0])
    ap.add_argument("--station", type=Path, required=True)
    ap.add_argument("--mock", action="store_true")
    a = ap.parse_args(argv)
    SelfCheck(a.station, a.mock, sys.stdout).run()
    return 0


if __name__ == "__main__":
    sys.exit(main())
