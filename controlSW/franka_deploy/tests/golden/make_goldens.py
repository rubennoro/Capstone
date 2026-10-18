"""Snapshot goldens for profiles without `station:` (used by tests/test_launcher_station.py).

Regenerate (only from code known to be unchanged apart from intended edits):
  cd controlSW/franka_deploy && env -u PYTHONPATH PYTHONPATH=. $RE tests/golden/make_goldens.py
Runs Launcher.snapshot() for profiles jetson (base) and mock into a tmp dir, normalises volatile / absolute fields and
writes bridge.yaml, franka_cartesian.yaml and deploy_resolved.yaml under tests/golden/snapshot_<profile>/.
"""
import sys
import tempfile
from pathlib import Path

import yaml

PKG = Path(__file__).resolve().parents[2]
GOLDEN = Path(__file__).resolve().parent
FILES = ("bridge.yaml", "franka_cartesian.yaml", "deploy_resolved.yaml")


def normalised_snapshot(profile: str, work: Path, config: Path | None = None, mutate=None) -> dict:
    """Run Launcher.snapshot() for a profile (laptop paths patched in) and return {file: normalised parsed yaml}."""
    from franka_deploy.launch.launcher import Launcher, load_config

    cfg = load_config(config or PKG / "configs" / "deploy.yaml", profile)
    cfg["work_root"] = str(work)
    cfg["python"] = {"bp": sys.executable, "realenv": sys.executable}   # snapshot() runs `pip freeze` with these
    if mutate:
        mutate(cfg)
    L = Launcher(cfg, interactive=False)
    L.snapshot()
    out = {}
    for f in FILES:
        d = yaml.safe_load((L.session / "config" / f).read_text())
        if f == "deploy_resolved.yaml":
            d["config_path"] = "<CONFIG_PATH>"
            d["work_root"] = "<WORK_ROOT>"
            d["python"] = {"bp": "<PY>", "realenv": "<PY>"}
        out[f] = d
    out["_session_files"] = sorted(p.name for p in (L.session / "config").iterdir())
    return out


if __name__ == "__main__":
    for prof in ("jetson", "mock"):
        with tempfile.TemporaryDirectory() as t:
            snap = normalised_snapshot(prof, Path(t))
        d = GOLDEN / f"snapshot_{prof}"
        d.mkdir(exist_ok=True)
        for f in FILES:
            (d / f).write_text(yaml.safe_dump(snap[f]))
        (d / "session_config_files.txt").write_text("\n".join(snap["_session_files"]) + "\n")
        print("wrote", d)
