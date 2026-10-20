"""T6: every ``sensor-rig`` command in docs/first_plugin.md runs (with --mock / fakes / temporary paths).

Blocks are marked with ``<!-- test: ... -->`` right before the fence. Directives (``;``-separated):
``vars`` (the variable block), ``append=ARGS``, ``replace=OLD=>NEW``, ``stdin=TEXT``, ``kalibr`` (Kalibr output files
from tests/data/kalibr in the cwd), ``fake-docker`` (a recorded session on the laptop + a fake ``docker``).
"""

from __future__ import annotations

import os
import re
import shlex
import stat
import subprocess
import sys
from pathlib import Path

import pytest

DOC = Path(__file__).resolve().parents[1] / "docs" / "first_plugin.md"
PROJECT = Path(__file__).resolve().parents[1]
KALIBR = Path(__file__).resolve().parent / "data" / "kalibr"
FENCE = re.compile(r"(?:<!-- test: (?P<dir>.*?) -->\n)?```bash\n(?P<body>.*?)```", re.S)


def blocks():
    out = []
    for m in FENCE.finditer(DOC.read_text()):
        body = m.group("body").replace("\\\n", " ")
        cmds = [ln.split("#")[0].strip() for ln in body.splitlines() if ln.strip().startswith("sensor-rig ")]
        out.append((m.group("dir"), body, cmds))
    return out


def directives(s: str | None) -> dict:
    d = {}
    for part in (s or "").split(";"):
        part = part.strip()
        if part:
            k, _, v = part.partition("=")
            d[k.strip()] = v
    return d


def test_every_sensor_rig_command_is_in_a_marked_block():
    bs = blocks()
    assert sum(len(c) for _, _, c in bs) >= 8
    for d, _, cmds in bs:
        if cmds:
            assert d is not None, f"unmarked block with sensor-rig commands: {cmds}"
    names = {ln.split("=")[0] for d, body, _ in bs if d == "vars" for ln in body.splitlines() if "=" in ln}
    assert names == {"RIG", "RIG_V1", "RAW", "LAPTOP_RAW", "EXPORT"}


FAKE_DOCKER = r'''#!PYTHON
import os, sys
args = sys.argv[1:]
if args[:2] == ["image", "inspect"]:
    print("false"); sys.exit(0)
mount = args[args.index("-v") + 1].split(":")[0]
name = args[-1]
out = []
for ln in open(os.path.join(os.environ["FAKE_TRAJ_DIR"], name + ".tum")):
    if ln.strip() and not ln.startswith("#"):
        p = ln.split(); s, frac = p[0].split(".")
        out.append(" ".join([str(int(s) * 10**9 + int(frac.ljust(9, "0")))] + p[1:]))
open(os.path.join(mount, "f_" + name + ".txt"), "w").write("\n".join(out) + "\n")
'''


def _laptop_session(laptop_raw: Path, rig_path: Path, traj_dir: Path) -> None:
    import shutil

    from sensor_rig.rig_config import load_rig
    from sensor_rig.timefmt import make_episode_name, parse_iso
    from wpa_helpers import START_ISO, make_raw_episode

    rig = load_rig(rig_path)
    t = parse_iso(START_ISO)
    cal = make_episode_name(t - 60_000_000_000, "11111", "rig1020", "gripper_calibration", "left")
    make_raw_episode(laptop_raw, rig, kind="gripper_calibration", name=cal, main_fps=30)
    dem = make_episode_name(t, "22222", "rig1020", "demonstration", "left")
    p = make_raw_episode(laptop_raw, rig, main_fps=30, name=dem, calibration_run=cal)
    traj_dir.mkdir(parents=True, exist_ok=True)
    shutil.move(str(p / "trajectory" / "trajectory.tum"), str(traj_dir / f"{dem}.tum"))
    shutil.rmtree(p / "trajectory")


def test_first_plugin_commands_run(tmp_path):
    cwd = tmp_path / "work"
    cwd.mkdir()
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    var = {"RIG": str(PROJECT / "sensor_rig" / "configs" / "rig_jetson_imx219_2p1_b0179.yaml"),
           "RIG_V1": str(tmp_path / "rig_v1.yaml"), "RAW": str(tmp_path / "raw"),
           "LAPTOP_RAW": str(tmp_path / "laptop_raw"), "EXPORT": str(tmp_path / "export")}
    env = dict(os.environ, PYTHONPATH=str(PROJECT), PYTHONUNBUFFERED="1",
               PATH=f"{bin_dir}{os.pathsep}{os.environ['PATH']}", FAKE_TRAJ_DIR=str(tmp_path / "traj"))
    ran = []
    for d, _, cmds in blocks():
        dv = directives(d)
        if "vars" in dv or not cmds:
            continue
        if "kalibr" in dv:
            for f in KALIBR.iterdir():
                (cwd / f.name).write_bytes(f.read_bytes())
        if "fake-docker" in dv:
            fd = bin_dir / "docker"
            fd.write_text(FAKE_DOCKER.replace("#!PYTHON", f"#!{sys.executable}"))
            fd.chmod(fd.stat().st_mode | stat.S_IEXEC)
            _laptop_session(Path(var["LAPTOP_RAW"]), Path(var["RIG_V1"]), tmp_path / "traj")
        for cmd in cmds:
            for k, v in sorted(var.items(), key=lambda kv: -len(kv[0])):
                cmd = cmd.replace(f"${k}", v)
            if "replace" in dv:
                old, new = dv["replace"].split("=>")
                assert old in cmd, (old, cmd)
                cmd = cmd.replace(old, new)
            if "append" in dv:
                cmd += " " + dv["append"]
            argv = shlex.split(cmd)
            assert argv[0] == "sensor-rig"
            p = subprocess.run([sys.executable, "-m", "sensor_rig", *argv[1:]], cwd=cwd, env=env,
                               input=(dv.get("stdin", "") + "\n") if "stdin" in dv else None,
                               capture_output=True, text=True, timeout=600)
            assert p.returncode == 0, f"{cmd}\nrc={p.returncode}\n{p.stdout[-3000:]}\n{p.stderr[-3000:]}"
            ran.append(argv[1])
    assert ran == ["doctor", "bench-encode", "collect", "stamp-offset", "calib-import", "rig-validate", "collect",
                   "process-session"]
    assert (Path(var["EXPORT"]) / ".sensor_rig" / "process_rig1020.json").is_file()
