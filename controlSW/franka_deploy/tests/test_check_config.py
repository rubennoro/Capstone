"""check-config validates the Jetson profile without hardware and fails on bad configs."""
import sys
from pathlib import Path

import yaml

from franka_deploy.launch.launcher import check_config, load_config

PKG = Path(__file__).resolve().parents[1]


def write(tmp_path, **over):
    ck = tmp_path / "m.ckpt"
    ck.write_bytes(b"x")
    z = tmp_path / "p.zarr.zip"
    z.write_bytes(b"x")
    cfg = yaml.safe_load(open(PKG / "configs" / "deploy.yaml"))
    cfg.update(work_root=str(tmp_path / "runs"), checkpoint=str(ck), prompt_zarr=str(z),
               python={"bp": sys.executable, "realenv": sys.executable})
    cfg["camera"]["rig_config_path"] = str(tmp_path / "missing_rig.yaml")
    cfg["bridge"]["override"]["backend"].update(panda_py_version="0.8.1", libfranka_version="0.13.3")
    for k, v in over.items():
        cfg[k] = v
    p = tmp_path / "deploy.yaml"
    p.write_text(yaml.safe_dump(cfg))
    return p


def test_check_config_fails_without_rig_and_passes_with_one(tmp_path, capsys):
    p = write(tmp_path)
    assert check_config(load_config(p, "jetson")) == 1
    assert "rig_not_default" in capsys.readouterr().out
    rig = Path(__import__("sensor_rig").__file__).parent / "sim" / "sim_rig.yaml"
    cfg = yaml.safe_load(open(p))
    cfg["camera"]["rig_config_path"] = str(rig)
    p.write_text(yaml.safe_dump(cfg))
    assert check_config(load_config(p, "jetson")) == 0
    out = capsys.readouterr().out
    assert "hardware-only" in out and "panda_py" in out


def test_check_config_refuses_repo_work_root(tmp_path):
    p = write(tmp_path, work_root=str(PKG / "runs"))
    assert check_config(load_config(p, "jetson")) == 1


def test_check_config_unpinned_panda_py(tmp_path):
    p = write(tmp_path)
    cfg = yaml.safe_load(open(p))
    cfg["bridge"]["override"]["backend"].update(panda_py_version=None)
    p.write_text(yaml.safe_dump(cfg))
    assert check_config(load_config(p, "jetson")) == 1
