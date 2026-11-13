"""The rig file for live use: the v2 rig file loads, broken files are refused with the key named, the CLI."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

from conftest import SMALL_RIG, V2_RIG, write_rig
from process_frames import FOLDER
from process_frames._dataproc.config import ConfigError
from process_frames.rig_file import load_live_rig, main


def test_v2_loads_at_its_resolutions_and_30_fps():
    rig = load_live_rig(V2_RIG)
    assert rig.rig_id == "rig_pi_v2"
    assert rig.cameras["main"].resolution == (1600, 1200) and rig.cameras["ultrawide"].resolution == (1280, 960)
    assert rig.cameras["main"].fps == 30.0 and rig.cameras["ultrawide"].fps == 30.0
    assert rig.cameras["main"].sensor == "imx519" and rig.cameras["ultrawide"].sensor == "imx219"


def _set(path_keys, value):
    def change(d):
        node = d
        for k in path_keys[:-1]:
            node = node[k]
        node[path_keys[-1]] = value
    return change


def _delete(path_keys):
    def change(d):
        node = d
        for k in path_keys[:-1]:
            node = node[k]
        del node[path_keys[-1]]
    return change


BROKEN = [
    ("missing key", _delete(["cameras", "main", "resolution"]), "cameras.main.resolution"),
    ("unknown key", _set(["cameras", "main", "colour"], "red"), "cameras.main.colour"),
    ("wrong schema", _set(["schema"], "dataproc.rig.v2"), "schema"),
    ("non-positive size", _set(["cameras", "ultrawide", "resolution"], [0, 96]), "cameras.ultrawide.resolution"),
    ("principal point outside", _set(["cameras", "main", "model", "intrinsics"], [100.0, 100.0, 500.0, 44.5]),
     "cameras.main.model.intrinsics"),
    ("ultrawide not fisheye", _set(["cameras", "ultrawide", "model", "type"], "pinhole-radtan"),
     "cameras.ultrawide.model.type"),
    ("sensor C0 does not allow", _set(["cameras", "main", "sensor"], "imx477"), "cameras.main.sensor"),
]


@pytest.mark.parametrize("name,change,key", BROKEN, ids=[b[0] for b in BROKEN])
def test_broken_rig_files_are_refused_naming_the_key(tmp_path, name, change, key):
    path = write_rig(tmp_path, "rig_broken", change)
    with pytest.raises(ConfigError) as e:
        load_live_rig(path)
    assert str(path) in str(e.value) and key in str(e.value)


def test_invalid_yaml_and_rig_id_mismatch(tmp_path):
    bad = tmp_path / "rig_bad.yaml"
    bad.write_text("schema: [unclosed\n")
    with pytest.raises(ConfigError, match="invalid YAML"):
        load_live_rig(bad)
    other = tmp_path / "rig_other.yaml"
    other.write_text(SMALL_RIG.read_text())             # rig_id: rig_small
    with pytest.raises(ConfigError, match="rig_id"):
        load_live_rig(other)


def test_main_json(capsys, tmp_path):
    assert main([str(V2_RIG), "--json"]) == 0
    out = json.loads(capsys.readouterr().out.strip())
    assert out == {"ok": True, "rig_id": "rig_pi_v2",
                   "cameras": {"main": {"sensor": "imx519", "width": 1600, "height": 1200, "fps": 30.0},
                               "ultrawide": {"sensor": "imx219", "width": 1280, "height": 960, "fps": 30.0}}}
    broken = write_rig(tmp_path, "rig_broken", _set(["cameras", "main", "sensor"], "imx477"))
    assert main([str(broken), "--json"]) == 1
    out = json.loads(capsys.readouterr().out.strip())
    assert out["ok"] is False and "cameras.main.sensor" in out["error"]
    assert main([str(V2_RIG)]) == 0
    assert capsys.readouterr().out.strip() == "rig_pi_v2: main imx519 1600x1200@30, ultrawide imx219 1280x960@30"


def test_module_runs_as_subprocess(tmp_path):
    done = subprocess.run([sys.executable, "-m", "process_frames.rig_file", str(SMALL_RIG), "--json"],
                          capture_output=True, text=True, cwd=FOLDER, timeout=60)
    assert done.returncode == 0, done.stderr
    assert json.loads(done.stdout)["rig_id"] == "rig_small"
    done = subprocess.run([sys.executable, "-m", "process_frames.rig_file", str(tmp_path / "nope.yaml"), "--json"],
                          capture_output=True, text=True, cwd=FOLDER, timeout=60)
    assert done.returncode == 1 and json.loads(done.stdout)["ok"] is False
