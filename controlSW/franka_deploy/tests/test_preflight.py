"""Launcher preflight checks: each one passes on good input and fails on a bad one."""
import os
import socket
import sys
from pathlib import Path

import numpy as np
import pytest

from franka_deploy.launch import preflight as pf
from franka_deploy.paths import REPO_ROOT


def test_dirs_outside_repo(tmp_path):
    assert pf.dirs_outside_repo(tmp_path)[0]
    assert not pf.dirs_outside_repo(REPO_ROOT / "results" / "x")[0]


def test_disk_free(tmp_path):
    assert pf.disk_free(tmp_path, 0.001)[0]
    assert not pf.disk_free(tmp_path, 1e9)[0]


def test_shm_clean(tmp_path):
    assert pf.shm_clean(str(tmp_path / "rmq_*"))[0]
    (tmp_path / "rmq_x_123_main").write_text("")
    assert not pf.shm_clean(str(tmp_path / "rmq_*"))[0]


def test_ports_free():
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    s.listen(1)
    busy = s.getsockname()[1]
    try:
        assert not pf.ports_free([f"tcp://localhost:{busy}"])[0]
    finally:
        s.close()
    assert pf.ports_free([f"tcp://localhost:{busy}"])[0]


def test_checkpoint_and_staged(tmp_path):
    ck = tmp_path / "m.ckpt"
    ck.write_bytes(b"x")
    assert pf.checkpoint_ok(ck)[0]
    assert not pf.checkpoint_ok(tmp_path / "m.pt")[0]
    assert not pf.checkpoint_ok(tmp_path / "missing.ckpt")[0]
    work = tmp_path / "server"
    assert not pf.checkpoint_staged(work, ck)[0]
    (work / "ckpt").mkdir(parents=True)
    (work / "ckpt" / "m.ckpt").symlink_to(ck)
    assert pf.checkpoint_staged(work, ck)[0]


def test_prompt_zarr_sha(tmp_path):
    z = tmp_path / "p.zarr.zip"
    z.write_bytes(b"abc")
    ok, d = pf.prompt_zarr_sha(z, None, tmp_path / "cache")
    assert ok and d == "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
    assert pf.prompt_zarr_sha(z, d)[0]
    assert not pf.prompt_zarr_sha(z, "0" * 64)[0]
    assert not pf.prompt_zarr_sha(tmp_path / "missing.zip", None)[0]


def test_imagecodecs_pin():
    ok, detail = pf.imagecodecs_pin(sys.executable)   # this env: whatever is installed
    assert detail.startswith("imagecodecs 2023.9.18 ") == ok


def test_imagecodecs_pin_wrong_version(monkeypatch):
    monkeypatch.setattr(pf, "IMAGECODECS_PIN", "0.0.0")
    assert not pf.imagecodecs_pin(sys.executable)[0]


def test_rig_not_default(tmp_path):
    pytest.importorskip("sensor_rig")
    assert not pf.rig_not_default("rig_default.yaml")[0]
    assert not pf.rig_not_default(None)[0]
    assert not pf.rig_not_default(tmp_path / "missing.yaml")[0]


def test_agent_hz():
    assert pf.agent_hz_ok(20)[0] and not pf.agent_hz_ok(10)[0]


def test_bridge_not_reflex():
    assert pf.bridge_not_reflex({"bridge_state": "idle", "error": None})[0]
    assert not pf.bridge_not_reflex({"bridge_state": "reflex", "error": "x"})[0]


def test_fk_near_home():
    home = [0.45, 0, 0.25]
    assert pf.fk_near_home([0.45, 0.01, 0.25, 1, 0, 0, 0], home, 0.05)[0]
    assert not pf.fk_near_home([0.55, 0, 0.25, 1, 0, 0, 0], home, 0.05)[0]


class StalledCam:
    camera_names = ["main", "ultrawide"]
    config = {"camera_configs": {"main": {"fps": 60}, "ultrawide": {"fps": 60}}}

    def get_latest_images_dict_THWC(self, n, extra=None):
        ts = 100.0 + np.arange(max(n, 8)) / 60.0       # never advances
        img = np.zeros((len(ts), 2, 2, 3), np.uint8)
        return {"main": img, "ultrawide": img}, ts


def test_camera_ok_detects_stall():
    assert not pf.camera_ok(StalledCam(), seconds=0.05)[0]
