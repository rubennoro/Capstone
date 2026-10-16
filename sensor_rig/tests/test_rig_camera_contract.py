"""RigCamera (real-env peripheral) honours the iPhone peripheral contract (P2 §6.4-6.6, R3 §6).

real-env's third-party deps (robotmq, robologger, robot_utils, loguru) are stubbed by ``realenv_stubs`` only when
they are missing, so on the Jetson the same tests run against the real packages.
"""

from __future__ import annotations

import json

import cv2
import numpy as np
import pytest

import realenv_stubs

realenv_stubs.install()
hydra = pytest.importorskip("hydra")
from hydra import compose, initialize_config_dir  # noqa: E402
from omegaconf import OmegaConf  # noqa: E402

rig_camera = pytest.importorskip("real_env.peripherals.rig_camera")
from real_env.peripherals.base_camera import BaseCamera  # noqa: E402

RigCamera = rig_camera.RigCamera


def compose_cfg(name: str, overrides=()):
    with initialize_config_dir(config_dir=str(realenv_stubs.REAL_ENV_CONFIGS), version_base=None):
        cfg = compose(config_name=name, overrides=list(overrides))
    OmegaConf.set_struct(cfg, False)
    cfg.config_str = json.dumps(OmegaConf.to_container(cfg, resolve=False))
    return cfg


def test_configs_compose_like_iphone():
    right = compose_cfg("rig_wrist_right")
    iphone = compose_cfg("iphone_wrist_right")
    assert right._target_ == "real_env.peripherals.rig_camera.RigCamera"
    for key in ("name", "server_endpoint", "logger_endpoint"):
        assert right[key] == iphone[key], key                    # drop-in: tasks need no change
    assert right.name == "right_wrist_camera_0"
    assert set(right.camera_configs) == {"main", "ultrawide"}
    for cam in ("main", "ultrawide"):
        cc = right.camera_configs[cam]
        assert (cc.type, cc.width, cc.height, cc.fps) == ("rgb", 960, 720, 60)
    assert right.camera_latency_s == 0.0 and right.display_resolution_WH is None
    assert right.video_codec == "libx264"
    mock = compose_cfg("rig_wrist_mock")
    assert mock.backend == "mock" and mock.publish_mode == "fixed_rate" and mock.name == right.name
    assert mock.backend_opts.episode_dir is None and mock.camera_configs == right.camera_configs


def test_get_images():
    """rig_wrist_mock.yaml via Hydra -> RigCamera (synthetic mock frames): 20 calls give exactly {main, ultrawide}
    (720,960,3) uint8 C-contiguous BGR, non-decreasing timestamps, and libx264 on the video logger."""
    cfg = compose_cfg("rig_wrist_mock", ["stats_period_s=0"])
    cam = hydra.utils.instantiate(cfg)
    try:
        assert isinstance(cam, BaseCamera)
        assert cam.video_logger.codec == "libx264"
        assert cam.publish_mode == "fixed_rate" and cam.publish_fps == 60
        last = -np.inf
        stamps = []
        for _ in range(20):
            frames, ts = cam.get_images_bgr_hwc_and_timestamp()
            assert set(frames) == {"main", "ultrawide"}
            for img in frames.values():
                assert img.shape == (720, 960, 3) and img.dtype == np.uint8 and img.flags.c_contiguous
            assert isinstance(ts, float) and ts >= last and ts > 0
            last = ts
            stamps.append(ts)
        assert len(set(stamps)) >= 5            # 30 fps physical main behind a 60 Hz fixed-rate publisher
    finally:
        cam.source.stop()


def _direct_camera(**kw):
    args = dict(
        name="right_wrist_camera_0", server_endpoint="tcp://localhost:16702", logger_endpoint="tcp://localhost:55556",
        camera_latency_s=0.0, display_resolution_WH=None,
        camera_configs={"main": {"type": "rgb", "width": 960, "height": 720, "fps": 60},
                        "ultrawide": {"type": "rgb", "width": 960, "height": 720, "fps": 60}},
        rig_config_path="rig_default.yaml", backend="mock", backend_opts={}, publish_mode="per_main_frame",
        stats_period_s=0, config_str=json.dumps({"camera_configs": {}}),
    )
    args.update(kw)
    return RigCamera(**args)


def test_base_run_publishes():
    """BaseCamera.run() with a recording RMQ server stopping after 10 timestamps: 10 RGB frames per camera and 10
    timestamps; published timestamp == main capture time (camera_latency_s = 0); bytes == BGR->RGB of our output."""
    cam = _direct_camera()
    server = realenv_stubs.RecordingRMQServer(stop_after_timestamps=10)
    cam.rmq_server = server
    packets = []
    orig = cam.source.next_packet

    def spy():
        p = orig()
        packets.append(p)
        return p

    cam.source.next_packet = spy
    with pytest.raises(realenv_stubs.StopRun):
        cam.run()                                   # run() stops the sources in its finally block
    assert not cam.source.sources["main"].running
    from robotmq import deserialize

    stamps = [deserialize(d) for d in server.data("timestamp")]
    assert len(stamps) == 10 and len(packets) == 10
    assert server.data("config") and json.loads(deserialize(server.data("config")[0])) == {"camera_configs": {}}
    assert stamps == [p.t_capture_s for p in packets]
    assert all(b > a for a, b in zip(stamps, stamps[1:]))
    for cam_name in ("main", "ultrawide"):
        blobs = server.data(cam_name)
        assert len(blobs) == 10
        for blob, p in zip(blobs, packets):
            assert blob == cv2.cvtColor(p.images[cam_name], cv2.COLOR_BGR2RGB).tobytes()
    topics = [t for t, _ in server.puts if t != "config"]
    assert topics[:3] == ["main", "ultrawide", "timestamp"]       # one frame per camera, then one timestamp


def test_rejects_bad_configs():
    with pytest.raises(ValueError, match="exactly main \\+ ultrawide"):
        _direct_camera(camera_configs={"main": {"type": "rgb", "width": 960, "height": 720, "fps": 60}})
    with pytest.raises(ValueError, match="non-uniform"):
        _direct_camera(camera_configs={"main": {"type": "rgb", "width": 960, "height": 540, "fps": 60},
                                       "ultrawide": {"type": "rgb", "width": 960, "height": 720, "fps": 60}})


def test_auto_publish_mode_and_latency_warning(capsys):
    cam = _direct_camera(publish_mode="auto", camera_latency_s=0.01)
    try:
        assert cam.publish_mode == "fixed_rate"     # rig_default main is 30 fps, config says 60
        assert "camera_latency_s=0.01 is non-zero" in capsys.readouterr().out
    finally:
        cam.source.stop()
