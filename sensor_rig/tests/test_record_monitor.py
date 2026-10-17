"""Live abort rules (pure), the record / bench-encode CLI on mocks, and the encoder bench."""

from __future__ import annotations

import json

import numpy as np
import pytest

from sensor_rig.record.monitor import LiveAbortConfig, evaluate_live_abort

S = 1_000_000_000


def snap(elapsed=3.0, main_fps=60.0, uw_fps=30.0, imu_rate=1000.0, qf_main=0, disk=100.0, stale_s=0.0):
    now = int(10 * S + elapsed * S)
    t_last = now - int(stale_s * S)

    def times(fps, dur):
        n = int(round(dur * fps))
        return (t_last - np.round(np.arange(n)[::-1] * S / fps)).astype(np.int64) if fps > 0 else np.zeros(0, np.int64)

    main = times(main_fps, elapsed)
    qf = main[-qf_main:] if qf_main else np.zeros(0, np.int64)
    return {"elapsed_s": elapsed, "now_ns": now, "disk_free_gb": disk,
            "cameras": {"main": {"fps_nominal": 60.0, "is_main": True, "frame_times": main, "queue_full_times": qf},
                        "ultrawide": {"fps_nominal": 30.0, "is_main": False, "frame_times": times(uw_fps, elapsed),
                                      "queue_full_times": np.zeros(0, np.int64)}},
            "imu": {"rate_nominal": 1000.0, "sample_times": times(imu_rate, elapsed + 1.0)}}


CFG = LiveAbortConfig()


def test_rules_ok():
    assert evaluate_live_abort(snap(), CFG) is None
    assert evaluate_live_abort(snap(main_fps=32), CFG) is None        # 32 >= 0.52 * 60 = 31.2
    assert evaluate_live_abort(snap(elapsed=0.5, main_fps=0), CFG) is None   # before after_s


def test_rules_fire():
    assert evaluate_live_abort(snap(main_fps=30), CFG).startswith("main_fps_low")
    assert evaluate_live_abort(snap(stale_s=0.8), CFG).startswith("main_fps_low")   # camera stalled
    assert evaluate_live_abort(snap(uw_fps=5), CFG) is None                          # UW: no fps rule
    assert evaluate_live_abort(snap(qf_main=3), CFG).startswith("queue_full")       # 3/120 = 2.5 % > 2 %
    assert evaluate_live_abort(snap(qf_main=2), CFG) is None                         # 2/120 = 1.7 %
    assert evaluate_live_abort(snap(imu_rate=850), CFG).startswith("imu_rate_low")
    assert evaluate_live_abort(snap(elapsed=1.5, imu_rate=850), CFG) is None        # before the 2 s window
    assert evaluate_live_abort(snap(disk=1.0), CFG).startswith("disk_low")


def test_config_from_dict():
    c = LiveAbortConfig.from_dict({"min_fps_ratio": 0.52, "after_s": 1.0, "min_disk_free_gb": 2.0})
    assert c.min_fps_ratio == 0.52
    with pytest.raises(ValueError):
        LiveAbortConfig.from_dict({"min_fps": 3})


def test_bench_encode_smoke():
    from sensor_rig.record.bench import bench_encode

    res = bench_encode(96, 64, quality=85, n=12, workers=(1, 2), print_fn=None)
    assert [r["workers"] for r in res] == [1, 2] and all(r["frames_per_s"] > 0 for r in res)


def test_cli_record_mock(test_rig_path, tmp_path, capsys):
    import argparse

    from sensor_rig.record import cli

    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers()
    cli.register(sub)
    args = parser.parse_args(["record", "--rig", str(test_rig_path), "--out", str(tmp_path), "--session", "rig-cli-test",
                              "--side", "left", "--kind", "gripper_calibration", "--duration", "0.6", "--mock",
                              "--warmup", "1.1", "--quiet"])
    assert args.func(args) == 0
    eps = [p for p in tmp_path.glob("*/*") if p.is_dir()]
    assert len(eps) == 1 and "_rig-cli-test_grippercalibration_left" in eps[0].name
    meta = json.loads((eps[0] / "meta.json").read_text())
    assert meta["status"] == "complete" and meta["sensors"]["main"]["frames"] >= 15
    # demonstration now finds the calibration automatically; bad args -> exit code 2
    args = parser.parse_args(["record", "--rig", str(test_rig_path), "--out", str(tmp_path), "--session", "rig-cli-test",
                              "--side", "left", "--duration", "-1", "--mock"])
    assert args.func(args) == 2
    args = parser.parse_args(["bench-encode", "--width", "64", "--height", "48", "--n", "4", "--workers", "1"])
    assert args.func(args) == 0
    assert "workers=1" in capsys.readouterr().out
