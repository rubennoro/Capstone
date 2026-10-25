"""B10: wireless-latency (QR clock). Pure parts run anywhere; the synth runs the REAL B5 receiver in a subprocess with a
StreamClient sender (process test: run under the suite lock with FD_TEST_PORT_OFFSET=2000)."""
import json
import os
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np
import pytest

from conftest import PKG_ROOT
from franka_deploy.tools import wireless_latency as wl

OFFSET = int(os.environ.get("FD_TEST_PORT_OFFSET", "0"))


# ------------------------------------------------------------------------------------------------ QR
def _jpeg(img, q=90):
    return cv2.imdecode(cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, q])[1], cv2.IMREAD_COLOR)


def test_qr_roundtrip_960x720_after_jpeg_q90_and_blur():
    t_ns, n = 123456789012345678, 4321
    frame = wl.qr_frame(wl.qr_text(t_ns, n), 960, 720)
    assert frame.shape == (720, 960, 3)
    blurred = cv2.GaussianBlur(frame, (0, 0), 2.0)                 # ~ a 2 px blur
    assert wl.decode_qr(_jpeg(blurred, 90)) == (t_ns, n)
    assert wl.decode_qr(_jpeg(frame, 90)) == (t_ns, n)


def test_decode_returns_none_without_a_code_and_for_foreign_codes():
    assert wl.decode_qr(np.full((720, 960, 3), 128, np.uint8)) is None
    foreign = wl.make_qr("hello world")
    assert wl.decode_qr(cv2.cvtColor(foreign, cv2.COLOR_GRAY2BGR)) is None


def test_many_distinct_timestamps_decode():
    base = 5_000_000_000_000
    for i in range(30):
        t = base + i * 33_333_333 + 17
        assert wl.decode_qr(_jpeg(wl.qr_frame(wl.qr_text(t, i)))) == (t, i)


# ------------------------------------------------------------------------------------------------ analysis
def _records(lat, n=900, hz=30.0, stream="main", jitter=0.002, map_err=0.0, drift_ms_per_min=0.0, seed=0, protocol="v1"):
    rng = np.random.default_rng(seed)
    recs = []
    for i in range(n):
        t_qr = 100.0 + i / hz
        t_arr = t_qr + lat + rng.normal(0, jitter) + drift_ms_per_min * 1e-3 * (i / hz) / 60.0
        recs.append({"stream": stream, "t_arr": t_arr, "t_map": t_qr + map_err + rng.normal(0, jitter / 4), "t_qr": t_qr, "counter": i})
    return recs


def test_analyze_v1_recovers_latency_and_passes():
    res = wl.analyze_records(_records(0.0812), {"main": 900}, "v1", {"display_hz": 60})
    d = res["streams"]["main"]
    assert d["ms"]["p50"] == pytest.approx(81.2, abs=0.5)
    assert d["decode_rate"] == 1.0 and all(res["pass"]["main"][k] for k in ("decode_rate", "spread", "stable"))
    sn = wl.snippet(res, {}, today="2026-10-21", out_dir="/o")
    assert any(s.startswith("sensor.clock.latency_s.main=0.08") for s in sn["set"])
    assert "status: measured" in sn["provenance_line"]
    assert any("display" in n.lower() or "refresh" in n.lower() for n in sn["notes"])      # the display-lag bias is stated
    assert any(s.startswith("sensor.task.camera_latency_s=") for s in sn["set"])


def test_analyze_flags_low_decode_rate_wide_spread_and_drift():
    res = wl.analyze_records(_records(0.08), {"main": 3000}, "v1", {"display_hz": 60})
    assert not res["pass"]["main"]["decode_rate"]                                           # 900 of 3000
    res = wl.analyze_records(_records(0.08, jitter=0.03), {"main": 900}, "v1", {})
    assert not res["pass"]["main"]["spread"]
    res = wl.analyze_records(_records(0.08, n=1800, drift_ms_per_min=30.0), {"main": 1800}, "v1", {})
    assert not res["pass"]["main"]["stable"] and res["streams"]["main"]["drift_ms_per_min"] == pytest.approx(30.0, rel=0.2)


def test_analyze_v2_residual_is_old_plus_error():
    recs = _records(0.05, map_err=0.0031, protocol="v2")
    res = wl.analyze_records(recs, {"main": 900}, "v2", {})
    assert res["streams"]["main"]["ms"]["p50"] == pytest.approx(3.1, abs=0.3)
    sn = wl.snippet(res, {"residual_latency_s": {"main": 0.002, "ultrawide": 0.0}})
    assert any(s.startswith("sensor.clock.residual_latency_s.main=0.0051") for s in sn["set"]), sn["set"]
    assert res["pass"]["main"]["spread"]                                                    # 0.5 ms spread <= 5 ms (v2)


def test_analyze_cli_roundtrip(tmp_path):
    frames = tmp_path / "frames.jsonl"
    with open(frames, "w") as f:
        f.write(json.dumps({"kind": "meta", "protocol": "v1", "n_frames": {"main": 900, "ultrawide": 0}, "display_hz": 120}) + "\n")
        for r in _records(0.07):
            f.write(json.dumps(r) + "\n")
    rc = wl.main(["analyze", "--frames", str(frames), "--out", str(tmp_path / "o")])
    assert rc == 0
    res = json.loads((tmp_path / "o" / "latency.json").read_text())
    assert res["streams"]["main"]["ms"]["p50"] == pytest.approx(70.0, abs=0.5) and res["bias"]["display_hz"] == 120
    assert (tmp_path / "o" / "snippet.txt").read_text().count("--set") >= 1


def test_help_exits_zero():
    for argv in (["--help"], ["measure", "--help"], ["synth", "--help"], ["analyze", "--help"], ["show", "--help"]):
        with pytest.raises(SystemExit) as e:
            wl.main(argv)
        assert e.value.code == 0


# ------------------------------------------------------------------------------------------------ synth through the real receiver
def _synth(tmp_path, protocol, extra=()):
    pytest.importorskip("real_env")
    pytest.importorskip("flask_socketio")
    pytest.importorskip("sensor_rig")
    import glob

    shm_before = set(glob.glob("/dev/shm/rmq_*"))
    env = {k: v for k, v in os.environ.items() if k != "PYTHONPATH"}
    env.update({"PYTHONPATH": str(PKG_ROOT), "PYTHONDONTWRITEBYTECODE": "1"})   # flask lives in ~/.local in RE
    out = tmp_path / f"synth_{protocol}"
    p = subprocess.run([sys.executable, "-m", "franka_deploy", "wireless-latency", "synth", "--protocol", protocol,
                        "--seconds", "8", "--port", str(5555 + OFFSET), "--camera-port", str(16702 + OFFSET),
                        "--out", str(out), *extra], capture_output=True, text=True, env=env, cwd=str(tmp_path), timeout=180)
    assert p.returncode == 0, p.stdout[-3000:] + p.stderr[-3000:]
    assert set(glob.glob("/dev/shm/rmq_*")) == shm_before, "synth left robotmq segments behind (breaks later shm_clean)"
    return json.loads((out / "synth.json").read_text()), out


def test_synth_v1_recovers_known_latency(tmp_path):
    r, out = _synth(tmp_path, "v1", ["--latency-s", "0.080"])
    for s in ("main", "ultrawide"):
        d = r["streams"][s]
        assert d["decode_rate"] >= 0.9, d
        assert abs(d["error_ms"]) <= 2.0, d                       # recovered p50 = 80 ms + the mint-to-arrival delay of the same frames
        assert 80.0 <= d["recovered_p50_ms"] <= 95.0
    assert (out / "snippet.txt").exists() and "sensor.clock.latency_s.main=0.08" in (out / "snippet.txt").read_text()


def test_synth_v2_clock_offset_and_drift_recovered(tmp_path):
    r, out = _synth(tmp_path, "v2", ["--latency-s", "0.040", "--offset-s", "12.345", "--drift-ppm", "30"])
    d = r["streams"]["main"]
    assert d["decode_rate"] >= 0.9, d
    assert abs(d["error_ms"]) <= 2.0, d                           # mapped stamp - QR time (expected 0) through ping sync
    assert d["arrival_p50_ms"] >= 40.0
