"""T15: Argus pipeline string (golden), PTS mapping, sensor-gap detection, latency tracking, rig -> params."""

from __future__ import annotations

import types

import numpy as np
import pytest

from sensor_rig.drivers.argus_logic import (
    GapDetector,
    LatencyTracker,
    argus_params_from_camera,
    argus_params_from_dict,
    build_argus_pipeline,
    detect_sensor_gaps,
    pts_to_monotonic,
)

GOLDEN_DEFAULT_MAIN = (
    'nvarguscamerasrc sensor-id=0 sensor-mode=3 exposuretimerange="100000 4000000" gainrange="1 8" '
    'ispdigitalgainrange="1 1" aelock=false awblock=false wbmode=1 '
    '! video/x-raw(memory:NVMM),width=1640,height=1232,framerate=30/1,format=NV12 '
    '! nvvidconv ! video/x-raw,format=BGRx '
    '! appsink name=sink emit-signals=false max-buffers=8 drop=false sync=false'
)


def test_pipeline_golden():
    s = build_argus_pipeline(0, 3, 1640, 1232, 30, exposure_time_range_ns=(100000, 4000000), gain_range=(1.0, 8.0),
                             wbmode=1, aelock=False, awblock=False)
    assert s == GOLDEN_DEFAULT_MAIN


def test_pipeline_locked_720p60_and_options():
    s = build_argus_pipeline(1, 4, 1280, 720, 60, exposure_time_range_ns=(4000000, 4000000), gain_range=(2, 2),
                             aelock=True, awblock=True, wbmode=0, tnr_mode=0, ee_mode=0, max_buffers=4)
    assert s == (
        'nvarguscamerasrc sensor-id=1 sensor-mode=4 exposuretimerange="4000000 4000000" gainrange="2 2" '
        'ispdigitalgainrange="1 1" tnr-mode=0 ee-mode=0 aelock=true awblock=true wbmode=0 '
        '! video/x-raw(memory:NVMM),width=1280,height=720,framerate=60/1,format=NV12 '
        '! nvvidconv ! video/x-raw,format=BGRx '
        '! appsink name=sink emit-signals=false max-buffers=4 drop=false sync=false'
    )
    assert "framerate=30000/1001" in build_argus_pipeline(0, None, 640, 480, 29.97002997)
    assert "sensor-mode" not in build_argus_pipeline(0, None, 640, 480, 30)


def test_params_from_rig_camera():
    cam = types.SimpleNamespace(sensor_id=0, sensor_mode=3, resolution=(1640, 1232), fps=30.0,
                                argus={"exposure_time_range_ns": [100000, 4000000], "gain_range": [1.0, 8.0],
                                       "aelock": False, "awblock": False, "wbmode": 1})
    assert build_argus_pipeline(**argus_params_from_camera(cam)) == GOLDEN_DEFAULT_MAIN


def test_params_p1_style_and_unknown_keys():
    p = argus_params_from_dict({"exposure_us": 4000, "gain": 2.0, "aelock": True, "awblock": True, "wbmode": 1})
    assert p["exposure_time_range_ns"] == (4000000, 4000000) and p["gain_range"] == (2.0, 2.0)
    with pytest.warns(UserWarning, match="exposure_typo"):
        argus_params_from_dict({"exposure_typo": 1})


def test_rig_default_pipeline_if_available():
    rig_config = pytest.importorskip("sensor_rig.rig_config")
    from pathlib import Path

    p = Path(rig_config.__file__).parent / "configs" / "rig_default.yaml"
    if not p.is_file():
        pytest.skip("rig_default.yaml not present")
    rig = rig_config.load_rig(p)
    s = build_argus_pipeline(**argus_params_from_camera(rig.cameras["main"]))
    assert s.startswith("nvarguscamerasrc sensor-id=0 ") and "appsink name=sink" in s


def test_pts_mapping():
    assert pts_to_monotonic(5_000_000_000, 1_234_567) == 5_001_234_567


def test_detect_sensor_gaps():
    period = 1e9 / 60
    t = np.round(np.arange(20) * period).astype(np.int64) + 10**12
    t = np.delete(t, [5, 12, 13, 14])          # one missing, then three missing
    assert detect_sensor_gaps(t, 60) == [(5, 1), (11, 3)]
    assert detect_sensor_gaps(t[:3], 60) == []
    jitter = t[:5] + np.array([0, 3_000_000, -2_000_000, 1_000_000, 0])   # jitter < 0.5 period: no gap
    assert detect_sensor_gaps(jitter, 60) == []


def test_gap_detector_seq_and_time():
    g = GapDetector(30)
    p = int(1e9 / 30)
    assert g.update(0, 0) == 0
    assert g.update(1, p) == 0
    assert g.update(3, 3 * p) == 1          # seq gap and time gap agree
    assert g.update(4, 4 * p) == 0
    assert g.update(8, 5 * p) == 3          # seq gap only (backend counted drops, stamps contiguous)
    assert g.update(None, 9 * p) == 3       # time gap only
    assert g.total_missing == 7


def test_latency_tracker():
    lt = LatencyTracker(window=50, baseline_n=20, drift_ns=5_000_000)
    for _ in range(30):
        lt.update(30_000_000)
    assert not lt.drifted and lt.median_ns == 30e6 and lt.min_ns == 30_000_000
    for _ in range(50):
        lt.update(37_000_000)
    assert lt.drifted
    lt.update(-1)
    assert lt.summary()["latency_negative"] == 1
