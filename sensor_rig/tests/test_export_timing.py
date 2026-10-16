"""T5 (D10b): time strings and the ultrawide slot pattern survive the round trip exactly."""

from __future__ import annotations

import numpy as np
import pytest

from fixtures_sim import *  # noqa: F401,F403
from sensor_rig.export.validate import validate_iphone_episode

pytestmark = pytest.mark.data


def test_T5_time_strings_exact(sim_export):
    se = sim_export("demo60")
    n = len(se.truth["t_ns"])
    assert se.json["rgbTimes"] == se.src.json["rgbTimes"][:n]
    assert se.json["poseTimes"] == se.json["rgbTimes"]
    assert se.json["ultrawideRGBTimes"] == se.src.json["ultrawideRGBTimes"][:n]
    assert se.json["recordingStartTime"] == se.src.json["recordingStartTime"]
    assert se.json["deviceIdentifier"] == "JetsonRig1,1"


def test_T5_calibration_pattern(sim_export):
    se = sim_export("calib")
    assert se.result.status == "exported", se.result
    n = len(se.truth["t_ns"])
    assert se.json["frameCount"] == n
    assert se.json["ultrawideRGBTimes"] == se.src.json["ultrawideRGBTimes"][:n]
    assert se.json["poseTimes"] == [] and se.json["rgbTimes"] == [] and se.json["hasRGB"] is False
    assert se.json["gripperCalibrationRunName"] == ""
    assert not (se.out_dir / f"{se.result.name}_rgb.mp4").exists()
    assert validate_iphone_episode(se.out_dir) == []


def test_T5_uw_30fps_hold_decimated(sim_export):
    se = sim_export("uw_hold")
    assert se.result.status == "exported", se.result
    uw = se.json["ultrawideRGBTimes"]
    occ = np.flatnonzero([bool(s) for s in uw])
    gaps = np.diff(occ)
    assert len(occ) > 10
    assert set(gaps.tolist()) <= {5, 6, 7}, sorted(set(gaps.tolist()))
    assert all(uw[i] == se.json["poseTimes"][i] for i in occ)  # one per slot, slot time string
    assert validate_iphone_episode(se.out_dir) == []
