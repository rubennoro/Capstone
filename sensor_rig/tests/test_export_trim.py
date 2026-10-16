"""T3 (D10b): VIO not yet initialised -> head trimmed; tracking lost mid-episode -> episode aborted."""

from __future__ import annotations

import json

import pytest

from fixtures_sim import *  # noqa: F401,F403
from sensor_rig.export.validate import validate_iphone_episode
from sensor_rig.timefmt import parse_iso

pytestmark = pytest.mark.data


def test_T3_init_delay_trims_head(sim_export):
    se = sim_export("init_delay")
    ref = sim_export("demo60")
    assert se.result.status == "exported", se.result
    t0 = parse_iso(se.src.json["rgbTimes"][0])
    first = parse_iso(se.json["poseTimes"][0])
    assert first >= t0 + 500_000_000
    assert first - (t0 + 500_000_000) < 20_000_000  # trimmed to the first slot with a pose, not more
    assert se.json["frameCount"] < ref.json["frameCount"]
    assert se.result.trim_head_s >= 0.5 - 1e-3
    uw = se.json["ultrawideRGBTimes"]
    assert len(uw) == se.json["frameCount"]
    assert any(uw) and all(s in ("", p) for s, p in zip(uw, se.json["poseTimes"]))
    assert validate_iphone_episode(se.out_dir) == []
    # the original iPhone UW frames inside the kept range are all present, in the same slots
    kept = set(se.json["poseTimes"])
    orig_uw = [s for s in se.src.json["ultrawideRGBTimes"][: len(se.truth["t_ns"])] if s and s in kept]
    assert [s for s in uw if s] == orig_uw


def test_T3_mid_episode_gap_aborts(sim_export):
    se = sim_export("gap")
    assert se.result.status == "aborted"
    assert se.result.reason == "pose_gap_mid_episode"
    assert not (se.out_root / se.result.name[:10] / se.result.name).exists()
    assert not any(p.name.startswith(".partial-") for p in se.out_root.iterdir())
    log = se.out_root / ".sensor_rig" / "export_log.jsonl"
    lines = [json.loads(x) for x in log.read_text().splitlines()]
    assert any(r["name"] == se.result.name and r["reason"] == "pose_gap_mid_episode" for r in lines)
