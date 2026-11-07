"""dataproc.pi_adapter: Pi-format episodes -> contract episodes (docs/PI_ADAPTER.md)."""

from __future__ import annotations

import json

import numpy as np
import pytest

from dataproc.pi_adapter import (DEFAULT_EXPOSURE_NS, PiEpisodeError, adapt_into, append_sources, converted_name,
                                 inspect_source, read_sources, scan_sources)
from dataproc.raw_episode import COMPLETE, VERIFIED, RawEpisode
from fixtures import build_episode
from pi_fixtures import contract_to_pi, write_pi_manifest
from test_process_common import tree_digest


@pytest.fixture
def contract(tmp_path):
    return build_episode(tmp_path / "contract" / "c", n_cam0=12, n_cam1=6)


@pytest.fixture
def pi(tmp_path, contract):
    return contract_to_pi(contract, tmp_path / "src" / "episode_000007", 7)


def _adapt(pi, tmp_path):
    inc = tmp_path / "incoming"
    inc.mkdir(exist_ok=True)
    return adapt_into(inspect_source(pi), inc, "rig_v1")


def test_converted_episode_passes_the_contract(tmp_path, pi, contract):
    out = _adapt(pi, tmp_path)
    assert out.name == converted_name(inspect_source(pi)) and out.name.endswith("_episode_000007")
    assert (out / VERIFIED).is_file() and (out / COMPLETE).is_file()
    rep = RawEpisode.open(out).validate(decode_videos=True)
    assert rep.ok, [p.message for p in rep.problems]
    assert not rep.problems
    assert not (tmp_path / "incoming" / ".adapt-tmp").exists()


def test_roles_follow_the_sensor_not_the_pi_file_name(tmp_path, pi, contract):
    out = _adapt(pi, tmp_path)
    m = json.loads((out / "meta.json").read_text())
    assert m["cameras"]["cam0"]["role"] == "main" and m["cameras"]["cam0"]["sensor"] == "imx519"
    assert m["cameras"]["cam1"]["role"] == "ultrawide" and m["cameras"]["cam1"]["sensor"] == "imx219"
    # the Pi's cam1.mp4 is the IMX519 (main): it must land as contract cam0.mp4
    assert (out / "cam0.mp4").read_bytes() == (pi / "cam1.mp4").read_bytes() == (contract / "cam0.mp4").read_bytes()
    assert (out / "cam1.mp4").read_bytes() == (contract / "cam1.mp4").read_bytes()


def test_imu_counts_become_si_units(tmp_path, pi, contract):
    out = _adapt(pi, tmp_path)
    a, b = RawEpisode.open(contract).imu, RawEpisode.open(out).imu
    np.testing.assert_array_equal(a.t_ns, b.t_ns)
    np.testing.assert_array_equal(a.sample_idx, b.sample_idx)
    np.testing.assert_allclose(b.gyro, a.gyro, atol=0.0175 * np.pi / 180)       # one LSB
    np.testing.assert_allclose(b.accel, a.accel, atol=0.000122 * 9.80665)


def test_meta_fields(tmp_path, pi):
    out = _adapt(pi, tmp_path)
    m = json.loads((out / "meta.json").read_text())
    raw = RawEpisode.open(out)
    starts = [raw.frames(c).t_ns[0] for c in ("cam0", "cam1")] + [raw.imu.t_ns[0]]
    stops = [raw.frames(c).t_ns[-1] for c in ("cam0", "cam1")] + [raw.imu.t_ns[-1]]
    assert m["button_start_ns"] == max(starts) and m["button_stop_ns"] == min(stops)   # whole recording
    assert m["clock"] == "CLOCK_BOOTTIME" and m["rig_id"] == "rig_v1" and m["episode_number"] == 7
    assert m["session_id"].startswith("pi-")
    assert m["pi_meta"]["episode"] == 7
    assert "cam0.exposure_ns" in m["adapter"]["synthesized"]
    assert set(raw.frames("cam0").exposure_ns) == {min(DEFAULT_EXPOSURE_NS, (10**9 // 60) // 2)}


def test_exposure_column_is_used_when_the_pi_records_it(tmp_path, contract):
    pi = contract_to_pi(contract, tmp_path / "src" / "episode_000001", 1, with_exposure=True)
    out = _adapt(pi, tmp_path)
    np.testing.assert_array_equal(RawEpisode.open(out).frames("cam0").exposure_ns,
                                  RawEpisode.open(contract).frames("cam0").exposure_ns)
    assert "cam0.exposure_ns" not in json.loads((out / "meta.json").read_text())["adapter"]["synthesized"]


def test_a_dropped_frame_becomes_a_sensor_seq_gap(tmp_path, pi):
    csv = pi / "cam1_frames.csv"                 # the Pi's cam1 = main, 60 fps
    lines = csv.read_text().splitlines()
    del lines[5]                                 # frame 4 lost: renumber the rest as the Pi would
    lines = [lines[0]] + [f"{i},{ln.split(',')[1]}" for i, ln in enumerate(lines[1:])]
    csv.write_text("\n".join(lines) + "\n")
    m = json.loads((pi / "meta.json").read_text())
    for c in m["cameras"]:
        if c["file"] == "cam1.mp4":
            c["frames"] -= 1
    (pi / "meta.json").write_text(json.dumps(m))
    write_pi_manifest(pi)
    out = _adapt(pi, tmp_path)
    ft = RawEpisode.open(out).frames("cam0")
    assert list(np.diff(ft.sensor_seq)).count(2) == 1
    assert json.loads((out / "meta.json").read_text())["cameras"]["cam0"]["dropped_frames"] == 1
    rep = RawEpisode.open(out).validate(decode_videos=False)
    assert rep.ok and "sensor_seq_gap" in rep.codes()


@pytest.mark.parametrize("breakage, problem", [
    ("incomplete", "no COMPLETE"),
    ("hash", "manifest check failed"),
    ("invalid", "marked it invalid"),
    ("problems", "reported problems"),
])
def test_bad_sources_are_refused(tmp_path, contract, breakage, problem):
    kw = {"incomplete": {"complete": False}, "invalid": {"valid": False},
          "problems": {"problems": ["imu gap"]}}.get(breakage, {})
    pi = contract_to_pi(contract, tmp_path / "src" / "episode_000001", 1, **kw)
    if breakage == "hash":
        with open(pi / "imu.csv", "a") as f:
            f.write("\n")
    src = inspect_source(pi)
    assert not src.ok and problem in src.problem
    inc = tmp_path / "incoming"
    inc.mkdir()
    with pytest.raises(PiEpisodeError):
        adapt_into(src, inc, "rig_v1")
    assert list(inc.iterdir()) == []


def test_unknown_camera_model_is_refused(tmp_path, pi):
    m = json.loads((pi / "meta.json").read_text())
    m["cameras"][0]["model"] = "imx477"
    (pi / "meta.json").write_text(json.dumps(m))
    write_pi_manifest(pi)
    inc = tmp_path / "incoming"
    inc.mkdir()
    with pytest.raises(PiEpisodeError, match="imx477"):
        adapt_into(inspect_source(pi), inc, "rig_v1")
    assert list(inc.iterdir()) == []


def test_source_is_never_modified(tmp_path, pi):
    before = tree_digest(pi.parent)
    _adapt(pi, tmp_path)
    scan_sources(pi.parent)
    assert tree_digest(pi.parent) == before


def test_source_records_roundtrip(tmp_path):
    p = tmp_path / "pi_sources.csv"
    assert read_sources(p) == []
    append_sources(p, [{"source_name": "a", "source_manifest_sha256": "x", "converted_name": "pi-a",
                        "task": "t", "kind": "demonstration", "status": "ingested", "adapted_unix_ns": 1}])
    append_sources(p, [{"source_name": "b", "source_manifest_sha256": "y", "task": "t"}])
    rows = read_sources(p)
    assert [r["source_name"] for r in rows] == ["a", "b"] and rows[0]["status"] == "ingested"
