"""T12: raw episode writer/reader round trip (+ clock/timefmt basics)."""

import json
import random

import numpy as np
import pytest

from sensor_rig import clock, timefmt
from sensor_rig.raw_episode import (
    CAM_DIRS,
    EUROC_IMU_HEADER,
    RAW_FORMAT,
    RawEpisode,
    RawEpisodeWriter,
    default_meta,
    find_raw_episodes,
    iter_raw_episodes,
)

NAME = "2026-10-20T15-02-11.123Z_48213_rig-pilot_demonstration_left"


@pytest.fixture
def rig_yaml(tmp_path):
    p = tmp_path / "some_rig.yaml"
    p.write_text("schema_version: 1\nrig_name: test\n")
    return p


def _write_frames(w, cam, times):
    cw = w.camera_writer(cam)
    for k, t in enumerate(times):
        p = w.frame_path(cam, t)
        p.write_bytes(b"\xff\xd8fakejpeg")
        cw.add(k, t, p.name, {"t_arrival_ns": t + 25_000_000, "bytes": 9})
    return cw


def test_round_trip(tmp_path, rig_yaml):
    root = tmp_path / "raw"
    c = clock.sample_clocks()
    w = RawEpisodeWriter(root, NAME, rig_yaml, {"clock": {"start": c.to_dict()}, "note": "hello"})
    hidden = root / "2026-10-20" / f".{NAME}.recording"
    assert hidden.is_dir() and not (root / "2026-10-20" / NAME).exists()
    assert list(iter_raw_episodes(root)) == []  # hidden while recording

    tm = 1_000_000_000 + np.arange(10, dtype=np.int64) * 16_666_667
    tu = 1_000_000_000 + np.arange(4, dtype=np.int64) * 33_333_333
    _write_frames(w, "main", tm)
    _write_frames(w, "ultrawide", tu)
    w.add_drop("main", 11, int(tm[-1]) + 1, "queue_full")
    with pytest.raises(ValueError):  # non-increasing stamp
        p = w.frame_path("main", tm[3])
        w.camera_writer("main").add(99, int(tm[3]), p.name)
    with pytest.raises(FileNotFoundError):
        w.camera_writer("main").add(99, int(tm[-1]) + 5, "missing.jpg")

    ti = 900_000_000 + np.arange(2000, dtype=np.int64) * 1_000_000
    gyro = np.random.default_rng(0).normal(size=(2000, 3))
    acc = np.random.default_rng(1).normal(size=(2000, 3)) + [0, 0, 9.81]
    w.imu_writer().add_batch(ti[:1000], gyro[:1000], acc[:1000])
    w.imu_writer().add_batch(ti[1000:], gyro[1000:], acc[1000:], aux={"chip_us": np.arange(1000)})
    with pytest.raises(ValueError):
        w.imu_writer().add_batch(ti[:1], gyro[:1], acc[:1])
    w.update_meta(sensors={"main": {"backend": "mock"}})
    final = w.finalize(stats={"sensors": {"main": {"fps_measured": 60.0}}})

    assert final == root / "2026-10-20" / NAME and not hidden.exists()
    ep = RawEpisode(final)
    assert ep.name == NAME and ep.kind == "demonstration" and not ep.is_calibration
    assert ep.meta["format"] == RAW_FORMAT and ep.meta["status"] == "complete"
    assert ep.meta["note"] == "hello" and ep.meta["side"] == "left" and ep.meta["sides_present"] == ["left"]
    assert ep.meta["sensors"]["main"]["frames"] == 10 and ep.meta["sensors"]["main"]["backend"] == "mock"
    assert ep.meta["sensors"]["main"]["fps_measured"] == 60.0
    assert ep.meta["sensors"]["imu0"]["samples"] == 2000
    np.testing.assert_array_equal(ep.camera_times("main"), tm)
    np.testing.assert_array_equal(ep.camera_times("ultrawide"), tu)
    assert all(p.is_file() for p in ep.camera_files("main"))
    t, g, a = ep.imu()
    np.testing.assert_array_equal(t, ti)
    np.testing.assert_allclose(g, gyro, rtol=1e-8, atol=1e-12)
    np.testing.assert_allclose(a, acc, rtol=1e-8)
    assert (final / "imu0" / "data.csv").read_text().splitlines()[0] == EUROC_IMU_HEADER
    assert (final / CAM_DIRS["main"] / "data.csv").read_text().splitlines()[0] == "#timestamp [ns],filename"
    assert (final / CAM_DIRS["main"] / "drops.csv").read_text().splitlines()[1].endswith(",queue_full,1")
    assert (final / "rig.yaml").read_text() == rig_yaml.read_text()
    assert ep.wall_offset_ns() == c.realtime_ns - c.monotonic_ns
    assert ep.trajectory_path() is None
    assert [e.name for e in iter_raw_episodes(root)] == [NAME]
    assert [e.name for e in find_raw_episodes(final)] == [NAME]
    assert [e.name for e in find_raw_episodes(root / "2026-10-20")] == [NAME]


def test_meta_keys_complete(tmp_path, rig_yaml):
    w = RawEpisodeWriter(tmp_path, NAME, rig_yaml)
    final = w.finalize()
    meta = json.loads((final / "meta.json").read_text())

    def keys(d, prefix=""):
        out = set()
        for k, v in d.items():
            out.add(prefix + k)
            if isinstance(v, dict):
                out |= keys(v, prefix + k + ".")
        return out

    assert keys(default_meta()) <= keys(meta)
    # empty streams still have their files
    ep = RawEpisode(final)
    assert len(ep.camera_times("main")) == 0 and len(ep.imu()[0]) == 0


@pytest.mark.parametrize("keep", [False, True])
def test_abort(tmp_path, rig_yaml, keep):
    w = RawEpisodeWriter(tmp_path, NAME, rig_yaml)
    _write_frames(w, "main", [1, 2, 3])
    w.abort("test", keep=keep)
    date = tmp_path / "2026-10-20"
    assert not (date / NAME).exists() and not (date / f".{NAME}.recording").exists()
    kept = date / f".{NAME}.aborted"
    assert kept.is_dir() == keep
    if keep:
        assert json.loads((kept / "meta.json").read_text())["abort_reason"] == "test"
    assert list(iter_raw_episodes(tmp_path)) == []


def test_bad_name(tmp_path, rig_yaml):
    with pytest.raises(ValueError):
        RawEpisodeWriter(tmp_path, "not_a_valid_name", rig_yaml)


def test_timefmt():
    ns = timefmt.parse_iso("2026-10-01T21:13:57.449Z")
    assert timefmt.iso_ms(ns) == "2026-10-01T21:13:57.449Z"
    assert timefmt.iso_ms(ns + 999_999) == "2026-10-01T21:13:57.449Z"  # floor
    assert timefmt.iso_ms(ns - 1) == "2026-10-01T21:13:57.448Z"
    assert timefmt.ISO_RE.match(timefmt.iso_ms(0))
    assert timefmt.sanitize_session("my_session 2!") == "my-session-2"
    with pytest.raises(ValueError):
        timefmt.sanitize_session("no-session")
    rid = timefmt.random_id(random.Random(0))
    assert len(rid) == 5 and "0" not in rid
    name = timefmt.make_episode_name(ns, rid, "firstreal", "gripper_calibration", "left")
    assert name == f"2026-10-01T21-13-57.449Z_{rid}_firstreal_grippercalibration_left"
    p = timefmt.parse_episode_name(name)
    assert p["kind"] == "gripper_calibration" and p["date"] == "2026-10-01" and p["side"] == "left"
    with pytest.raises(ValueError):
        timefmt.parse_episode_name("a_b_c_d")


def test_clock():
    s = clock.sample_clocks(5)
    assert s.uncertainty_ns >= 0 and abs(s.monotonic_ns - clock.now_ns()) < 1e9
    a = clock.ClockSample(1000, 500, 0, 0)
    b = clock.ClockSample(1000 + 1_000_000_000, 500 + 999_000_000, 0, 0)
    m = clock.RawToMono(a, b)
    assert m(500) == 1000 and m(500 + 999_000_000) == 1000 + 1_000_000_000
    assert m(np.array([500], dtype=np.int64))[0] == 1000
    assert clock.RawToMono(a)(600) == 1100
