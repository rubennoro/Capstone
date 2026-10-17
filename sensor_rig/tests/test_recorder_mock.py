"""T13: the recorder end to end with mock backends on a SimClock (faster than real time).

* 60 fps main + 30 fps ultrawide + 1 kHz static IMU, 3 s episode: frame counts within +-1 of expected, data.csv
  ascending, every listed file exists (and no unlisted ones), IMU covers [start - 1 s, stop + 1 s], RawEpisode reads
  it back, MockCamera.replay re-serves it.
* Slow encoder (50 ms per frame) + queue of 5: queue_full drops in drops.csv; CSV rows == files; every frame is
  either written or dropped.
* Sensor gaps (frames lost before userspace) -> sensor_gap rows. Live abort (stalled main camera) deletes / keeps the
  episode. Gripper-calibration bookkeeping.
"""

from __future__ import annotations

import dataclasses
import json
import time
from pathlib import Path

import numpy as np
import pytest

from sensor_rig.drivers.base import SimClock
from sensor_rig.drivers.mock_camera import MockCamera
from sensor_rig.drivers.mock_imu import MockImu
from sensor_rig.raw_episode import RawEpisode
from sensor_rig.record.monitor import LiveAbortConfig
from sensor_rig.record.recorder import EpisodeAborted, Recorder, RecorderError

SPEED = 4.0
MAIN = (160, 120, 60.0)
UW = (128, 96, 30.0)


@pytest.fixture
def small_rig(test_rig):
    """The WP-A test rig with small 60/30 fps cameras (recorder tests don't need valid intrinsics for the size)."""
    cams = dict(test_rig.cameras)
    cams["main"] = dataclasses.replace(cams["main"], resolution=MAIN[:2], fps=MAIN[2])
    cams["ultrawide"] = dataclasses.replace(cams["ultrawide"], resolution=UW[:2], fps=UW[2])
    return test_rig.replace(cameras=cams)


def make_backends(clock, main_kw=None, uw_kw=None):
    cams = {
        "main": MockCamera.synthetic("main", *MAIN, clock=clock, **(main_kw or {})),
        "ultrawide": MockCamera.synthetic("ultrawide", *UW, clock=clock, **(uw_kw or {})),
    }
    imu = MockImu.static(rate_hz=1000, clock=clock)
    return cams, imu


def make_recorder(rig, root, clock, cams, imu, **kw):
    kw.setdefault("warmup_s", 1.2)          # >= IMU pre-roll, so the episode gets a full second of IMU before start
    kw.setdefault("monitor_period_s", 0.5)
    return Recorder(rig, root, cameras=cams, imu=imu, clock=clock, **kw)


def read_csv_rows(path: Path) -> list[list[str]]:
    lines = path.read_text().splitlines()
    return [ln.split(",") for ln in lines[1:] if ln.strip()]


def check_camera_dir(cam_dir: Path) -> np.ndarray:
    rows = read_csv_rows(cam_dir / "data.csv")
    t = np.array([int(r[0]) for r in rows], dtype=np.int64)
    assert np.all(np.diff(t) > 0), "data.csv not strictly ascending"
    listed = {r[1] for r in rows}
    assert all(r[1] == f"{r[0]}.jpg" for r in rows)
    on_disk = {p.name for p in (cam_dir / "data").iterdir()}
    assert listed == on_disk, "files and data.csv rows differ"
    meta_rows = read_csv_rows(cam_dir / "frames_meta.csv")
    assert len(meta_rows) == len(rows)
    return t


def record(rec, seconds, **ep_kw):
    ep_kw.setdefault("kind", "demonstration")
    ep_kw.setdefault("session", "rig-test")
    ep_kw.setdefault("side", "left")
    ep_kw.setdefault("allow_no_calibration", True)
    name = rec.start_episode(**ep_kw)
    t_end = rec.clock.now_ns() + int(seconds * 1e9)
    rec.clock.sleep_until(t_end)
    return name, rec.stop_episode()


def test_recorder_basic_episode(small_rig, tmp_path):
    clock = SimClock(speed=SPEED)
    cams, imu = make_backends(clock)
    rec = make_recorder(small_rig, tmp_path, clock, cams, imu)
    with rec:
        name, path = record(rec, 3.0)
    assert path == tmp_path / name[:10] / name and path.is_dir()
    assert not any(p.name.startswith(".") for p in (tmp_path / name[:10]).iterdir())
    meta = json.loads((path / "meta.json").read_text())
    assert meta["status"] == "complete" and meta["name"] == name
    start, stop = meta["record_start_mono_ns"], meta["record_stop_mono_ns"]
    assert stop - start >= 3e9
    for cam, (_, _, fps) in (("main", MAIN), ("ultrawide", UW)):
        t = check_camera_dir(path / f"cam_{cam}")
        expected = (stop - start) / 1e9 * fps
        assert abs(len(t) - expected) <= 1.0 + 1e-9, f"{cam}: {len(t)} frames, expected {expected:.1f}"
        assert t[0] >= start and t[-1] <= stop
        assert t[0] - start <= 1e9 / fps and stop - t[-1] <= 1e9 / fps
        assert read_csv_rows(path / f"cam_{cam}" / "drops.csv") == []
        s = meta["sensors"][cam]
        assert s["frames"] == len(t) and s["backend"] == "mock" and s["jpeg_quality"] == 90
        assert s["fps_measured"] == pytest.approx(fps, rel=0.02)
    # IMU covers [start - 1 s, stop + 1 s]
    ep = RawEpisode(path)
    t_imu, gyro, accel = ep.imu()
    assert t_imu[0] <= start - 1e9 + 2e6 and t_imu[0] >= start - 1e9
    assert t_imu[-1] >= stop + 1e9 - 2e6 and t_imu[-1] <= stop + 1e9
    assert np.all(np.diff(t_imu) > 0)
    np.testing.assert_allclose(np.diff(t_imu), 1e6, atol=1)
    np.testing.assert_allclose(accel.mean(axis=0), [0, 0, 9.80665], atol=0.01)
    assert meta["sensors"]["imu0"]["samples"] == len(t_imu)
    # reader round trip + clock / naming bookkeeping
    assert len(ep.camera_times("main")) == meta["sensors"]["main"]["frames"]
    assert ep.read_image("main", 0).shape == (MAIN[1], MAIN[0], 3)
    assert meta["clock"]["start"]["realtime_ns"] - meta["clock"]["start"]["monotonic_ns"] == clock.realtime_offset_ns
    assert meta["recording_start_time"].startswith("2026-10-20T15:")
    assert meta["gripper_calibration_run_name"] == "" and meta["sides_present"] == ["left"]
    assert (path / "rig.yaml").is_file() and (path / "logs" / "recorder.log").is_file()


def test_replay_recorded_episode(small_rig, tmp_path):
    clock = SimClock(speed=SPEED)
    cams, imu = make_backends(clock)
    with make_recorder(small_rig, tmp_path, clock, cams, imu) as rec:
        _, path = record(rec, 1.0)
    t_rec = RawEpisode(path).camera_times("ultrawide")
    cam = MockCamera.replay(path, "ultrawide", realtime=False)
    with cam:
        frames = []
        while (f := cam.read(timeout_s=1.0)) is not None:
            frames.append(f)
    assert len(frames) == len(t_rec)
    np.testing.assert_array_equal(np.array([f.t_ns for f in frames]) - frames[0].t_ns, t_rec - t_rec[0])
    assert frames[0].image.shape == (UW[1], UW[0], 3)
    assert len(cam.read_recent(4)) == 4 and cam.latest().seq == len(t_rec) - 1
    cam2 = MockCamera.replay(path, "ultrawide", realtime=False, preserve_timestamps=True)
    with cam2:
        assert cam2.read().t_ns == t_rec[0]
    cam3 = MockCamera.replay(path, "ultrawide", realtime=True, loop=True, clock=SimClock(speed=50.0))
    with cam3:
        got = [cam3.read(timeout_s=1.0) for _ in range(len(t_rec) + 3)]
    t3 = np.array([f.t_ns for f in got])
    assert np.all(np.diff(t3) > 0) and got[len(t_rec)].seq == len(t_rec)


class SlowEncoder:
    def __init__(self, delay_s=0.05):
        from sensor_rig.record.jpeg import OpenCvJpeg

        self.inner = OpenCvJpeg(90)
        self.quality = 90
        self.delay_s = delay_s

    def encode(self, img):
        time.sleep(self.delay_s)
        return self.inner.encode(img)


def test_recorder_slow_encoder_queue_full(small_rig, tmp_path):
    clock = SimClock(speed=SPEED)
    cams, imu = make_backends(clock)
    rec = make_recorder(small_rig, tmp_path, clock, cams, imu, encoder=SlowEncoder(0.05), queue_frames=5,
                        live_abort=LiveAbortConfig(max_queue_full_frac=1.1))
    with rec:
        name, path = record(rec, 3.0)
    meta = json.loads((path / "meta.json").read_text())
    for cam in ("main", "ultrawide"):
        d = path / f"cam_{cam}"
        t = check_camera_dir(d)
        drops = read_csv_rows(d / "drops.csv")
        qf = [r for r in drops if r[2] == "queue_full"]
        assert len(qf) > 0, f"{cam}: expected queue_full drops"
        assert {r[2] for r in drops} <= {"queue_full"}
        # every episode frame is either written or dropped, each ep_seq exactly once
        seqs = sorted([int(r[0]) for r in read_csv_rows(d / "frames_meta.csv")] + [int(r[0]) for r in qf])
        assert seqs == list(range(len(seqs)))
        assert meta["sensors"][cam]["drops"]["queue_full"] == len(qf)
        assert meta["sensors"][cam]["frames"] == len(t)


def test_recorder_sensor_gaps(small_rig, tmp_path):
    clock = SimClock(speed=SPEED)
    # source frames 150..152 and 200 are lost before userspace (~2.5 s and ~3.3 s after the camera starts)
    cams, imu = make_backends(clock, main_kw={"drop_seqs": {150, 151, 152, 200}})
    with make_recorder(small_rig, tmp_path, clock, cams, imu, warmup_s=1.2) as rec:
        _, path = record(rec, 3.0)
    drops = read_csv_rows(path / "cam_main" / "drops.csv")
    gaps = [(int(r[3])) for r in drops if r[2] == "sensor_gap"]
    assert sorted(gaps) == [1, 3]
    t = check_camera_dir(path / "cam_main")
    assert int(np.max(np.diff(t))) > 3.5e9 / 60


def test_live_abort_stalled_main(small_rig, tmp_path):
    clock = SimClock(speed=SPEED)
    # main delivers ~1.6 s of frames, then stalls (warm-up 1.2 s passes, the episode is aborted mid-way)
    cams, imu = make_backends(clock, main_kw={"n_frames": 96})
    rec = make_recorder(small_rig, tmp_path, clock, cams, imu, keep_aborted=True)
    with rec:
        name = rec.start_episode("demonstration", "rig-test", "left", allow_no_calibration=True)
        clock.sleep(3.0)
        assert not rec.recording
        with pytest.raises(EpisodeAborted, match="main_fps_low"):
            rec.stop_episode()
    date = tmp_path / name[:10]
    assert not (date / name).exists()
    kept = date / f".{name}.aborted"
    assert kept.is_dir()
    meta = json.loads((kept / "meta.json").read_text())
    assert meta["status"] == "aborted" and meta["abort_reason"].startswith("main_fps_low")
    assert rec.episodes[-1]["status"] == "aborted"


def test_warmup_refuses_dead_camera(small_rig, tmp_path):
    clock = SimClock(speed=SPEED)
    cams, imu = make_backends(clock, uw_kw={"n_frames": 3})
    rec = make_recorder(small_rig, tmp_path, clock, cams, imu, warmup_s=0.8)
    with pytest.raises(RecorderError, match="ultrawide"):
        rec.open()


def test_calibration_bookkeeping_and_abort_delete(small_rig, tmp_path):
    clock = SimClock(speed=SPEED)
    cams, imu = make_backends(clock)
    with make_recorder(small_rig, tmp_path, clock, cams, imu) as rec:
        with pytest.raises(RecorderError, match="no gripper calibration"):
            rec.start_episode("demonstration", "rig test", "right")
        cal_name, cal_path = record(rec, 0.5, kind="gripper_calibration", session="rig test", side="right",
                                    allow_no_calibration=False)
        assert "_rig-test_grippercalibration_right" in cal_name
        demo = rec.start_episode("demonstration", "rig test", "right")
        rec.clock.sleep(0.3)
        rec.abort_episode("test abort")                      # keep=False -> deleted
        assert not any(p.name.startswith(f".{demo}") or p.name == demo for p in (tmp_path / demo[:10]).iterdir())
        name, path = record(rec, 0.5, session="rig test", side="right", allow_no_calibration=False)
        head, _ = record(rec, 0.3, session="rig test", side="head", allow_no_calibration=False)
        with pytest.raises(RecorderError):
            rec.start_episode("demonstration", "no-session", "left")
    meta = json.loads((path / "meta.json").read_text())
    assert meta["gripper_calibration_run_name"] == cal_name
    assert meta["kind"] == "demonstration" and meta["session"] == "rig-test" and meta["side"] == "right"
    cal_meta = json.loads((cal_path / "meta.json").read_text())
    assert cal_meta["kind"] == "gripper_calibration" and cal_meta["gripper_calibration_run_name"] == ""
    state = json.loads((tmp_path / ".sensor_rig_state.json").read_text())
    assert state == {"rig-test|right|default": cal_name}


def test_resolution_mismatch_refused(small_rig, tmp_path):
    clock = SimClock(speed=SPEED)
    cams = {"main": MockCamera.synthetic("main", 100, 100, 60, clock=clock),
            "ultrawide": MockCamera.synthetic("ultrawide", *UW, clock=clock)}
    rec = make_recorder(small_rig, tmp_path, clock, cams, MockImu.static(clock=clock))
    with pytest.raises(RecorderError, match="calibration resolution"):
        rec.open()


def test_stream_writer_drops_nonmonotonic_stamps(test_rig, tmp_path):
    """F3: duplicate / backwards stamps are dropped before encoding, so no row ever points at an overwritten or
    deleted JPEG."""
    from sensor_rig.drivers.base import Frame
    from sensor_rig.raw_episode import RawEpisodeWriter
    from sensor_rig.record.jpeg import OpenCvJpeg
    from sensor_rig.record.stream_writer import CameraStreamWriter

    name = "2026-10-20T15-00-00.000Z_12345_rig-test_demonstration_left"
    w = RawEpisodeWriter(tmp_path, name, test_rig, {})
    sw = CameraStreamWriter("main", w, OpenCvJpeg(90), 30.0, start_ns=0, n_workers=2)
    stamps = [1000, 2000, 2000, 1500, 3000]
    imgs = [np.full((24, 32, 3), 40 * i, np.uint8) for i in range(len(stamps))]
    res = [sw.submit(Frame(img, t, i, t + 10, {})) for i, (img, t) in enumerate(zip(imgs, stamps))]
    assert res == ["queued", "queued", "dropped", "dropped", "queued"]
    st = sw.finish()
    assert st["drops"]["nonmonotonic_ts"] == 2
    path = w.finalize()
    t = check_camera_dir(path / "cam_main")
    assert list(t) == [1000, 2000, 3000]
    ep = RawEpisode(path)
    assert ep.read_image("main", 1)[0, 0, 0] == pytest.approx(40, abs=3)     # frame 1's JPEG, not frame 2's
    drops = read_csv_rows(path / "cam_main" / "drops.csv")
    assert len(drops) == 2
    seqs = sorted([int(r[0]) for r in read_csv_rows(path / "cam_main" / "frames_meta.csv")] +
                  [int(r[0]) for r in drops])
    assert seqs == list(range(len(stamps)))


class _BrokenImu(MockImu):
    """Static IMU that emits one batch with a repeated stamp ``fail_after_s`` after start (a backend bug)."""

    def __init__(self, clock, fail_after_s):
        base = MockImu.static(rate_hz=1000, clock=clock)
        super().__init__(1000, base._sample_fn, clock=clock)
        self.fail_after_ns = int(fail_after_s * 1e9)
        self._broken = False

    def _push(self, batch):
        if not self._broken and self.t_start_ns is not None and len(batch) > 1 and \
                batch.t_ns[0] - self.t_start_ns > self.fail_after_ns:
            batch.t_ns = batch.t_ns.copy()
            batch.t_ns[1] = batch.t_ns[0]
            self._broken = True
        super()._push(batch)


def test_imu_pump_failure_aborts_episode(small_rig, tmp_path):
    """F7: an exception in the IMU pump (here ImuWriter rejecting a non-increasing stamp) is recorded and aborts the
    episode cleanly instead of killing the pump silently."""
    clock = SimClock(speed=SPEED)
    cams, _ = make_backends(clock)
    imu = _BrokenImu(clock, fail_after_s=2.0)
    rec = make_recorder(small_rig, tmp_path, clock, cams, imu, keep_aborted=True)
    with rec:
        name = rec.start_episode("demonstration", "rig-test", "left", allow_no_calibration=True)
        clock.sleep(2.0)
        assert not rec.recording
        assert "imu" in rec.pump_errors and "strictly increasing" in rec.pump_errors["imu"]
        assert rec.status["pump_errors"]["imu"]
        with pytest.raises(EpisodeAborted, match="imu_error"):
            rec.stop_episode()
        with pytest.raises(RecorderError, match="imu"):
            rec.start_episode("demonstration", "rig-test", "left", allow_no_calibration=True)
    kept = tmp_path / name[:10] / f".{name}.aborted"
    meta = json.loads((kept / "meta.json").read_text())
    assert meta["status"] == "aborted" and meta["abort_reason"].startswith("imu_error")


def test_replay_keeps_camera_offsets(small_rig, tmp_path):
    """Cameras replayed from one episode keep their original main <-> ultrawide relative timing (shared origin and
    live start), in live (realtime) and offline replays, via replay_set and via make_camera."""
    from sensor_rig.drivers.base import make_camera

    clock = SimClock(speed=SPEED)
    cams, imu = make_backends(clock)
    with make_recorder(small_rig, tmp_path, clock, cams, imu) as rec:
        _, path = record(rec, 1.0)
    ep = RawEpisode(path)
    rec_main, rec_uw = ep.camera_times("main"), ep.camera_times("ultrawide")
    offset_rec = int(rec_uw[0] - rec_main[0])

    rec_t = {"main": rec_main, "ultrawide": rec_uw}
    origin = min(int(rec_main[0]), int(rec_uw[0]))

    def first_two(cams_):
        """Live stamp of each camera's first delivered frame, mapped back to the episode origin via its source
        sequence number (``Frame.seq`` = source frame index), so a frame lost to the bounded queue under CPU load
        cannot fake an offset: returns {cam: t_live - (t_rec[seq] - origin)}, which must equal the shared live
        start for every camera."""
        for c in cams_.values():
            c.start()
        try:
            out = {}
            for n, c in cams_.items():
                f = c.read(timeout_s=2.0)
                out[n] = int(f.t_ns) - (int(rec_t[n][f.seq]) - origin)
            return out
        finally:
            for c in cams_.values():
                c.stop()

    assert offset_rec == int(rec_uw[0]) - int(rec_main[0])
    live_clock = SimClock(speed=20.0)
    got = first_two(MockCamera.replay_set(path, clock=live_clock, realtime=True))
    assert abs(got["ultrawide"] - got["main"]) < 1_000_000      # same live origin -> recorded offsets kept
    start = live_clock.now_ns() + 50_000_000
    via_factory = {n: make_camera(small_rig, n, backend="mock", episode_dir=path, clock=live_clock, realtime=True,
                                  restamp=True, start_now_ns=start) for n in ("main", "ultrawide")}
    got = first_two(via_factory)
    assert abs(got["ultrawide"] - got["main"]) < 1_000_000
    assert got["main"] == start and got["ultrawide"] == start   # the episode origin maps to start_now_ns
    # synthetic mocks ignore replay-only options instead of raising
    syn = make_camera(small_rig, "main", backend="mock", clock=live_clock, realtime=True, loop=True, restamp=True,
                      start_now_ns=start)
    assert isinstance(syn, MockCamera) and syn.realtime
