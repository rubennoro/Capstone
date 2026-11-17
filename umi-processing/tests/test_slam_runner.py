"""umiproc.slam.runner: command, checks, output clearing and the result, with an injected runner (no native binary),
plus the real subprocess path with a stand-in script."""

import json
import os
import shutil
import stat
import subprocess
from pathlib import Path

import numpy as np
import pytest

from umiproc.slam.log import SUMMARY_FORMAT
from umiproc.slam.runner import (SlamRunError, default_binary, default_vocabulary, run_umi_slam,
                                 subprocess_runner)

FIX = Path(__file__).parent / "fixtures" / "slam"
T0 = 5_000_000_000
DT = 33_366_700


@pytest.fixture()
def env(tmp_path):
    """A fake fork folder (binary + vocabulary), a 40-frame EuRoC folder, settings and a mask."""
    fork = tmp_path / "fork"
    (fork / "umiproc_build").mkdir(parents=True)
    (fork / "Vocabulary").mkdir()
    binary = fork / "umiproc_build" / "umiproc_slam"
    binary.write_text("#!/bin/sh\nexit 99\n")
    binary.chmod(binary.stat().st_mode | stat.S_IEXEC)
    (fork / "Vocabulary" / "ORBvoc.txt").write_text("voc")
    seq = tmp_path / "seq"
    (seq / "mav0" / "cam0" / "data").mkdir(parents=True)
    (seq / "mav0" / "imu0").mkdir(parents=True)
    times = T0 + DT * np.arange(40, dtype=np.int64)
    (seq / "times.txt").write_text("".join(f"{t}\n" for t in times))
    (seq / "mav0" / "imu0" / "data.csv").write_text("#t\n")
    settings = tmp_path / "settings.yaml"
    settings.write_text("%YAML:1.0\n")
    mask = tmp_path / "mask.png"
    mask.write_bytes(b"png")
    return {"tmp": tmp_path, "fork": fork, "binary": binary, "voc": fork / "Vocabulary" / "ORBvoc.txt",
            "seq": seq, "times": times, "settings": settings, "mask": mask, "work": tmp_path / "work"}


def _summary_line(**over) -> str:
    d = {"format": SUMMARY_FORMAT, "exit_code": 0, "error": None, "frames_total": 40, "frames_fed": 40,
         "lost_frames": 0, "not_tracking_frames": 0, "stopped_early": False, "stop_reason": None,
         "state_counts": {"2": 40}, "maps": [{"id": 0, "keyframes": 3, "map_points": 9, "imu_initialized": True,
                                              "inertial_ba1": True, "inertial_ba2": True}],
         "t0_ns": T0, "map_removed": False, "options": {},
         "pacing": {"track_ms": {"median": 20.0, "p95": 25.0, "max": 30.0}, "frames_over_period": 0,
                    "feed_rate_hz": 29.9, "recording_rate_hz": 30.0, "feed_wall_s": 1.3}}
    d.update(over)
    return "UMIPROC_SUMMARY " + json.dumps(d) + "\n"


class FakeRunner:
    """Records the call; writes a log, the CSV (all frames tracked) and t0, and optionally the map."""

    def __init__(self, rc=0, write_map=False, summary=True, frames=40, raise_timeout=False):
        self.rc, self.write_map, self.summary, self.frames, self.raise_timeout = rc, write_map, summary, frames, raise_timeout
        self.calls = []

    def __call__(self, cmd, cwd, log_path, timeout_s):
        self.calls.append((list(cmd), Path(cwd), Path(log_path), timeout_s))
        assert Path(cwd).is_dir()
        args = list(cmd)
        out = Path(args[args.index("--output_trajectory_csv") + 1])
        if self.write_map and "--save_map" in args:
            Path(args[args.index("--save_map") + 1]).write_bytes(b"atlas")
        if self.raise_timeout:
            raise subprocess.TimeoutExpired(args, timeout_s)
        rows = "".join(f"{k},{k * DT / 1e9:.6f},2,false,false,0,0,0,0,0,0,1\n" for k in range(self.frames))
        out.write_text("frame_idx,timestamp,state,is_lost,is_keyframe,x,y,z,q_x,q_y,q_z,q_w\n" + rows)
        Path(str(out) + ".t0_ns").write_text(f"{T0}\n")
        log = "frames 40\nINIT_RELOCALIZE success!\npKFCur->mTimeStamp0\n"
        if self.summary:
            log += _summary_line(frames_fed=self.frames, exit_code=self.rc)
        Path(log_path).write_text(log)
        return self.rc


def _run(env, runner, **kw):
    return run_umi_slam(env["seq"], env["settings"], env["work"], binary=env["binary"], vocabulary=env["voc"],
                        runner=runner, **kw)


def test_localize_command_and_result(env):
    fake = FakeRunner()
    loadmap = env["tmp"] / "map.osa"
    loadmap.write_bytes(b"atlas")
    res = _run(env, fake, load_map=loadmap, mask_img=env["mask"], max_lost_frames=30)
    (cmd, cwd, log_path, timeout_s), = fake.calls
    assert cwd == env["work"].resolve() and log_path == res.log_path
    assert cmd[0] == str(env["binary"].resolve())
    pairs = dict(zip(cmd[1::2], cmd[2::2]))
    assert pairs["--vocabulary"] == str(env["voc"].resolve()) and pairs["--setting"] == str(env["settings"].resolve())
    assert pairs["--sequence"] == str(env["seq"].resolve())
    assert pairs["--output_trajectory_csv"] == str(env["work"].resolve() / "trajectory.csv")
    assert pairs["--load_map"] == str(loadmap.resolve()) and pairs["--mask_img"] == str(env["mask"].resolve())
    assert pairs["--max_lost_frames"] == "30"
    assert "--save_map" not in cmd and "--init_tag_id" not in cmd and "--no_realtime" not in cmd
    for flag in ("--vocabulary", "--setting", "--sequence", "--output_trajectory_csv", "--load_map", "--mask_img"):
        assert Path(pairs[flag]).is_absolute()
    assert timeout_s == pytest.approx(3.0 * 39 * DT / 1e9 + 300.0)
    assert res.ok and res.returncode == 0 and not res.timed_out and not res.map_saved
    assert res.summary.frames_fed == 40 and res.log.relocalization_times_s == [0.0]
    tr = res.trajectory()
    assert tr.n_tracked == 40 and np.array_equal(tr.t_ns, env["times"])


def test_map_command(env):
    fake = FakeRunner(write_map=True)
    res = _run(env, fake, save_map=env["tmp"] / "map.osa", init_tag_id=13, init_tag_size=0.16, aruco_dict_id=0,
               realtime=False, timeout_s=12)
    cmd = fake.calls[0][0]
    pairs = dict(zip(cmd[1::2], cmd[2::2]))
    assert pairs["--save_map"] == str((env["tmp"] / "map.osa").resolve())
    assert pairs["--init_tag_id"] == "13" and pairs["--init_tag_size"] == "0.16" and pairs["--aruco_dict_id"] == "0"
    assert cmd[-1] == "--no_realtime" and fake.calls[0][3] == 12
    assert res.ok and res.map_saved


def test_old_outputs_cleared_before_the_run(env):
    env["work"].mkdir()
    old_csv = env["work"] / "trajectory.csv"
    old_csv.write_text("old")
    Path(str(old_csv) + ".t0_ns").write_text("1")
    (env["work"] / "umiproc_slam.log").write_text(_summary_line())
    old_map = env["tmp"] / "map.osa"
    old_map.write_bytes(b"old atlas")

    def crash(cmd, cwd, log_path, timeout_s):      # dies before writing anything
        return -11
    res = _run(env, crash, save_map=old_map)
    assert not old_csv.exists() and not Path(str(old_csv) + ".t0_ns").exists() and not old_map.exists()
    assert res.summary is None and not res.ok and res.returncode == -11
    assert res.log.events == ()
    with pytest.raises(SlamRunError, match="no summary"):
        res.trajectory()


@pytest.mark.parametrize("rc", [1, 3])
def test_failed_run_removes_the_map(env, rc):
    res = _run(env, FakeRunner(rc=rc, write_map=True), save_map=env["tmp"] / "map.osa")
    assert res.returncode == rc and not res.ok and not res.map_saved and not (env["tmp"] / "map.osa").exists()
    assert res.summary.exit_code == rc


def test_timeout(env):
    res = _run(env, FakeRunner(write_map=True, raise_timeout=True), save_map=env["tmp"] / "map.osa")
    assert res.timed_out and res.returncode is None and not res.ok and not res.map_saved


def test_stopped_early_trajectory_uses_frames_fed(env):
    res = _run(env, FakeRunner(rc=3, frames=25))
    tr = res.trajectory()
    assert tr.n_frames == 25 and tr.n_frames_total == 40


@pytest.mark.parametrize("what,match", [
    ("binary", "umiproc_slam binary not found"),
    ("voc", "ORB vocabulary not found"),
    ("times", "times.txt not found"),
    ("imu", "IMU file not found"),
    ("frames", "frame folder not found"),
    ("settings", "settings not found"),
])
def test_missing_inputs(env, what, match):
    target = {"binary": env["binary"], "voc": env["voc"], "times": env["seq"] / "times.txt",
              "imu": env["seq"] / "mav0" / "imu0" / "data.csv", "settings": env["settings"],
              "frames": env["seq"] / "mav0" / "cam0" / "data"}[what]
    shutil.rmtree(target) if target.is_dir() else target.unlink()
    fake = FakeRunner()
    with pytest.raises(SlamRunError, match=match):
        _run(env, fake)
    assert fake.calls == []


def test_bad_map_and_mask_arguments(env):
    fake = FakeRunner()
    with pytest.raises(SlamRunError, match="map to load not found"):
        _run(env, fake, load_map=env["tmp"] / "nope.osa")
    with pytest.raises(SlamRunError, match="mask image not found"):
        _run(env, fake, mask_img=env["tmp"] / "nope.png")
    same = env["tmp"] / "m.osa"
    same.write_bytes(b"x")
    with pytest.raises(SlamRunError, match="same file"):
        _run(env, fake, load_map=same, save_map=same)
    with pytest.raises(SlamRunError, match="folder for the map"):
        _run(env, fake, save_map=env["tmp"] / "no_such_dir" / "m.osa")
    env["binary"].chmod(0o644)
    with pytest.raises(SlamRunError, match="not executable"):
        _run(env, fake)
    assert fake.calls == []


def test_default_paths_follow_the_environment(monkeypatch, tmp_path):
    monkeypatch.delenv("UMIPROC_SLAM_BINARY", raising=False)
    monkeypatch.setenv("ORB_SLAM3_UMI_DIR", str(tmp_path / "f"))
    assert default_binary() == tmp_path / "f" / "umiproc_build" / "umiproc_slam"
    assert default_vocabulary() == tmp_path / "f" / "Vocabulary" / "ORBvoc.txt"
    monkeypatch.delenv("ORB_SLAM3_UMI_DIR")
    assert default_binary() == Path.home() / "ORB_SLAM3_umi" / "umiproc_build" / "umiproc_slam"


def test_real_subprocess_with_a_stand_in_binary(env):
    """The default runner: cwd, stdout+stderr into the log, the exit code, and a timeout that kills."""
    env["binary"].write_text(
        "#!/bin/sh\n"
        "pwd > cwd.txt\n"
        "echo to-stderr >&2\n"
        f"printf '%s' '{_summary_line(exit_code=3).strip()}'\n"
        "echo\n"
        "exit 3\n")
    res = _run(env, None)
    assert res.returncode == 3 and res.summary.exit_code == 3 and not res.ok
    assert (env["work"] / "cwd.txt").read_text().strip() == str(env["work"].resolve())
    assert "to-stderr" in res.log_path.read_text()
    env["binary"].write_text("#!/bin/sh\nexec sleep 30\n")
    res = run_umi_slam(env["seq"], env["settings"], env["work"], binary=env["binary"], vocabulary=env["voc"],
                       timeout_s=0.5)
    assert res.timed_out and res.returncode is None


def test_subprocess_runner_direct(tmp_path):
    log = tmp_path / "log.txt"
    assert subprocess_runner(["/bin/sh", "-c", "echo hi; exit 2"], tmp_path, log, 10) == 2
    assert log.read_text() == "hi\n"
    assert os.path.getsize(log) == 3


def test_truncated_summary_keeps_the_result(env):
    def crash_mid_line(cmd, cwd, log_path, timeout_s):
        Path(log_path).write_text("INIT_RELOCALIZE success!\npKFCur->mTimeStamp0.5\nUMIPROC_SUMMARY {\"format\": \"umi")
        return -11
    res = _run(env, crash_mid_line)
    assert res.summary is None and "not JSON" in res.summary_error and res.returncode == -11
    assert res.log.relocalization_times_s == [0.5]


def test_variant_binary_from_the_environment(monkeypatch, env):
    """UMIPROC_SLAM_BINARY selects another build (e.g. umiproc_build_0006); the vocabulary stays the main one."""
    variant = env["tmp"] / "umiproc_build_0006" / "umiproc_slam"
    variant.parent.mkdir()
    shutil.copy2(env["binary"], variant)
    monkeypatch.setenv("UMIPROC_SLAM_BINARY", str(variant))
    monkeypatch.setenv("ORB_SLAM3_UMI_DIR", str(env["fork"]))
    assert default_binary() == variant and default_vocabulary() == env["voc"]
    fake = FakeRunner()
    res = run_umi_slam(env["seq"], env["settings"], env["work"], runner=fake)
    assert res.command[0] == str(variant) and res.command[res.command.index("--vocabulary") + 1] == str(env["voc"])
