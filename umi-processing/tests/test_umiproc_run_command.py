"""``umiproc run`` end to end without the native binary (a fake umiproc_slam runner, synthetic finger tags): the
localization run's options, waiting for the gripper calibration then finishing without a second SLAM run, a demo
split into two episodes (one raw copy), UMI's drops (> 5 lost, stopped, a new map) into ``failed/`` with reasons,
a piece dropped by a pose check, the copied finger-tag coverage, the out-of-date refusal, the skipped map and
calibration recordings, the CLI, and the effective rig in the copied stages."""

import csv
import json
import os

import numpy as np
import pytest
import yaml

from umiproc import cli as umicli
from umiproc.calibrate.gripper import calibrate_folder
from umiproc.paths import rig_file, rig_lens_file
from umiproc.run import command as RC
from umiproc.run import pieces as RP
from umiproc.run.recording import EPISODE, FAILED, LOCALIZE_FILE, PIECE_FILE, SKIPPED, WAITING
from umiproc.status import folder_status, format_folder_status
from umiproc_testlib import FakeSlam, clone_session, fake_finger_tags, session_template, tx_slam_tag_example

quiet = lambda *a, **k: None  # noqa: E731
LOST = 4                                     # Tracking::LOST


@pytest.fixture(scope="module")
def template(tmp_path_factory):
    """A passed session folder, demo-shaped recordings 1-4 and 20 (the calibration clip), built once."""
    return session_template(tmp_path_factory.mktemp("run_template"), demos=(1, 2, 3, 4, 20), n_frames=40)


@pytest.fixture
def env(template, tmp_path, monkeypatch):
    e = clone_session(template, tmp_path, monkeypatch)
    fake_finger_tags(monkeypatch)
    real = RP.pose_checks

    def pose_checks(*a, **k):                # the fixture IMU is noise: the copied scale check cannot measure it
        r = real(*a, **k)
        r["reasons"] = [x for x in r["reasons"] if x != "scale_unobservable"]
        r["ok"] = not r["reasons"]
        return r

    monkeypatch.setattr(RP, "pose_checks", pose_checks)
    return e


def moving(k, n):
    return (0.002 * k, 0.001 * k, 0.0, 0.0, 0.0, 0.0, 1.0)


class Slam(FakeSlam):
    """FakeSlam for the localization runs, by demo: ``lost`` = frame indices not tracked; ``stop_after`` = stop
    (exit 3, --max_lost_frames) after that many frames; ``maps`` = maps with keyframes at the end; ``jump`` = add
    0.5 m to x from that frame on."""

    def __init__(self, per_demo=None):
        super().__init__(pose=moving)
        self.per_demo = per_demo or {}
        self.current = {}

    def __call__(self, cmd, cwd, log_path, timeout_s):
        args = list(cmd)
        seq = os.path.basename(os.path.dirname(args[args.index("--sequence") + 1]))   # .work/run/<pi_name>/euroc
        d = next((v for k, v in self.per_demo.items() if seq.endswith(k)), {})
        lost = set(d.get("lost", ()))
        jump = d.get("jump")
        self.state = lambda k, n, mapping: LOST if k in lost else 2
        self.pose = (lambda k, n: moving(k, n) if jump is None or k < jump
                     else (0.5 + 0.002 * k,) + moving(k, n)[1:])
        nm = d.get("maps", 1)
        self.maps = lambda mapping: [{"id": i, "keyframes": 12, "map_points": 300, "imu_initialized": True,
                                      "inertial_ba1": True, "inertial_ba2": True} for i in range(nm)]
        rc = super().__call__(cmd, cwd, log_path, timeout_s)
        stop = d.get("stop_after")
        if stop is not None:                 # rewrite as a run that stopped at --max_lost_frames
            out = args[args.index("--output_trajectory_csv") + 1]
            lines = open(out).read().splitlines()
            open(out, "w").write("\n".join(lines[:stop + 1]) + "\n")
            text = open(log_path).read()
            s = json.loads(text.split("UMIPROC_SUMMARY ")[1])
            s.update(exit_code=3, frames_fed=stop, stopped_early=True, stop_reason="max_lost_frames")
            open(log_path, "w").write(text.split("UMIPROC_SUMMARY ")[0] + "Lost tracking on 30 >= 30 frames. "
                                      "Stopping.\nUMIPROC_SUMMARY " + json.dumps(s) + "\n")
            return 3
        return rc


def _run(env, start="episode_000001", end="episode_000001", slam=None, **kw):
    slam = slam if slam is not None else Slam()
    res = RC.run_range(env["root"], env["folder"], start, end, source=env["src"], log=quiet, runner=slam, **kw)
    return res, slam


def _calibrate(env, name="episode_000020"):
    o = calibrate_folder(env["root"], env["folder"], name, source=env["src"], log=quiet)
    assert o.passed, o.message


def _sources(env):
    with open(env["root"].path / "pi_sources.csv", newline="") as f:
        return {r["source_name"]: r for r in csv.DictReader(f)}


def _ledger(env):
    with open(env["root"].tasks / env["folder"] / "episodes.csv", newline="") as f:
        return list(csv.DictReader(f))


def _opt(cmd, k):
    return cmd[cmd.index(k) + 1] if k in cmd else None


# --------------------------------------------------------------------------------------------- waiting -> finished

def test_waiting_for_the_gripper_calibration_then_finished_without_slam(env):
    from umiproc.dp.export.zarr_writer import load_episode_zarr

    root, task_dir = env["root"], env["root"].tasks / env["folder"]
    res, slam = _run(env, end="episode_000002")
    assert res.exit_code == RC.EXIT_WAITING and [o.state for o in res.outcomes] == [WAITING, WAITING]
    assert len(slam.calls) == 2
    cmd = slam.calls[0]
    assert _opt(cmd, "--load_map").endswith("map_atlas.osa") and _opt(cmd, "--max_lost_frames") == "30"
    assert _opt(cmd, "--mask_img").endswith("slam_mask.png") and "--no_realtime" not in cmd
    assert "--save_map" not in cmd and "--init_tag_id" not in cmd
    pi = res.outcomes[0].pi_name
    st = task_dir / ".staging" / pi
    loc = json.loads((st / "processed" / LOCALIZE_FILE).read_text())
    assert loc["phase"] == "localized" and loc["rules"]["ok"] and loc["pieces"][0]["pose"]["ok"]
    assert loc["fingerprint_hash"] == res.fingerprint_hash and loc["map"]["pi_name"] == res.map.pi_name
    assert (st / "processed" / "localize" / "trajectory.csv").is_file()
    assert not (root.path / ".work").exists()                       # the EuRoC frames are gone
    text = format_folder_status(folder_status(root.path, env["folder"]))
    assert "waiting in .staging/: 2" in text and "waiting for the gripper calibration" in text
    rep = json.loads((task_dir / "report.json").read_text())
    assert rep["totals"]["waiting"] == 2 and rep["calibration"]["usable"] is False
    # calibrate, then run again: only widths and export, no second localization
    _calibrate(env)
    res2, slam2 = _run(env, end="episode_000002")
    assert res2.exit_code == RC.EXIT_OK and slam2.calls == []
    assert [(o.state, o.slots) for o in res2.outcomes] == [(EPISODE, ["episode_1"]), (EPISODE, ["episode_2"])]
    e1 = task_dir / "episode_1"
    assert (e1 / "raw" / "COMPLETE").is_file() and not (e1 / "raw").is_symlink()
    z = load_episode_zarr(e1 / "processed" / "episode.zarr")
    assert z.name == pi and z.task_label == "block_raise" and z.n_steps > 60
    assert np.all(z.gripper_width >= 0) and np.all(z.gripper_width <= 0.081)
    status = json.loads((e1 / "processed" / "status.json").read_text())
    assert [status[s]["status"] for s in ("check", "slam", "grid", "pose", "width", "validate", "images",
                                          "export")] == ["done"] * 8
    assert status["width"]["outputs"]["calibration"].startswith("pi-")
    piece = json.loads((e1 / "processed" / PIECE_FILE).read_text())
    assert (piece["piece"], piece["pieces"], piece["raw_of"], piece["world"]) == (1, 1, None, "table_tag")
    assert _sources(env)["episode_000001"]["status"] == "ingested"
    assert not (task_dir / ".staging").exists() or not any((task_dir / ".staging").iterdir())


def test_poses_are_exported_in_the_table_frame(env):
    from scipy.spatial.transform import Rotation

    from umiproc.dp.config import load_rig
    from umiproc.dp.export.zarr_writer import load_episode_zarr
    from umiproc.dp.stages.pose import T_imu_tcp

    _calibrate(env)
    res, _ = _run(env)
    e1 = env["root"].tasks / env["folder"] / "episode_1"
    z = load_episode_zarr(e1 / "processed" / "episode.zarr")
    loc = json.loads((e1 / "processed" / LOCALIZE_FILE).read_text())
    t_lo = loc["pieces"][0]["finish"]["values"]["t_lo_ns"]
    times = [int(x) for x in (e1 / "processed" / "localize" / "times.txt").read_text().split()]
    # the fake camera position is linear in the frame index: interpolate at the first exported row
    k = np.interp(t_lo, times, np.arange(len(times)))
    T_W_cam = np.eye(4)
    T_W_cam[:3, 3] = moving(k, 0)[:3]
    rig = load_rig(rig_file(env["root"].path, "rig_pi_v3"))
    T_cam_imu = np.linalg.inv(np.asarray(rig.cameras["ultrawide"].T_imu_cam))
    T = (np.asarray(rig.T_world_offset) @ np.linalg.inv(tx_slam_tag_example()) @ T_W_cam @ T_cam_imu
         @ T_imu_tcp(rig))
    np.testing.assert_allclose(z.eef_pos[0], T[:3, 3], atol=1e-5)
    np.testing.assert_allclose(Rotation.from_rotvec(z.eef_rot_axis_angle[0]).as_matrix(), T[:3, :3], atol=1e-5)


# --------------------------------------------------------------------------------------------- split

def test_a_demo_split_at_its_lost_frames_gives_two_episodes_with_one_raw_copy(env):
    from umiproc.dp.export.zarr_writer import load_episode_zarr

    _calibrate(env)
    res, _ = _run(env, slam=Slam({"000001": {"lost": [18, 19]}}))
    o = res.outcomes[0]
    assert res.exit_code == RC.EXIT_OK and (o.state, o.slots) == (EPISODE, ["episode_1", "episode_2"])
    assert [(p["index"], p["frames"], p["outcome"]) for p in o.pieces] == [(1, [0, 18], EPISODE),
                                                                          (2, [20, 39], EPISODE)]
    task_dir = env["root"].tasks / env["folder"]
    e1, e2 = task_dir / "episode_1", task_dir / "episode_2"
    assert e2.joinpath("raw").is_symlink() and os.readlink(e2 / "raw") == os.path.join("..", "episode_1", "raw")
    assert (e2 / "raw" / "COMPLETE").is_file() and not (e1 / "raw").is_symlink()
    names = [load_episode_zarr(e / "processed" / "episode.zarr").name for e in (e1, e2)]
    assert names == [f"{o.pi_name}_piece1", f"{o.pi_name}_piece2"]
    p2 = json.loads((e2 / "processed" / PIECE_FILE).read_text())
    assert (p2["piece"], p2["pieces"], p2["raw_of"], p2["frames"]) == (2, 2, "episode_1", [20, 39])
    slots = [(r["slot"], r["pi_name"]) for r in _ledger(env) if r["slot"].startswith("episode_")]
    assert slots == [("episode_1", o.pi_name), ("episode_2", o.pi_name)]
    f = folder_status(env["root"].path, env["folder"])
    assert [e["slot"] for e in f["episodes"]] == ["episode_1", "episode_2"]
    text = format_folder_status(f)
    assert "episodes:             2 (from 1 recordings)" in text and "(pieces 1, 2 of 2)" in text
    assert "episode_1, episode_2" in text
    # the copied packaging reads both
    from umiproc.dp.package.collect import FolderInfo, Selection, collect_folder

    info = FolderInfo(env["folder"], task_dir, "block_raise", "rig_pi_v3")
    sel = Selection([info])
    collect_folder(info, sel)
    assert sorted(e.pi_name for e in sel.episodes) == sorted(names)
    # an already-made episode in the range is skipped quietly
    res2, slam2 = _run(env)
    assert slam2.calls == [] and res2.outcomes[0].state == SKIPPED and res2.outcomes[0].quiet


def test_a_piece_failing_a_pose_check_is_dropped_the_other_kept(env):
    _calibrate(env)
    res, _ = _run(env, slam=Slam({"000001": {"lost": [18, 19], "jump": 30}}))
    o = res.outcomes[0]
    assert (o.state, o.slots) == (EPISODE, ["episode_1"])
    assert o.pieces[1]["outcome"] == FAILED and o.pieces[1]["stage"] == "pose"
    assert "speed" in o.pieces[1]["reasons"] and "piece 2 dropped: speed" in o.describe()


# --------------------------------------------------------------------------------------------- drops

def test_umis_drops_go_to_failed_with_their_reason(env):
    _calibrate(env)
    slam = Slam({"000001": {"lost": range(10, 16)},                     # 6 lost
                 "000002": {"lost": range(5, 39), "stop_after": 35},    # stopped at --max_lost_frames
                 "000003": {"maps": 2},                                 # a new map
                 "000004": {"lost": range(0, 3)}})                     # 3 lost at the start: fine
    res, _ = _run(env, end="episode_000004", slam=slam)
    by = {o.name: o for o in res.outcomes}
    assert res.exit_code == RC.EXIT_OK
    assert (by["episode_000001"].state, by["episode_000001"].stage, by["episode_000001"].reasons) == \
        (FAILED, "slam", ["too_many_lost"])
    assert by["episode_000002"].reasons == ["stopped_lost"]
    assert by["episode_000003"].reasons == ["new_map"]
    assert by["episode_000004"].state == EPISODE
    task_dir = env["root"].tasks / env["folder"]
    reason = json.loads((task_dir / "failed" / by["episode_000001"].pi_name / "reason.json").read_text())
    assert reason["stage"] == "slam" and reason["reasons"] == ["too_many_lost"]
    assert "6 lost (limit 5)" in reason["message"] and reason["rules"]["n_lost"] == 6
    src = _sources(env)
    assert [src[f"episode_00000{k}"]["status"] for k in (1, 2, 3, 4)] == ["failed", "failed", "failed", "ingested"]
    text = format_folder_status(folder_status(env["root"].path, env["folder"]))
    assert "failed/:              3" in text and "slam: too_many_lost" in text
    # --retry-failed runs them again from scratch (a new localization)
    res2, slam2 = _run(env, end="episode_000003", slam=Slam(), retry_failed=True)
    assert len(slam2.calls) == 3 and [o.state for o in res2.outcomes] == [EPISODE, EPISODE, EPISODE]


def test_finger_tag_coverage_below_80_percent_fails_at_validate(env, monkeypatch):
    _calibrate(env)
    fake_finger_tags(monkeypatch, visible=lambda i, n: {0, 1} if i % 2 else set())
    res, _ = _run(env)
    o = res.outcomes[0]
    assert (o.state, o.stage) == (FAILED, "validate") and any("tag_coverage" in r for r in o.reasons)


# --------------------------------------------------------------------------------------------- refusals, skips, CLI

def test_an_out_of_date_map_is_refused_and_nothing_changes(env):
    p = rig_file(env["root"].path, "rig_pi_v3")
    doc = yaml.safe_load(p.read_text())
    doc["imu"]["vio_inflation"] = {"noise_density": 10.0, "random_walk": 10.0}
    p.write_text(yaml.safe_dump(doc, sort_keys=False))
    for dry in (False, True):
        with pytest.raises(RC.RunInputError, match="OUT OF DATE") as e:
            _run(env, dry_run=dry)
        assert e.value.code == "out_of_date" and "sensors.imu" in str(e.value)
    assert set(_sources(env)) == {"episode_000900"}
    assert not (env["root"].tasks / env["folder"] / ".staging").exists()


def test_the_map_and_calibration_recordings_in_the_range_are_skipped(env):
    _calibrate(env)
    res, slam = _run(env, start="episode_000020", end="episode_000900")
    whys = {o.name: (o.state, o.message) for o in res.outcomes}
    assert whys["episode_000900"][0] == SKIPPED and "map recording" in whys["episode_000900"][1]
    assert whys["episode_000020"][0] == SKIPPED and "calibration" in whys["episode_000020"][1]
    assert slam.calls == []


def test_dry_run_and_the_cli(env, capsys):
    root = str(env["root"].path)
    base = ["--source", str(env["src"]), "--root", root]
    rc = umicli.main(["run", env["folder"], "episode_000001", "episode_000002", "--dry-run"] + base)
    out = capsys.readouterr().out
    assert rc == 0 and "episode_000001: take" in out and "up to date" in out and "nothing was changed" in out
    assert not (env["root"].tasks / env["folder"] / ".staging").exists()
    rc = umicli.main(["run", "nope", "episode_000001", "episode_000002"] + base)
    assert rc == 2 and "nothing was changed" in capsys.readouterr().err
    # the fake SLAM build's binary exits 99: the localization fails with slam_error (exit 0: it is listed)
    rc = umicli.main(["run", env["folder"], "episode_000001", "episode_000001"] + base)
    out = capsys.readouterr().out
    assert rc == 0 and "failed (slam: slam_error)" in out and "slam_error: not a recording problem" in out


def test_the_effective_rig_reaches_the_copied_stages(env):
    from umiproc.dp.calibrate.task import _load_rig
    from umiproc.dp.process import load_folder_context
    from umiproc.lens.lensfile import dump_lens, lens_document

    r = env["root"]
    nominal = load_folder_context(r, env["folder"], slam="none", log=quiet).rig
    rig_lens_file(r.path, "rig_pi_v3").write_text(dump_lens(lens_document(
        rig_id="rig_pi_v3", sensor="imx219", resolution=(1280, 960), intrinsics=[470.5, 469.0, 641.0, 478.0],
        distortion=[0.021, -0.006, 0.0012, -0.0003], date="2026-11-17T10:00:00Z", source={}, board={}, metrics={})))
    ctx = load_folder_context(r, env["folder"], slam="none", log=quiet)
    cam = ctx.rig.cameras["ultrawide"]
    assert cam.model.intrinsics[0] == 470.5 and nominal.cameras["ultrawide"].model.intrinsics[0] != 470.5
    assert ctx.prep.ultrawide.phys.fx == 470.5                            # the training images' remap
    assert ctx.sections["cameras.ultrawide"]["model"]["intrinsics"][0] == 470.5   # the stage hashes
    assert _load_rig(r, "rig_pi_v3", quiet).cameras["ultrawide"].model.intrinsics[0] == 470.5
    # a lens file for another resolution cannot be used: the folder context says so
    rig_lens_file(r.path, "rig_pi_v3").write_text(dump_lens(lens_document(
        rig_id="rig_pi_v3", sensor="imx219", resolution=(640, 480), intrinsics=[235.0, 235.0, 320.0, 240.0],
        distortion=[0.0, 0.0, 0.0, 0.0], date="2026-11-17T10:00:00Z", source={}, board={}, metrics={})))
    from umiproc.dp.process import ProcessError

    with pytest.raises(ProcessError, match="measured at 640x480"):
        load_folder_context(r, env["folder"], slam="none", log=quiet)


def test_an_unexpected_error_fails_only_its_piece_or_recording(env, monkeypatch):
    _calibrate(env)
    real = RP.finish_piece

    def finish_piece(piece, *a, **k):
        if piece.index == 2:
            raise RuntimeError("boom")
        return real(piece, *a, **k)

    monkeypatch.setattr(RP, "finish_piece", finish_piece)
    res, _ = _run(env, slam=Slam({"000001": {"lost": [18, 19]}}))
    o = res.outcomes[0]
    assert res.exit_code == RC.EXIT_OK and o.slots == ["episode_1"] and o.pieces[1]["reasons"] == ["exception"]
    assert (env["root"].tasks / env["folder"] / "episode_1" / "processed" / "error_piece2.txt").is_file()

    def orientation_check(*a, **k):
        raise RuntimeError("bang")

    monkeypatch.setattr(RP, "orientation_check", orientation_check)
    res, _ = _run(env, start="episode_000002", end="episode_000003")
    assert [(o.state, o.stage, o.reasons) for o in res.outcomes] == [(FAILED, "pose", ["exception"])] * 2
    assert "bang" in res.outcomes[0].message
    d = env["root"].tasks / env["folder"] / "failed" / res.outcomes[0].pi_name
    assert "RuntimeError: bang" in (d / "processed" / "error.txt").read_text()
