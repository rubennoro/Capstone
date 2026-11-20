"""convert.py: one episode end to end, every §5.3 check, status.json, the up-to-date rule, the command
(PLAN.md §4.1, §5, §11 convert bullet).

One module-scoped raw task holds a good episode and one episode per fault (most are copies of the good one with a file
edited; ``make_episode`` costs ~1 s and a conversion ~4 s). One module-scoped ``teleop_train convert`` run over it
gives the statuses, zarrs and printed rows most tests read. Convert never writes under the raw root, so several out
roots can share it; a test that edits raw files works on a clone (``clone``).
"""
from __future__ import annotations

import contextlib
import csv
import datetime
import hashlib
import io
import json
import shutil
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pytest
import yaml
from scipy.spatial.transform import Rotation

import synthetic_episode as se
from teleop_train import cli, convert, paths
from teleop_train import images as images_mod
from teleop_train import trajectory as tj
from teleop_train._copied.zarr_writer import load_episode_zarr
import teleop_train.episode_reader as er_mod
from teleop_train.convert import (CONVERT_VERSION, SETTINGS, STATUS_SCHEMA, EpisodeResult, convert_episode,
                                  convert_task, is_up_to_date, parse_episode_list, read_status)

TASK = "pick_test"
STATUS_KEYS = {"schema", "state", "reason", "episode", "outcome", "raw_episode_dir", "meta_sha256", "window_ns",
               "kept_ns", "raw_s", "kept_s", "n_steps", "n_uw", "pose_source", "F_T_TCP", "rig_id", "rig_sha256",
               "settings", "code_sha256", "convert_version", "teleop_train", "dropped", "created"}
POSE_PREFIXES = ("O_T_TCP_", "O_T_TCP_target_")


# --------------------------------------------------------------------------------------------- raw-file edits

def _read_rows(path: Path) -> tuple[list[str], list[list[str]]]:
    with open(path, newline="", encoding="utf-8") as f:
        rows = list(csv.reader(f))
    return rows[0], rows[1:]


def _write_rows(path: Path, header: list[str], rows: list[list[str]]) -> None:
    path.write_text("".join(",".join(r) + "\n" for r in [header, *rows]), encoding="utf-8")


def edit_states(ep_dir: Path, fn) -> None:
    """fn(header_index: dict, rows: list[list[str]]) -> new rows, applied to robot_states.csv."""
    path = ep_dir / "robot_states.csv"
    header, rows = _read_rows(path)
    _write_rows(path, header, fn({c: i for i, c in enumerate(header)}, rows))


def edit_meta(ep_dir: Path, fn) -> None:
    meta = yaml.safe_load((ep_dir / "meta.yaml").read_text(encoding="utf-8"))
    fn(meta)
    (ep_dir / "meta.yaml").write_text(yaml.safe_dump(meta, sort_keys=False), encoding="utf-8")


def copy_episode(src: Path, task_dir: Path, number: int) -> Path:
    """A copy of a raw episode under another number (meta.episode follows the folder name)."""
    dst = task_dir / f"episode_{number:03d}"
    shutil.copytree(src, dst)

    def rename(m):
        m["episode"], m["episode_number"] = dst.name, number
    edit_meta(dst, rename)
    return dst


def drop_state_rows(start_s: float, length_s: float):
    """Leaves out the arm states with t_ns in [T0 + start, T0 + start + length)."""
    lo, hi = se.T0_NS + round(start_s * 1e9), se.T0_NS + round((start_s + length_s) * 1e9)

    def fn(ix, rows):
        return [r for r in rows if not (lo <= int(r[ix["t_ns"]]) < hi)]
    return fn


def freeze_poses(width=None):
    """Every pose = the first row's (both pose sources); width(t_ns) -> metres replaces the widths when given."""
    def fn(ix, rows):
        first = {c: rows[0][i] for c, i in ix.items() if c.startswith(POSE_PREFIXES)}
        for r in rows:
            for c, v in first.items():
                r[ix[c]] = v
            r[ix["gripper_width_m"]] = repr(float(width(int(r[ix["t_ns"]])) if width else 0.08))
        return rows
    return fn


def identity_rotations(ix, rows):
    for r in rows:
        for p in POSE_PREFIXES:
            for a in range(3):
                for b in range(3):
                    r[ix[f"{p}{a}{b}"]] = repr(1.0 if a == b else 0.0)
    return rows


def swap_state_times(ix, rows):
    i = ix["t_ns"]
    rows[50][i], rows[51][i] = rows[51][i], rows[50][i]
    return rows


def non_rigid_row(ix, rows):
    for p in POSE_PREFIXES:
        rows[10][ix[f"{p}00"]] = repr(float(rows[10][ix[f"{p}00"]]) * 1.01)
    return rows


def short_blip(t_ns: int) -> float:
    """Width 0.08 -> 0.07 m over 0.2 s at T0 + 3 s (50 mm/s), else still: about 1.5 s kept."""
    s = (t_ns - se.T0_NS) / 1e9 - 3.0
    return 0.08 - 0.01 * min(max(s / 0.2, 0.0), 1.0)


# --------------------------------------------------------------------------------------------- the module task

#: number -> (expected state, reason fragment) for the module run
EXPECTED = {
    0: ("converted", ""),
    1: ("converted", ""),                                   # outcome fail: converted, marked fail
    2: ("excluded", "COMPLETE: missing"),
    3: ("excluded", "meta_version 2"),
    4: ("excluded", "rig is null"),
    5: ("excluded", "must strictly increase"),
    6: ("excluded", "video_index 6 != row number 5"),
    7: ("excluded", "frames but main_frames.csv has"),      # decoded frame count != CSV rows
    8: ("excluded", "consecutive arm states inside the kept span"),
    9: ("excluded", "consecutive arm states inside the kept span"),
    10: ("converted", ""),                                  # a state gap in the still start, outside the kept span
    11: ("excluded", "consecutive main frames inside the kept span"),
    12: ("converted", ""),                                  # one dropped main frame passes
    13: ("excluded", "consecutive ultrawide frames inside the kept span"),
    14: ("excluded", "no motion found"),
    15: ("excluded", "shorter than 2 s"),
    16: ("excluded", "the rig expects 640x480"),
    17: ("excluded", "is not a rigid transform"),
    18: ("excluded", "zero rotation vector"),
}


@dataclass
class ModuleRun:
    raw_root: Path
    out_root: Path
    tp: paths.TaskPaths
    good: se.SyntheticEpisode
    raw_hash: str
    code: int
    out: str
    err: str


def run_cli(*argv: str) -> tuple[int, str, str]:
    out, err = io.StringIO(), io.StringIO()
    with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
        code = cli.main(list(argv))
    return code, out.getvalue(), err.getvalue()


def cli_convert(raw_root: Path, out_root: Path, *extra: str) -> tuple[int, str, str]:
    return run_cli("convert", TASK, "--raw-root", str(raw_root), "--out-root", str(out_root), *extra)


@pytest.fixture(scope="module")
def mod(tmp_path_factory) -> ModuleRun:
    base = tmp_path_factory.mktemp("convert")
    raw_root, out_root = base / "raw", base / "out"
    task_dir = se.make_task(raw_root, TASK)
    good = se.make_episode(task_dir, 0)
    g = good.episode_dir

    edit_meta(copy_episode(g, task_dir, 1), lambda m: m.update(outcome="fail"))
    (copy_episode(g, task_dir, 2) / "COMPLETE").unlink()
    edit_meta(copy_episode(g, task_dir, 3), lambda m: m.update(meta_version=2))
    edit_meta(copy_episode(g, task_dir, 4), lambda m: m.update(rig=None, rig_note="rig.yaml: FileNotFoundError"))
    edit_states(copy_episode(g, task_dir, 5), swap_state_times)
    e6 = copy_episode(g, task_dir, 6)
    header, rows = _read_rows(e6 / "main_frames.csv")
    rows[5][header.index("video_index")] = "6"
    _write_rows(e6 / "main_frames.csv", header, rows)
    se.make_episode(task_dir, 7, extra_main_packets=1)
    edit_states(copy_episode(g, task_dir, 8), drop_state_rows(3.0, 0.1))
    edit_states(copy_episode(g, task_dir, 9), drop_state_rows(3.0, 2.0))
    edit_states(copy_episode(g, task_dir, 10), drop_state_rows(0.3, 0.3))
    se.make_episode(task_dir, 11, main_gap=(3.0, 0.15))
    se.make_episode(task_dir, 12, main_gap=(3.0, 0.034))
    se.make_episode(task_dir, 13, uw_gap=(3.0, 0.25))
    edit_states(copy_episode(g, task_dir, 14), freeze_poses())
    edit_states(copy_episode(g, task_dir, 15), freeze_poses(short_blip))
    se.make_episode(task_dir, 16, frame_size=(320, 240))
    edit_states(copy_episode(g, task_dir, 17), non_rigid_row)
    edit_states(copy_episode(g, task_dir, 18), identity_rotations)

    raw_hash = se.tree_hash(raw_root)
    code, out, err = cli_convert(raw_root, out_root)
    tp = paths.task_paths(TASK, raw_root, out_root)
    return ModuleRun(raw_root, out_root, tp, good, raw_hash, code, out, err)


def ep(n: int) -> str:
    return f"episode_{n:03d}"


def status_of(m: ModuleRun, n: int) -> dict:
    s = read_status(m.tp.status_json(ep(n)))
    assert s is not None
    return s


def clone(m: ModuleRun, dst: Path, numbers=(0,)) -> paths.TaskPaths:
    """A copy of the raw task (task.yaml + the given episodes) and of their outputs, with the marker pointing at the
    copied raw folder."""
    raw_task = dst / "raw" / TASK
    out_task = dst / "out" / TASK
    raw_task.mkdir(parents=True)
    out_task.mkdir(parents=True)
    shutil.copy2(m.tp.raw_task_dir / "task.yaml", raw_task / "task.yaml")
    shutil.copy2(m.tp.out_task_dir / "task.yaml", out_task / "task.yaml")
    for n in numbers:
        shutil.copytree(m.tp.raw_task_dir / ep(n), raw_task / ep(n))
        shutil.copytree(m.tp.out_task_dir / ep(n), out_task / ep(n))
    marker = yaml.safe_load(m.tp.marker.read_text(encoding="utf-8"))
    marker["raw_task_dir"] = str(raw_task.resolve())
    (out_task / paths.MARKER_FILE).write_text(yaml.safe_dump(marker), encoding="utf-8")
    return paths.task_paths(TASK, dst / "raw", dst / "out")


def kept_grid(status: dict) -> np.ndarray:
    lo, hi = status["window_ns"]
    k0, k1 = status["kept_ns"]
    g = tj.make_grid(lo, hi)
    return g[(g >= k0) & (g <= k1)]


# --------------------------------------------------------------------------------------------- module run + §5.3

def test_module_run_exit_code_and_rows(mod):
    assert mod.code == 0, mod.err
    assert mod.err == ""
    lines = mod.out.splitlines()
    assert lines[0].split()[:4] == ["episode", "outcome", "raw", "s"]
    rows = lines[1:-1]
    assert len(rows) == len(EXPECTED)
    for n, row in zip(sorted(EXPECTED), rows):
        state, frag = EXPECTED[n]
        assert row.split()[0] == str(n)
        assert state in row and frag in row, row
    n_conv = sum(1 for s, _ in EXPECTED.values() if s == "converted")
    assert lines[-1] == (f"{n_conv} converted, 0 up to date, {len(EXPECTED) - n_conv} excluded; "
                         f"{n_conv} of {len(EXPECTED)} usable")


@pytest.mark.parametrize("n", sorted(EXPECTED))
def test_check_rows(mod, n):
    """Every §5.3 row: the state, the one-line reason, and no episode.zarr unless converted."""
    state, frag = EXPECTED[n]
    s = status_of(mod, n)
    assert s["state"] == state
    assert frag in s["reason"]
    assert "\n" not in s["reason"]
    assert (s["reason"] == "") == (state == "converted")
    assert mod.tp.episode_zarr(ep(n)).is_dir() == (state == "converted")
    assert set(s) == STATUS_KEYS


def test_fail_outcome_converted_and_marked(mod):
    s = status_of(mod, 1)
    assert s["state"] == "converted" and s["outcome"] == "fail"
    assert " fail " in mod.out.splitlines()[2]
    assert status_of(mod, 0)["outcome"] == "success"


def test_excluded_status_nulls(mod):
    s = status_of(mod, 2)                                # COMPLETE missing: nothing read, outcome from meta
    assert s["outcome"] == "success"
    for k in ("window_ns", "kept_ns", "raw_s", "kept_s", "n_steps", "n_uw", "F_T_TCP", "rig_id", "rig_sha256",
              "dropped"):
        assert s[k] is None, k
    assert s["pose_source"] == "target" and s["settings"] == SETTINGS
    s = status_of(mod, 8)                                # a gap: read, windowed and trimmed, then refused
    assert s["window_ns"] is not None and s["kept_ns"] is not None and s["raw_s"] > s["kept_s"] > 0
    assert s["n_steps"] is None and s["n_uw"] is None
    assert s["F_T_TCP"] == se.F_T_TCP.tolist() and s["rig_id"] == "rig_pi_v3_640"
    s = status_of(mod, 14)                               # no motion: windowed, nothing kept
    assert s["raw_s"] > 0 and s["kept_ns"] is None


def test_seq_gap_reason_names_the_gap(mod):
    assert "gap of 2010.0 ms" in status_of(mod, 9)["reason"]


def test_main_frame_within_limit(mod, tmp_path, monkeypatch):
    """'every grid step has a main frame within 50 ms': with 30 fps frames, a 10 ms limit fails."""
    monkeypatch.setattr(images_mod, "MAIN_MAX_DT_NS", 10_000_000)
    tp = paths.task_paths(TASK, mod.raw_root, tmp_path / "out")
    paths.prepare_out_task(tp)
    r = convert_episode(tp, ep(0), "target")
    assert r.state == "excluded" and "no main frame within 10 ms" in r.reason
    assert not tp.episode_zarr(ep(0)).exists()


# --------------------------------------------------------------------------------------------- status.json, zarr

def test_converted_status(mod):
    s = status_of(mod, 0)
    text = mod.tp.status_json(ep(0)).read_text(encoding="utf-8")
    assert json.loads(text) == s                         # JSON-native values only
    assert s["schema"] == STATUS_SCHEMA and s["convert_version"] == CONVERT_VERSION
    assert s["episode"] == ep(0) and s["state"] == "converted" and s["reason"] == ""
    assert s["raw_episode_dir"] == str(mod.tp.raw_episode_dir(ep(0)))
    assert s["meta_sha256"] == hashlib.sha256((mod.good.episode_dir / "meta.yaml").read_bytes()).hexdigest()
    assert s["settings"] == SETTINGS
    assert s["pose_source"] == "target"
    assert s["F_T_TCP"] == se.F_T_TCP.tolist()
    rig = se.load_rig()
    assert s["rig_id"] == rig["rig_id"]
    assert s["rig_sha256"] == hashlib.sha256(
        json.dumps(rig, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    lo, hi = s["window_ns"]
    k0, k1 = s["kept_ns"]
    assert all(isinstance(v, int) for v in (lo, hi, k0, k1))
    assert lo <= k0 < k1 <= hi
    assert s["raw_s"] == pytest.approx((hi - lo) / 1e9) and s["kept_s"] == pytest.approx((k1 - k0) / 1e9)
    assert isinstance(s["n_steps"], int) and isinstance(s["n_uw"], int)
    assert s["dropped"] == {"states": 0, "main": 0, "ultrawide": 0}
    assert set(s["teleop_train"]) == {"version", "commit", "dirty"}
    assert datetime.datetime.fromisoformat(s["created"]).utcoffset() == datetime.timedelta(0)


def test_settings_from_module_constants():
    assert SETTINGS == {
        "grid_hz": tj.GRID_HZ, "trans_speed_mps": tj.TRANS_SPEED_MPS, "rot_speed_deg_s": tj.ROT_SPEED_DEG_S,
        "width_rate_mps": tj.WIDTH_RATE_MPS, "smooth_s": tj.SMOOTH_S, "margin_s": tj.MARGIN_S,
        "max_state_gap_ns": tj.MAX_STATE_GAP_NS, "max_main_gap_ns": tj.MAX_MAIN_GAP_NS,
        "max_uw_gap_ns": tj.MAX_UW_GAP_NS, "min_kept_s": tj.MIN_KEPT_S,
        "main_max_dt_ns": images_mod.MAIN_MAX_DT_NS, "uw_rate_hz": images_mod.UW_RATE_HZ,
        "rigid_tol": er_mod.RIGID_TOL}
    assert json.loads(json.dumps(SETTINGS)) == SETTINGS


def test_zarr_read_back(mod):
    """Shapes, names, the kept span against the truth (tolerance as test_trajectory.test_trim_against_truth) and the
    arrays against the truth on the kept grid."""
    s = status_of(mod, 0)
    a = load_episode_zarr(mod.tp.episode_zarr(ep(0)))
    n, m = s["n_steps"], s["n_uw"]
    assert a.name == f"{TASK}_{ep(0)}" and a.task_label == TASK
    assert a.main_rgb.shape == (n, 224, 224, 3) and a.uw_rgb.shape == (m, 224, 224, 3)
    assert a.eef_pos.shape == (n, 3) and a.eef_rot_axis_angle.shape == (n, 3) and a.gripper_width.shape == (n, 1)
    assert a.uw_upsample_local.shape == (n,) and int(a.uw_upsample_local[-1]) == m - 1
    assert 40 <= m <= 70                                 # ~10 Hz over ~5.7 s

    truth = mod.good.truth
    half_s = (round(tj.SMOOTH_S * tj.GRID_HZ) // 2) / tj.GRID_HZ
    early = (half_s + 0.010 + 1 / tj.GRID_HZ) * 1e9
    late = (1 / tj.GRID_HZ) * 1e9
    k0, k1 = s["kept_ns"]
    start, end = truth.motion_start_ns - 0.5e9, truth.motion_end_ns + 0.5e9
    assert start - early - 1 <= k0 <= start + late + 1
    assert end - late - 1 <= k1 <= end + early + 1

    t = kept_grid(s)
    assert len(t) == n and t[0] == k0 and t[-1] == k1
    # interpolation is exact apart from the 100 Hz corners (<= 0.15 mm here); a one-step (16.7 ms) shift would be 0.8 mm
    np.testing.assert_allclose(a.eef_pos, truth.position(t), atol=2e-4)
    rot_err = np.degrees((Rotation.from_rotvec(a.eef_rot_axis_angle.astype(np.float64)).inv()
                          * truth.rotation(t)).magnitude())
    assert rot_err.max() < 0.05
    np.testing.assert_allclose(a.gripper_width[:, 0], truth.width(t), atol=2e-4)


def _index_from_224(img: np.ndarray) -> int:
    """se.frame_image's index bands read back from a prepared 224x224 main image (640x480 letterboxed: 80 px of
    padding on top, scale 224/640)."""
    s, top, band = 224 / 640, 80, se.WIDTH // se.INDEX_BITS
    y0, y1 = round((top + 60) * s), round((top + 180) * s)
    value = 0
    for b in range(se.INDEX_BITS):
        x0, x1 = round((b * band + band // 4) * s), round(((b + 1) * band - band // 4) * s)
        if img[y0:y1, x0:x1].mean() > 128:
            value |= 1 << b
    return value


@pytest.mark.parametrize("n", [0, 12])                    # 12: a single dropped main frame
def test_each_image_is_the_frame_nearest_its_grid_step(mod, n):
    """End to end: zarr main_rgb[k] shows the main frame whose mid-exposure is nearest kept grid step k."""
    s = status_of(mod, n)
    a = load_episode_zarr(mod.tp.episode_zarr(ep(n)))
    with open(mod.tp.raw_episode_dir(ep(n)) / "main_frames.csv", newline="") as f:
        rows = list(csv.DictReader(f))
    t_mid = np.array([int(r["t_mid_ns"]) for r in rows], dtype=np.int64)
    vidx = np.array([int(r["video_index"]) for r in rows], dtype=np.int64)
    t = kept_grid(s)
    nearest = vidx[np.abs(t[:, None] - t_mid[None, :]).argmin(axis=1)]
    got = np.array([_index_from_224(img) for img in a.main_rgb])
    assert np.array_equal(got, nearest)


def test_single_dropped_main_frame_converts_fully(mod):
    s = status_of(mod, 12)
    assert s["state"] == "converted" and s["n_steps"] == status_of(mod, 0)["n_steps"]


# --------------------------------------------------------------------------------------------- up to date

def test_second_run_is_up_to_date(mod):
    code, out, err = cli_convert(mod.raw_root, mod.out_root)
    assert code == 0, err
    rows = out.splitlines()[1:-1]
    for n, row in zip(sorted(EXPECTED), rows):
        if n == 2:                                       # no COMPLETE: always checked again
            assert " excluded " in row
        else:
            assert "  up to date" in row, row
            if EXPECTED[n][0] == "excluded":
                assert "up to date  excluded: " in row and EXPECTED[n][1] in row, row
    n_conv = sum(1 for s, _ in EXPECTED.values() if s == "converted")
    assert out.splitlines()[-1] == (f"0 converted, {len(EXPECTED) - 1} up to date, 1 excluded; "
                                    f"{n_conv} of {len(EXPECTED)} usable")


def test_up_to_date_result(mod):
    r = convert_episode(mod.tp, ep(0), "target")
    assert r.state == "up to date" and r.reason == "" and r.usable
    assert r.status == status_of(mod, 0)
    assert (r.outcome, r.raw_s, r.kept_s, r.n_steps) == ("success", r.status["raw_s"], r.status["kept_s"],
                                                         r.status["n_steps"])
    r = convert_episode(mod.tp, ep(3), "target")
    assert r.state == "up to date" and "meta_version 2" in r.reason and not r.usable


def test_is_up_to_date_rule(mod):
    s = status_of(mod, 0)
    sha = s["meta_sha256"]
    zp = mod.tp.episode_zarr(ep(0))
    assert is_up_to_date(s, sha, "target", zp)
    assert is_up_to_date(s, sha, "target")
    assert not is_up_to_date(None, sha, "target")
    assert not is_up_to_date(s, None, "target")
    assert not is_up_to_date(s, sha, "measured")
    assert not is_up_to_date(s, "0" * 64, "target")
    assert not is_up_to_date(s, sha, "target", zp.parent / "missing.zarr")
    for key, value in (("schema", "other"), ("convert_version", CONVERT_VERSION + 1), ("state", "weird"),
                       ("settings", {**SETTINGS, "margin_s": 0.4})):
        assert not is_up_to_date({**s, key: value}, sha, "target", zp), key
    excluded = status_of(mod, 3)
    assert is_up_to_date(excluded, excluded["meta_sha256"], "target", mod.tp.episode_zarr(ep(3)))


def test_pose_source_change_reconverts(mod, tmp_path):
    tp = clone(mod, tmp_path)
    [r] = convert_task(tp, "measured")
    assert r.state == "converted" and r.status["pose_source"] == "measured"
    measured = load_episode_zarr(tp.episode_zarr(ep(0)))
    target = load_episode_zarr(mod.tp.episode_zarr(ep(0)))
    assert measured.eef_pos.shape == target.eef_pos.shape
    np.testing.assert_allclose(measured.eef_pos - target.eef_pos, np.tile(se.MEASURED_OFFSET_M, (len(target.eef_pos), 1)),
                               atol=1e-6)
    np.testing.assert_array_equal(measured.eef_rot_axis_angle, target.eef_rot_axis_angle)
    assert [r.state for r in convert_task(tp, "measured")] == ["up to date"]
    assert [r.state for r in convert_task(tp, "target")] == ["converted"]


def test_force_reconverts_identically(mod, tmp_path):
    tp = clone(mod, tmp_path)
    before = se.tree_hash(tp.raw_task_dir)
    seen: list[EpisodeResult] = []
    [r] = convert_task(tp, "target", force=True, progress=seen.append)
    assert r.state == "converted" and seen == [r]
    assert se.tree_hash(tp.raw_task_dir) == before
    a = load_episode_zarr(tp.episode_zarr(ep(0)))
    b = load_episode_zarr(mod.tp.episode_zarr(ep(0)))
    for k in ("main_rgb", "uw_rgb", "uw_upsample_local", "eef_pos", "eef_rot_axis_angle", "gripper_width"):
        np.testing.assert_array_equal(getattr(a, k), getattr(b, k), err_msg=k)   # deterministic (rule 5)


def test_meta_edit_reconverts(mod, tmp_path):
    tp = clone(mod, tmp_path)
    meta = tp.raw_episode_dir(ep(0)) / "meta.yaml"
    meta.write_text(meta.read_text(encoding="utf-8") + "# note\n", encoding="utf-8")
    [r] = convert_task(tp, "target")
    assert r.state == "converted"
    assert r.status["meta_sha256"] == hashlib.sha256(meta.read_bytes()).hexdigest()


def test_missing_zarr_reconverts(mod, tmp_path):
    tp = clone(mod, tmp_path)
    shutil.rmtree(tp.episode_zarr(ep(0)))
    [r] = convert_task(tp, "target")
    assert r.state == "converted" and tp.episode_zarr(ep(0)).is_dir()


def test_stale_zarr_removed_when_excluded(mod, tmp_path):
    tp = clone(mod, tmp_path)
    assert tp.episode_zarr(ep(0)).is_dir()
    edit_meta(tp.raw_episode_dir(ep(0)), lambda m: m.update(meta_version=2))
    [r] = convert_task(tp, "target")
    assert r.state == "excluded" and "meta_version 2" in r.reason
    assert not tp.episode_zarr(ep(0)).exists()
    assert read_status(tp.status_json(ep(0)))["state"] == "excluded"


def test_stale_zarr_removed_when_complete_disappears(mod, tmp_path):
    tp = clone(mod, tmp_path)
    (tp.raw_episode_dir(ep(0)) / "COMPLETE").unlink()
    [r] = convert_task(tp, "target")                    # meta.yaml unchanged, but no COMPLETE: never up to date
    assert r.state == "excluded" and "COMPLETE" in r.reason
    assert not tp.episode_zarr(ep(0)).exists()


# --------------------------------------------------------------------------------------------- helpers

@pytest.mark.parametrize("text, expected", [
    ("3,7-9", {3, 7, 8, 9}), ("0", {0}), (" 1 , 2 ", {1, 2}), ("5-5", {5}), ("2,2,1-3", {1, 2, 3}),
])
def test_parse_episode_list(text, expected):
    assert parse_episode_list(text) == expected


@pytest.mark.parametrize("text", ["", " ", "a", "3-", "-3", "9-7", "1,,2", "1.5", "+3"])
def test_parse_episode_list_refuses(text):
    with pytest.raises(ValueError, match="--episodes"):
        parse_episode_list(text)


def test_read_status(tmp_path):
    p = tmp_path / "status.json"
    assert read_status(p) is None
    p.write_text("{not json", encoding="utf-8")
    assert read_status(p) is None
    p.write_text("[1, 2]", encoding="utf-8")
    assert read_status(p) is None
    p.write_text('{"a": 1}', encoding="utf-8")
    assert read_status(p) == {"a": 1}


def test_convert_episode_refuses_bad_arguments(mod):
    with pytest.raises(ValueError, match="pose_source"):
        convert_episode(mod.tp, ep(0), "both")
    with pytest.raises(paths.PathsError):
        convert_episode(mod.tp, "ep_0", "target")


def test_convert_task_filters_by_number(mod, tmp_path):
    tp = paths.task_paths(TASK, mod.raw_root, tmp_path / "out")
    results = convert_task(tp, "target", episodes={14, 3, 99})
    assert [r.episode for r in results] == [ep(3), ep(14)]
    assert all(r.state == "excluded" for r in results)
    assert tp.marker.is_file() and (tp.out_task_dir / "task.yaml").is_file()


# --------------------------------------------------------------------------------------------- the command

def test_cli_nothing_usable_exits_1(mod, tmp_path):
    code, out, err = cli_convert(mod.raw_root, tmp_path / "out", "--episodes", "2-3,99")
    assert code == 1
    assert out.splitlines()[-1] == "0 converted, 0 up to date, 2 excluded; 0 of 2 usable"
    assert "no raw episode folder for number(s) 99" in err


def test_cli_only_excluded_up_to_date_exits_1(mod):
    code, out, _ = cli_convert(mod.raw_root, mod.out_root, "--episodes", "3")
    assert code == 1 and " up to date " in out and "0 of 1 usable" in out


def test_cli_up_to_date_converted_exits_0(mod):
    code, out, _ = cli_convert(mod.raw_root, mod.out_root, "--episodes", "0")
    assert code == 0 and "1 of 1 usable" in out


def test_cli_bad_episode_list(mod, tmp_path):
    code, out, err = cli_convert(mod.raw_root, tmp_path / "out", "--episodes", "9-7")
    assert code == 1 and out == ""
    assert err.startswith("teleop_train convert: --episodes") and err.count("\n") == 1
    assert not (tmp_path / "out").exists()


def test_cli_handheld_folder_refused(mod, tmp_path):
    (tmp_path / "out" / TASK).mkdir(parents=True)
    code, _, err = cli_convert(mod.raw_root, tmp_path / "out")
    assert code == 1 and paths.MARKER_FILE in err and err.count("\n") == 1


def test_cli_other_raw_folder_refused(mod, tmp_path):
    other = tmp_path / "raw2"
    (other / TASK).mkdir(parents=True)
    shutil.copy2(mod.tp.raw_task_dir / "task.yaml", other / TASK / "task.yaml")
    code, _, err = run_cli("convert", TASK, "--raw-root", str(other), "--out-root", str(mod.out_root))
    assert code == 1 and "was made from" in err


def test_cli_missing_task(tmp_path):
    code, _, err = run_cli("convert", "nope", "--raw-root", str(tmp_path / "raw"), "--out-root", str(tmp_path / "o"))
    assert code == 1 and "task.yaml: missing" in err


def test_cli_no_episodes(tmp_path):
    se.make_task(tmp_path / "raw", TASK)
    code, out, err = cli_convert(tmp_path / "raw", tmp_path / "out")
    assert code == 1 and "no episodes to convert" in err


def test_cli_force_and_pose_source_options(mod, tmp_path):
    tp = clone(mod, tmp_path)
    code, out, _ = run_cli("convert", TASK, "--raw-root", str(tmp_path / "raw"), "--out-root", str(tmp_path / "out"),
                           "--force", "--pose-source", "measured", "--episodes", "0")
    assert code == 0 and " converted" in out.splitlines()[1]
    assert read_status(tp.status_json(ep(0)))["pose_source"] == "measured"


# --------------------------------------------------------------------------------------------- last: raw unchanged

def test_raw_root_unchanged(mod):
    """Rule 1: after every conversion above, the shared raw root is byte for byte what it was."""
    assert se.tree_hash(mod.raw_root) == mod.raw_hash
