"""teleop_train.episode_reader: one raw episode into arrays (PLAN.md §5.1) and every file-level refusal (§5.3)."""
from __future__ import annotations

import csv
import dataclasses
import hashlib
import shutil
from pathlib import Path
from typing import Callable

import numpy as np
import pytest
import yaml

import synthetic_episode as se
from teleop_train.episode_reader import (OUTCOMES, POSE_COLUMNS, RIGID_TOL, EpisodeError, RawEpisode,
                                         count_video_frames, pose_columns, read_episode, rigid_error)


@pytest.fixture
def task_dir(tmp_path):
    return se.make_task(tmp_path / "raw")


@pytest.fixture(scope="module")
def template(tmp_path_factory) -> se.SyntheticEpisode:
    """One default episode, made once per module (generating the JPEGs is the slow part) and copied per test."""
    return se.make_episode(se.make_task(tmp_path_factory.mktemp("template")), 0)


@pytest.fixture
def fresh(template):
    """fresh(task_dir, number=0, meta_edit=None): a copy of the template as episode_<number>, meta.yaml edited
    (episode / episode_number follow the number, then meta_edit)."""
    def make(task_dir: Path, number: int = 0, meta_edit: Callable[[dict], None] | None = None) -> se.SyntheticEpisode:
        name = f"episode_{number:03d}"
        dst = task_dir / name
        shutil.copytree(template.episode_dir, dst)
        meta = yaml.safe_load((dst / "meta.yaml").read_text(encoding="utf-8"))
        meta.update(episode=name, episode_number=number)
        if meta_edit is not None:
            meta_edit(meta)
        (dst / "meta.yaml").write_text(yaml.safe_dump(meta, sort_keys=False), encoding="utf-8")
        return dataclasses.replace(template, episode_dir=dst, meta=meta)
    return make


def edit_csv(path: Path, fn: Callable[[list[str], list[list[str]]], None]) -> None:
    """Rewrite a CSV after fn(header, rows) changed them in place."""
    with open(path, newline="", encoding="utf-8") as f:
        rows = list(csv.reader(f))
    header, data = rows[0], rows[1:]
    fn(header, data)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f, lineterminator="\n")
        w.writerow(header)
        w.writerows(data)


def set_cell(path: Path, row: int, column: str, value: str) -> None:
    def fn(header, data):
        data[row][header.index(column)] = value
    edit_csv(path, fn)


def drop_column(path: Path, column: str) -> None:
    def fn(header, data):
        i = header.index(column)
        del header[i]
        for r in data:
            del r[i]
    edit_csv(path, fn)


def read_refused(ep_dir: Path, match: str, pose_source: str = "target") -> str:
    root = ep_dir.parent.parent
    before = se.tree_hash(root)
    with pytest.raises(EpisodeError, match=match) as e:
        read_episode(ep_dir, pose_source)
    msg = str(e.value)
    assert "\n" not in msg and msg
    assert se.tree_hash(root) == before
    return msg


# ---- a good episode -------------------------------------------------------------------------------------------

def test_read_good_episode(task_dir, fresh):
    ep = fresh(task_dir, 4)
    before = se.tree_hash(task_dir.parent)
    r = read_episode(ep.episode_dir)
    assert se.tree_hash(task_dir.parent) == before
    assert isinstance(r, RawEpisode)
    assert r.episode_dir == ep.episode_dir and r.name == "episode_004" and r.task == "pick_test"
    assert r.outcome == "success" and r.pose_source == "target"
    assert r.meta["episode_number"] == 4 and r.meta["pc_hello"]["commit"] == "abc1234"
    assert r.meta_sha256 == hashlib.sha256((ep.episode_dir / "meta.yaml").read_bytes()).hexdigest()
    assert r.rig == se.load_rig()
    assert r.F_T_TCP.dtype == np.float64 and r.F_T_TCP.shape == (4, 4)
    np.testing.assert_array_equal(r.F_T_TCP, se.F_T_TCP)

    S = len(ep.t_state_ns)
    assert r.t_state_ns.dtype == np.int64 and r.seq.dtype == np.int64
    assert r.T_state.shape == (S, 4, 4) and r.T_state.dtype == np.float64
    assert r.width_m.shape == (S,) and r.width_m.dtype == np.float64
    np.testing.assert_array_equal(r.t_state_ns, ep.t_state_ns)
    np.testing.assert_array_equal(r.seq, 1000 + np.arange(S))
    # floats are written in repr form: read back exactly
    np.testing.assert_array_equal(r.T_state, ep.truth.target(ep.t_state_ns))
    np.testing.assert_array_equal(r.width_m, ep.truth.width(ep.t_state_ns))
    assert r.dropped_states == 0

    for t, idx, want in ((r.main_t_ns, r.main_video_index, ep.main_t_ns), (r.uw_t_ns, r.uw_video_index, ep.uw_t_ns)):
        assert t.dtype == np.int64 and idx.dtype == np.int64
        np.testing.assert_array_equal(t, want)
        np.testing.assert_array_equal(idx, np.arange(len(want)))
    assert r.dropped_frames == {"main": 0, "ultrawide": 0}
    assert r.main_video == ep.episode_dir / "main.mkv" and r.uw_video == ep.episode_dir / "ultrawide.mkv"


def test_measured_pose_source(task_dir, fresh):
    ep = fresh(task_dir)
    r = read_episode(str(ep.episode_dir), "measured")
    assert r.pose_source == "measured"
    np.testing.assert_array_equal(r.T_state, ep.truth.measured(ep.t_state_ns))
    t = read_episode(ep.episode_dir, "target")
    np.testing.assert_allclose(r.T_state[:, :3, 3] - t.T_state[:, :3, 3],
                               np.tile(se.MEASURED_OFFSET_M, (len(ep.t_state_ns), 1)), atol=1e-12)


def test_pose_columns():
    assert POSE_COLUMNS == {"target": "O_T_TCP_target", "measured": "O_T_TCP"}
    assert pose_columns("target") == [f"O_T_TCP_target_{r}{c}" for r in range(4) for c in range(4)]
    assert pose_columns("measured")[:2] == ["O_T_TCP_00", "O_T_TCP_01"] and pose_columns("measured")[-1] == "O_T_TCP_33"
    with pytest.raises(ValueError):
        pose_columns("commanded")


def test_unknown_pose_source_is_value_error(task_dir, fresh):
    ep = fresh(task_dir)
    with pytest.raises(ValueError) as e:
        read_episode(ep.episode_dir, "commanded")
    assert type(e.value) is ValueError


def test_fail_outcome_is_read(task_dir, fresh):
    ep = fresh(task_dir, meta_edit=lambda m: m.update(outcome="fail"))
    assert read_episode(ep.episode_dir).outcome == "fail"
    assert OUTCOMES == ("success", "fail")


def test_gaps_are_not_refused_here(task_dir):
    # the gap limits apply to the kept span and belong to convert (§5.3); the reader only reads
    ep = se.make_episode(task_dir, 0, state_gap=(3.0, 2.0), main_gap=(3.0, 0.5), uw_gap=(3.0, 0.5))
    r = read_episode(ep.episode_dir)
    assert np.diff(r.t_state_ns).max() >= 2_000_000_000
    np.testing.assert_array_equal(r.main_video_index, np.arange(len(ep.main_t_ns)))


def test_rigid_error():
    T = np.tile(np.eye(4), (3, 1, 1))
    T[1, :3, :3] *= 1.001
    T[2, 0, 0] = np.nan
    err = rigid_error(T)
    assert err[0] == 0.0 and err[1] > RIGID_TOL and np.isinf(err[2])
    B = np.eye(4)
    B[3, 0] = 1e-3
    assert rigid_error(B) > RIGID_TOL
    M = np.eye(4)
    M[0, 0] = -1.0                                       # a reflection: orthonormal, det -1
    assert rigid_error(M) > RIGID_TOL


# ---- dropped mapped times -------------------------------------------------------------------------------------

def test_blank_state_t_ns_dropped_and_counted(task_dir):
    ep = se.make_episode(task_dir, 0, blank_state_t_ns=5)
    r = read_episode(ep.episode_dir)
    assert r.dropped_states == 5
    np.testing.assert_array_equal(r.t_state_ns, ep.t_state_ns[5:])
    np.testing.assert_array_equal(r.seq, 1005 + np.arange(len(ep.t_state_ns) - 5))
    np.testing.assert_array_equal(r.T_state, ep.truth.target(ep.t_state_ns[5:]))
    np.testing.assert_array_equal(r.width_m, ep.truth.width(ep.t_state_ns[5:]))


def test_blank_main_t_mid_ns_keeps_video_index(task_dir):
    ep = se.make_episode(task_dir, 0, blank_main_t_ns=4)
    r = read_episode(ep.episode_dir)
    assert r.dropped_frames == {"main": 4, "ultrawide": 0}
    np.testing.assert_array_equal(r.main_t_ns, ep.main_t_ns[4:])
    np.testing.assert_array_equal(r.main_video_index, np.arange(4, len(ep.main_t_ns)))


def test_blank_t_mid_ns_in_the_middle(task_dir, fresh):
    ep = fresh(task_dir)
    csv_path = ep.episode_dir / "ultrawide_frames.csv"
    for k in (10, 11, 50):
        set_cell(csv_path, k, "t_mid_ns", "")
    r = read_episode(ep.episode_dir)
    assert r.dropped_frames == {"main": 0, "ultrawide": 3}
    keep = np.setdiff1d(np.arange(len(ep.uw_t_ns)), [10, 11, 50])
    np.testing.assert_array_equal(r.uw_video_index, keep)
    np.testing.assert_array_equal(r.uw_t_ns, ep.uw_t_ns[keep])


def test_blank_t_ns_in_the_middle(task_dir, fresh):
    ep = fresh(task_dir)
    set_cell(ep.episode_dir / "robot_states.csv", 100, "t_ns", "")
    r = read_episode(ep.episode_dir)
    assert r.dropped_states == 1
    np.testing.assert_array_equal(r.t_state_ns, np.delete(ep.t_state_ns, 100))
    np.testing.assert_array_equal(r.seq, np.delete(1000 + np.arange(len(ep.t_state_ns)), 100))


def test_video_index_checked_on_dropped_rows_too(task_dir):
    ep = se.make_episode(task_dir, 0, blank_main_t_ns=3)
    set_cell(ep.episode_dir / "main_frames.csv", 1, "video_index", "7")
    read_refused(ep.episode_dir, r"main_frames.csv line 3: video_index 7 != row number 1")


# ---- count_video_frames ---------------------------------------------------------------------------------------

def test_count_video_frames(task_dir):
    ep = se.make_episode(task_dir, 0, extra_main_packets=3)
    assert count_video_frames(ep.episode_dir / "main.mkv") == len(ep.main_t_ns) + 3
    assert count_video_frames(ep.episode_dir / "ultrawide.mkv") == len(ep.uw_t_ns)


def test_count_video_frames_unreadable(tmp_path):
    bad = tmp_path / "x.mkv"
    bad.write_bytes(b"not a matroska file at all" * 10)
    with pytest.raises(EpisodeError, match="x.mkv: unreadable") as e:
        count_video_frames(bad)
    assert "\n" not in str(e.value)
    with pytest.raises(EpisodeError, match="missing"):
        count_video_frames(tmp_path / "none.mkv")
    empty = tmp_path / "empty.mkv"
    empty.write_bytes(b"")
    with pytest.raises(EpisodeError, match="empty.mkv"):
        count_video_frames(empty)


def test_count_video_frames_header_only(tmp_path):
    """A writer closed without frames leaves only the header: no player opens it, nor does av (an EpisodeError)."""
    path = tmp_path / "h.mkv"
    se.write_mkv(path, [], se.WIDTH, se.HEIGHT)
    with pytest.raises(EpisodeError, match="h.mkv: unreadable") as e:
        count_video_frames(path)
    assert "\n" not in str(e.value)


def test_extra_packets_refused(task_dir):
    ep = se.make_episode(task_dir, 0, extra_main_packets=2)
    n = len(ep.main_t_ns)
    read_refused(ep.episode_dir, rf"main.mkv: {n + 2} frames but main_frames.csv has {n} rows")


def test_missing_csv_row_refused(task_dir, fresh):
    ep = fresh(task_dir)
    edit_csv(ep.episode_dir / "ultrawide_frames.csv", lambda h, d: d.pop())
    n = len(ep.uw_t_ns)
    read_refused(ep.episode_dir, rf"ultrawide.mkv: {n} frames but ultrawide_frames.csv has {n - 1} rows")


# ---- meta.yaml and COMPLETE refusals --------------------------------------------------------------------------

META_FAULTS = [
    ("version_2", lambda m: m.update(meta_version=2), "meta_version 2"),
    ("version_true", lambda m: m.update(meta_version=True), "meta_version True"),
    ("version_missing", lambda m: m.pop("meta_version"), "meta_version None"),
    ("outcome_bad", lambda m: m.update(outcome="aborted"), "outcome 'aborted'"),
    ("outcome_missing", lambda m: m.pop("outcome"), "outcome None"),
    ("rig_not_mapping", lambda m: m.update(rig=["a"]), "rig is not a mapping"),
    ("hello_missing", lambda m: m.pop("pc_hello"), "F_T_TCP missing"),
    ("ftcp_missing", lambda m: m["pc_hello"].pop("F_T_TCP"), "F_T_TCP missing"),
    ("ftcp_3x4", lambda m: m["pc_hello"].update(F_T_TCP=se.F_T_TCP[:3].tolist()), "not a 4x4"),
    ("ftcp_flat", lambda m: m["pc_hello"].update(F_T_TCP=se.F_T_TCP.reshape(-1).tolist()), "not a 4x4"),
    ("ftcp_text", lambda m: m["pc_hello"].update(F_T_TCP=[["1", 0, 0, 0]] + se.F_T_TCP[1:].tolist()), "not a 4x4"),
    ("ftcp_bool", lambda m: m["pc_hello"].update(F_T_TCP=[[True, 0, 0, 0]] + se.F_T_TCP[1:].tolist()), "not a 4x4"),
    ("ftcp_scaled", lambda m: m["pc_hello"].update(F_T_TCP=(se.F_T_TCP * [[1.01], [1.01], [1.01], [1]]).tolist()),
     "F_T_TCP is not a rigid transform"),
    ("ftcp_nan", lambda m: m["pc_hello"].update(F_T_TCP=[[float("nan"), 0, 0, 0]] + se.F_T_TCP[1:].tolist()),
     "F_T_TCP is not a rigid transform"),
    ("task_other", lambda m: m.update(task="other_task"), "task 'other_task' != task folder 'pick_test'"),
    ("episode_other", lambda m: m.update(episode="episode_009"), "episode 'episode_009' != folder 'episode_000'"),
]


@pytest.mark.parametrize("edit,match", [(e, m) for _, e, m in META_FAULTS], ids=[i for i, _, _ in META_FAULTS])
def test_meta_refusals(task_dir, edit, match, fresh):
    ep = fresh(task_dir, meta_edit=edit)
    read_refused(ep.episode_dir, match)


def test_ftcp_with_int_entries_is_fine(task_dir, fresh):
    ep = fresh(task_dir, meta_edit=lambda m: m["pc_hello"].update(
        F_T_TCP=[[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]))
    np.testing.assert_array_equal(read_episode(ep.episode_dir).F_T_TCP, np.eye(4))


def test_rig_null_says_why(task_dir, fresh):
    ep = fresh(task_dir, meta_edit=lambda m: m.update(rig=None, rig_note="/x/rig_pi_v3_640.yaml: FileNotFoundError"))
    msg = read_refused(ep.episode_dir, "rig is null")
    assert "FileNotFoundError" in msg


def test_rig_null_without_note(task_dir, fresh):
    ep = fresh(task_dir, meta_edit=lambda m: m.update(rig=None, rig_note="line one\nline two"))
    msg = read_refused(ep.episode_dir, "rig is null")
    assert "line one" in msg and "line two" not in msg


def test_no_complete(task_dir, fresh):
    ep = fresh(task_dir)
    (ep.episode_dir / "COMPLETE").unlink()
    read_refused(ep.episode_dir, "COMPLETE: missing")


def test_meta_missing(task_dir, fresh):
    ep = fresh(task_dir)
    (ep.episode_dir / "meta.yaml").unlink()
    read_refused(ep.episode_dir, "meta.yaml: missing")


@pytest.mark.parametrize("content,match", [
    (b"{unclosed: [\n", "meta.yaml: unreadable"),
    (b"\xff\xfe\x00bad", "meta.yaml: unreadable"),
    (b"- just\n- a list\n", "meta.yaml: not a mapping"),
    (b"", "meta.yaml: not a mapping"),
    (b"!!python/object:os.system {}\n", "meta.yaml: unreadable"),
])
def test_meta_unreadable(task_dir, content, match, fresh):
    ep = fresh(task_dir)
    (ep.episode_dir / "meta.yaml").write_bytes(content)
    read_refused(ep.episode_dir, match)


def test_not_a_folder(tmp_path):
    with pytest.raises(EpisodeError, match="not a folder"):
        read_episode(tmp_path / "episode_000")


# ---- missing files and columns --------------------------------------------------------------------------------

@pytest.mark.parametrize("name", ["robot_states.csv", "main_frames.csv", "ultrawide_frames.csv", "main.mkv",
                                  "ultrawide.mkv"])
def test_missing_file(task_dir, name, fresh):
    ep = fresh(task_dir)
    (ep.episode_dir / name).unlink()
    read_refused(ep.episode_dir, f"{name}: missing")


@pytest.mark.parametrize("name,column", [
    ("robot_states.csv", "t_ns"), ("robot_states.csv", "seq"), ("robot_states.csv", "gripper_width_m"),
    ("robot_states.csv", "O_T_TCP_target_23"), ("main_frames.csv", "video_index"), ("main_frames.csv", "t_mid_ns"),
    ("ultrawide_frames.csv", "t_mid_ns"),
])
def test_missing_column(task_dir, name, column, fresh):
    ep = fresh(task_dir)
    drop_column(ep.episode_dir / name, column)
    read_refused(ep.episode_dir, f"{name}: column {column} missing")


def test_missing_column_of_the_other_pose_source_is_fine(task_dir, fresh):
    ep = fresh(task_dir)
    drop_column(ep.episode_dir / "robot_states.csv", "O_T_TCP_target_23")
    np.testing.assert_array_equal(read_episode(ep.episode_dir, "measured").T_state,
                                  ep.truth.measured(ep.t_state_ns))
    read_refused(ep.episode_dir, "column O_T_TCP_target_23 missing", "target")


def test_empty_csv(task_dir, fresh):
    ep = fresh(task_dir)
    (ep.episode_dir / "robot_states.csv").write_text("")
    read_refused(ep.episode_dir, "robot_states.csv: empty")


def test_header_only_csv(task_dir, fresh):
    ep = fresh(task_dir)
    edit_csv(ep.episode_dir / "robot_states.csv", lambda h, d: d.clear())
    read_refused(ep.episode_dir, r"robot_states.csv: 0 rows with a mapped t_ns \(0 without\), need at least 2")


def test_wrong_field_count(task_dir, fresh):
    ep = fresh(task_dir)
    edit_csv(ep.episode_dir / "robot_states.csv", lambda h, d: d[3].pop())
    read_refused(ep.episode_dir, r"robot_states.csv line 5: \d+ fields for \d+ columns")


# ---- too few rows ---------------------------------------------------------------------------------------------

def test_too_few_states(task_dir, fresh):
    ep = fresh(task_dir)
    n = len(ep.t_state_ns)
    edit_csv(ep.episode_dir / "robot_states.csv", lambda h, d: [r.__setitem__(h.index("t_ns"), "") for r in d[1:]])
    read_refused(ep.episode_dir, rf"robot_states.csv: 1 rows with a mapped t_ns \({n - 1} without\)")


def test_too_few_frames(task_dir, fresh):
    ep = fresh(task_dir)
    n = len(ep.main_t_ns)
    edit_csv(ep.episode_dir / "main_frames.csv", lambda h, d: [r.__setitem__(h.index("t_mid_ns"), "") for r in d])
    read_refused(ep.episode_dir, rf"main_frames.csv: 0 rows with a mapped t_mid_ns \({n} without\)")


def test_too_few_ultrawide_frames(task_dir, fresh):
    ep = fresh(task_dir)
    csv_path = ep.episode_dir / "ultrawide_frames.csv"
    edit_csv(csv_path, lambda h, d: [r.__setitem__(h.index("t_mid_ns"), "") for r in d[1:]])
    read_refused(ep.episode_dir, r"ultrawide_frames.csv: 1 rows with a mapped t_mid_ns")


# ---- times ----------------------------------------------------------------------------------------------------

def test_t_ns_not_increasing(task_dir, fresh):
    ep = fresh(task_dir)
    p = ep.episode_dir / "robot_states.csv"
    set_cell(p, 20, "t_ns", str(int(ep.t_state_ns[19])))
    read_refused(ep.episode_dir, rf"robot_states.csv line 22: t_ns {ep.t_state_ns[19]} not after {ep.t_state_ns[19]}")


def test_t_ns_going_back_across_a_dropped_row(task_dir, fresh):
    ep = fresh(task_dir)
    p = ep.episode_dir / "robot_states.csv"
    set_cell(p, 20, "t_ns", "")
    set_cell(p, 21, "t_ns", str(int(ep.t_state_ns[19]) - 1))
    read_refused(ep.episode_dir, r"robot_states.csv line 23: t_ns \d+ not after \d+")


@pytest.mark.parametrize("cam", ["main", "ultrawide"])
def test_t_mid_ns_not_increasing(task_dir, cam, fresh):
    ep = fresh(task_dir)
    t = ep.main_t_ns if cam == "main" else ep.uw_t_ns
    set_cell(ep.episode_dir / f"{cam}_frames.csv", 30, "t_mid_ns", str(int(t[28])))
    read_refused(ep.episode_dir, rf"{cam}_frames.csv line 32: t_mid_ns {t[28]} not after {t[29]}")


@pytest.mark.parametrize("cam", ["main", "ultrawide"])
def test_video_index_not_row_number(task_dir, cam, fresh):
    ep = fresh(task_dir)
    set_cell(ep.episode_dir / f"{cam}_frames.csv", 12, "video_index", "13")
    read_refused(ep.episode_dir, rf"{cam}_frames.csv line 14: video_index 13 != row number 12")


@pytest.mark.parametrize("name,column,value,match", [
    ("robot_states.csv", "t_ns", "1.5", "t_ns '1.5' is not an integer"),
    ("robot_states.csv", "seq", "", "seq '' is not an integer"),
    ("robot_states.csv", "O_T_TCP_target_03", "", "O_T_TCP_target_03 '' is not a number"),
    ("robot_states.csv", "gripper_width_m", "wide", "gripper_width_m 'wide' is not a number"),
    ("robot_states.csv", "gripper_width_m", "", "gripper_width_m '' is not a number"),
    ("robot_states.csv", "gripper_width_m", "nan", "gripper_width_m 'nan' is not finite"),
    ("robot_states.csv", "gripper_width_m", "inf", "gripper_width_m 'inf' is not finite"),
    ("main_frames.csv", "t_mid_ns", "12x", "t_mid_ns '12x' is not an integer"),
    ("main_frames.csv", "video_index", "", "video_index '' is not an integer"),
    ("ultrawide_frames.csv", "t_mid_ns", "None", "t_mid_ns 'None' is not an integer"),
])
def test_bad_values(task_dir, name, column, value, match, fresh):
    ep = fresh(task_dir)
    set_cell(ep.episode_dir / name, 7, column, value)
    msg = read_refused(ep.episode_dir, match)
    assert f"{name} line 9" in msg


# ---- rigid transforms -----------------------------------------------------------------------------------------

@pytest.mark.parametrize("column,value", [
    ("O_T_TCP_target_00", "1.001"),            # scaled rotation
    ("O_T_TCP_target_01", "0.01"),             # sheared
    ("O_T_TCP_target_30", "0.5"),              # bottom row
    ("O_T_TCP_target_33", "2.0"),
    ("O_T_TCP_target_12", "nan"),
])
def test_pose_not_rigid(task_dir, column, value, fresh):
    ep = fresh(task_dir)
    set_cell(ep.episode_dir / "robot_states.csv", 40, column, value)
    read_refused(ep.episode_dir, r"robot_states.csv line 42: O_T_TCP_target is not a rigid transform")
    # the other pose source is untouched and still reads
    read_episode(ep.episode_dir, "measured")


def test_measured_pose_not_rigid(task_dir, fresh):
    ep = fresh(task_dir)
    set_cell(ep.episode_dir / "robot_states.csv", 0, "O_T_TCP_11", "0.9")
    read_refused(ep.episode_dir, r"line 2: O_T_TCP is not a rigid transform", "measured")
    read_episode(ep.episode_dir, "target")


def test_reflection_refused(task_dir, fresh):
    ep = fresh(task_dir)
    p = ep.episode_dir / "robot_states.csv"
    for c in ("O_T_TCP_target_00", "O_T_TCP_target_10", "O_T_TCP_target_20"):
        def neg(h, d, c=c):
            d[5][h.index(c)] = repr(-float(d[5][h.index(c)]))
        edit_csv(p, neg)
    read_refused(ep.episode_dir, "line 7: O_T_TCP_target is not a rigid transform")


def test_tiny_rounding_is_rigid(task_dir, fresh):
    ep = fresh(task_dir)
    p = ep.episode_dir / "robot_states.csv"
    set_cell(p, 3, "O_T_TCP_target_00", repr(1.0 + 1e-9))
    read_episode(ep.episode_dir)


def test_rows_of_dropped_states_are_not_parsed_for_poses(task_dir):
    # a row with no mapped time is dropped as a whole, so its other cells do not matter
    ep = se.make_episode(task_dir, 0, blank_state_t_ns=2)
    set_cell(ep.episode_dir / "robot_states.csv", 0, "O_T_TCP_target_00", "")
    assert read_episode(ep.episode_dir).dropped_states == 2


def test_raw_folder_unchanged_by_reads(task_dir, fresh):
    for n in range(2):
        fresh(task_dir, n)
    before = se.tree_hash(task_dir.parent)
    for n in range(2):
        read_episode(task_dir / f"episode_{n:03d}")
        read_episode(task_dir / f"episode_{n:03d}", "measured")
    assert se.tree_hash(task_dir.parent) == before
