"""``cameras.ultrawide.slam_mask`` in the stage hashes: only the slam stage hashes the mask's polygons (so adding or
changing a mask re-runs slam and, through the deps, pose, validate, images and export); a rig without a mask hashes
exactly as before masks existed; the calibration hash never depends on the mask; the ``source`` note is not hashed.
The re-runs themselves: test_process_rerun.py and tests/run/test_run_range.py."""

from __future__ import annotations

from pathlib import Path

import yaml

from dataproc.config import TaskConfig, parse_rig
from dataproc.files import canonical_json
from dataproc.process import EpisodeProcessor
from dataproc.report import STAGE_NAMES
from dataproc.stages.base import Calibration, ProcessContext, rig_sections
from dataproc.stages.calibration import calibration_inputs_hash

RIG_V1 = Path(__file__).resolve().parents[1] / "configs" / "rigs" / "rig_v1.yaml"
KEY = "cameras.ultrawide.slam_mask"
POLYGONS = [[[0, 520], [820, 520], [820, 616], [0, 616]], [[300, 200], [720, 200], [410, 450]]]    # 820x616
#: rig_sections keys and the cameras.ultrawide section keys before masks existed
SECTIONS = {"rig_id", "cameras.main", "cameras.ultrawide", "cameras.resolution_fps", "virtual_ultrawide", "tcp",
            "imu", "imu.time_offset_ns", "aruco", "world_offset", "gripper"}
CAMERA_SECTION = {"resolution", "fps", "model", "T_imu_cam", "timestamp_ref", "time_offset_ns"}


def _rig(mask: dict | None = None):
    d = yaml.safe_load(RIG_V1.read_text())
    if mask is not None:
        d["cameras"]["ultrawide"]["slam_mask"] = mask
    return parse_rig(d)


def _floats(polygons) -> list:
    return [[[float(x), float(y)] for x, y in poly] for poly in polygons]


def _processor(rig, tmp_path) -> EpisodeProcessor:
    """An EpisodeProcessor on a context made from ``rig`` alone: computing the hashes reads nothing but the
    episode's MANIFEST.sha256 (missing here, hashed as empty)."""
    task = TaskConfig(label="block_raise", rig=rig.rig_id, created="2026-11-16T00:00:00+00:00")
    ctx = ProcessContext(root=None, folder="block_raise", task_dir=tmp_path, task=task, rig=rig, ledger=[],
                         backend=None, backend_name="orbslam3", force=False,
                         calibration=Calibration("done", "cal", token="gripper_range:0"))
    return EpisodeProcessor(ctx, tmp_path / "episode_1", pi_name="s01_episode_000001")


def _inputs(rig, tmp_path) -> dict:
    """``{stage: canonical_json(inputs)}`` (the inputs hold numpy arrays; this is what the hash is taken of)."""
    p = _processor(rig, tmp_path)
    return {st.name: canonical_json(st.inputs(p.ctx, p.ep)) for st in p.stages}


def _hashes(rig, tmp_path) -> dict:
    return _processor(rig, tmp_path).inputs_hashes()


def test_no_mask_hashes_as_before(tmp_path):
    rig = _rig()
    s = rig_sections(rig)
    assert set(s) == SECTIONS and set(s["cameras.ultrawide"]) == CAMERA_SECTION
    # the slam inputs are exactly what they were before masks existed, so are their hashes
    assert _inputs(rig, tmp_path)["slam"] == canonical_json({"backend": "orbslam3",
                                                              "cameras.ultrawide": s["cameras.ultrawide"],
                                                              "imu": s["imu"]})


def test_mask_changes_only_the_slam_inputs(tmp_path):
    plain, masked = _inputs(_rig(), tmp_path), _inputs(_rig({"source": "a note", "polygons": POLYGONS}), tmp_path)
    assert list(plain) == list(STAGE_NAMES)
    assert {name for name in plain if masked[name] != plain[name]} == {"slam"}
    s0 = rig_sections(_rig())
    assert masked["slam"] == canonical_json({"backend": "orbslam3", "cameras.ultrawide": s0["cameras.ultrawide"],
                                             "imu": s0["imu"], KEY: {"polygons": _floats(POLYGONS)}})  # + polygons
    s = rig_sections(_rig({"source": "a note", "polygons": POLYGONS}))
    assert set(s) == SECTIONS | {KEY} and set(s["cameras.ultrawide"]) == CAMERA_SECTION
    assert s[KEY] == {"polygons": _floats(POLYGONS)}


def test_mask_reruns_slam_pose_validate_images_export(tmp_path):
    plain, masked = _hashes(_rig(), tmp_path), _hashes(_rig({"polygons": POLYGONS}), tmp_path)
    assert {name for name in plain if masked[name] != plain[name]} == {"slam", "pose", "validate", "images", "export"}
    moved = [POLYGONS[0], [[300, 200], [720, 200], [410, 451]]]                         # one point 1 px lower
    again = _hashes(_rig({"polygons": moved}), tmp_path)
    assert {name for name in plain if again[name] != masked[name]} == {"slam", "pose", "validate", "images", "export"}


def test_calibration_hash_ignores_the_mask(tmp_path):
    plain = calibration_inputs_hash(tmp_path / "raw", _rig(), "rig_v1")
    assert calibration_inputs_hash(tmp_path / "raw", _rig({"polygons": POLYGONS}), "rig_v1") == plain


def test_source_note_is_not_hashed(tmp_path):
    a = _hashes(_rig({"source": "traced from episode_000115", "polygons": POLYGONS}), tmp_path)
    assert _hashes(_rig({"source": "another note", "polygons": POLYGONS}), tmp_path) == a
    assert _hashes(_rig({"polygons": POLYGONS}), tmp_path) == a


def test_ints_and_floats_hash_the_same(tmp_path):
    ints = [[[0, 0], [720, 0], [720, 600]]]
    a = _hashes(_rig({"polygons": ints}), tmp_path)
    assert _hashes(_rig({"polygons": [[[0.0, 0.0], [720.0, 0.0], [720.0, 600.0]]]}), tmp_path) == a
    assert _hashes(_rig({"polygons": [[[0, 0], [720.5, 0], [720, 600]]]}), tmp_path)["slam"] != a["slam"]
