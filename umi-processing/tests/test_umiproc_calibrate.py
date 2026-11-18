"""``umiproc calibrate``: the gripper range of a session folder from a calibration recording (the copied fit on
synthetic finger tags, plus UMI's 80 % finger-tag rule), what a pass records, that a failure records nothing, the
input errors, and that the effective rig (the lens file) is what the tags are measured with."""

import csv
import json

import pytest

from umiproc import cli as umicli
from umiproc.calibrate import gripper as G
from umiproc.calibrate.cli import format_outcome
from umiproc.paths import rig_lens_file
from umiproc.status import folder_status, format_folder_status
from umiproc_testlib import clone_session, fake_finger_tags, next_lines, session_template, tree_digest

quiet = lambda *a, **k: None  # noqa: E731


@pytest.fixture(scope="module")
def template(tmp_path_factory):
    """A passed session folder and calibration-clip-shaped recordings 1-3 (rig_pi_v3's cameras), built once."""
    return session_template(tmp_path_factory.mktemp("calibrate_template"), demos=(1, 2, 3), n_frames=30)


@pytest.fixture
def env(template, tmp_path, monkeypatch):
    return clone_session(template, tmp_path, monkeypatch)


def _sources(root):
    p = root.path / "pi_sources.csv"
    if not p.is_file():
        return []
    with open(p, newline="") as f:
        return list(csv.DictReader(f))


def _cal(env, name="episode_000001", folder=None):
    return G.calibrate_folder(env["root"], folder or env["folder"], name, source=env["src"], log=quiet)


# --------------------------------------------------------------------------------------------- the 80 % rule

def test_finger_tag_rule_is_umis_min_of_the_two_rates():
    d = {"n_frames": 100, "left_id": 0, "right_id": 1, "left_rate": 0.95, "right_rate": 0.80, "both_rate": 0.78}
    r = G.finger_tag_rule(d)
    assert r["ok"] and r["rate"] == 0.80                     # UMI: min(left, right) >= 0.8, inclusive
    assert not G.finger_tag_rule({**d, "right_rate": 0.79})["ok"]
    assert not G.finger_tag_rule({})["ok"] and G.finger_tag_rule({})["n_frames"] == 0
    assert "tag 0 seen in 95 % and tag 1 in 79 %" in G.finger_tag_rule({**d, "right_rate": 0.79})["message"]


# --------------------------------------------------------------------------------------------- pass

def test_a_passing_calibration_is_recorded_in_the_folder(env, monkeypatch):
    fake_finger_tags(monkeypatch)
    root, folder = env["root"], env["folder"]
    o = _cal(env)
    assert (o.passed, o.exit_code) == (True, G.EXIT_OK), o.message
    task_dir = root.tasks / folder
    gr = json.loads((task_dir / "gripper_range.json").read_text())
    assert gr["min_width"] == pytest.approx(0.044, abs=1e-6) and gr["max_width"] == pytest.approx(0.124, abs=1e-6)
    slot = task_dir / o.slot
    assert o.slot == f"calibration/{o.pi_name}" and (slot / "raw" / "COMPLETE").is_file()
    st = json.loads((slot / "processed" / "status.json").read_text())["gripper_range"]
    assert st["status"] == "done" and st["diagnostics"]["detection_rate"] == 1.0
    rep = json.loads((slot / "processed" / G.REPORT_NAME).read_text())
    assert rep["finger_tag_rule"]["ok"] and rep["lens"]["provenance"] == "nominal, uncalibrated"
    rows = _sources(root)
    assert [(r["kind"], r["status"], r["task"]) for r in rows] == [("map", "ingested", folder),
                                                                   ("calibration", "ingested", folder)]
    with open(task_dir / "episodes.csv", newline="") as f:
        assert [r["kind"] for r in csv.DictReader(f)] == ["map", "calibration"]
    assert not any(root.incoming.iterdir())
    text = format_outcome(o, env["src"])
    assert "calibrate: PASSED" in text and "UMI's finger-tag rule PASS" in text and "umiproc run" in text
    assert "measured opening 0.080 m (expected 0.080 m +- 0.010 m)" in text
    f = folder_status(root.path, folder)
    assert f["gripper_range"] and "opening 0.080 m" in format_folder_status(f)
    # the same recording again: already taken (exit 2)
    with pytest.raises(G.CalibrateInputError, match="already taken"):
        _cal(env)


# --------------------------------------------------------------------------------------------- failures

def test_below_80_percent_fails_and_records_nothing(env, monkeypatch):
    # tag 1 hidden in 30 % of the frames: the copied fit (its floor is 20 %) passes, UMI's rule does not
    fake_finger_tags(monkeypatch, visible=lambda i, n: {0, 1} if i % 10 >= 3 else {0})
    root, src = env["root"], env["src"]
    before = tree_digest(src)
    o = _cal(env)
    assert (o.passed, o.exit_code) == (False, G.EXIT_FAILED)
    assert o.fit.ok and o.reasons == ["finger_tags_below_80"]
    assert not o.rule["ok"] and o.rule["right_rate"] == pytest.approx(0.7, abs=0.04)
    task_dir = root.tasks / env["folder"]
    assert not (task_dir / "gripper_range.json").exists() and not (task_dir / "calibration").exists()
    assert [r["kind"] for r in _sources(root)] == ["map"]
    assert not any(root.incoming.iterdir()) and tree_digest(src) == before
    text = format_outcome(o, src)
    assert "calibrate: FAILED" in text and "UMI's finger-tag rule FAIL" in text and "the right one" in text
    assert "nothing was recorded" in text
    line, cmds = next_lines(text)
    assert "section 6" in line and cmds == [f"umiproc calibrate {env['folder']} <GRIPPER_EPISODE> --source {src}"]
    # fixed: the same recording passes now (nothing was taken)
    fake_finger_tags(monkeypatch)
    assert _cal(env).passed


def test_a_failed_fit_explains_itself(env, monkeypatch):
    fake_finger_tags(monkeypatch, visible=lambda i, n: {0})          # tag 1 never seen
    o = _cal(env)
    assert not o.passed and "gripper_range_fit" in o.reasons and o.exit_code == G.EXIT_FAILED
    text = format_outcome(o, env["src"])
    assert "tag 1 (right finger) was never seen" in text
    assert [r["kind"] for r in _sources(env["root"])] == ["map"]


def test_input_errors(env, monkeypatch, tmp_path):
    fake_finger_tags(monkeypatch)
    with pytest.raises(G.CalibrateInputError, match="does not exist") as e:
        _cal(env, folder="nope")
    assert e.value.code == "folder"
    with pytest.raises(G.CalibrateInputError) as e:
        _cal(env, name="episode_000077")
    assert e.value.code == "source_missing"
    # a folder whose map is incomplete (an interrupted map: no report.json)
    from umiproc.map.build import REPORT_NAME
    for p in (env["root"].tasks / env["folder"] / "map").rglob(REPORT_NAME):
        p.rename(p.with_name("report.json.bak"))
    with pytest.raises(G.CalibrateInputError, match="INCOMPLETE") as e:
        _cal(env)
    assert e.value.code == "map"
    assert [r["kind"] for r in _sources(env["root"])] == ["map"]


def test_one_calibration_per_session_once_episodes_exist(env, monkeypatch):
    fake_finger_tags(monkeypatch)
    assert _cal(env).passed
    o2 = _cal(env, name="episode_000002")                    # no episode yet: it replaces the first
    assert o2.passed and o2.replaced and "replaces the earlier calibration" in format_outcome(o2, env["src"])
    (env["root"].tasks / env["folder"] / "episode_1" / "raw").mkdir(parents=True)
    with pytest.raises(G.CalibrateInputError, match="one calibration per session") as e:
        _cal(env, name="episode_000003")
    assert e.value.code == "already_calibrated"


def test_the_lens_file_is_what_the_tags_are_measured_with(env, monkeypatch):
    from umiproc.dp.stages import calibration as SC
    from umiproc.lens.lensfile import dump_lens, lens_document

    fake_finger_tags(monkeypatch)
    rig_lens_file(env["root"].path, "rig_pi_v3").write_text(dump_lens(lens_document(
        rig_id="rig_pi_v3", sensor="imx219", resolution=(1280, 960), intrinsics=[470.5, 469.0, 641.0, 478.0],
        distortion=[0.021, -0.006, 0.0012, -0.0003], date="2026-11-17T10:00:00Z", source={}, board={}, metrics={})))
    seen = {}
    real = SC.fit_calibration

    def spy(raw_dir, rig, prep, task_rig):
        seen["rig"], seen["prep"] = rig, prep
        return real(raw_dir, rig, prep, task_rig)

    monkeypatch.setattr(SC, "fit_calibration", spy)
    o = _cal(env)
    assert o.passed and o.lens["provenance"] == "calibrated"
    assert seen["rig"].cameras["ultrawide"].model.intrinsics[0] == 470.5
    assert seen["prep"].ultrawide.phys.fx == 470.5          # the remap to the virtual view


def test_cli(env, monkeypatch, capsys):
    fake_finger_tags(monkeypatch)
    root = str(env["root"].path)
    rc = umicli.main(["calibrate", env["folder"], "episode_000042", "--source", str(env["src"]), "--root", root])
    assert rc == 2 and "nothing was created" in capsys.readouterr().err
    rc = umicli.main(["calibrate", env["folder"], "episode_000001", "--source", str(env["src"]), "--root", root])
    out = capsys.readouterr().out
    assert rc == 0 and "calibrate: PASSED" in out
    line, cmds = next_lines(out)
    assert "section 7)" in line and cmds == [f"umiproc run {env['folder']} <FIRST_DEMO> <LAST_DEMO> --source "
                                             f"{env['src']} --root {root}"]
