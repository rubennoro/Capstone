"""``umiproc map`` end to end without the native binary (a stand-in build and a fake umiproc_slam runner): the two
SLAM runs it makes and their options, what a failed map leaves (nothing attached, nothing recorded, the report in
maps_failed/, the same recording can be run again), what a passed map creates (the task folder with the map next to
raw/, both ledgers), the input errors (exit 2, nothing created), and what ``umiproc status`` says about the map
(passed, fingerprint, out of date), and a pass that is interrupted or that intake refuses after every check
passed (report.json before take(), the map kept in maps_failed/, status's INCOMPLETE hint)."""

import csv
import json
from pathlib import Path

import pytest
import yaml

from umiproc import cli as umicli
from umiproc.dp.paths import DataRoot
from umiproc.map import build as B
from umiproc.map import checks as C
from umiproc.paths import rig_file, rig_lens_file
from umiproc.status import folder_status, format_folder_status, format_root_status, root_status
from umiproc_testlib import FakeSlam, fake_slam_build, pi_map_recording, pi_recording, tree_digest

RIG = "rig_pi_v3"
quiet = lambda *a, **k: None  # noqa: E731


@pytest.fixture
def env(tmp_path, monkeypatch):
    src = tmp_path / "pi_incoming"
    src.mkdir()
    pi_map_recording(src, 1)
    binary = fake_slam_build(tmp_path / "fork", monkeypatch)
    return {"root": DataRoot(tmp_path / "root"), "src": src, "binary": binary, "tmp": tmp_path}


def _map(env, name="episode_000001", task="block_raise", runner=None, **kw):
    runner = runner if runner is not None else FakeSlam()
    out = B.build_map(env["root"], task, name, source=env["src"], rig_id=RIG, log=quiet, runner=runner, **kw)
    return out, runner


def _sources(root):
    p = root.path / "pi_sources.csv"
    if not p.is_file():
        return []
    with open(p, newline="") as f:
        return list(csv.DictReader(f))


def _pass_m4_m5(monkeypatch):
    monkeypatch.setattr(C, "check_m4", lambda *a, **k: C.CheckResult("M4", True, {}, "ok (test)"))
    monkeypatch.setattr(C, "check_m5", lambda *a, **k: C.CheckResult("M5", True, {}, "ok (test)"))


def _nothing_taken(root):
    assert not (root.tasks).exists() or not any(root.tasks.iterdir())
    assert _sources(root) == []
    assert not root.incoming.exists() or not any(root.incoming.iterdir())
    assert not (root.path / B.WORK_DIR).exists()


# --------------------------------------------------------------------------------------------- the runs

def test_two_runs_with_umis_options(env):
    o, fake = _map(env)
    assert len(fake.calls) == 2
    mapping, reloc = fake.calls

    def opt(cmd, k):
        return cmd[cmd.index(k) + 1] if k in cmd else None

    assert opt(mapping, "--save_map").endswith(B.ATLAS_NAME) and opt(mapping, "--load_map") is None
    assert (opt(mapping, "--init_tag_id"), float(opt(mapping, "--init_tag_size")), opt(mapping, "--aruco_dict_id")) \
        == ("13", 0.16, "0")
    assert opt(reloc, "--load_map") == opt(mapping, "--save_map") and opt(reloc, "--save_map") is None
    assert opt(reloc, "--max_lost_frames") == "30" and opt(reloc, "--init_tag_id") is None
    for cmd in (mapping, reloc):
        assert opt(cmd, "--mask_img").endswith("slam_mask.png") and "--no_realtime" not in cmd
        assert opt(cmd, "--setting").endswith("orbslam3_settings.yaml")
        seq = Path(opt(cmd, "--sequence"))
        assert seq.name == "euroc"
    # the EuRoC folder had every ultrawide frame at the native size, unmasked
    rep = o.report
    assert rep["diagnostics"]["euroc"]["size"] == [1280, 960] and "slam_mask" not in rep["diagnostics"]["euroc"]
    assert rep["diagnostics"]["build"]["patches_label"] == "0001-0002 (2 patches)"
    assert rep["diagnostics"]["build"]["binary"] == str(env["binary"])
    assert rep["diagnostics"]["vio_inflation"] == {"noise_density": 1.0, "random_walk": 1.0}
    assert rep["diagnostics"]["lens"]["provenance"] == "nominal, uncalibrated"
    ir = rep["runs"]["mapping"]["init_resets"]              # the mapping CSV's states reach the M1/M2 hint
    assert (ir["source"], ir["resets"], ir["suspect"]) == ("mapping trajectory", 0, False)
    assert o.checks[0].values["init_resets"] == ir


# --------------------------------------------------------------------------------------------- failing

def test_a_failed_map_attaches_nothing_and_can_run_again(env):
    root, src = env["root"], env["src"]
    before = tree_digest(src)
    o, _ = _map(env)                         # stationary fake trajectory: M4 (scale) warns, M5 (no tag) blocks
    assert (o.passed, o.exit_code) == (False, B.EXIT_FAILED)
    assert [c.id for c in o.checks] == ["M1", "M2", "M3", "M4", "M5"]
    assert [c.ok for c in o.checks] == [True, True, True, False, False]
    assert [c.id for c in o.checks if c.failed] == ["M5"] and [c.id for c in o.checks if c.warned] == ["M4"]
    assert o.checks[3].values["scale_status"] == "scale_unobservable"
    assert o.checks[3].values["orientation_windows"] == 0 and o.checks[3].values["orientation_ok"] is False
    assert "not enough tracked span" in o.checks[3].values["orientation"]["reason"]
    assert o.checks[4].values["frames_with_tag"] == 0 and "never detected" in o.checks[4].hints[0]
    _nothing_taken(root)
    assert tree_digest(src) == before
    where = o.where
    assert where.parent == root.path / B.FAILED_DIR / o.pi_name
    rep = json.loads((where / B.REPORT_NAME).read_text())
    assert rep["passed"] is False and rep["kept_in"] == str(where)
    assert [c["id"] for c in rep["checks"] if c["passed"] is not True] == ["M4", "M5"]
    assert rep["warnings"] == ["M4"] and [c["id"] for c in rep["checks"] if c["blocking"]] == ["M1", "M5"]
    for c in rep["checks"]:
        assert c["limits"] and c["message"]
    names = sorted(p.name for p in where.iterdir())
    assert B.ATLAS_NAME not in names                      # no atlas, no frames
    assert {B.MAPPING_DIR, B.RELOCALIZE_DIR, "orbslam3_settings.yaml", "slam_mask.png", B.FINGERPRINT_NAME,
            B.TAG_DETECTIONS_NAME, B.REPORT_NAME} <= set(names)
    assert (where / B.MAPPING_DIR / "trajectory.csv").is_file() and (where / B.RELOCALIZE_DIR /
                                                                       "trajectory.csv.t0_ns").is_file()
    fp = json.loads((where / B.FINGERPRINT_NAME).read_text())
    assert fp["hash"] == rep["fingerprint_hash"]
    # the same recording again (A/B): runs again, kept separately
    o2, _ = _map(env)
    assert o2.exit_code == B.EXIT_FAILED and o2.where != where and o2.where.parent == where.parent
    _nothing_taken(root)
    st = root_status(root.path, RIG)
    assert len(st["failed_maps"]) == 2
    text = format_root_status(st, "test")
    assert "failed maps" in text and "M5 failed" in text and "M4, M5" not in text     # only what blocked


def test_no_map_with_keyframes_skips_the_relocalization(env):
    o, fake = _map(env, runner=FakeSlam(maps=lambda mapping: []))
    assert len(fake.calls) == 1 and o.exit_code == B.EXIT_FAILED
    assert [(c.id, c.passed) for c in o.checks] == [("M1", False), ("M2", None), ("M3", None), ("M4", None),
                                                    ("M5", None)]
    _nothing_taken(env["root"])


def test_a_crashed_mapping_run_fails_m1(env):
    o, fake = _map(env, runner=FakeSlam(rc=1))
    assert o.exit_code == B.EXIT_FAILED and len(fake.calls) == 1
    assert "did not finish" in o.checks[0].message
    _nothing_taken(env["root"])


# --------------------------------------------------------------------------------------------- passing

def test_a_passed_map_creates_the_session_folder(env, monkeypatch):
    _pass_m4_m5(monkeypatch)
    root = env["root"]
    o, _ = _map(env)
    assert (o.passed, o.exit_code, o.folder) == (True, B.EXIT_OK, "block_raise")
    d = root.tasks / "block_raise" / "map" / o.pi_name
    assert o.where == d and (d / "raw" / "COMPLETE").is_file()
    for name in (B.ATLAS_NAME, "orbslam3_settings.yaml", "slam_mask.png", B.FINGERPRINT_NAME, B.TAG_DETECTIONS_NAME,
                 B.REPORT_NAME):
        assert (d / name).is_file(), name
    assert (d / B.MAPPING_DIR / "umiproc_slam.log").is_file() and (d / B.RELOCALIZE_DIR / "trajectory.csv").is_file()
    rep = json.loads((d / B.REPORT_NAME).read_text())
    assert rep["passed"] is True and rep["folder"] == "block_raise" and rep["slot"] == f"map/{o.pi_name}"
    rows = _sources(root)
    assert [(r["kind"], r["status"], r["task"], r["converted_name"]) for r in rows] == \
        [("map", "ingested", "block_raise", o.pi_name)]
    with open(root.tasks / "block_raise" / "episodes.csv", newline="") as f:
        assert [(r["slot"], r["kind"]) for r in csv.DictReader(f)] == [(f"map/{o.pi_name}", "map")]
    assert not (root.path / B.FAILED_DIR).exists() and not (root.path / B.WORK_DIR).exists()
    assert not any(root.incoming.iterdir())
    # status: passed, fingerprint, up to date
    f = folder_status(root.path, "block_raise")
    m = f["map_info"][0]
    assert m["passed"] is True and m["up_to_date"] is True and m["checks"] == {f"M{i}": True for i in range(1, 6)}
    text = format_folder_status(f)
    assert f"map/{o.pi_name}: passed" in text and rep["fingerprint_hash"][:12] in text and "up to date" in text
    assert "label block_raise, map passed," in format_root_status(root_status(root.path, RIG), "test")
    # the data-root rig copy is edited (IMU A/B setting): the map is out of date
    p = rig_file(root.path, RIG)
    doc = yaml.safe_load(p.read_text())
    doc["imu"]["vio_inflation"] = {"noise_density": 10.0, "random_walk": 10.0}
    p.write_text(yaml.safe_dump(doc, sort_keys=False))
    f = folder_status(root.path, "block_raise")
    assert f["map_info"][0]["up_to_date"] is False and "sensors.imu" in f["map_info"][0]["differences"]
    assert "OUT OF DATE" in format_folder_status(f)
    assert "map passed, OUT OF DATE" in format_root_status(root_status(root.path, RIG), "test")
    # the same clip again: refused (already taken); another clip: the next session folder
    with pytest.raises(B.MapInputError, match="already taken"):
        _map(env)
    pi_map_recording(env["src"], 2)
    o2, _ = _map(env, name="episode_000002")
    assert o2.folder == "block_raise_2"


def test_the_effective_lens_is_used_everywhere(env, monkeypatch):
    from umiproc.lens.lensfile import dump_lens, lens_document
    from umiproc.sensors import ensure_rig_files

    root = env["root"]
    ensure_rig_files(root.path, RIG, log=quiet)
    rig_lens_file(root.path, RIG).write_text(dump_lens(lens_document(
        rig_id=RIG, sensor="imx219", resolution=(1280, 960), intrinsics=[580.5, 579.25, 641.0, 478.0],
        distortion=[0.021, -0.006, 0.0012, -0.0003], date="2026-11-16T18:00:00Z", source={}, board={}, metrics={})))
    o, fake = _map(env)
    where = o.where
    tags = json.loads((where / B.TAG_DETECTIONS_NAME).read_text())
    assert tags["lens"]["K"][0][0] == 580.5 and tags["lens"]["D"] == [0.021, -0.006, 0.0012, -0.0003]
    settings = (where / "orbslam3_settings.yaml").read_text()
    assert "Camera1.fx: 580.5" in settings and "lens: calibrated 2026-11-16T18:00:00Z" in settings
    fp = json.loads((where / B.FINGERPRINT_NAME).read_text())["record"]
    assert fp["sensors"]["ultrawide"]["lens"]["provenance"] == "calibrated"
    assert o.report["diagnostics"]["lens"]["provenance"] == "calibrated"


def test_warnings_do_not_block_the_map(env, monkeypatch):
    """M2, M3, M4 and M5's spread/tilt only warn: the folder is created, the report lists the warnings, and status
    and the CLI say so. Only M1 and M5's sightings block."""
    monkeypatch.setattr(C, "check_m4", lambda *a, **k: C.CheckResult("M4", False, {}, "scale off (test)", ["h4"]))
    monkeypatch.setattr(C, "check_m5", lambda *a, **k: C.CheckResult("M5", False, {"sightings": 30},
                                                                     "spread 2 cm (test)", ["h5"], blocking=False))
    root = env["root"]
    o, _ = _map(env)
    assert (o.passed, o.exit_code, o.folder) == (True, B.EXIT_OK, "block_raise")
    assert [c.id for c in o.checks if c.warned] == ["M4", "M5"] and not any(c.failed for c in o.checks)
    rep = json.loads((o.where / B.REPORT_NAME).read_text())
    assert rep["passed"] is True and rep["warnings"] == ["M4", "M5"]
    assert "warnings: M4, M5" in o.message
    f = folder_status(root.path, "block_raise")
    assert f["map_info"][0]["warnings"] == ["M4", "M5"]
    text = format_folder_status(f)
    assert "M4 warning" in text and "M5 warning" in text and "FAILED" not in text
    assert "map passed with warnings (M4, M5)" in format_root_status(root_status(root.path, RIG), "test")
    out = __import__("umiproc.map.cli", fromlist=["format_outcome"]).format_outcome(o, "block_raise", "episode_000001")
    assert "map: PASSED" in out and "(warnings: M4, M5;" in out and "M4 WARN" in out and "M5 WARN" in out
    assert "to improve it (a warning, it does not block)" in out


def test_blocking_defaults():
    assert C.BLOCKING == {"M1": True, "M2": False, "M3": False, "M4": False, "M5": True}
    assert C.not_evaluated("M5", "x").failed and C.not_evaluated("M4", "x").warned
    assert C.not_evaluated("M2", "x").warned and not C.not_evaluated("M2", "x").failed


# --------------------------------------------------------------------------------------------- input errors

def test_input_errors_create_nothing(env, monkeypatch, tmp_path):
    root = env["root"]
    with pytest.raises(B.MapInputError) as e:
        _map(env, name="episode_000099")
    assert e.value.code == "source_missing"
    pi_recording(env["src"], 3)                              # 64x48 at 200 Hz: not this rig's settings
    with pytest.raises(B.MapInputError, match="does not fit rig rig_pi_v3") as e:
        _map(env, name="episode_000003")
    assert e.value.code == "rig_mismatch"
    with pytest.raises(B.MapInputError, match="task name"):
        _map(env, task="bad name")
    monkeypatch.setenv("UMIPROC_SLAM_BINARY", str(tmp_path / "nowhere" / "umiproc_slam"))
    with pytest.raises(B.MapInputError, match="native/build.sh") as e:
        _map(env)
    assert e.value.code == "slam_build"
    _nothing_taken(root)
    assert not (root.path / B.FAILED_DIR).exists()


def test_a_recording_intake_refuses_is_quarantined_without_slam(env, monkeypatch):
    from umiproc import recordings as R

    _pass_m4_m5(monkeypatch)
    root = env["root"]
    other = env["tmp"] / "other_source"
    other.mkdir()
    pi_map_recording(other, 1, seed=77)                      # the same Pi episode number, another recording
    R.take_recording(root, other, "episode_000001", R.MAP, rig_id=RIG, task="earlier", log=quiet)
    o, fake = _map(env)
    assert (o.passed, o.exit_code) == (False, B.EXIT_INPUT) and fake.calls == []
    assert "duplicate" in o.report["refused"] and "quarantine" in o.message
    rows = _sources(root)
    assert [r["status"] for r in rows] == ["ingested", "quarantined"]
    assert not (root.tasks / "block_raise").exists()


# --------------------------------------------------------------------------------------------- the CLI

def test_cli_exit_codes_and_messages(env, capsys):
    # the stand-in binary exits 99: the mapping run does not finish (M1), nothing else is evaluated
    rc = umicli.main(["map", "block_raise", "episode_000001", "--source", str(env["src"]),
                      "--root", str(env["root"].path)])
    out = capsys.readouterr()
    assert rc == 1
    assert "map: FAILED for episode_000001" in out.err and "M1 FAIL" in out.err and "did not finish" in out.err
    assert "M2 n/a" in out.err and "what to change" in out.err and "nothing was attached" in out.err
    assert "maps_failed" in out.err and "SLAM build:" in out.err and "noise_density 1.0" in out.err
    assert f"map: data root {env['root'].path} (from --root)" in out.out
    rc = umicli.main(["map", "block_raise", "episode_000042", "--source", str(env["src"]),
                      "--root", str(env["root"].path)])
    assert rc == 2 and "nothing was created" in capsys.readouterr().err
    rc = umicli.main(["map", "block_raise", "episode_000001", "--root", "/data"])
    assert rc == 2 and "refused" in capsys.readouterr().err
    _nothing_taken(env["root"])


# --------------------------------------------------------------------------------------------- an interrupted pass

def test_a_pass_writes_the_report_into_the_work_folder_before_taking(env, monkeypatch):
    from umiproc import recordings as R

    _pass_m4_m5(monkeypatch)
    real_take, seen = R.take, {}

    def take(root, staged, kind, **kw):
        work = env["root"].path / B.WORK_DIR / "map" / staged.pi_name
        seen["report"] = json.loads((work / B.REPORT_NAME).read_text())
        seen["atlas"] = (work / B.ATLAS_NAME).is_file()
        return real_take(root, staged, kind, **kw)

    monkeypatch.setattr(R, "take", take)
    o, _ = _map(env)
    assert o.passed and seen["atlas"]
    assert seen["report"]["passed"] is True
    assert (seen["report"]["folder"], seen["report"]["slot"]) == ("block_raise", f"map/{o.pi_name}")
    rep = json.loads((o.where / B.REPORT_NAME).read_text())
    assert (rep["folder"], rep["slot"]) == ("block_raise", f"map/{o.pi_name}")


@pytest.mark.parametrize("error", ["IntakeError", "PiIntakeError", "RecordingError"])
def test_an_intake_error_after_a_pass_keeps_the_map_in_maps_failed(env, monkeypatch, error):
    from umiproc import recordings as R
    from umiproc.dp.intake import IntakeError
    from umiproc.dp.pi_intake import PiIntakeError

    _pass_m4_m5(monkeypatch)
    exc = {"IntakeError": IntakeError, "PiIntakeError": PiIntakeError, "RecordingError": R.RecordingError}[error]

    def take(*a, **k):
        raise exc("the disk is full (test)")

    monkeypatch.setattr(R, "take", take)
    o, _ = _map(env)
    assert (o.passed, o.exit_code) == (False, B.EXIT_INPUT) and "the disk is full" in o.message
    _nothing_taken(env["root"])
    where = o.where
    assert where.parent == env["root"].path / B.FAILED_DIR / o.pi_name
    assert (where / B.ATLAS_NAME).is_file() and (where / B.MAPPING_DIR / "umiproc_slam.log").is_file()
    rep = json.loads((where / B.REPORT_NAME).read_text())
    assert rep["passed"] is False and error in rep["intake_error"] and rep["folder"] is None
    assert [c["passed"] for c in rep["checks"]] == [True] * 5
    o2, _ = _map(env)                                       # nothing was taken: it can be run again
    assert o2.exit_code == B.EXIT_INPUT and o2.where != where


def test_status_on_an_interrupted_pass(env, monkeypatch):
    _pass_m4_m5(monkeypatch)
    o, _ = _map(env)
    slot = o.where
    # as after a hard kill between take() and the last os.replace: the slot without report.json, the rest in .work
    work = env["root"].path / B.WORK_DIR / "map" / o.pi_name
    work.mkdir(parents=True)
    (slot / B.REPORT_NAME).rename(work / B.REPORT_NAME)
    text = format_folder_status(folder_status(env["root"].path, "block_raise"))
    assert "INCOMPLETE: the recording was taken but its map was not finished" in text
    assert "already taken" in text and "record a new mapping clip (it creates a new session folder)" in text
    assert f"its map is still in {work}" in text
    with pytest.raises(B.MapInputError, match="already taken"):
        _map(env)
    assert (work / B.REPORT_NAME).is_file()                 # refused before the work folder is touched


# --------------------------------------------------------------------------------------------- --first-map

TWO_MAPS = [{"id": 0, "keyframes": 8, "map_points": 200, "imu_initialized": True, "inertial_ba1": True,
             "inertial_ba2": True},
            {"id": 1, "keyframes": 4, "map_points": 90, "imu_initialized": False, "inertial_ba1": False,
             "inertial_ba2": False}]
ONE_MAP = [TWO_MAPS[0]]


def _full_states(k):
    """The whole clip: initialising, the first piece tracked on fed frames 2-21, lost, a new map from frame 24."""
    if k < 2:
        return 1
    if k < 22:
        return 2
    if k < 24:
        return 3
    return 0 if k == 24 else (1 if k < 27 else 2)


def _two_piece_fake(cut_maps=ONE_MAP):
    """A fake whose first (whole-clip) mapping run ends with two maps; later runs track from frame 2 on."""
    holder = {}

    def state(k, n, mapping):
        return _full_states(k) if len(holder["f"].calls) == 1 else (1 if k < 2 else 2)

    def maps(mapping):
        return TWO_MAPS if len(holder["f"].calls) == 1 else cut_maps

    holder["f"] = FakeSlam(state=state, maps=maps)
    return holder["f"]


@pytest.fixture
def env40(env):
    pi_map_recording(env["src"], 2, n_frames=40)
    return env


def _times(cmd):
    seq = Path(cmd[cmd.index("--sequence") + 1])
    return seq, [int(x) for x in (seq / "times.txt").read_text().split()]


def test_first_map_cuts_the_clip_after_the_first_piece_and_maps_it_again(env40, monkeypatch):
    _pass_m4_m5(monkeypatch)
    seen = []
    fake = _two_piece_fake()

    def spy(cmd, cwd, log_path, timeout_s):              # the sequences, read while they still exist
        seen.append(_times(cmd))
        return fake(cmd, cwd, log_path, timeout_s)

    o, _ = _map(env40, name="episode_000002", runner=spy, first_map=True)
    assert len(fake.calls) == 3                           # whole clip, cut clip, re-localization of the cut clip
    (seq_full, t_full), (seq_cut, t_cut), (seq_rel, t_rel) = seen
    assert seq_full.name == "euroc" and seq_cut.name == seq_rel.name == "euroc_first_map"
    assert len(t_full) >= 30 and t_cut == t_rel == t_full[:22]          # fed frames 0-21: the piece's last tracked
    assert "--save_map" in fake.calls[1] and "--load_map" in fake.calls[2]
    assert o.passed and o.checks[0].ok
    fm = o.report["first_map"]
    assert (fm["applied"], fm["frames"], fm["frames_full"], fm["maps_with_keyframes"]) == (True, 22, len(t_full), 2)
    assert (fm["piece"]["first_tracked"], fm["piece"]["last_tracked"], fm["piece"]["tracked"]) == (2, 21, 20)
    assert fm["end_s"] == pytest.approx((t_full[21] - t_full[0]) / 1e9, abs=1e-3)
    assert [(p["start"], p["last_tracked"]) for p in fm["pieces"]] == [(2, 21), (27, len(t_full) - 1)]
    assert fm["full_run"]["log"] == f"{B.MAPPING_FULL_DIR}/umiproc_slam.log"
    assert o.report["runs"]["mapping_full"]["maps"] == TWO_MAPS and o.report["runs"]["mapping"]["maps"] == ONE_MAP
    assert o.report["runs"]["mapping"]["frames_fed"] == 22 and o.report["diagnostics"]["frames"] == 22
    d = o.where
    assert (d / B.MAPPING_FULL_DIR / "trajectory.csv").is_file() and (d / B.ATLAS_NAME).is_file()
    assert sorted(p.parent.name for p in d.glob(f"*/{B.RUN_PNG_NAME}")) == sorted(
        [B.MAPPING_DIR, B.MAPPING_FULL_DIR, B.RELOCALIZE_DIR])
    assert not (env40["root"].path / B.WORK_DIR).exists()
    text = format_outcome_text(o)
    assert "--first-map: the whole clip gave 2 maps" in text and "22 of" in text and "figures:" in text


def format_outcome_text(o):
    from umiproc.map.cli import format_outcome

    return format_outcome(o, "block_raise", "episode_000002")


def test_first_map_cut_that_still_gives_two_maps_fails_and_keeps_both_runs(env40):
    fake = _two_piece_fake(cut_maps=TWO_MAPS)
    o, _ = _map(env40, name="episode_000002", runner=fake, first_map=True)
    assert o.exit_code == B.EXIT_FAILED and not o.checks[0].ok and o.report["first_map"]["applied"]
    assert (o.where / B.MAPPING_FULL_DIR / "umiproc_slam.log").is_file()
    assert not any("--first-map" in h for h in o.checks[0].hints)    # no advice to do what was done
    _nothing_taken(env40["root"])


def test_without_the_flag_m1_names_the_first_piece(env40):
    fake = FakeSlam(state=lambda k, n, mapping: _full_states(k) if mapping else 2, maps=lambda mapping: TWO_MAPS)
    o, _ = _map(env40, name="episode_000002", runner=fake)
    assert len(fake.calls) == 2 and o.exit_code == B.EXIT_FAILED
    hint = o.checks[0].hints[-1]
    assert "first map piece tracked" in hint and "fed frames 2-21" in hint and "--first-map" in hint
    assert o.report["first_map"] is None and not (o.where / B.MAPPING_FULL_DIR).exists()


def test_first_map_with_one_map_cuts_nothing(env40, monkeypatch):
    _pass_m4_m5(monkeypatch)
    o, fake = _map(env40, name="episode_000002", first_map=True)
    assert len(fake.calls) == 2 and o.passed
    assert o.report["first_map"] == {"applied": False, "reason": "the whole clip gave one map (or none): nothing to cut"}
    assert not (o.where / B.MAPPING_FULL_DIR).exists()


def test_cli_passes_first_map(env40, monkeypatch, capsys):
    seen = {}

    def fake_build(root, task, map_dir, **kw):
        seen.update(kw)
        raise B.MapInputError("stop here")

    monkeypatch.setattr("umiproc.map.cli.build_map", fake_build)
    assert umicli.main(["map", "block_raise", "episode_000002", "--source", str(env40["src"]), "--root",
                        str(env40["root"].path), "--first-map"]) == B.EXIT_INPUT
    assert seen["first_map"] is True
