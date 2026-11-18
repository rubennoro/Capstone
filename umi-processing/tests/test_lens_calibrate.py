"""umiproc lens-calibrate end to end, on synthetic Pi-format recordings of the board rendered through a known fisheye
model (umiproc.lens.synth): the fit recovers the truth, every check passes, --apply installs the lens file with a
backup; a clip with too few views fails check (a); unusable input is refused with a clear message."""

import argparse
import contextlib
import io
import json
import os
import time

import cv2
import numpy as np
import pytest

from umiproc.cli import main
from umiproc.dp.pi_adapter import converted_name, inspect_source
from umiproc.dp.pi_format import write_pi_manifest
from umiproc.lens import calibrate, lensfile, synth
from umiproc.paths import repo_config
from umiproc.slam_config import load_slam_config, mask_keep

PI_NAME = "episode_000900"


def _out(root, episode_dir):
    """``<root>/lens/<pi_name>``: the converted name pi-YYYY-MM-DD_episode_NNNNNN, not the Pi's directory name."""
    name = converted_name(inspect_source(episode_dir))
    assert name.startswith("pi-") and name.endswith("_" + episode_dir.name)
    return root / "lens" / name


def _run(argv, env_root=None):
    """``umiproc <argv>`` in-process: (exit code, stdout, stderr); ``env_root`` sets $UMIPROC_ROOT."""
    out, err = io.StringIO(), io.StringIO()
    with pytest.MonkeyPatch.context() as mp:
        if env_root is not None:
            mp.setenv("UMIPROC_ROOT", str(env_root))
        with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
            code = main(argv)
    return code, out.getvalue(), err.getvalue()


@pytest.fixture(scope="module")
def keep():
    return np.array(mask_keep(load_slam_config(repo_config("slam", "rig_pi_v3.yaml"))))


@pytest.fixture(scope="module")
def recording(tmp_path_factory, keep):
    """40 poses (grid, edges, corners, random) over the image outside the mask, each held 3 frames, 2 moving frames
    between poses (~200 frames), the gripper painted over the mask."""
    src = tmp_path_factory.mktemp("incoming")
    model = synth.truth_model()
    poses = synth.calibration_poses(model, keep, n_extra=16)
    assert len(poses) >= 40
    info = synth.make_recording(src / PI_NAME, model, poses, keep=keep)
    return src, info


def _listing(d):
    return sorted((p.name, p.stat().st_size, p.stat().st_mtime_ns) for p in d.iterdir())


@pytest.fixture(scope="module")
def calibrated(recording, tmp_path_factory):
    src, _ = recording
    root = tmp_path_factory.mktemp("root")
    before = _listing(src / PI_NAME)
    code, out, err = _run(["lens-calibrate", PI_NAME, "--source", str(src)], env_root=root)
    assert _listing(src / PI_NAME) == before                     # the source is only read
    return code, out, err, root


def test_synthetic_recording_is_a_valid_pi_recording(recording):
    src, info = recording
    s = inspect_source(src / PI_NAME, verify_hashes=True)
    assert s.ok, s.problem
    cam = [c for c in s.meta["cameras"] if c["model"] == "imx219"][0]
    assert cam["file"] == "cam0.mp4" and cam["size"] == [1280, 960] and cam["frames"] == info["frames"]
    n = len(info["poses"])
    assert n >= 40 and info["frames"] == 3 * n + 2 * (n - 1)          # 3 held frames per pose, 2 moving between


def test_exit_code_and_outputs(calibrated, recording):
    code, out, err, root = calibrated
    assert code == 0, out + err
    d = _out(root, recording[0] / PI_NAME)
    assert [p.name for p in (root / "lens").iterdir()] == [d.name]
    assert {p.name for p in d.iterdir()} == {"lens.yaml", "report.json", "coverage.png"}
    assert "every check within its limit" in out
    for cid in "abcdefgh":
        assert f"PASS ({cid})" in out
    assert not (root / "rigs").exists()                         # nothing applied without --apply
    cmd = out.strip().splitlines()[-1].strip()
    assert cmd.startswith(f"umiproc lens-calibrate {PI_NAME} --source ") and cmd.endswith(" --apply")
    assert "--root" not in cmd or str(root) in cmd


def _model(path):
    return lensfile.load_lens(path).model


def test_fit_recovers_the_truth(calibrated, recording, keep):
    *_, root = calibrated
    m = _model(_out(root, recording[0] / PI_NAME) / "lens.yaml")
    t = synth.truth_model()
    assert m.fx == pytest.approx(t.fx, rel=0.005) and m.fy == pytest.approx(t.fy, rel=0.005)
    assert abs(m.cx - t.cx) <= 2.0 and abs(m.cy - t.cy) <= 2.0
    # reprojecting a dense grid of rays: fitted vs true model, everywhere outside the mask
    u, v = np.meshgrid(np.arange(0, 1280, 4.0), np.arange(0, 960, 4.0))
    uv = np.stack([u.ravel(), v.ravel()], 1)
    uv = uv[keep[uv[:, 1].astype(int), uv[:, 0].astype(int)] != 0]
    rays, ok = t.unproject(uv)
    assert ok.all()
    proj, ok = m.project(rays)
    assert ok.all()
    err = np.linalg.norm(proj - uv, axis=1)
    assert err.max() < 0.3, f"max {err.max():.3f} px at {uv[err.argmax()]}"


def test_report(calibrated, recording):
    src, info = recording
    *_, root = calibrated
    rep = json.loads((_out(root, src / PI_NAME) / "report.json").read_text())
    assert rep["schema"] == "umiproc.lens_report.v1" and rep["passed"] is True
    checks = {c["id"]: c for c in rep["checks"]}
    assert sorted(checks) == list("abcdefgh") and all(c["passed"] and c["evaluated"] for c in checks.values())
    assert rep["warnings"] == [] and not any(c["blocking"] for c in checks.values())      # every check only warns
    assert checks["a"]["value"] >= 25
    assert checks["b"]["value"] <= 0.5 and checks["c"]["value"] <= 0.6
    assert checks["d"]["value"] >= 0.85
    assert 140 <= checks["h"]["value"] <= 180
    assert rep["source"]["manifest_sha256"] == inspect_source(src / PI_NAME).manifest_sha256
    assert rep["source"]["episode_dir"] == str(src / PI_NAME)
    assert rep["source"]["pi_name"] == _out(root, src / PI_NAME).name
    assert rep["detector"]["corner_shift_corrected_px"] in (0.0, 0.5)
    assert rep["model_agreement"]["agree"] is True
    assert rep["frames"]["decoded"] == info["frames"]
    assert rep["selection"]["fit"] == len(rep["views"]["fit"]) and rep["selection"]["held_out"] == len(
        rep["views"]["held_out"])
    assert all(v["rms_px"] < 0.5 for v in rep["views"]["fit"] + rep["views"]["held_out"])
    cells = {c["cell"]: c for c in rep["coverage"]["cells"]}
    assert not cells["bottom"]["checked"] and cells["bottom-left"]["checked"]
    lf = lensfile.load_lens(_out(root, src / PI_NAME) / "lens.yaml")
    assert lf.source["manifest_sha256"] == rep["source"]["manifest_sha256"]
    assert lf.metrics["fit_views"] == checks["a"]["value"] and lf.metrics["checks_outside_limits"] == []
    assert [lf.model.fx, lf.model.fy, lf.model.cx, lf.model.cy] == rep["fit"]["intrinsics"]


def test_coverage_png(calibrated, recording):
    *_, root = calibrated
    img = cv2.imread(str(_out(root, recording[0] / PI_NAME) / "coverage.png"))
    assert img.shape == (960, 1280, 3)


def test_apply_installs_and_backs_up(recording, tmp_path):
    src, _ = recording
    rigs = tmp_path / "rigs"
    rigs.mkdir()
    old = rigs / "rig_pi_v3.lens.yaml"
    old.write_text("old lens file\n")
    code, out, err = _run(["lens-calibrate", PI_NAME, "--source", str(src), "--root", str(tmp_path), "--apply"])
    assert code == 0, out + err
    assert old.read_bytes() == (_out(tmp_path, src / PI_NAME) / "lens.yaml").read_bytes()
    backups = sorted(rigs.glob("rig_pi_v3.lens.yaml.bak-*"))
    assert len(backups) == 1 and backups[0].read_text() == "old lens file\n"
    assert "applied" in out and backups[0].name in out
    assert lensfile.load_lens(old).provenance == "measured"


def test_focal_length_check_only_warns(recording, tmp_path, monkeypatch):
    """(f) outside its limit (made so here by a 0.1 % tolerance; the synthetic truth is 0.3 % off the nominal):
    reported and printed as a warning, but lens.yaml is written and --apply installs it."""
    src, _ = recording
    monkeypatch.setattr(calibrate, "FOCAL_TOLERANCE", 0.001)
    code, out, err = _run(["lens-calibrate", PI_NAME, "--source", str(src), "--root", str(tmp_path), "--apply"])
    assert code == 0, out + err
    assert "WARN (f) focal length vs nominal" in out and "FAIL" not in out
    assert "lens written, with checks outside their limits" in out and "applied" in out
    d = _out(tmp_path, src / PI_NAME)
    rep = json.loads((d / "report.json").read_text())
    f = [c for c in rep["checks"] if c["id"] == "f"][0]
    assert rep["passed"] is True and rep["warnings"] == ["f"]
    assert f["passed"] is False and f["evaluated"] is True and f["blocking"] is False and f["hint"]
    assert (d / "lens.yaml").read_bytes() == (tmp_path / "rigs" / "rig_pi_v3.lens.yaml").read_bytes()
    lf = lensfile.load_lens(d / "lens.yaml")
    assert lf.metrics["focal_deviation"] == f["value"] and lf.metrics["checks_outside_limits"] == ["f"]


def test_check_status_words():
    """Every check only warns (UMI has no lens gate): WARN, never FAIL."""
    assert calibrate.Check("f", 0.2, "x", False).line().startswith("WARN (f)")
    assert calibrate.Check("f", None, "x", False, evaluated=False).line().startswith("WARN (not evaluated) (f)")
    assert calibrate.Check("g", 50.0, "x", False).line().startswith("WARN (g)")
    assert calibrate.Check("g", 50.0, "x", True).line().startswith("PASS (g)")
    assert calibrate.Check("f", 0.2, "x", False).warned and calibrate.Check("g", 50.0, "x", False).warned
    assert not calibrate.Check("f", None, "x", False, evaluated=False).warned
    assert all(not calibrate.Check(cid, 0, "x", False).blocking for cid in "abcdefgh")


@pytest.fixture(scope="module")
def few_views(tmp_path_factory, keep):
    src = tmp_path_factory.mktemp("incoming_few")
    model = synth.truth_model()
    poses = synth.calibration_poses(model, keep)[:8]
    synth.make_recording(src / "episode_000901", model, poses, keep=keep, episode=901)
    return src


def test_too_few_views_only_warns(few_views, tmp_path):
    """Fewer than 25 fit views (but enough for a fit): (a) is a warning; lens.yaml is written, labelled with it, and
    --apply installs it."""
    d = _out(tmp_path, few_views / "episode_000901")
    code, out, err = _run(["lens-calibrate", "episode_000901", "--source", str(few_views), "--apply"],
                          env_root=tmp_path)
    assert code == 0, out + err
    assert "WARN (a) fit views" in out and "FAIL" not in out
    assert "pause" in out                                     # what to record differently, as a hint
    rep = json.loads((d / "report.json").read_text())
    a = [c for c in rep["checks"] if c["id"] == "a"][0]
    assert rep["passed"] is True and "a" in rep["warnings"] and a["passed"] is False and a["value"] < 25 and a["hint"]
    lf = lensfile.load_lens(d / "lens.yaml")
    assert "a" in lf.metrics["checks_outside_limits"]
    installed = tmp_path / "rigs" / "rig_pi_v3.lens.yaml"
    assert installed.read_bytes() == (d / "lens.yaml").read_bytes() and "applied" in out


# --------------------------------------------------------------------------------------------- bad input

def _tiny(src, name="episode_000902", size=(1280, 960), n=4):
    frames = [np.full((size[1], size[0]), 90, np.uint8) for _ in range(n)]
    return synth.write_pi_recording(src / name, frames, episode=902)


def test_no_board_in_view(tmp_path):
    """No fit at all: the one case that still fails (exit 1, no lens.yaml, a stale one removed, --apply does
    nothing)."""
    _tiny(tmp_path / "in")
    stale = _out(tmp_path / "root", tmp_path / "in" / "episode_000902") / "lens.yaml"
    stale.parent.mkdir(parents=True)
    stale.write_text("from an earlier run\n")
    code, out, _ = _run(["lens-calibrate", "episode_000902", "--source", str(tmp_path / "in"), "--root",
                         str(tmp_path / "root"), "--apply"])
    assert code == 1 and "WARN (a)" in out and "no fit" in out and "NO LENS" in out
    rep = json.loads((stale.parent / "report.json").read_text())
    assert rep["fit"] is None and rep["frames"]["with_board"] == 0 and rep["passed"] is False
    assert not all(c["evaluated"] for c in rep["checks"])
    assert not stale.exists() and not (tmp_path / "root" / "rigs").exists()


def test_a_reused_pi_episode_number_gets_its_own_folder(tmp_path):
    """The Pi restarts its numbering: episode_000902 recorded on two days lands in two folders, and the second run
    leaves the first one's results alone."""
    root = tmp_path / "root"
    days = {"day1": (2026, 11, 10), "day2": (2026, 11, 15)}
    for d, (y, mo, da) in days.items():
        ep = _tiny(tmp_path / d)
        t = time.mktime((y, mo, da, 12, 0, 0, 0, 0, -1))           # local noon: the converted name's local date
        os.utime(ep / "COMPLETE", (t, t))
    first = root / "lens" / "pi-2026-11-10_episode_000902"
    code, out, _ = _run(["lens-calibrate", "episode_000902", "--source", str(tmp_path / "day1"), "--root", str(root)])
    assert code == 1 and str(first / "report.json") in out
    before = _listing(first)
    code, out, _ = _run(["lens-calibrate", "episode_000902", "--source", str(tmp_path / "day2"), "--root", str(root)])
    second = root / "lens" / "pi-2026-11-15_episode_000902"
    assert code == 1 and str(second / "report.json") in out
    assert sorted(p.name for p in (root / "lens").iterdir()) == [first.name, second.name]
    assert _listing(first) == before
    for d in (first, second):
        rep = json.loads((d / "report.json").read_text())
        assert rep["source"]["pi_name"] == d.name


@pytest.mark.parametrize("damage,message", [
    ("no_complete", "no COMPLETE"),
    ("hash", "manifest check failed"),
    ("resolution", "needs 1280x960"),
    ("no_imx219", "exactly one camera with model 'imx219'"),
    ("invalid", "marked it invalid"),
])
def test_unusable_recordings_are_refused(tmp_path, damage, message):
    ep = _tiny(tmp_path / "in", size=(640, 480) if damage == "resolution" else (1280, 960))
    if damage == "no_complete":
        (ep / "COMPLETE").unlink()
    elif damage == "hash":
        with open(ep / "cam0_frames.csv", "a") as f:
            f.write("\n")
    elif damage in ("no_imx219", "invalid"):
        meta = json.loads((ep / "meta.json").read_text())
        if damage == "no_imx219":
            meta["cameras"][0]["model"] = "imx477"
        else:
            meta["valid"] = False
        (ep / "meta.json").write_text(json.dumps(meta))
        write_pi_manifest(ep)
    code, out, err = _run(["lens-calibrate", ep.name, "--source", str(tmp_path / "in"), "--root",
                           str(tmp_path / "root")])
    assert code == 1 and message in err, err
    assert not (tmp_path / "root" / "lens").exists()


def test_bad_arguments(tmp_path):
    code, _, err = _run(["lens-calibrate", "episode_000999", "--source", str(tmp_path), "--root", str(tmp_path)])
    assert code == 1 and "not a directory" in err
    code, _, err = _run(["lens-calibrate", "episode_000999", "--root", "/data"])
    assert code == 1 and "refused" in err
    code, _, err = _run(["lens-calibrate", "episode_000999", "--source", str(tmp_path), "--root", str(tmp_path),
                         "--rig", "rig_nope"])
    assert code == 1 and "no rig file" in err and "rig_pi_v3" in err
    code, _, err = _run(["lens-calibrate", "bad name!", "--source", str(tmp_path), "--root", str(tmp_path)])
    assert code == 1 and "invalid episode directory name" in err
    code, _, err = _run(["lens-calibrate", "x", "--max-views", "3", "--root", str(tmp_path)])
    assert code == 1 and "--max-views" in err


def test_defaults():
    parser = argparse.ArgumentParser()
    calibrate.add_arguments(parser)
    args = parser.parse_args(["episode_000001"])
    assert (args.source, args.rig, args.max_views, args.root, args.apply) == ("/data/incoming", "rig_pi_v3", 60,
                                                                               None, False)
