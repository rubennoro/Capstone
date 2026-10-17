"""T4: ``sensor-rig process-session`` (raw session -> pose -> export -> validate), VIO hooks incl. a fake docker."""

from __future__ import annotations

import json
import os
import shutil
import stat
import sys
from pathlib import Path

import cv2
import numpy as np
import pytest

from sensor_rig.cli import main as cli_main
from sensor_rig.pose import vio_hooks
from sensor_rig.process.session import UsageError, pipeline_line, process_session
from sensor_rig.raw_episode import RawEpisode
from wpa_helpers import CALIB_NAME, make_raw_episode

SESSION = "rig-test"


def make_session(root: Path, rig, n_demos=2, bad_cal_demo=False):
    """Calibration + demos (session rig-test); the demos' trajectories are moved to root/traj/<NAME>.tum."""
    raw = root / "raw"
    traj = root / "traj"
    traj.mkdir(parents=True, exist_ok=True)
    make_raw_episode(raw, rig, kind="gripper_calibration", name=CALIB_NAME, main_fps=30)
    names = []
    for i in range(n_demos):
        p = make_raw_episode(raw, rig, main_fps=30, rid=f"{i + 1}" * 5)
        names.append(p.name)
    if bad_cal_demo:
        p = make_raw_episode(raw, rig, main_fps=30, rid="99999",
                             calibration_run="2026-10-20T14-00-00.000Z_12345_rig-test_grippercalibration_left")
        names.append(p.name)
    for n in names:
        d = raw / n[:10] / n / "trajectory"
        shutil.move(str(d / "trajectory.tum"), str(traj / f"{n}.tum"))
        shutil.rmtree(d)
    return raw, traj, names


def run(root, raw, traj, **kw):
    kw.setdefault("iphone_raw", root / "no_iphone_raw")
    lines = []
    rc, doc = process_session(raw, SESSION, root / "export", "import-dir", traj_dir=traj, log=lines.append, **kw)
    return rc, doc, lines


def by_name(doc):
    return {e["name"]: e for e in doc["episodes"]}


def test_happy_path_and_report(tmp_path, test_rig):
    raw, traj, names = make_session(tmp_path, test_rig)
    rc, doc, lines = run(tmp_path, raw, traj)
    assert rc == 0, "\n".join(lines)
    eps = by_name(doc)
    assert eps[CALIB_NAME]["status"] == "exported" and "skipped" in eps[CALIB_NAME]["vio"]
    for n in names:
        e = eps[n]
        assert e["status"] == "exported" and e["export"]["grid_mode"] == "fill" and e["export"]["lag"]
        assert e["pose_import"]["gyro_resid_rms"] is not None and e["calibration_run"] == CALIB_NAME
        assert (tmp_path / "export" / n[:10] / n / f"{n}.json").is_file()
    rep = json.loads((tmp_path / "export" / ".sensor_rig" / f"process_{SESSION}.json").read_text())
    assert rep["counts"] == {"calibrations_exported": 1, "demos_exported": 2, "demos_failed": 0}
    assert "--sessions rig-test" in rep["pipeline_command"] and "--dataset rig-rig-test-single" in rep["pipeline_command"]
    assert any(ln.startswith("then (not run here): scripts/run_pipeline.sh all") for ln in lines)


def test_missing_calibration_and_partial(tmp_path, test_rig):
    raw, traj, names = make_session(tmp_path, test_rig, n_demos=1, bad_cal_demo=True)
    rc, doc, lines = run(tmp_path, raw, traj)
    eps = by_name(doc)
    bad = [n for n in names if "_99999_" in n][0]
    assert eps[bad]["status"] == "failed" and eps[bad]["step"] == "check" and "not a kept calibration" in eps[bad]["reason"]
    assert eps[names[0]]["status"] == "exported"                     # the other demo continued
    assert rc == 1
    rc2, _, _ = run(tmp_path, raw, traj, allow_partial=True)
    assert rc2 == 0


def test_failing_episode_does_not_stop_others(tmp_path, test_rig):
    raw, traj, names = make_session(tmp_path, test_rig, n_demos=3)
    (traj / f"{names[1]}.tum").unlink()                              # no trajectory for one demo
    rc, doc, _ = run(tmp_path, raw, traj)
    eps = by_name(doc)
    assert eps[names[1]]["status"] == "failed" and eps[names[1]]["step"] == "vio"
    assert eps[names[0]]["status"] == eps[names[2]]["status"] == "exported" and rc == 1


def test_pose_import_refusal_and_force(tmp_path, test_rig):
    raw, traj, names = make_session(tmp_path, test_rig, n_demos=1)
    lines = (traj / f"{names[0]}.tum").read_text().splitlines()
    vals = [ln.split() for ln in lines if ln and not ln.startswith("#")]
    for v in vals[40:]:
        v[1] = f"{float(v[1]) + 0.5:.9f}"                            # 50 cm jump
    (traj / f"{names[0]}.tum").write_text("\n".join(" ".join(v) for v in vals) + "\n")
    rc, doc, _ = run(tmp_path, raw, traj)
    e = by_name(doc)[names[0]]
    assert rc == 1 and e["step"] == "pose_import" and "translation jump" in e["reason"]
    rc, doc, _ = run(tmp_path, raw, traj, force_episodes=[names[0]])
    assert by_name(doc)[names[0]]["pose_import"]["forced"] is True


def test_resume_and_overwrite(tmp_path, test_rig):
    raw, traj, names = make_session(tmp_path, test_rig, n_demos=1)
    assert run(tmp_path, raw, traj)[0] == 0
    out_json = tmp_path / "export" / names[0][:10] / names[0] / f"{names[0]}.json"
    m0 = out_json.stat().st_mtime_ns
    rc, doc, _ = run(tmp_path, raw, traj)
    e = by_name(doc)[names[0]]
    assert rc == 0 and e["pose_import"].get("resumed") and e["export"]["status"] == "skipped"
    assert out_json.stat().st_mtime_ns == m0
    rc, doc, _ = run(tmp_path, raw, traj, overwrite=True)
    e = by_name(doc)[names[0]]
    assert rc == 0 and not e["pose_import"].get("resumed") and e["export"]["status"] == "exported"
    assert out_json.stat().st_mtime_ns != m0


def test_session_name_rules(tmp_path, test_rig):
    raw, traj, _ = make_session(tmp_path, test_rig, n_demos=1)
    iph = tmp_path / "iphone_raw" / "2026-10-01"
    iph.mkdir(parents=True)
    (iph / "2026-10-01T21-10-44.057Z_15647_rig-test_grippercalibration_left").mkdir()
    with pytest.raises(UsageError, match="already exists"):
        run(tmp_path, raw, traj, iphone_raw=tmp_path / "iphone_raw")
    with pytest.raises(UsageError, match="rig-only"):
        process_session(raw, "firstreal", tmp_path / "export", "import-dir", traj_dir=traj,
                        iphone_raw=tmp_path / "none")
    rc = cli_main(["process-session", "--raw", str(raw), "--session", "firstreal", "--out", str(tmp_path / "x"),
                   "--vio", "import-dir", "--traj-dir", str(traj)])
    assert rc == 2


def test_no_calibration_exported_fails(tmp_path, test_rig):
    raw, traj, names = make_session(tmp_path, test_rig, n_demos=1)
    cal_dir = raw / CALIB_NAME[:10] / CALIB_NAME
    shutil.move(str(cal_dir), str(cal_dir.parent / f".{CALIB_NAME}.discarded"))   # discarded in collect
    rc, doc, _ = run(tmp_path, raw, traj)
    assert rc == 1 and by_name(doc)[names[0]]["step"] == "check"


def test_pipeline_line_inside_and_outside_repo(tmp_path):
    from sensor_rig.process import session as S

    cmd, note = pipeline_line("rig1020", S.REPO / "data" / "rig_export" / "rig1020", "block_raise")
    assert "--raw-dir data/rig_export/rig1020" in cmd and note is None
    cmd, note = pipeline_line("rig1020", tmp_path / "exp", "block_raise")
    assert "--raw-dir data/rig_export/rig1020" in cmd and "rsync" in note


def test_cmd_hook(tmp_path, test_rig):
    raw, traj, names = make_session(tmp_path, test_rig, n_demos=1)
    tmpl = f"cp {traj}/{{name}}.tum {{out}} && test -f {{settings}} && test -f {{mask}}"
    lines = []
    rc, doc = process_session(raw, SESSION, tmp_path / "export", "cmd", vio_cmd=tmpl, log=lines.append,
                              iphone_raw=tmp_path / "none")
    e = by_name(doc)[names[0]]
    assert rc == 0, "\n".join(lines)
    assert e["vio"]["format"] == "tum" and e["vio"]["mask"]["masked_fraction"] > 0.0


FAKE_DOCKER = r'''#!/usr/bin/env python3
import os, sys
args = sys.argv[1:]
log = os.environ["FAKE_DOCKER_LOG"]
open(log, "a").write(" ".join(args) + "\n")
if args[:2] == ["image", "inspect"]:
    print("false"); sys.exit(0)
assert args[0] == "run"
mount = args[args.index("-v") + 1].split(":")[0]
name = args[-1]
src = os.path.join(os.environ["FAKE_TRAJ_DIR"], name + ".tum")
out = []
for ln in open(src):
    if not ln.strip() or ln.startswith("#"):
        continue
    p = ln.split()
    s, frac = p[0].split(".")
    out.append(" ".join([str(int(s) * 10**9 + int(frac.ljust(9, "0")))] + p[1:]))   # EuRoC: t in ns
open(os.path.join(mount, "f_" + name + ".txt"), "w").write("\n".join(out) + "\n")
'''


def test_orbslam3_docker_hook_with_fake_docker(tmp_path, test_rig, monkeypatch):
    raw, traj, names = make_session(tmp_path, test_rig, n_demos=1)
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    d = bin_dir / "docker"
    d.write_text(FAKE_DOCKER.replace("#!/usr/bin/env python3", f"#!{sys.executable}"))
    d.chmod(d.stat().st_mode | stat.S_IEXEC)
    monkeypatch.setenv("PATH", f"{bin_dir}{os.pathsep}{os.environ['PATH']}")
    monkeypatch.setenv("FAKE_DOCKER_LOG", str(tmp_path / "docker.log"))
    monkeypatch.setenv("FAKE_TRAJ_DIR", str(traj))
    lines = []
    rc, doc = process_session(raw, SESSION, tmp_path / "export", "orbslam3-docker", log=lines.append,
                              iphone_raw=tmp_path / "none")
    assert rc == 0, "\n".join(lines)
    e = by_name(doc)[names[0]]
    assert e["vio"]["mask_used"] is False and e["vio"]["format"] == "orbslam3-euroc"
    calls = (tmp_path / "docker.log").read_text().splitlines()
    run_call = [c for c in calls if c.startswith("run ")][0]
    assert vio_hooks.ORB_BIN in run_call and "/data/orbslam3_settings.yaml /data /data/times.txt" in run_call
    assert run_call.endswith(names[0]) and "-w /data" in run_call and "--mask" not in run_call
    prep = tmp_path / "export" / ".sensor_rig" / "vio" / SESSION / names[0]
    assert (prep / "mask_ultrawide.png").is_file() and (prep / "orbslam3_settings.yaml").is_file()
    # resume: the container is not run again
    n_runs = len([c for c in calls if c.startswith("run ")])
    process_session(raw, SESSION, tmp_path / "export", "orbslam3-docker", log=lines.append,
                    iphone_raw=tmp_path / "none")
    calls = (tmp_path / "docker.log").read_text().splitlines()
    assert len([c for c in calls if c.startswith("run ")]) == n_runs


def test_docker_command_with_mask(tmp_path):
    cmd = vio_hooks.orbslam3_docker_command("img:1", tmp_path, "EP", mask=tmp_path / "m.png", user="1:2")
    assert cmd[:5] == ["docker", "run", "--rm", "--user", "1:2"]
    assert f"{tmp_path.resolve()}:/data" in cmd and f"{(tmp_path / 'm.png').resolve()}:/data/mask.png:ro" in cmd
    assert cmd[-2:] == ["--mask", "/data/mask.png"]


def test_docker_missing_is_a_vio_failure(tmp_path, test_rig, monkeypatch):
    raw, traj, names = make_session(tmp_path, test_rig, n_demos=1)
    monkeypatch.setenv("PATH", str(tmp_path / "empty"))
    rc, doc = process_session(raw, SESSION, tmp_path / "export", "orbslam3-docker", log=lambda m: None,
                              iphone_raw=tmp_path / "none")
    e = by_name(doc)[names[0]]
    assert rc == 1 and e["step"] == "vio" and "not found" in e["reason"]


def test_finger_mask(test_rig, tmp_path):
    from sensor_rig.pose.mask import finger_mask, write_finger_mask

    m = finger_mask(test_rig)
    w, h = test_rig.cameras["ultrawide"].resolution
    assert m.shape == (h, w) and set(np.unique(m)) <= {0, 255}
    frac = (m == 0).mean()
    assert 0.02 < frac < 0.5
    assert (m[: h // 3] == 255).all()                                  # top of the image is never masked
    info = write_finger_mask(test_rig, "ultrawide", tmp_path / "m.png", (w // 2, h // 2))
    assert cv2.imread(str(tmp_path / "m.png"), cv2.IMREAD_GRAYSCALE).shape == (h // 2, w // 2)
    assert info["masked_fraction"] > 0


@pytest.mark.parametrize("cmd", ["collect", "record"])
def test_shared_session_name_check_before_sensors_open(cmd, tmp_path, capsys, monkeypatch):
    """collect and record refuse the names process-session would refuse, before any sensor is opened."""
    from sensor_rig import naming
    from sensor_rig.record import recorder as rec_mod

    opened = []
    monkeypatch.setattr(rec_mod.Recorder, "open", lambda self: opened.append(1))
    extra = ["--side", "left", "--duration", "1"] if cmd == "record" else []
    rc = cli_main([cmd, "--mock", "--out", str(tmp_path / "raw"), "--session", "block1020", *extra])
    err = capsys.readouterr().err
    assert rc == 2 and "not a rig-only name" in err and not opened
    # collision with an iPhone session in data/raw (here: a fake data/raw)
    iph = tmp_path / "iphone_raw" / "2026-10-01"
    iph.mkdir(parents=True)
    (iph / "2026-10-01T21-10-44.057Z_15647_rig1020_demonstration_left").mkdir()
    monkeypatch.setattr(naming, "DEFAULT_IPHONE_RAW", tmp_path / "iphone_raw")
    monkeypatch.setattr(naming.session_name_problems, "__defaults__", (tmp_path / "iphone_raw",))
    rc = cli_main([cmd, "--mock", "--out", str(tmp_path / "raw"), "--session", "rig1020", *extra])
    assert rc == 2 and "already exists" in capsys.readouterr().err and not opened


def test_session_name_problems_messages(tmp_path):
    from sensor_rig.naming import session_name_problems

    assert session_name_problems("rig1020", tmp_path / "none") == []
    assert session_name_problems("rig1020", None) == []
    p = session_name_problems("firstreal", tmp_path / "none")
    assert len(p) == 1 and "rig-only" in p[0]


def test_dockerfile_patches_for_headless_opencv42_build():
    """The (unbuilt) image must patch the OpenCV 4.4 requirement (Ubuntu 20.04 ships 4.2) and the Pangolin viewer
    (no X display in a container), and fail the build if a patch did not apply."""
    df = (Path(__file__).resolve().parents[1] / "docker" / "orbslam3" / "Dockerfile").read_text()
    assert "FROM ubuntu:20.04" in df and "ARG ORB_SLAM3_REF=v1.0-release" in df
    assert "sed -i 's/find_package(OpenCV 4.4)/find_package(OpenCV 4.2)/' CMakeLists.txt" in df
    assert "grep -q 'find_package(OpenCV 4.2)' CMakeLists.txt" in df
    assert ("sed -i 's/ORB_SLAM3::System::IMU_MONOCULAR, true)/ORB_SLAM3::System::IMU_MONOCULAR, false)/'" in df
            and "Examples/Monocular-Inertial/mono_inertial_euroc.cc" in df)
    assert "grep -q 'IMU_MONOCULAR, false)'" in df
    assert df.index("grep -q 'IMU_MONOCULAR, false)'") < df.index("./build.sh")
    assert 'sensor_rig.mask_supported="false"' in df
