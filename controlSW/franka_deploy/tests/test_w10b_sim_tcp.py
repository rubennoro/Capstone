"""W10B: one TCP in sim (I-TCP via the launcher), real profiles unchanged, warning surfacing, shm glob, validate modes.

    env -u PYTHONPATH PYTHONPATH=. $RE -m pytest -q tests/test_w10b_sim_tcp.py
The sim-iphone cases need mujoco + franka_sim (skipped otherwise); the import-guard cases run a subprocess that blocks
mujoco and franka_sim."""
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import yaml

from franka_deploy.launch import launcher as lm
from franka_deploy.launch import preflight as pf
from franka_deploy.station import StationError, load_station, station_from_deploy_cfg

PKG = Path(__file__).resolve().parents[1]
DEP = PKG / "configs" / "deploy.yaml"
FILE_F_T_TCP = yaml.safe_load(open(PKG / "configs" / "franka_cartesian.yaml"))["F_T_TCP"]
PLACEHOLDER_SET = "tooling.F_T_TCP=" + json.dumps(FILE_F_T_TCP)


def _mujoco():
    try:
        lm._franka_sim_model()
        return True
    except ImportError:
        return False


needs_mujoco = pytest.mark.skipif(not _mujoco(), reason="needs mujoco + franka_sim")


# ------------------------------------------------------------------------------------------ real / fake: unchanged
@pytest.mark.parametrize("prof", ["real-iphone", "real-rig"])
def test_real_profiles_keep_the_file_tcp_and_still_fail_on_it(prof):
    cfg = lm.load_config(DEP, prof)
    assert "_sim_tcp" not in cfg and not any("F_T_TCP" in str(x) for x in cfg.get("station_set", []))
    st = station_from_deploy_cfg(cfg)
    assert st.get("tooling.F_T_TCP") is None and st.prov("tooling.F_T_TCP").status == "hw"
    ctrl, _ = lm.build_controller_cfg(cfg, st)
    assert ctrl["F_T_TCP"] == FILE_F_T_TCP                       # the controller keeps franka_cartesian.yaml's value
    assert lm.sim_tcp_check(cfg, st) is None                      # no sim_tcp preflight on real
    ok, detail = pf.station_placeholders(st, "real", cfg.get("allow_placeholders", []))
    assert not ok and "tooling.F_T_TCP" in detail                 # an unmeasured TCP still fails a real run
    with pytest.raises(lm.ConfigUsageError):                      # B10: real --set accepts only host.*
        lm.load_config(DEP, prof, station_set=[PLACEHOLDER_SET])


@pytest.mark.parametrize("prof", ["sim-iphone-fake", "sim-rig-fake", "station-mock"])
def test_fake_backend_sims_are_not_filled(prof):
    cfg = lm.load_config(DEP, prof)
    assert "_sim_tcp" not in cfg
    st = station_from_deploy_cfg(cfg)
    assert lm.build_controller_cfg(cfg, st)[0]["F_T_TCP"] == FILE_F_T_TCP and lm.sim_tcp_check(cfg, st) is None


def test_null_tcp_with_a_desk_end_effector_refuses_the_controller_fragment():
    spec = {"station": {"robot": "panda", "tooling": "franka_hand_iphumi_iphone16pro"}}
    assert "F_T_TCP" not in load_station(spec).controller_fragment()
    st = load_station(spec, sets=["robot.desk_end_effector=franka_hand"])
    with pytest.raises(StationError, match="F_T_TCP is null"):
        st.controller_fragment()


# ------------------------------------------------------------------------------------------------- sim MuJoCo
@needs_mujoco
@pytest.mark.parametrize("prof", ["sim-iphone", "sim-rig"])
def test_sim_mujoco_profiles_drive_the_model_tcp(prof):
    cfg = lm.load_config(DEP, prof)
    rec = cfg["_sim_tcp"]
    assert rec["provenance"] == "derived (sim)" and rec["yaml_status"] == "hw" and rec["yaml_value"] is None
    assert cfg["station_set"][-1] == rec["set"]
    st = station_from_deploy_cfg(cfg)
    T_sim, _ = lm.sim_tcp(st)
    assert np.allclose(st.get("tooling.F_T_TCP"), T_sim, atol=1e-12)            # station_resolved sees it
    ctrl, _ = lm.build_controller_cfg(cfg, st)
    assert np.allclose(ctrl["F_T_TCP"], T_sim, atol=1e-12)                       # and the controller
    home = lm.apply_station_home(cfg, st)
    T_ee = np.eye(4)
    T_ee[:3, :3] = lm._quat_wxyz_to_R(home["home_O_T_EE_xyz_wxyz"][3:])
    T_ee[:3, 3] = home["home_O_T_EE_xyz_wxyz"][:3]
    assert np.allclose((T_ee @ T_sim)[:3, 3], st.get("robot.home_tcp.xyz_m"), atol=2e-6)   # the start pose follows
    ok, detail = lm.sim_tcp_check(cfg, st)
    assert ok and "launcher-derived" in detail and "0.000 mm" in detail, detail
    assert abs(T_sim[2, 3] - 0.0973) < 1e-3                                     # (0, 0, 97.3 mm) at extension 0


@needs_mujoco
def test_stale_explicit_set_is_kept_and_refused_by_preflight():
    cfg = lm.load_config(DEP, "sim-iphone", station_set=[PLACEHOLDER_SET])
    assert "_sim_tcp" not in cfg and cfg["station_set"][-1] == PLACEHOLDER_SET
    ok, detail = lm.sim_tcp_check(cfg, station_from_deploy_cfg(cfg))
    assert not ok and "given" in detail, detail
    good = lm.load_config(DEP, "sim-iphone")["_sim_tcp"]["set"]                 # an explicit correct value passes
    cfg = lm.load_config(DEP, "sim-iphone", station_set=[good])
    assert "_sim_tcp" not in cfg and lm.sim_tcp_check(cfg, station_from_deploy_cfg(cfg))[0]


@needs_mujoco
def test_controller_override_mismatch_is_refused():
    cfg = lm.load_config(DEP, "sim-iphone")
    cfg["controller"]["override"]["F_T_TCP"] = FILE_F_T_TCP
    assert not lm.sim_tcp_check(cfg, station_from_deploy_cfg(cfg))[0]


@needs_mujoco
def test_sim_tcp_follows_extension_and_side():
    sets = ["tooling.finger_adapter.extension_m=0.10", 'tooling.tcp_x_along_hand="-y"']
    cfg = lm.load_config(DEP, "sim-iphone", station_set=sets)
    T = np.asarray(cfg["_sim_tcp"]["F_T_TCP"])
    assert abs(T[2, 3] - 0.1973) < 1e-3 and abs(np.degrees(np.arctan2(T[1, 0], T[0, 0])) + 135.0) < 0.1


def test_check_config_reports_sim_tcp():
    import contextlib
    import io

    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        lm.check_config(lm.load_config(DEP, "real-iphone"))
    assert "sim_tcp" not in buf.getvalue()
    if _mujoco():
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            lm.check_config(lm.load_config(DEP, "sim-iphone"))
        assert "OK   sim_tcp:" in buf.getvalue() and "derived (sim)" in buf.getvalue()


# ---------------------------------------------------------------------------------------- import guard (no mujoco)
GUARD = r"""
import sys, importlib.abc
class Block(importlib.abc.MetaPathFinder):
    def find_spec(self, name, path=None, target=None):
        if name.split(".")[0] in ("mujoco", "franka_sim"):
            raise ImportError(f"blocked {name}")
        return None
sys.meta_path.insert(0, Block())
import contextlib, io
from franka_deploy.launch import launcher as lm
from franka_deploy.station import station_from_deploy_cfg
for prof in ("real-iphone", "real-rig", "sim-iphone-fake"):
    cfg = lm.load_config(sys.argv[1], prof)
    assert "_sim_tcp" not in cfg, prof
    with contextlib.redirect_stdout(io.StringIO()):
        lm.check_config(cfg)
cfg = lm.load_config(sys.argv[1], "sim-iphone")
assert cfg["_sim_tcp"].get("error") and "F_T_TCP" not in str(cfg.get("station_set")), cfg.get("_sim_tcp")
ok, detail = lm.sim_tcp_check(cfg, station_from_deploy_cfg(cfg))
assert not ok and "cannot derive" in detail, detail
assert not any(m.split(".")[0] in ("mujoco", "franka_sim") for m in sys.modules)
print("GUARD_OK")
"""


def test_franka_deploy_works_without_mujoco_and_sim_refuses():
    env = {k: v for k, v in os.environ.items() if k != "PYTHONPATH"}
    env["PYTHONPATH"] = str(PKG)
    r = subprocess.run([sys.executable, "-c", GUARD, str(DEP)], capture_output=True, text=True, env=env, cwd=str(PKG),
                       timeout=100)
    assert "GUARD_OK" in r.stdout, r.stdout + r.stderr


# ---------------------------------------------------------------------------------------- warning surfacing (item 6)
def test_bridge_tcp_warning_reaches_the_launcher_summary(tmp_path, capsys):
    L = lm.Launcher.__new__(lm.Launcher)
    L.session, L.summary, L.events, L.t0 = tmp_path, {}, [], 0.0
    (tmp_path / "logs").mkdir()
    (tmp_path / "logs" / "bridge.log").write_text(
        "[mujoco] WARNING: the controller F_T_TCP differs from the sim's actual TCP (model.F_T_TCP) by 131.0 mm / 135.4 deg: "
        "grasps will miss. Use franka_sim.model.derived_tcp(station).\n[mujoco] robot=panda\n")
    found = L.bridge_tcp_warnings()
    assert len(found) == 1 and L.summary["warnings"] == found and "131.0 mm" in capsys.readouterr().out
    (tmp_path / "logs" / "bridge.log").write_text("[mujoco] robot=panda\n")
    L.summary = {}
    assert L.bridge_tcp_warnings() == [] and "warnings" not in L.summary


# ---------------------------------------------------------------------------------------------- shm glob (item 7)
def test_cleanup_shm_is_prefix_anchored():
    from franka_deploy.harness.dryrun import _rmq_user, cleanup_shm

    if not os.path.isdir("/dev/shm"):
        pytest.skip("no /dev/shm")
    u, pid, other = _rmq_user(), 4000001, 4000002
    mine, foreign = Path(f"/dev/shm/rmq_{u}_{pid}_srv_topic"), Path(f"/dev/shm/rmq_{u}_{other}_srv_{pid}_topic")
    try:
        mine.write_bytes(b"x")
        foreign.write_bytes(b"x")
        assert cleanup_shm(pid) == [mine.name]
        assert foreign.exists() and not mine.exists()
    finally:
        for p in (mine, foreign):
            p.unlink(missing_ok=True)


# ------------------------------------------------------------------------------------------- validate modes (item 4)
def _episode(tmp_path, run_mode, video):
    d = tmp_path / f"ep_{run_mode}"
    (d / "config").mkdir(parents=True, exist_ok=True)
    (d / "config" / "deploy_resolved.yaml").write_text(yaml.safe_dump({"run_mode": run_mode, "camera": {"video": video}}))
    return d


def test_validate_mode_auto_and_sim_rules(tmp_path):
    from franka_deploy.tools import validate_episode as ve

    assert ve.episode_mode(str(_episode(tmp_path, "sim", False))) == "sim"
    assert ve.episode_mode(str(_episode(tmp_path, "real", True))) == "real"
    assert ve.episode_mode(str(tmp_path / "nothing")) == "real"               # no snapshot: strict
    assert ve.episode_mode(str(_episode(tmp_path, "sim", False)), "real") == "real"
    assert not ve._video_expected(str(_episode(tmp_path, "sim", False)))


def test_validate_mode_auto_survives_a_corrupt_snapshot(tmp_path):
    """W10B review finding 4: yaml.YAMLError is not a ValueError; auto falls back to real (strict) instead of raising."""
    from franka_deploy.tools import validate_episode as ve

    d = tmp_path / "ep_bad"
    (d / "config").mkdir(parents=True)
    for text in ("run_mode: [sim\n  : {", "- just\n- a list\n", "\t\tbad: : :\n"):
        (d / "config" / "deploy_resolved.yaml").write_text(text)
        assert ve.episode_mode(str(d)) == "real"
        assert ve._video_expected(str(d)) is True
    assert ve.episode_mode(str(d), "sim") == "sim"                              # an explicit mode never reads the file
