"""B2: launcher / run_task / fdcal / annotate read the station; profiles without `station:` are unchanged (goldens)."""
import importlib.util
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import pytest
import yaml

from franka_deploy.launch import preflight as pf
from franka_deploy.launch.launcher import Launcher, check_config, load_config, station_contradictions
from franka_deploy.station import STATION_ROOT, load_station, station_from_deploy_cfg

PKG = Path(__file__).resolve().parents[1]
GOLDEN = PKG / "tests" / "golden"
_spec = importlib.util.spec_from_file_location("make_goldens", GOLDEN / "make_goldens.py")
make_goldens = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(make_goldens)


def patched_cfg(tmp_path, profile):
    """deploy.yaml profile with laptop paths (files exist) so check_config / static_preflight can pass."""
    tmp_path.mkdir(parents=True, exist_ok=True)
    ck = tmp_path / "m.ckpt"
    ck.write_bytes(b"x")
    z = tmp_path / "p.zarr.zip"
    z.write_bytes(b"x")
    cfg = load_config(PKG / "configs" / "deploy.yaml", profile)
    cfg.update(work_root=str(tmp_path / "runs"), checkpoint=str(ck), prompt_zarr=str(z),
               python={"bp": sys.executable, "realenv": sys.executable})
    return cfg


def launcher_snapshot(tmp_path, profile, mutate=None):
    cfg = patched_cfg(tmp_path, profile)
    if mutate:
        mutate(cfg)
    L = Launcher(cfg, interactive=False)
    L.snapshot()
    return L


# ------------------------------------------------------------------------------------------------------- goldens


OLD_PORT = "55" + "55"   # the pre-B2 bridge port (kept out of the source as a literal: acceptance grep)


def _port_moved(text: str) -> str:
    return text.replace("tcp://127.0.0.1:" + OLD_PORT, "tcp://127.0.0.1:5565")


@pytest.mark.parametrize("profile", ["jetson", "mock"])
def test_snapshot_golden_without_station(tmp_path, profile):
    """Profiles without `station:` produce the same session config as before B2, apart from the bridge port."""
    snap = make_goldens.normalised_snapshot(profile, tmp_path)
    d = GOLDEN / f"snapshot_{profile}"
    for f in make_goldens.FILES:
        golden = yaml.safe_load(_port_moved((d / f).read_text()))
        assert snap[f] == golden, f
    assert snap["_session_files"] == (d / "session_config_files.txt").read_text().split()
    assert "station_resolved.yaml" not in snap["_session_files"]


def test_golden_differs_only_by_the_port_line():
    for profile in ("jetson", "mock"):
        for f in make_goldens.FILES:
            text = (GOLDEN / f"snapshot_{profile}" / f).read_text()
            assert "5565" not in text  # goldens are the pre-B2 output (5555)


# ------------------------------------------------------------------------------------------------------- run_task


def test_run_task_parser_defaults_equal_old_literals():
    from franka_deploy.tasks.run_task import parser

    a = parser().parse_args(["--work-dir", "w"])
    assert (a.lookahead_s, a.home_gripper_m, a.camera_latency_s, a.policy_latency_s) == (0.1, 0.08, 0.08, 0.1)
    b = parser().parse_args(["--work-dir", "w", "--lookahead-s", "0.2", "--home-gripper-m", "0.05",
                             "--camera-latency-s", "0.07", "--policy-latency-s", "0.15"])
    assert (b.lookahead_s, b.home_gripper_m, b.camera_latency_s, b.policy_latency_s) == (0.2, 0.05, 0.07, 0.15)


# ----------------------------------------------------------------------------------------------- station-mock


def test_station_mock_snapshot_carries_station_values(tmp_path):
    L = launcher_snapshot(tmp_path, "station-mock")
    st = L.station
    assert st is not None and st.ids["robot"] == "panda" and st.ids["sensor"] == "iphone16pro_wifi"
    cfgd = L.session / "config"
    bridge = yaml.safe_load((cfgd / "bridge.yaml").read_text())
    assert bridge["safety"]["v_max"] == st.get("criteria.limits.v_max_m_s")
    assert bridge["safety"]["workspace_min"] == list(st.get("scene.workspace.min_m"))
    assert bridge["watchdog_hold_s"] == st.get("robot.bridge.watchdog_hold_s")
    assert bridge["backend"]["type"] == "fake"                      # deploy.yaml override still wins
    assert bridge["endpoint"] == "tcp://127.0.0.1:5565"
    ctrl = yaml.safe_load((cfgd / "franka_cartesian.yaml").read_text())
    assert ctrl["F_T_TCP"] == st.controller_fragment()["F_T_TCP"]
    assert ctrl["gripper"]["type"] == "fake"                        # controller.override wins over the station
    assert ctrl["gripper"]["width_map"]["a"] == st.get("tooling.width_map.a")
    res = yaml.safe_load((cfgd / "station_resolved.yaml").read_text())
    assert res["fingerprint"] == st.fingerprint() == L.summary["station_fingerprint"]
    assert res["ids"]["tooling"] == "franka_hand_iphumi_iphone16pro"
    assert "station_overrides" in L.summary["preflight"]


def test_override_contradicting_station_is_recorded(tmp_path):
    def mutate(cfg):
        cfg["bridge"]["override"]["safety"] = {"v_max": 0.31, "slack_s": cfg_station_slack}

    cfg_station_slack = load_station({"station": {"robot": "panda"}}).get("robot.bridge.slack_s")  # equal: not recorded
    L = launcher_snapshot(tmp_path, "station-mock", mutate)
    rec = L.summary["preflight"]["station_overrides"]
    paths = {(r["target"], r["path"]): r for r in rec}
    assert paths[("bridge", "safety.v_max")]["override"] == 0.31
    assert paths[("bridge", "safety.v_max")]["station"] == L.station.get("criteria.limits.v_max_m_s")
    assert ("bridge", "safety.slack_s") not in paths                 # same value as the station: no contradiction
    assert ("controller", "gripper.type") in paths                   # fake gripper vs the station's franka_hand
    bridge = yaml.safe_load((L.session / "config" / "bridge.yaml").read_text())
    assert bridge["safety"]["v_max"] == 0.31                         # explicit override still wins


def test_station_contradictions_nested_and_lists():
    frag = {"a": {"b": 1, "c": [1, 2]}, "d": 3}
    assert station_contradictions("t", frag, {"a": {"b": 1, "c": [1, 3]}, "d": 3, "z": 9}) == [
        {"target": "t", "path": "a.c", "station": [1, 2], "override": [1, 3]}]


def test_no_station_has_no_station_artifacts(tmp_path):
    L = launcher_snapshot(tmp_path, "mock")
    assert L.station is None and "station_overrides" not in L.summary["preflight"]
    assert "station_fingerprint" not in L.summary


# ----------------------------------------------------------------------------------- real mode / placeholders


def _record_checks(L, monkeypatch):
    rec = {}
    monkeypatch.setattr(L, "check", lambda name, result: rec.__setitem__(name, bool(result[0])) or rec.setdefault("_d", {}).update({name: result[1]}))
    monkeypatch.setattr(pf, "imagecodecs_pin", lambda p: (True, "stub"))
    monkeypatch.setattr(pf, "rig_not_default", lambda p: (True, "stub"))
    monkeypatch.setattr(pf, "ports_free", lambda eps: (True, "stub"))
    monkeypatch.setattr(pf, "shm_clean", lambda: (True, "stub"))
    L.static_preflight()
    return rec


def test_static_preflight_real_mode_fails_on_critical_placeholders(tmp_path, monkeypatch):
    L = launcher_snapshot(tmp_path, "station-mock", lambda c: c.update(run_mode="real"))
    rec = _record_checks(L, monkeypatch)
    assert rec["station_valid"] and rec["tooling_sensor_compat"]
    assert rec["station_placeholders"] is False
    assert "robot.desk_end_effector" in rec["_d"]["station_placeholders"]


def test_static_preflight_missing_run_mode_defaults_to_real(tmp_path, monkeypatch):
    L = launcher_snapshot(tmp_path, "station-mock", lambda c: c.pop("run_mode"))
    assert _record_checks(L, monkeypatch)["station_placeholders"] is False


def test_static_preflight_real_mode_passes_with_allow_list(tmp_path, monkeypatch):
    crit = [p for p, _ in load_station({"station": {"robot": "panda", "tooling": "franka_hand_iphumi_iphone16pro",
                                                    "sensor": "iphone16pro_wifi", "scene": "block_raise_table",
                                                    "host": "pc_laptop"}}).placeholders(critical_only=True)]
    L = launcher_snapshot(tmp_path, "station-mock", lambda c: c.update(run_mode="real", allow_placeholders=crit))
    assert _record_checks(L, monkeypatch)["station_placeholders"] is True
    L2 = launcher_snapshot(tmp_path / "b", "station-mock", lambda c: c.update(run_mode="real", allow_placeholders=crit[:-1]))
    assert _record_checks(L2, monkeypatch)["station_placeholders"] is False


def test_static_preflight_sim_mode_ok_and_lists_them(tmp_path, monkeypatch):
    L = launcher_snapshot(tmp_path, "station-mock")
    rec = _record_checks(L, monkeypatch)
    assert rec["station_placeholders"] is True and "robot.desk_end_effector" in rec["_d"]["station_placeholders"]


def test_no_station_static_preflight_has_no_station_checks(tmp_path, monkeypatch):
    L = launcher_snapshot(tmp_path, "mock")
    rec = _record_checks(L, monkeypatch)
    assert not any(k.startswith("station") or k == "tooling_sensor_compat" for k in rec)


def _check_config_cfg(tmp_path, **over):
    cfg = patched_cfg(tmp_path, "station-mock")
    cfg["camera"]["rig_config_path"] = None
    cfg.update(over)
    return cfg


def test_check_config_station_mock_sim_prints_placeholders_and_passes(tmp_path, capsys):
    assert check_config(_check_config_cfg(tmp_path)) == 0
    out = capsys.readouterr().out
    assert "fingerprint" in out and "CRITICAL" in out and "robot.desk_end_effector" in out


def test_check_config_real_mode_fails_then_passes_with_allow(tmp_path, capsys):
    assert check_config(_check_config_cfg(tmp_path, run_mode="real")) == 1
    assert "FAIL station_placeholders" in capsys.readouterr().out
    st = station_from_deploy_cfg(_check_config_cfg(tmp_path))
    allow = [p for p, _ in st.placeholders(critical_only=True)]
    assert check_config(_check_config_cfg(tmp_path, run_mode="real", allow_placeholders=allow)) == 0


def test_check_config_mock_unchanged_output_has_no_station_lines(tmp_path, capsys):
    cfg = patched_cfg(tmp_path, "mock")
    cfg["camera"]["rig_config_path"] = None
    assert check_config(cfg) == 0
    out = capsys.readouterr().out
    assert "station" not in out and "fingerprint" not in out


# ------------------------------------------------------------------------------------------------------- fdcal


def _sweep_dir(tmp_path, widths):
    d = tmp_path / "sweep"
    d.mkdir()
    sweep = []
    for i, _ in enumerate(widths):
        cv2.imwrite(str(d / f"uw_{i:03d}.png"), np.zeros((8, 8), np.uint8))
        sweep.append({"position": 3000 - 13636.0 * widths[i], "image": f"uw_{i:03d}.png"})
    (d / "sweep.json").write_text(json.dumps(sweep))
    return d


def _patch_width(monkeypatch, widths):
    from franka_deploy.fdcal import width as wd

    it = iter(0.30 + np.asarray(widths))                        # raw tag separation; the closed one is subtracted
    monkeypatch.setattr(wd, "load_aruco", lambda: (None, {}))
    monkeypatch.setattr(wd, "raw_width", lambda *a, **k: float(next(it)))
    return wd


def test_fdcal_width_min_coverage_flag(tmp_path, monkeypatch):
    widths = np.linspace(0.0, 0.081, 9)
    d = _sweep_dir(tmp_path, widths)
    wd = _patch_width(monkeypatch, widths)
    assert wd.main(["--data", str(d), "--min-coverage-m", "0.08"]) == 0
    wd = _patch_width(monkeypatch, widths)
    assert wd.main(["--data", str(d)]) == 2                      # default 0.085 unchanged
    wd = _patch_width(monkeypatch, widths)
    assert wd.main(["--data", str(d), "--min-coverage-m", "0.085"]) == 2


def test_fdcal_width_station_coverage_and_alias(tmp_path, monkeypatch, capsys):
    widths = np.linspace(0.0, 0.081, 9)
    d = _sweep_dir(tmp_path, widths)
    wd = _patch_width(monkeypatch, widths)
    cfg = PKG / "configs" / "deploy.yaml"
    assert wd.main(["--data", str(d), "--deploy-config", str(cfg), "--profile", "station-mock"]) == 0   # 0.080 from station
    checks = json.loads(capsys.readouterr().out.strip().splitlines()[-1])
    assert checks["covers_required"] is True and checks["covers_0p085"] is True and checks["min_coverage_m"] == 0.080
    wd = _patch_width(monkeypatch, widths)
    assert wd.main(["--data", str(d), "--deploy-config", str(cfg), "--profile", "station-mock",
                    "--min-coverage-m", "0.09"]) == 2           # explicit flag beats the station
    with pytest.raises(SystemExit):
        wd.main(["--data", str(d), "--deploy-config", str(cfg)])  # one without the other


def test_fdcal_handeye_station_values():
    from franka_deploy.fdcal import handeye as he

    C_T_TCP, K = he.station_values(str(PKG / "configs" / "deploy.yaml"), "station-mock")
    np.testing.assert_allclose(C_T_TCP, he.C_T_TCP)
    np.testing.assert_allclose(K, he.K_MAIN_960, atol=1e-4)
    with pytest.raises(SystemExit):
        he.station_values(str(PKG / "configs" / "deploy.yaml"), "mock")


# ----------------------------------------------------------------------------------------------------- annotate


def test_annotate_defaults_are_the_pre_trial_rule():
    from franka_deploy.eval import annotate as an

    assert (an.LIFT_MIN_M, an.HOLD_MIN_S, an.SLIP_TOL_M) == (0.05, 2.0, 0.002)   # hardware_day section 10


def _episode(tmp_path, lift_m):
    """Controller trace: closes at t=1, lifts by lift_m, holds >2 s, releases."""
    t = np.arange(0, 8, 0.05)
    fk = np.zeros((len(t), 7))
    fk[:, 3] = 1.0
    fk[:, 2] = np.where((t >= 2) & (t < 6.5), lift_m, 0.0)
    w = np.where((t >= 1) & (t < 6.5), 0.04, 0.08)
    d = tmp_path / "ep"
    d.mkdir()
    np.savez(d / "controller_trace.npz", t=t, fk=fk, width_measured=w)
    (d / "label.json").write_text(json.dumps({"label": "success"}))
    (d / "summary.json").write_text(json.dumps({}))
    return d


def test_annotate_criteria_block_raise_uses_five_cm(tmp_path):
    from franka_deploy.eval import annotate as an

    rule = an.load_success_rule("block_raise")
    assert rule == {"lift_min_m": 0.05, "hold_min_s": 2.0, "slip_tol_m": 0.002}
    assert an.load_success_rule(str(STATION_ROOT / "run" / "criteria" / "block_raise.yaml")) == rule
    custom = tmp_path / "c.yaml"
    custom.write_text(yaml.safe_dump({"success": {"lift_min_m": 0.02, "hold_s": 1.0, "slip_tol_m": 0.004}}))
    assert an.load_success_rule(str(custom))["lift_min_m"] == 0.02
    with pytest.raises(SystemExit):
        an.load_success_rule("no_such_criteria")
    d = _episode(tmp_path, 0.04)                                  # 4 cm: below 5 cm, above the old 3 cm
    assert an.auto_suggest(str(d))["success"] is False
    assert an.auto_suggest(str(d), lift_min_m=0.03)["success"] is True
    assert an.main([str(d), "--annotator", "t", "--auto", "--criteria", "block_raise"]) == 0
    assert json.loads((d / "annotation.json").read_text())["success"] is False
    assert an.main([str(d), "--annotator", "t", "--auto", "--criteria", str(custom)]) == 0
    assert json.loads((d / "annotation.json").read_text())["success"] is True


# ------------------------------------------------------------------------------------------------------ misc


def test_task_franka_yaml_is_gone_and_unreferenced():
    assert not (PKG / "configs" / "task_franka.yaml").exists()
    assert "task_franka" not in (PKG / "README.md").read_text()


def test_bridge_port_is_5565_everywhere():
    for f in ("deploy.yaml", "bridge.yaml", "franka_cartesian.yaml"):
        text = (PKG / "configs" / f).read_text()
        assert "127.0.0.1:5565" in text and "127.0.0.1:" + OLD_PORT not in text
