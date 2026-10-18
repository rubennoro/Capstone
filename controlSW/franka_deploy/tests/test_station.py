"""Station config loader (B1): every shipped YAML validates, every error class is caught, and the derived fragments
reproduce today's bridge.yaml / franka_cartesian.yaml / run_task.py literals. Pure: stdlib + numpy + yaml only."""

from __future__ import annotations

import dataclasses
import os
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import yaml

from franka_deploy import station as S
from franka_deploy.station import PROFILES, STATION_ROOT, StationError, load_station, station_from_deploy_cfg

DEPLOY_DIR = Path(__file__).resolve().parents[1]
CFG = DEPLOY_DIR / "configs"
IPHONE = PROFILES["sim-iphone"]


def _yaml(p):
    return yaml.safe_load(Path(p).read_text())


def _copy_root(tmp_path) -> Path:
    root = tmp_path / "configs"
    shutil.copytree(STATION_ROOT, root)
    return root


def _edit(root, kind, group, gid, fn):
    p = root / kind / group / f"{gid}.yaml"
    d = yaml.safe_load(p.read_text())
    fn(d)
    p.write_text(yaml.safe_dump(d, sort_keys=False))


def _errors(root, spec=None, sets=()):
    with pytest.raises(StationError) as ei:
        load_station(spec or IPHONE, root, sets=sets)
    return "\n".join(ei.value.problems)


# ------------------------------------------------------------------------------------------------ shipped files


def test_every_yaml_validates():
    assert S._validate_all(STATION_ROOT) == []


def test_file_layout_complete():
    for kind, groups in S.KINDS.items():
        for g in groups:
            assert list((STATION_ROOT / kind / g).glob("*.yaml")), (kind, g)
    for name in ("panda", "fr3"):
        assert (STATION_ROOT / "station/robot" / f"{name}.yaml").is_file()
    assert (STATION_ROOT / "README.md").is_file()


@pytest.mark.parametrize("name", sorted(PROFILES))
def test_profiles_load(name):
    st = load_station(PROFILES[name])
    assert st.fingerprint() == load_station(PROFILES[name]).fingerprint()
    assert yaml.safe_load(st.to_yaml())["ids"] == st.ids


def test_fr3_loads_and_differs():
    spec = {"station": {**IPHONE["station"], "robot": "fr3"}, "run": IPHONE["run"]}
    assert load_station(spec).fingerprint() != load_station(IPHONE).fingerprint()


def test_cli_validate_all_exits_zero():
    assert S.main(["validate", "--all"]) == 0
    assert S.main(["validate", "--spec", "station.robot=panda", "--spec", "run.link=ideal"]) == 0
    assert S.main(["validate", "--spec", "station.robot=nope"]) == 1


def test_interpolation_resolved():
    st = load_station(IPHONE)
    assert st.get("policy.checkpoint") == st.get("host.checkpoint")
    assert st.get("policy.task.exec_h") == 8
    assert st.get("tooling.hand.endpoint") == "tcp://localhost:5566"
    assert "${" not in st.to_yaml()


def test_values_requested_by_spec():
    st = load_station(IPHONE)
    wm = st.get("tooling.width_map")
    assert (wm["a"], wm["b"], wm["train_closed_m"], wm["train_min_m"], wm["train_max_m"], wm["coverage_required_m"]) == (
        1.0, 0.008, 0.0431, 0.0447, 0.0844, 0.080)
    assert st.get("criteria.success") == {"lift_min_m": 0.05, "hold_s": 2.0, "timeout_s": 60, "slip_tol_m": 0.002}
    assert st.get("tooling.hand.port") == 5566
    assert st.get("robot.home_tcp") == {"xyz_m": [0.45, 0.0, 0.25], "max_dist_m": 0.05}
    t = st.get("policy.task")
    assert (t["agent_hz"], t["exec_h"], t["inference_timeout_s"]) == (20, 8, 2.0)
    lap = load_station({"station": {"host": "pc_laptop"}})
    assert lap.get("host.work_root") == "/home/ruben/franka_runs"
    assert lap.get("host.checkpoint").endswith("best_slim_epoch=0050-val_action_mse_error_all=0.0001481.ckpt")
    assert lap.get("host.prompt_zarr").endswith("firstreal-single.zarr.zip")
    jet = _yaml(CFG / "deploy.yaml")
    jh = load_station({"station": {"host": "jetson_orin_nano"}})
    for k in ("work_root", "checkpoint", "prompt_zarr", "prompt_zarr_sha256", "disk_budget_gb", "extra_path"):
        assert jh.get(f"host.{k}") == jet[k], k
    assert jh.get("host.python") == jet["python"]
    assert jh.get("host.inference_device") == jet["inference"]["device"]


def test_subsets_allowed():
    assert load_station({}).groups == {}
    assert set(load_station({"run": {"link": "ideal"}}).groups) == {"link"}


# ---------------------------------------------------------------------------------------------- error detection


def test_missing_unit_suffix(tmp_path):
    root = _copy_root(tmp_path)

    def f(d):
        d["foo"] = 3
        d["provenance"]["foo"] = {"status": "decision", "source": "test"}

    _edit(root, "station", "host", "pc_laptop", f)
    assert "'foo' is numeric without a unit suffix" in _errors(root)


def test_missing_provenance(tmp_path):
    root = _copy_root(tmp_path)
    _edit(root, "station", "host", "pc_laptop", lambda d: d.__setitem__("bar_s", 1.0))
    assert "'bar_s' has no provenance" in _errors(root)


def test_provenance_for_unknown_path(tmp_path):
    root = _copy_root(tmp_path)
    _edit(root, "station", "host", "pc_laptop",
          lambda d: d["provenance"].__setitem__("ghost_s", {"status": "decision", "source": "x"}))
    assert "provenance 'ghost_s' matches no value" in _errors(root)


def test_bad_status(tmp_path):
    root = _copy_root(tmp_path)
    _edit(root, "station", "host", "pc_laptop", lambda d: d["provenance"]["work_root"].__setitem__("status", "guess"))
    assert "status must be one of" in _errors(root)


def test_missing_source(tmp_path):
    root = _copy_root(tmp_path)
    _edit(root, "station", "host", "pc_laptop", lambda d: d["provenance"]["work_root"].__setitem__("source", ""))
    assert "source missing" in _errors(root)


def test_interpolation_cycle(tmp_path):
    root = _copy_root(tmp_path)

    def f(d):
        d["a_ref"], d["b_ref"] = "${host.b_ref}", "${host.a_ref}"
        d["provenance"]["a_ref"] = d["provenance"]["b_ref"] = {"status": "decision", "source": "test"}

    _edit(root, "station", "host", "pc_laptop", f)
    assert "interpolation cycle" in _errors(root)


def test_interpolation_unknown_target(tmp_path):
    root = _copy_root(tmp_path)
    _edit(root, "run", "policy", "bp_run001", lambda d: d.__setitem__("checkpoint", "${host.nothing}"))
    assert "no such value" in _errors(root)
    assert "no such value" in "\n".join(_err_of(lambda: load_station({"run": {"policy": "bp_run001"}})))


def _err_of(fn):
    with pytest.raises(StationError) as ei:
        fn()
    return ei.value.problems


def test_unknown_critical_path(tmp_path):
    root = _copy_root(tmp_path)
    _edit(root, "station", "robot", "panda", lambda d: d["critical"].append("no.such.path"))
    assert "critical path 'no.such.path' does not exist" in _errors(root)


def test_incompatible_tooling_sensor():
    spec = {"station": {"tooling": "franka_hand_iphumi_iphone16pro", "sensor": "rig1_wifi"}}
    msg = "\n".join(_err_of(lambda: load_station(spec)))
    assert "not compatible with sensor 'rig1_wifi'" in msg


def test_missing_schema_and_id(tmp_path):
    root = _copy_root(tmp_path)
    _edit(root, "station", "host", "pc_laptop", lambda d: d.pop("schema"))
    assert "schema must be" in _errors(root)
    root = _copy_root(tmp_path / "b")
    _edit(root, "station", "host", "pc_laptop", lambda d: d.pop("id"))
    assert "id missing" in _errors(root)
    root = _copy_root(tmp_path / "c")
    _edit(root, "station", "host", "pc_laptop", lambda d: d.__setitem__("id", "other"))
    assert "does not match the file name" in _errors(root)
    root = _copy_root(tmp_path / "d")
    _edit(root, "station", "host", "pc_laptop", lambda d: d.__setitem__("schema", "station.robot/v1"))
    assert "schema must be 'station.host/v<N>'" in _errors(root)


def test_missing_file_and_unknown_group():
    assert "file not found" in "\n".join(_err_of(lambda: load_station({"station": {"robot": "nope"}})))
    assert "unknown station group" in "\n".join(_err_of(lambda: load_station({"station": {"bogus": "x"}})))


def test_all_errors_listed_together(tmp_path):
    root = _copy_root(tmp_path)
    _edit(root, "station", "host", "pc_laptop", lambda d: d.update(foo=1, bar_s=2.0))
    msg = _errors(root)
    assert "'foo'" in msg and "'bar_s'" in msg


def test_non_suffixed_allow_list_is_enforced():
    ok = S._unit_ok
    for path in (("width_map", "a"), ("width_map", "b"), ("streams", "main", "K"), ("F_T_TCP",), ("home_O_T_EE_xyz_wxyz",),
                 ("n_demos",), ("seed",), ("ports", "main"), ("latency_s", "main"), ("limits", "v_max_m_s"),
                 ("pred_vs_demo_pos_err_mm_p95_max", "k4"), ("collisions_max",), ("collision_boxes", "0", "mass_kg")):
        assert ok(path), path
    for path in (("foo",), ("v_max",), ("width_map", "c"), ("impedance", "stiffness"), ("limits", "speed")):
        assert not ok(path), path
    assert "K" in S.NON_SUFFIXED_ALLOW and "*_T_*" in S.NON_SUFFIXED_ALLOW


# ------------------------------------------------------------------------------------------------- sets, hashes


def test_sets_override():
    st = load_station(IPHONE, sets=["link.latency_s=0.5", "tooling.width_map.b=0.01", "link.stall.duration_s=[0.1, 0.2]",
                                    "tooling.collision_boxes.0.mass_kg=0.2"])
    assert st.get("link.latency_s") == 0.5 and st.get("tooling.width_map.b") == 0.01
    assert st.get("link.stall.duration_s") == [0.1, 0.2] and st.get("tooling.collision_boxes.0.mass_kg") == 0.2
    assert st.overrides[0] == "link.latency_s=0.5"
    assert yaml.safe_load(st.to_yaml())["overrides"] == list(st.overrides)
    assert st.fingerprint() != load_station(IPHONE).fingerprint()


@pytest.mark.parametrize("bad", ["link.nope=1", "nogroup.x=1", "link.latency_s", "scene.workspace.min_m.7=1", "link.=1"])
def test_sets_bad_path_errors(bad):
    with pytest.raises(StationError):
        load_station(IPHONE, sets=[bad])


def test_fingerprint_stable_and_sensitive():
    st = load_station(IPHONE)
    assert len(st.fingerprint()) == 16
    rev = dataclasses.replace(st, groups={g: dict(reversed(list(d.items()))) for g, d in reversed(list(st.groups.items()))})
    assert rev.fingerprint() == st.fingerprint()
    assert load_station(IPHONE, sets=["scene.workspace.cone_max_deg=61.0"]).fingerprint() != st.fingerprint()


# --------------------------------------------------------------------------------------------- placeholders


EXPECTED_CRITICAL_IPHONE = [
    "robot.desk_end_effector", "robot.fci.libfranka_version", "robot.impedance.collision_thresholds",
    "tooling.width_map.a", "tooling.width_map.b", "tooling.F_T_TCP", "tooling.payload.mass_kg",
    "sensor.clock.latency_s.main", "sensor.clock.latency_s.ultrawide",
    "scene.workspace.min_m", "scene.workspace.max_m",
]
EXPECTED_CRITICAL_RIG_EXTRA = ["sensor.rig_config_path", "sensor.streams.main.K", "sensor.streams.ultrawide.K"]


def test_critical_placeholders_iphone_exact():
    for name in ("sim-iphone", "real-iphone"):
        got = [p for p, _ in load_station(PROFILES[name]).placeholders(critical_only=True)]
        assert got == EXPECTED_CRITICAL_IPHONE, name


def test_critical_placeholders_rig():
    got = [p for p, _ in load_station(PROFILES["real-rig"]).placeholders(critical_only=True)]
    base = [p for p in EXPECTED_CRITICAL_IPHONE if not p.startswith("sensor.")]
    assert sorted(got) == sorted(base + EXPECTED_CRITICAL_RIG_EXTRA)


def test_placeholders_superset_and_prov():
    st = load_station(IPHONE)
    allp = st.placeholders()
    crit = st.placeholders(critical_only=True)
    assert set(p for p, _ in crit) < set(p for p, _ in allp)
    assert all(pv.status in ("placeholder", "hw") for _, pv in allp)
    p = st.prov("tooling.width_map.a")
    assert p.status == "placeholder" and p.source and p.path == "width_map"
    assert st.prov("tooling.width_map.train_min_m").status == "measured"
    assert st.get("nothing.here", 5) == 5
    with pytest.raises(KeyError):
        st.get("tooling.nothing")


# -------------------------------------------------------------------------------- fragments == today's configs


def test_bridge_fragment_equals_bridge_yaml():
    b = _yaml(CFG / "bridge.yaml")
    f = load_station(IPHONE).bridge_fragment()
    pp, fpp = b["backend"]["panda_py"], f["backend"]["panda_py"]
    assert np.array_equal(np.array(fpp["impedance"], float), np.array(pp["impedance"], float))
    for k in ("damping_ratio", "nullspace_stiffness", "filter_coeff", "hostname"):
        assert fpp[k] == pp[k], k
    assert fpp["collision"] == pp["collision"] and fpp["NE_T_EE"] == pp["NE_T_EE"]
    assert f["backend"]["panda_py_version"] == b["backend"]["panda_py_version"]
    assert f["backend"]["libfranka_version"] == b["backend"]["libfranka_version"]
    for k in ("workspace_min", "workspace_max", "cone_axis", "cone_max_deg", "v_max", "w_max", "slack_s"):
        assert f["safety"][k] == b["safety"][k], k
    assert f["watchdog_hold_s"] == b["watchdog_hold_s"] and f["watchdog_stop_s"] == b["watchdog_stop_s"]
    assert f["home_O_T_EE_xyz_wxyz"] == b["home_O_T_EE_xyz_wxyz"]


def test_home_equals_deploy_yaml():
    d = _yaml(CFG / "deploy.yaml")
    st = load_station(IPHONE)
    assert st.get("robot.home_tcp.xyz_m") == d["home"]["xyz"]
    assert st.get("robot.home_tcp.max_dist_m") == d["home"]["max_dist_m"]
    assert st.get("criteria.limits.v_max_m_s") == d["task"]["v_max"]
    assert st.get("criteria.limits.w_max_rad_s") == d["task"]["w_max"]
    for k_st, k_d in (("agent_hz", "agent_hz"), ("exec_h", "exec_h"), ("inference_timeout_s", "inference_timeout_s")):
        assert st.get(f"policy.task.{k_st}") == d["task"][k_d]
    assert st.get("policy.prompt_index") == d["task"]["prompt_index"]


def test_controller_fragment_equals_franka_cartesian():
    c = _yaml(CFG / "franka_cartesian.yaml")
    f = load_station(IPHONE).controller_fragment()
    assert np.array_equal(np.array(f["F_T_TCP"], float), np.array(c["F_T_TCP"], float))
    g = f["gripper"]
    assert g["type"] == "franka_hand" and g["endpoint"] == "tcp://localhost:5566"
    assert (g["width_map"]["a"], g["width_map"]["b"]) == (1.0, 0.008)
    assert load_station({"station": {"tooling": "franka_hand_iphumi_rig1"}}).controller_fragment()["F_T_TCP"] == f["F_T_TCP"]


def test_controller_fragment_ee_frame():
    st = load_station(IPHONE)
    robot = dict(st.groups["robot"], desk_end_effector="franka_hand",
                 NE_T_EE=[[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0.1034], [0, 0, 0, 1]])
    st2 = dataclasses.replace(st, groups={**st.groups, "robot": robot})
    F_T_TCP = np.array(st.groups["tooling"]["F_T_TCP"], float)
    expect = F_T_TCP.copy()
    expect[2, 3] -= 0.1034
    assert np.allclose(st2.controller_fragment()["F_T_TCP"], expect)
    assert st2.bridge_fragment()["backend"]["panda_py"]["NE_T_EE"] is not None


def test_task_args_equal_run_task_literals():
    st = load_station(IPHONE)
    assert st.task_args() == {"lookahead_s": 0.1, "home_gripper_m": 0.08, "camera_latency_s": 0.08,
                              "policy_latency_s": 0.1, "max_image_age_s": st.get("sensor.task.max_image_age_s")}
    assert st.task_args()["max_image_age_s"] == 0.35
    # run_task.py flags (B2: were literals) default to the same values the config mirrors
    from franka_deploy.tasks.run_task import parser

    d = parser().parse_args(["--work-dir", "w"])
    assert (d.lookahead_s, d.home_gripper_m, d.camera_latency_s, d.policy_latency_s) == (0.1, 0.08, 0.08, 0.1)
    # station-only (no run groups): the same literals
    assert load_station({"station": IPHONE["station"]}).task_args()["policy_latency_s"] == 0.1


def test_camera_and_sender_config():
    st = load_station(IPHONE)
    c = st.camera_config("tcp://localhost:16702", "tcp://localhost:55556")
    assert c["ports"] == {"main": 5555, "ultrawide": 5556} and c["streams"]["main"]["fps_hz"] == 60
    assert c["endpoint"] == "tcp://localhost:16702" and c["name"] == "right_wrist_camera_0"
    s = st.sender_config()
    assert s["link"]["id"] == "wifi_good" and s["sim"]["image_source"] == "render"
    assert s["streams"]["main"]["K"][0][0] == 703.1714
    r = load_station(PROFILES["sim-rig"])
    assert r.sender_config()["protocol"] == "iphumi_socketio_v2" and r.camera_config("a", "b")["ports"] == {"main": 5555}


def test_station_from_deploy_cfg():
    assert station_from_deploy_cfg({"work_root": "/x"}) is None
    st = station_from_deploy_cfg({"station": IPHONE["station"], "run": IPHONE["run"], "station_set": ["link.seed=9"]})
    assert st.get("link.seed") == 9 and st.ids["robot"] == "panda"
    assert station_from_deploy_cfg({"station": {"robot": "fr3"}}).ids == {"robot": "fr3"}


def test_link_files_match_design():
    ids = ("ideal", "wifi_good", "wifi_busy", "wifi_bad", "wifi_good_skew", "wifi_busy_skew")
    g = {i: load_station({"run": {"link": i}}) for i in ids}

    def v(i, k):
        return g[i].get("link." + k)

    assert [v("ideal", k) for k in ("latency_s", "jitter_s", "bandwidth_mbit_s", "stall.rate_hz")] == [0.0] * 4
    assert (v("wifi_good", "latency_s"), v("wifi_good", "jitter_s"), v("wifi_good", "stall.rate_hz"),
            v("wifi_good", "bandwidth_mbit_s")) == (0.008, 0.004, 0.02, 400.0)
    assert v("wifi_busy", "stall.duration_s") == [0.05, 0.25] and v("wifi_busy", "bandwidth_mbit_s") == 150.0
    assert v("wifi_bad", "stall.duration_s")[1] == 0.5 and v("wifi_bad", "stall.rate_hz") == 1.0
    for n in ("wifi_good", "wifi_busy"):
        assert v(f"{n}_skew", "clock") == {"offset_s": 12.345, "drift_ppm": 30.0}
        assert v(f"{n}_skew", "latency_s") == v(n, "latency_s")


# --------------------------------------------------------------------------------------------- import hygiene


def test_import_does_not_pull_mujoco_or_robotmq():
    code = ("import sys, franka_deploy.station\n"
            "bad = [m for m in ('mujoco', 'robotmq', 'torch', 'scipy', 'cv2') if m in sys.modules]\n"
            "assert not bad, bad\n")
    env = {k: v for k, v in os.environ.items() if k != "PYTHONPATH"}
    env["PYTHONPATH"] = str(DEPLOY_DIR)
    r = subprocess.run([sys.executable, "-c", code], env=env, cwd=str(DEPLOY_DIR), capture_output=True, text=True)
    assert r.returncode == 0, r.stderr


def test_sets_apply_before_interpolation():
    """B2: an override of a referenced value reaches every dependent value (policy.checkpoint = ${host.checkpoint})."""
    spec = {"station": {"host": "pc_laptop"}, "run": {"policy": "bp_run001"}}
    base = load_station(spec)
    st = load_station(spec, sets=["host.checkpoint=/tmp/other.ckpt", "policy.contract.exec_h=4"])
    assert st.get("host.checkpoint") == "/tmp/other.ckpt"
    assert st.get("policy.checkpoint") == "/tmp/other.ckpt" != base.get("policy.checkpoint")
    assert st.get("policy.task.exec_h") == 4               # "${policy.contract.exec_h}" follows the override too
    assert st.overrides == ("host.checkpoint=/tmp/other.ckpt", "policy.contract.exec_h=4")
    # overriding the dependent value directly still wins over the reference
    assert load_station(spec, sets=["policy.checkpoint=/x.ckpt"]).get("policy.checkpoint") == "/x.ckpt"


def test_station_groups_are_deeply_frozen():
    st = load_station(IPHONE)
    with pytest.raises(TypeError):
        st.groups["tooling"]["C_T_TCP"] = 1
    with pytest.raises(TypeError):
        st.groups["new"] = {}
    assert isinstance(st.groups["tooling"]["C_T_TCP"], tuple) and isinstance(st.groups["tooling"]["C_T_TCP"][0], tuple)
    fp = st.fingerprint()
    got = st.get("tooling.C_T_TCP")                         # accessors return mutable copies
    got[0][0] = 99.0
    frag = st.controller_fragment()
    frag["F_T_TCP"][0][0] = 99.0
    st.bridge_fragment()["safety"]["v_max"] = 99.0
    assert st.get("tooling.C_T_TCP")[0][0] == 1.0 and st.fingerprint() == fp
    assert yaml.safe_load(st.to_yaml())["fingerprint"] == fp   # to_yaml thaws
