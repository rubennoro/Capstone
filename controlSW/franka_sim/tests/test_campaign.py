"""A8 Stage 1b campaign: pure helpers (alignment + window metric, Wilson aggregation, resume, infeasible labelling,
overlay construction) and one lock-wrapped process test (2 demos x speed 0.5 on sim-iphone, end to end).

    cd controlSW/franka_sim && MUJOCO_GL=egl env -u PYTHONPATH $RE -m pytest -q tests/test_campaign.py
    process test: FD_TEST_PORT_OFFSET=1000 A8_PROCESS=1 flock $LOCK ... -k process   (needs the firstreal zarr)
"""

import json
import math
import os
from pathlib import Path

import numpy as np
import pytest

from franka_sim import campaign as C

S = Path(os.environ.get("S", "/tmp/claude-1000/-home-ruben-Capstone/299e5dab-9afb-41ab-aa6c-ac52f8fb5552/scratchpad"))
ZARR = Path(os.environ.get("ZARR") or S / "deploy_work" / "prompt_zarr" / "firstreal-single.zarr.zip")
STAGE1 = Path(__file__).resolve().parents[2] / "docs" / "sim_plan" / "results" / "stage1" / "panda_ext0.1_my_stage1.json.gz"
CRIT = {"track_pos_m": 0.01, "track_rot_deg": 3, "track_ok_fraction_min": 0.9, "reflex_max": 0}


# ---------------------------------------------------------------------------------------------- synthetic paths

def _demo(n=1200, v=0.05):
    """Demo TCP poses at 60 Hz: x moves at v m/s, yaw turns slowly."""
    T = np.tile(np.eye(4), (n, 1, 1))
    t = np.arange(n) / 60.0
    T[:, 0, 3] = v * t
    th = 0.2 * t
    T[:, 0, 0], T[:, 0, 1], T[:, 1, 0], T[:, 1, 1] = np.cos(th), -np.sin(th), np.sin(th), np.cos(th)
    return T


def _schedule(n, speed=0.5, t0=100.0, lookahead=0.1):
    """Pause-mode calls: 8 actions at 20 Hz, 0.15 s pause, pointer advances 7 * 3 * speed frames."""
    calls, f, t = [], 0.0, t0
    while f < n - 1:
        calls.append({"replay_frame": f, "timestamps": t + np.arange(8) / 20.0})
        f = min(f + 7 * 3 * speed, n - 1)
        t += 7 / 20.0 + 0.15
    calls.append({"replay_frame": float(n - 1), "timestamps": t + np.arange(8) / 20.0})
    return calls, C.frame_schedule(calls, speed, lookahead)


def _measured(T, tk, fk, lag=0.0, hz=200.0, offset=None, spike=None):
    t = np.arange(tk[0] - 0.5, tk[-1] + 1.0, 1.0 / hz)
    f = np.interp(t - lag, tk, fk, left=fk[0], right=fk[-1])
    Tm = C._interp_T(T, f)
    if offset is not None:
        Tm[1:, 1, 3] += offset                         # after the first sample (the reference is anchored there)
    if spike is not None:
        a, b, d = spike
        m = (t >= t[0] + a) & (t < t[0] + b)
        Tm[m, 2, 3] += d
    return t, Tm


def test_perfect_track_is_one():
    T = _demo()
    _, (tk, fk) = _schedule(len(T))
    t, Tm = _measured(T, tk, fk)
    r = C.track_metric(t, Tm, tk, fk, T, 0.01, 3.0, 0.9)
    assert r["track_fraction"] == pytest.approx(1.0) and r["track_ok"] is True
    assert r["pos_err_mm"]["max"] < 0.5 and r["rot_err_deg"]["max"] < 0.05


def test_constant_2cm_offset_is_zero():
    T = _demo()
    _, (tk, fk) = _schedule(len(T))
    t, Tm = _measured(T, tk, fk, offset=0.02)
    r = C.track_metric(t, Tm, tk, fk, T, 0.01, 3.0, 0.9)
    assert r["track_fraction"] <= 0.01 and r["track_ok"] is False


def test_spike_shorter_than_window_passes_long_one_fails():
    T = _demo()
    _, (tk, fk) = _schedule(len(T))
    t, Tm = _measured(T, tk, fk, spike=(8.0, 8.3, 0.05))          # 0.3 s < (1 - 0.9) x 5 s
    r = C.track_metric(t, Tm, tk, fk, T, 0.01, 3.0, 0.9)
    assert 0.9 <= r["track_fraction"] < 1.0 and r["track_ok"] is True
    t, Tm = _measured(T, tk, fk, spike=(8.0, 9.5, 0.05))          # 1.5 s > 0.5 s
    r = C.track_metric(t, Tm, tk, fk, T, 0.01, 3.0, 0.9)
    assert r["track_ok"] is False and 6.0 < r["worst_window_t_s"] + 5.0 and r["worst_window_t_s"] < 9.6


def test_time_alignment_absorbs_a_pure_delay():
    T = _demo(v=0.15)
    _, (tk, fk) = _schedule(len(T), speed=1.0)
    t, Tm = _measured(T, tk, fk, lag=0.12)
    r = C.track_metric(t, Tm, tk, fk, T, 0.01, 3.0, 0.9)
    assert r["lag_s"] == pytest.approx(0.12, abs=0.011) and r["track_fraction"] == pytest.approx(1.0)


def test_frame_schedule_monotone_and_lookahead():
    calls = [{"replay_frame": 0.0, "timestamps": 10 + np.arange(8) / 20}, {"replay_frame": 10.5, "timestamps": 10.5 + np.arange(8) / 20}]
    tk, fk = C.frame_schedule(calls[::-1], 0.5, 0.1)
    assert tk[0] == pytest.approx(10.1) and fk[0] == 0.0 and fk[7] == pytest.approx(7 * 1.5)
    assert np.all(np.diff(tk) > 0) and np.all(np.diff(fk) >= 0) and fk[-1] == pytest.approx(10.5 + 7 * 1.5)


def test_window_fraction_short_series_is_one_window():
    t = np.arange(0, 2, 0.02)
    ok = np.ones_like(t, bool)
    ok[:10] = False
    assert C.window_fraction(t, ok, 5.0)[0] == pytest.approx(0.9)


# ---------------------------------------------------------------------------------------------- aggregation

def _row(demo, speed=0.5, ok=True, infeasible=False, fault=None, reflexes=(), pad=0, gt=True):
    r = {"run": C.run_id(demo, speed), "demo": demo, "speed": speed, "stage1_infeasible": infeasible,
         "system_fault": bool(fault), "fault_reason": fault, "reflexes": list(reflexes), "pad_table": pad,
         "track": {"track_ok": ok, "track_fraction": 1.0 if ok else 0.5}, "sim_gt": {"success": gt},
         "clamp_count": 1, "settle_timeouts": 0, "wall_s": 60.0, "worst_window": {}}
    r["track_ok"], r["track_fraction"] = r["track"]["track_ok"], r["track"]["track_fraction"]
    r["cause"] = C.classify(r)
    return r


def test_aggregate_wilson_infeasible_and_faults_excluded():
    from franka_deploy.eval.metrics import wilson

    rows = [_row(d) for d in range(18)] + [_row(18, ok=False), _row(19, ok=False, infeasible=True),
                                           _row(20, fault="launcher rc 2"), _row(21, reflexes=["cartesian_reflex"], ok=False, pad=2)]
    a = C.aggregate(rows, CRIT)["0.5"]
    k, n = a["track_ok"]["k"], a["track_ok"]["n"]
    assert (k, n) == (18, 20)                      # 19 infeasible and 20 faulted are not in the denominator
    p, lo, hi = wilson(18, 20)
    assert a["track_ok"]["p"] == pytest.approx(p, abs=1e-4) and a["track_ok"]["ci95"] == pytest.approx([lo, hi], abs=1e-4)
    assert a["track_ok"]["holds"] is True and a["stage1_infeasible"] == [19] and a["system_faults"] == 1
    assert a["runs_with_reflex"] == 1 and a["runs_with_pad_table"] == 1 and a["pad_table_contacts_total"] == 2
    assert a["causes"]["system_fault"] == 1 and a["causes"]["reflex"] == 1 and a["causes"]["overshoot"] == 2
    assert {f["demo"] for f in a["failing"]} == {18, 19, 20, 21}


def test_classify_order():
    r = _row(0, ok=False)
    r["worst_window"] = {"pad_table": True, "safety_clamp": True}
    assert C.classify(r) == "pad_table"
    r["worst_window"] = {"safety_clamp": True, "hand": True}
    assert C.classify(r) == "safety_clamp"
    r["worst_window"] = {"hand": True}
    assert C.classify(r) == "hand"
    assert C.classify(_row(1)) is None
    assert C.classify(_row(2, fault="x", reflexes=["r"])) == "system_fault"


def test_campaign_text_renders():
    rows = [_row(0), _row(1, ok=False)]
    doc = {"schema": C.SCHEMA, "profile": "sim-iphone", "robot": "panda", "fingerprint": "f", "overrides": ["a=b"],
           "criteria": CRIT, "v_max_m_s": 0.25, "grasp_height": "scene", "anchor_dz_m": 0.01, "jobs": 2,
           "aggregates": C.aggregate(rows, CRIT), "rows": rows}
    txt = C.campaign_text(doc)
    assert "d00_s0.5" in txt and "overshoot" in txt and "1/2" in txt


# ---------------------------------------------------------------------------------------------- resume

def _clean_stored(**kw):
    r = {"final": True, "fingerprint": "fp1", "launcher_rc": 0, "launcher": {"exit_reason": "quit"},
         "sim_reset": {"kind": "sim_reset", "reply": {"ok": True}}, "episode": "/e", "replay_frames": {"completed": True},
         "reflexes": [], "track": {"track_ok": True}}
    r.update(kw)
    return r


def test_resume_skips_final_rows_with_the_same_fingerprint_and_retries_faults(tmp_path):
    c = C.Campaign.__new__(C.Campaign)
    c.out, c.fingerprint, c.speeds, c.demos = tmp_path, "fp1", [0.5, 1.0], [0, 1, 2]
    (tmp_path / "rows").mkdir()
    (tmp_path / "rows" / "d00_s0.5.json").write_text(json.dumps(_clean_stored(run="d00_s0.5")))
    (tmp_path / "rows" / "d01_s0.5.json").write_text(json.dumps(_clean_stored(run="d01_s0.5", fingerprint="OLD")))
    (tmp_path / "rows" / "d02_s0.5.json").write_text(json.dumps(_clean_stored(run="d02_s0.5", final=False)))
    (tmp_path / "rows" / "d00_s1.json").write_text("{not json")
    (tmp_path / "rows" / "d01_s1.json").write_text(json.dumps(_clean_stored(run="d01_s1", launcher_rc=2)))   # a fault
    rows, todo = c.pending()
    assert list(rows) == ["d00_s0.5"]
    assert todo == [(1, 0.5), (2, 0.5), (0, 1.0), (1, 1.0), (2, 1.0)]


def test_fault_rules():
    assert C.fault_reasons(_clean_stored()) == []
    # a policy exception after a reflex (the task cannot schedule on a reflexed bridge) is the reflex, not a fault
    r = _clean_stored(reflexes=["joint_reflex"], policy_exceptions=["EEF traj scheduling failed"],
                      replay_frames={"completed": False})
    assert C.fault_reasons(r) == [] and C.finalize(r)["cause"] == "reflex"
    r = _clean_stored(policy_exceptions=["boom"])
    assert C.fault_reasons(r) == ["policy_exception boom"]
    r = _clean_stored(episode=None, collect_fault="no episode dir", refused_start="FK 52 mm from home")
    assert C.finalize(r)["cause"] == "system_fault" and "start refused" in r["fault_reason"]
    assert "launcher timeout" in C.fault_reasons(_clean_stored(timeout_fault="launcher timeout (200 s)"))[0]


# ---------------------------------------------------------------------------------------------- stage1 fixture + overlay

def _fixture_stage1(tmp_path):
    from franka_sim import frames as fr

    n = 50
    T = np.tile(np.eye(4), (n, 1, 1))
    T[:, :3, :3] = np.array([[1, 0, 0], [0, -1, 0], [0, 0, -1.0]])      # z down
    T[:, 2, 3] = np.linspace(0.30, 0.05, n)
    bTw = fr.anchor_placement(T[30], [0.6, 0.1, 0.01], math.radians(30))
    names = ["a", "b"]
    s1 = {"schema": "stage1/v1", "robot": "panda", "station_sets": ["tooling.finger_adapter.extension_m=0.1"],
          "grasp_tcp_height_m": 0.01,
          "recommendation": {"base_T_world": {k: bTw.tolist() for k in names},
                             "start_q": {"a": [0.1] * 7, "b": None},
                             "start_tcp_pose7": {"a": fr.T_to_pose7(bTw @ T[0]).tolist(), "b": None},
                             "home": {"q": [0.0] * 7}},
          "demos": [{"name": "a", "feasible": True, "reason": None, "grasp_index": 30, "n_frames": n},
                    {"name": "b", "feasible": False, "reason": "unreachable", "grasp_index": 30, "n_frames": n}]}
    p = tmp_path / "stage1.json"
    p.write_text(json.dumps(s1))
    return p, T, bTw


def test_stage1_rows_and_overlay_from_a_fixture(tmp_path):
    p, T, bTw = _fixture_stage1(tmp_path)
    s1 = C.load_stage1(p)
    rows = C.stage1_demo_rows(s1)
    assert [r["feasible"] for r in rows] == [True, False] and rows[1]["start_q"] is None
    wm = {"a": 1.0, "b": 0.0431, "gap_min_m": 0.0, "gap_max_m": 0.08}
    w = np.full(len(T), 0.0831)
    ov = C.overlay(rows[0], T, w, wm)
    assert ov["q_init"] == [0.1] * 7 and ov["ik_ok"] is True
    assert ov["home_xyz"] == pytest.approx((bTw @ T[0])[:3, 3], abs=1e-6)
    assert ov["eraser_xy_yaw"][:2] == pytest.approx([0.6, 0.1], abs=1e-6)          # the grasp TCP lands there
    Tg = bTw @ T[30]
    yaw = ov["eraser_xy_yaw"][2]
    ey = np.array([-math.sin(yaw), math.cos(yaw)])                                  # eraser y = closing axis (TCP x)
    assert abs(float(ey @ Tg[:2, 0] / np.linalg.norm(Tg[:2, 0]))) == pytest.approx(1.0, abs=1e-6)
    assert ov["hand_gap_m"] == pytest.approx(0.04, abs=1e-9)
    # anchor lift: the start q is re-solved by the given ik (seeded at the stage1 q), the eraser xy is unchanged
    seen = {}

    def ik(Tt, q0):
        seen["T"], seen["q0"] = Tt, q0
        return np.full(7, 0.2), True

    ov2 = C.overlay(rows[0], T, w, wm, dz=0.01, ik=ik)
    assert ov2["q_init"] == [0.2] * 7 and seen["q0"].tolist() == [0.1] * 7
    assert seen["T"][2, 3] == pytest.approx((bTw @ T[0])[2, 3] + 0.01)
    assert ov2["eraser_xy_yaw"] == ov["eraser_xy_yaw"] and ov2["anchor_dz_m"] == 0.01
    # a demo whose start stage1 could not reach: best-effort IK from the home q; home = where the arm really is
    ov3 = C.overlay(rows[1], T, w, wm, ik=lambda Tt, q0: (q0 + 0.3, False), q_seed=[0.0] * 7,
                    fk=lambda q: np.eye(4) + np.pad(np.array([[0, 0, 0, 0.5]]), ((0, 3), (0, 0))))
    assert ov3["ik_ok"] is False and ov3["q_init"] == [0.3] * 7 and ov3["home_xyz"] == [0.5, 0.0, 0.0]
    # widths map through tooling.width_map and clip to the stroke
    assert C.gap_from_width(0.2, wm) == 0.08 and C.gap_from_width(0.0, wm) == 0.0


def test_parse_and_port_offset():
    assert C.parse_demos("all", 3) == [0, 1, 2] and C.parse_demos("2,0", 3) == [0, 2]
    with pytest.raises(ValueError):
        C.parse_demos("5", 3)
    assert C.parse_demos("all", [7, 3]) == [3, 7] and C.parse_demos([7], [3, 7]) == [7]   # a Stage 1 subset
    with pytest.raises(ValueError):
        C.parse_demos("0", [3, 7])
    assert C.parse_speeds("0.5,1.0") == [0.5, 1.0] and C.run_id(3, 1.0) == "d03_s1"
    prof = {"station_set": ["tooling.hand.endpoint=tcp://127.0.0.1:5565"]}
    C.apply_port_offset(prof, 1100)
    assert "tooling.hand.endpoint=tcp://127.0.0.1:6665" in prof["station_set"]
    assert "sensor.transport.ports.main=6655" in prof["station_set"] and prof["bridge"]["endpoint"].endswith(":6665")
    assert prof["sim_state_endpoint"].endswith(":6670")


def test_cli_registered():
    from franka_sim import __main__ as m
    from franka_sim import cli

    m._register_builtins()
    assert "campaign" in cli.registered()


@pytest.mark.skipif(not (STAGE1.is_file() and ZARR.is_file()), reason="needs the committed stage1.json and the firstreal zarr")
def test_campaign_on_a_stage1_subset_maps_demos_by_name(tmp_path):
    """W10: eval --stage all --episodes N gives a Stage 1 run over a subset; 1b maps its demos to zarr indices by name."""
    s1 = C.load_stage1(STAGE1)
    full = [d["name"] for d in s1["demos"]]
    s1["demos"] = [s1["demos"][10], s1["demos"][32]]
    p = tmp_path / "stage1.json"
    p.write_text(json.dumps(s1))
    s1["station_sets"] = s1["station_sets"] + ["tooling.hand.endpoint=tcp://127.0.0.1:5565", "sensor.transport.bind_host=127.0.0.1"]
    p.write_text(json.dumps(s1))                       # an eval-made Stage 1 also records the profile's wiring sets
    c = C.Campaign(C.DEPLOY_PKG / "configs" / "deploy.yaml", "sim-iphone", p, "all", "0.5", tmp_path / "o", zarr=ZARR)
    assert c.demos == [10, 32] and sorted(c.s1_rows) == [10, 32]
    assert not any(x.startswith(C.WIRING_SET_PREFIXES) for x in c.run_sets(c.plan(10, 0.5)))   # the port shift survives
    assert c.plan(10, 0.5)["name"] == full[10] and c.plan(32, 0.5)["stage1_feasible"] is False
    with pytest.raises(ValueError):
        C.Campaign(C.DEPLOY_PKG / "configs" / "deploy.yaml", "sim-iphone", p, "0", "0.5", tmp_path / "o2", zarr=ZARR)


@pytest.mark.skipif(not (STAGE1.is_file() and ZARR.is_file()), reason="needs the committed stage1.json and the firstreal zarr")
def test_campaign_plan_on_the_committed_stage1(tmp_path):
    """The real stage1.json + model: the overlays match Stage 1 (dz = 0 keeps its q; F_T_TCP override = model TCP)."""
    c = C.Campaign(C.DEPLOY_PKG / "configs" / "deploy.yaml", "sim-iphone", STAGE1, "0,32", "0.5", tmp_path,
                   grasp_height="stage1", zarr=ZARR)
    assert c.anchor_dz == 0.0 and c.crit == CRIT and c.v_max == pytest.approx(0.25)
    assert "tooling.tcp_x_along_hand=-y" in c.overrides and any(o.startswith("tooling.F_T_TCP=") for o in c.overrides)
    p0 = c.plan(0, 0.5)
    assert p0["q_init"] == pytest.approx(c.s1["recommendation"]["start_q"][p0["name"]], abs=1e-6)
    Tfk = c.kin_tcp.fk(np.asarray(p0["q_init"]))
    assert np.linalg.norm(Tfk[:3, 3] - np.asarray(p0["home_xyz"])) < 2e-3
    assert c.plan(32, 0.5)["stage1_feasible"] is False
    c2 = C.Campaign(C.DEPLOY_PKG / "configs" / "deploy.yaml", "sim-iphone", STAGE1, "0", "0.5", tmp_path, zarr=ZARR)
    assert c2.anchor_dz == pytest.approx(0.01) and c2.fingerprint != c.fingerprint
    p = c2.plan(0, 0.5)
    assert p["ik_ok"] and p["home_xyz"][2] == pytest.approx(p0["home_xyz"][2] + 0.01, abs=1e-4)
    (tmp_path / "r").mkdir()
    dep = c2.derive(p, tmp_path / "r", 1)
    import yaml

    prof = yaml.safe_load(dep.read_text())["profiles"]["sim-iphone"]
    assert prof["bridge"]["override"]["backend"]["mujoco"]["q_init"] == pytest.approx(p["q_init"])
    assert prof["run"]["policy"] == "demo_replay" and prof["bridge"]["endpoint"].endswith(str(5565 + c2.port_base + 100))
    assert c2.script(p).startswith("1:sim_reset:{") and ",3:c," in c2.script(p)


# ---------------------------------------------------------------------------------------------- process (lock-wrapped)

@pytest.mark.skipif(os.environ.get("A8_PROCESS") != "1" or not (STAGE1.is_file() and ZARR.is_file()),
                    reason="process test: A8_PROCESS=1 under flock $LOCK (launches 2 full sim stacks)")
def test_process_two_demos_end_to_end(tmp_path):
    out = Path(os.environ.get("A8_PROCESS_OUT") or tmp_path) / "proc"
    doc = C.Campaign(C.DEPLOY_PKG / "configs" / "deploy.yaml", "sim-iphone", STAGE1, "0,1", "0.5", out, zarr=ZARR,
                     validate_n=2).run()
    assert (out / "campaign.json").is_file() and (out / "campaign.txt").is_file()
    assert doc["schema"] == C.SCHEMA and len(doc["rows"]) == 2
    a = doc["aggregates"]["0.5"]
    assert a["system_faults"] == 0, [r["fault_reason"] for r in doc["rows"]]
    for r in doc["rows"]:
        assert r["sim_gt"] is not None and r["track"]["track_fraction"] is not None
        assert r["replay_frames"]["completed"]
    # validate: everything passes except the two by-design items of a demo_replay sim session (no checkpoint sha, no video)
    assert all(c["validate_rc"] == 0 or c["validate_ok_except_sim_known"] for c in doc["checks"]), doc["checks"]
    assert [c["eval_rc"] for c in doc["checks"]] == [0, 0], doc["checks"]
    # resume: a rerun runs nothing
    t = __import__("time").monotonic()
    C.Campaign(C.DEPLOY_PKG / "configs" / "deploy.yaml", "sim-iphone", STAGE1, "0,1", "0.5", out, zarr=ZARR, validate_n=0).run()
    assert __import__("time").monotonic() - t < 30
