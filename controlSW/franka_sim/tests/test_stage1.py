"""Stage 1 (A3): synthetic demos in tmp_path, hand-computed clamp slowdown, out-of-reach flag, determinism, the
sets -> model path, and a real-data smoke test. Run with the real-env interpreter (mujoco, zarr, scipy)."""

import json
import os
from pathlib import Path

for _k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):   # before numpy loads BLAS (when run alone)
    os.environ.setdefault(_k, "1")

import numpy as np  # noqa: E402
import pytest  # noqa: E402
from scipy.spatial.transform import Rotation  # noqa: E402

pytest.importorskip("mujoco")

from franka_sim import frames as fr  # noqa: E402
from franka_sim import stage1_replay as s1  # noqa: E402

IDS = {"robot": "panda", "tooling": "franka_hand_iphumi_iphone16pro", "sensor": "iphone16pro_wifi",
       "scene": "block_raise_table"}
RUN = {"criteria": "block_raise", "sim": "stage1"}
ZARR = os.environ.get("ZARR") or str(Path(os.environ.get("S", "/nonexistent")) / "deploy_work" / "prompt_zarr" /
                                     "firstreal-single.zarr.zip")


def _write_npz(path: Path, T_zup: np.ndarray, width: np.ndarray) -> Path:
    """One demo given as z-up TCP poses, stored the way the dataset stores them (ARKit y-up, rotvec + position)."""
    T = fr.T_YUP_ZUP @ T_zup
    np.savez(path, pos=T[:, :3, 3], rotvec=Rotation.from_matrix(T[:, :3, :3]).as_rotvec(), width=width,
             ends=np.array([len(T)]))
    return path


def _line_demo(n_down=180, n_hold=60, n_up=180, drop=0.15, speed_ok=True):
    """TCP z down; descends ``drop`` m to the grasp point (closing the gripper), then rises. 0.05 m/s."""
    R = np.diag([1.0, -1.0, -1.0])
    n = n_down + n_hold + n_up
    z = np.r_[np.linspace(drop, 0.0, n_down), np.zeros(n_hold), np.linspace(0.0, drop, n_up)]
    T = np.tile(np.eye(4), (n, 1, 1))
    T[:, :3, :3] = R
    T[:, 2, 3] = z
    w = np.r_[np.full(n_down, 0.08), np.linspace(0.08, 0.045, n_hold), np.full(n_up, 0.045)]
    return T, w


@pytest.fixture(scope="module")
def demo_npz(tmp_path_factory):
    T, w = _line_demo()
    return _write_npz(tmp_path_factory.mktemp("s1") / "line.npz", T, w)


@pytest.fixture(scope="module")
def ctx():
    return s1.Ctx(s1.station_spec(IDS, RUN, "panda", 0.10, "+y"))


def _demos(path):
    return [s1.prepare_demo(d) for d in s1.load_demos(path)]


def test_grasp_index_and_loading(demo_npz):
    d = _demos(demo_npz)[0]
    assert d["grasp_from"] == "phases"
    assert 180 <= d["i_grasp"] <= 245                   # closing happens while holding at the bottom
    assert np.allclose(d["T_zup"][d["i_grasp"], 2, 3], 0.0, atol=1e-6)   # z-up height at the bottom = 0
    assert np.allclose(d["T_zup"][0, 2, 3], 0.15, atol=1e-6)


def test_straight_line_reaches_with_no_clamps(demo_npz):
    spec = s1.station_spec(IDS, RUN, "panda", 0.10, "+y")
    res = s1.run_combo(spec, _demos(demo_npz), grid=[(0.5, 0.0, 0.0), (0.5, 0.0, np.pi)], top_k=1, log=lambda *_: None)
    assert res["schema"] == "stage1/v1"
    d = res["demos"][0]
    assert d["reach_fraction"] == 1.0
    assert d["clamp_fraction"] == 0.0
    assert d["slowdown"] == pytest.approx(1.0, abs=0.01)
    assert res["recommendation"]["home"]["ik_ok"]
    assert set(res["recommendation"]["base_T_world"]) == {"ep0"}
    assert res["summary"]["n_demos"] == 1


def test_overspeed_segment_clamps_and_slows(ctx):
    """0.6 m/s for 0.5 s between two rests: hand computation (v_max 0.25 m/s) says the filtered target needs 1.2 s for
    the 0.3 m, so total = 0.5 + 1.2 = 1.7 s against 1.5 s: slowdown 1.133."""
    hz = s1.DEMO_HZ
    x = np.r_[np.zeros(30), np.linspace(0.0, 0.3, 31)[1:], np.full(30, 0.3)]
    T = np.tile(np.eye(4), (len(x), 1, 1))
    T[:, 0, 3] = x
    st = s1.clamp_stats(ctx, T, duration_s=(len(x) - 1) / hz)
    expected = (0.5 + 0.3 / ctx.safety["v_max"]) / ((len(x) - 1) / hz)
    assert st["clamp_fraction"] > 0.4
    assert st["slowdown"] > 1.0
    assert st["slowdown"] == pytest.approx(expected, rel=0.10)
    assert 0.1 < st["max_backlog_m"] < 0.3


def test_out_of_reach_is_flagged_with_first_index(demo_npz):
    spec = s1.station_spec(IDS, RUN, "panda", 0.10, "+y")
    res = s1.run_combo(spec, _demos(demo_npz), grid=[(3.0, 0.0, 0.0)], top_k=1, log=lambda *_: None)
    d = res["demos"][0]
    assert d["feasible"] is False and d["reason"] == "unreachable"
    assert d["first_fail"] == 0 and d["reach_fraction"] < 0.1
    assert res["summary"]["infeasible"]["ep0"]["reason"] == "unreachable"


def test_deterministic(demo_npz):
    spec = s1.station_spec(IDS, RUN, "panda", 0.10, "+y")
    a, b = (s1.run_combo(spec, _demos(demo_npz), grid=[(0.5, 0.0, 0.0), (0.45, 0.1, 1.0)], top_k=1,
                         log=lambda *_: None) for _ in range(2))
    assert json.dumps(s1.strip_times(a), sort_keys=True, default=float) == \
        json.dumps(s1.strip_times(b), sort_keys=True, default=float)


def test_sets_reach_the_model():
    c0 = s1.Ctx(s1.station_spec(IDS, RUN, "panda", 0.0, "+y"))
    c1 = s1.Ctx(s1.station_spec(IDS, RUN, "panda", 0.05, "+y"))
    c2 = s1.Ctx(s1.station_spec(IDS, RUN, "fr3", 0.05, "-y"))
    assert c1.sm.F_T_TCP[2, 3] - c0.sm.F_T_TCP[2, 3] == pytest.approx(0.05, abs=1e-6)
    assert c2.sm.robot == "fr3"
    assert c2.sm.F_T_TCP[2, 3] == pytest.approx(c1.sm.F_T_TCP[2, 3], abs=1e-6)
    assert not np.allclose(c1.sm.F_T_TCP[:3, :3], c2.sm.F_T_TCP[:3, :3])
    assert c1.st.get("tooling.finger_adapter.extension_m") == 0.05
    assert c2.st.get("tooling.tcp_x_along_hand") == "-y"


def test_cli_registered():
    from franka_sim import cli
    from franka_sim.__main__ import _register_builtins

    _register_builtins()
    assert "stage1" in cli.registered()


@pytest.mark.skipif(not Path(ZARR).is_file(), reason="$ZARR (firstreal-single) not available")
def test_real_data_smoke(tmp_path):
    from franka_sim import __main__ as m

    rc = m.main(["stage1", "--zarr", ZARR, "--ids", ",".join(f"{k}={v}" for k, v in IDS.items()), "--quick",
                 "--demos", "0,1", "--robots", "panda", "--out", str(tmp_path)])
    assert rc == 0
    f = next(tmp_path.glob("panda_ext*/stage1.json"))
    r = json.loads(f.read_text())
    assert r["schema"] == "stage1/v1" and len(r["demos"]) == 2
    assert {"recommendation", "placements", "summary", "station_fingerprint"} <= set(r)
    assert (tmp_path / "design_summary.json").is_file() and (tmp_path / "summary.txt").is_file()


@pytest.mark.parametrize("robot", ["panda", "fr3"])
@pytest.mark.parametrize("side", ["+y", "-y"])
def test_link_groups_and_pairs(robot, side):
    """Pair lists come from Kin.groups link0..7 (A2fix): the tooling is checked against links 0-6 (incl. 5 and 6), the
    table against links 1-7 + hand + tooling (base and pads excluded), self includes link 0 and link 7."""
    c = s1.Ctx(s1.station_spec(IDS, RUN, robot, 0.10, side))
    k = c.kin
    assert sorted(c.link_geoms) == list(range(8)) and all(c.link_geoms[i] for i in range(8))
    for i in range(8):
        assert all(f"link{i}" in k._gname(g) for g in c.link_geoms[i])
    tool_links = {b for _, b in c._tool_pairs}
    for i in (0, 5, 6):
        assert set(c.link_geoms[i]) <= tool_links
    assert not set(c.link_geoms[7]) & tool_links
    table_movers = {a for a, _ in c._table_pairs}
    assert set(c.link_geoms[7]) <= table_movers and set(c.link_geoms[1]) <= table_movers
    assert not set(c.link_geoms[0]) & table_movers and not set(k.groups["pads"]) & table_movers
    self_geoms = {x for p in c._self_pairs for x in p}
    assert set(c.link_geoms[0]) <= self_geoms and set(c.link_geoms[7]) <= self_geoms
    assert not any(set(p) <= set(c.link_geoms[0] + c.link_geoms[1]) for p in c._self_pairs)   # adjacent skipped
    d_tool, d_table, d_self = c.distances(c.home_q)
    assert d_table > 0.05 and d_self > 0.0 and d_tool > c.cfg["tooling_clearance_min_m"]


_ORDER_SCRIPT = r"""
import json, sys
import numpy as np
order, zarr = sys.argv[1], sys.argv[2]
from franka_sim import stage1_replay as s1
IDS = {"robot": "panda", "tooling": "franka_hand_iphumi_iphone16pro", "sensor": "iphone16pro_wifi",
       "scene": "block_raise_table"}
spec = s1.station_spec(IDS, {"criteria": "block_raise", "sim": "stage1"}, "panda", 0.10, "-y")
ids = [0, 1, 2]
if order == "a":                          # demos first, placement given as 0.4, demos in order
    demos = [s1.prepare_demo(d) for d in s1.load_demos(zarr, ids)]
    s1._init_worker(spec, None, demos)
    gx, todo = 0.4, [0, 1, 2]
else:                                     # model first, another placement tracked before, linspace gx, reversed
    s1._init_worker(spec, None, [])
    demos = [s1.prepare_demo(d) for d in s1.load_demos(zarr, ids)]
    s1._DEMOS[:] = demos
    s1._coarse_task((0, (0.6, 0.2, 1.0), [2]))
    gx, todo = float(np.linspace(0.30, 0.70, 5)[1]), [2, 1, 0]
out = {}
for di in todo:
    _, _, r = s1._fine_task((0, di, (gx, -0.3, np.deg2rad(270.0)), None))
    out[str(ids[di])] = {k: (round(v, 6) if isinstance(v, float) else v) for k, v in r.items()
                         if k in ("feasible", "reason", "first_fail", "grasp_branch", "min_margin_deg",
                                  "min_tooling_clearance_m", "min_table_m", "min_self_m", "reach_fraction")}
    out[str(ids[di])]["start_q"] = None if r["start_q"] is None else [round(x, 5) for x in r["start_q"]]
    out[str(ids[di])]["grasp_from"] = demos[di]["grasp_from"]
print("RESULT " + json.dumps(out, sort_keys=True))
"""


@pytest.mark.skipif(not Path(ZARR).is_file(), reason="$ZARR (firstreal-single) not available")
def test_fine_task_order_independent_across_processes():
    """A3 review finding 1: the same fine-task result for 3 real demos in two fresh processes that load the demos
    before / after the model, track another placement first, use 0.4 vs linspace's 0.39999999999999997 and run the
    demos in opposite orders. PYTHONPATH holds franka_sim only (franka_deploy must come from franka_sim.model)."""
    import subprocess
    import sys

    root = Path(s1.__file__).resolve().parents[1]
    env = {**os.environ, "PYTHONPATH": str(root), "OMP_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1"}
    procs = [subprocess.Popen([sys.executable, "-c", _ORDER_SCRIPT, o, ZARR], env=env, stdout=subprocess.PIPE,
                              stderr=subprocess.PIPE, text=True, cwd=str(root)) for o in ("a", "b")]
    res = []
    for p in procs:
        out, err = p.communicate(timeout=300)
        assert p.returncode == 0, err[-2000:]
        res.append(json.loads(next(ln for ln in out.splitlines() if ln.startswith("RESULT "))[7:]))
    assert res[0] == res[1]
    assert all(r["grasp_from"] == "phases" for r in res[0].values())
    assert sum(r["feasible"] for r in res[0].values()) == 3


def test_flange_speed_and_cone_gate(demo_npz):
    """time_above_vmax is on the flange path (same as the clamp), the TCP number is separate; the cone gate is off by
    default and, when on, fails a demo whose flange tilts past cone_max_deg."""
    spec = s1.station_spec(IDS, RUN, "panda", 0.10, "+y")
    dem = _demos(demo_npz)
    res = s1.run_combo(spec, dem, grid=[(0.5, 0.0, 0.0)], top_k=1, log=lambda *_: None)
    d = res["demos"][0]
    assert {"time_above_vmax_fraction", "tcp_time_above_vmax_fraction", "flange_speed_max_m_s"} <= set(d)
    assert res["summary"]["gates"] == {"cone": False, "box": False}
    assert {"worst_joint_margin_feasible_deg", "worst_joint_margin_all_demos_deg", "n_demos_over_cone",
            "cone_max_angle_deg"} <= set(res["summary"])
    assert d["feasible"] and d["cone_violation_fraction"] == 0.0           # straight down: inside the cone
    c = s1.Ctx(spec)
    c.cfg["gate_cone"] = True
    c.safety = {**c.safety, "cone_max_deg": 0.0}                            # every frame is "over" a 0 deg cone
    T_ee = dem[0]["T_zup"] @ c.TCP_T_EE
    r = s1._gate_safety(c, T_ee, np.arange(len(T_ee)), {"feasible": True, "first_fail": None, "reason": None})
    assert r["feasible"] is False and r["reason"] == "cone" and r["first_fail"] == 0
