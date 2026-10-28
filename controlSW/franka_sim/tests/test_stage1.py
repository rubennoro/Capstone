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


# --------------------------------------------------------------------------------- A9d: sweep axis by finger mode

HEAD_REF = "dad0ca3"          # stage1_replay.py before A9d (box-mode reference)
A9_NEW_KEYS = {"finger_mode", "swept_key", "swept_value", "flange_to_tip_m", "tcp_at", "pad_length_m",
               "flange_to_tip_m_by_robot"}
GRID2 = [(0.5, 0.0, 0.0), (0.5, 0.0, np.pi)]


def _args(npz, out, **kw):
    import argparse

    a = dict(zarr=str(npz), ids=",".join(f"{k}={v}" for k, v in IDS.items()), run="criteria=block_raise,sim=stage1",
             root=None, set=[], robots="panda", finger_extensions=None, lengths=None, finger_mode=None,
             camera_sides="+y,-y", demos="all", jobs=1, quick=True, out=str(out))
    a.update(kw)
    return argparse.Namespace(**a)


def _sweep(mod, monkeypatch, args):
    monkeypatch.setattr(mod, "placement_grid", lambda cfg, quick=False: list(GRID2))
    return mod.sweep(args, log=lambda *_: None)


def _no_times(x):
    if isinstance(x, dict):
        return {k: _no_times(v) for k, v in x.items() if k != "wall_s"}
    if isinstance(x, list):
        return [_no_times(v) for v in x]
    return x


def _superset(new, old, path="", extra=A9_NEW_KEYS):
    """``new`` has every key/value of ``old`` (recursively); keys only in ``new`` must be A9d keys."""
    if isinstance(old, dict):
        assert isinstance(new, dict), path
        assert set(old) <= set(new), (path, set(old) - set(new))
        assert set(new) - set(old) <= extra, (path, set(new) - set(old))
        for k in old:
            _superset(new[k], old[k], f"{path}.{k}", extra)
    elif isinstance(old, list):
        assert isinstance(new, list) and len(new) == len(old), path
        for i, (a, b) in enumerate(zip(new, old)):
            _superset(a, b, f"{path}[{i}]", extra)
    else:
        assert new == old, (path, new, old)


def _head_module(tmp_path):
    import importlib.util
    import subprocess

    repo = Path(s1.__file__).resolve().parents[3]
    r = subprocess.run(["git", "-C", str(repo), "show", f"{HEAD_REF}:controlSW/franka_sim/franka_sim/stage1_replay.py"],
                       capture_output=True, text=True)
    if r.returncode != 0:
        pytest.skip(f"git show {HEAD_REF} failed: {r.stderr[:200]}")
    p = tmp_path / "stage1_replay_head.py"
    p.write_text(r.stdout)
    spec = importlib.util.spec_from_file_location("_stage1_replay_head", p)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_box_sweep_matches_head(demo_npz, tmp_path, monkeypatch):
    """Box mode: the same combo dirs and, in stage1.json + design_summary.json, every pre-A9d key with the same value;
    the only additions are the A9d design keys (finger_mode, swept_key/value, flange_to_tip_m, tcp_at, pad_length_m)."""
    head = _head_module(tmp_path)
    kw = dict(finger_extensions="0.05,0.1")
    _sweep(head, monkeypatch, _args(demo_npz, tmp_path / "head", **kw))
    summ = _sweep(s1, monkeypatch, _args(demo_npz, tmp_path / "new", **kw))
    old_files = sorted(p.relative_to(tmp_path / "head") for p in (tmp_path / "head").rglob("*.json"))
    new_files = sorted(p.relative_to(tmp_path / "new") for p in (tmp_path / "new").rglob("*.json"))
    assert old_files == new_files and len(old_files) == 5           # 4 combos + design_summary.json
    for f in old_files:
        _superset(_no_times(json.loads((tmp_path / "new" / f).read_text())),
                  _no_times(json.loads((tmp_path / "head" / f).read_text())), str(f))
    for r in summ["rows"]:
        assert r["finger_mode"] == "box" and r["swept_key"] == "extension_m" and r["swept_value"] == r["extension_m"]
    f2t = {r["extension_m"]: r["flange_to_tip_m"] for r in summ["rows"]}
    assert f2t[0.1] - f2t[0.05] == pytest.approx(0.05, abs=1e-9)
    res = json.loads(next((tmp_path / "new").glob("panda_ext0.1_py/stage1.json")).read_text())
    # box: TCP at the pad centre, tip = pad distal end
    assert res["design"]["flange_to_tip_m"] == pytest.approx(res["F_T_TCP"][2][3] + res["design"]["pad_length_m"] / 2,
                                                             abs=1e-12)
    rec = summ["recommended"]
    assert rec["swept_key"] == "extension_m" and rec["swept_value"] == rec["extension_m"]
    assert rec["flange_to_tip_m"] == pytest.approx(f2t[rec["extension_m"]])
    txt = (tmp_path / "new" / "summary.txt").read_text()
    assert f"extension_m = {rec['extension_m']} m (flange_to_tip {rec['flange_to_tip_m']:.4f} m)" in txt
    # --lengths is the same axis as --finger-extensions in box mode
    summ2 = _sweep(s1, monkeypatch, _args(demo_npz, tmp_path / "new2", lengths="0.05,0.1"))
    assert _no_times(summ2) == _no_times(summ)
    with pytest.raises(SystemExit, match="not both"):
        s1.sweep(_args(demo_npz, tmp_path / "x", lengths="0.1", finger_extensions="0.1"), log=lambda *_: None)


@pytest.fixture(scope="module")
def box_kit(tmp_path_factory):
    from franka_sim import finger_kit as FK

    d = FK.make_box_kit(tmp_path_factory.mktemp("kit") / "box_kit")
    return d, FK.kit_digest(d)


def _kit_sets(kit):
    d, sha = kit
    return ["tooling.finger_adapter.mode=kit", f"tooling.finger_adapter.kit={d}",
            f'tooling.finger_adapter.kit_sha256="{sha}"']


def test_kit_sweep_lengths_and_flange_to_tip(demo_npz, box_kit, tmp_path, monkeypatch):
    """Kit mode on a make_box_kit kit: --lengths sweeps adapter_length_m (2 lengths x 2 sides); every row reports
    flange_to_tip_m, and the box-equivalent L gives the box design's flange_to_tip (box and kit compare directly)."""
    from franka_sim import finger_kit as FK
    from franka_sim import model as M

    box = s1.Ctx(s1.station_spec(IDS, RUN, "panda", 0.10, "+y"))
    fl = box.sm.names["bodies"]["finger_l"]
    J = FK.jaw_T(*M.jaw_mount(box.sm.model, fl))
    L_eq = FK.box_equivalent_placement(J, M._finger_tip_z(box.sm.model, fl), 0.10,
                                       box.st.get("tooling.finger_adapter.pad_size_m"))["L"]
    assert L_eq > 0.0
    summ = _sweep(s1, monkeypatch, _args(demo_npz, tmp_path, lengths=f"0.0,{L_eq!r}", set=_kit_sets(box_kit)))
    dirs = sorted(p.parent.name for p in tmp_path.glob("*/stage1.json"))
    assert dirs == sorted([s1.combo_dir("panda", L, sd, "kit") for L in (0.0, L_eq) for sd in ("+y", "-y")])
    assert all(d.startswith("panda_len") for d in dirs)
    assert len(summ["rows"]) == 4
    for r in summ["rows"]:
        assert r["finger_mode"] == "kit" and r["swept_key"] == "adapter_length_m"
        assert np.isfinite(r["flange_to_tip_m"])
    f2t = {(r["swept_value"], r["camera_side"]): r["flange_to_tip_m"] for r in summ["rows"]}
    for sd in ("+y", "-y"):
        assert f2t[(L_eq, sd)] - f2t[(0.0, sd)] == pytest.approx(L_eq, abs=1e-9)
        assert f2t[(L_eq, sd)] == pytest.approx(s1.flange_to_tip_m(box.sm), abs=1e-9)
    res = json.loads(next(tmp_path.glob("panda_len0_py/stage1.json")).read_text())
    assert res["design"]["tcp_at"] == "tip" and res["design"]["adapter_length_m"] == 0.0
    assert res["design"]["flange_to_tip_m"] == pytest.approx(res["F_T_TCP"][2][3], abs=1e-12)
    assert res["design"]["finger_kit"]["sha"] == box_kit[1]
    rec = summ["recommended"]
    assert rec["finger_mode"] == "kit" and rec["swept_key"] == "adapter_length_m"
    assert rec["adapter_length_m"] == rec["swept_value"] and rec["swept_value"] in (0.0, L_eq)
    assert rec["flange_to_tip_m"] == pytest.approx(f2t[(rec["swept_value"], rec["camera_side"])])
    txt = (tmp_path / "summary.txt").read_text()
    assert "len_mm" in txt and "f2tip_mm" in txt and "adapter_length_m = " in txt and "flange_to_tip" in txt
    # eval wrapper: the new fields reach the report metrics, the old keys are kept
    from franka_sim.eval import stage1 as E

    m, _ = E.metrics_from(res)
    assert m["finger_mode"] == "kit" and m["swept_key"] == "adapter_length_m" and m["swept_value"] == 0.0
    assert m["flange_to_tip_m"] == res["design"]["flange_to_tip_m"]
    assert m["finger_extension_m"] == res["design"]["extension_m"] and m["camera_side"] == "+y"


def test_stock_sweep_has_no_length_axis(demo_npz, tmp_path, monkeypatch):
    summ = _sweep(s1, monkeypatch, _args(demo_npz, tmp_path, finger_mode="stock"))
    dirs = sorted(p.parent.name for p in tmp_path.glob("*/stage1.json"))
    assert dirs == ["panda_stock_my", "panda_stock_py"]
    for r in summ["rows"]:
        assert r["finger_mode"] == "stock" and r["swept_key"] is None and r["swept_value"] is None
        assert 0.1 < r["flange_to_tip_m"] < 0.2
    rec = summ["recommended"]
    assert rec["swept_key"] is None and rec["flange_to_tip_m"] is not None
    res = json.loads((tmp_path / "panda_stock_py" / "stage1.json").read_text())
    assert res["design"]["tcp_at"] == "stock_pads"
    assert "finger mode stock, no length axis (flange_to_tip" in (tmp_path / "summary.txt").read_text()
    with pytest.raises(SystemExit, match="no length axis"):
        s1.sweep(_args(demo_npz, tmp_path / "x", finger_mode="stock", lengths="0.1"), log=lambda *_: None)
    with pytest.raises(ValueError, match="no length axis"):
        s1.station_spec(IDS, RUN, "panda", 0.1, "+y", key=None)


def test_eval_metrics_from_pre_a9d_json():
    """A stage1.json written before A9d (design = extension_m + camera_side only) still maps; new fields are None."""
    from franka_sim.eval import stage1 as E

    row = {"name": "ep0", "feasible": True, "reach_fraction": 1.0, "min_margin_deg": 30.0, "min_tooling_clearance_m": 0.04,
           "n_collisions": 0, "clamp_fraction": 0.0, "slowdown": 1.0, "mapped_p99_le_stroke": True, "gap_p99_m": 0.05}
    res = {"summary": {"n_feasible": 1, "worst_joint_margin_all_demos_deg": 30.0, "worst_joint_margin_feasible_deg": 30.0,
                       "worst_tooling_clearance_m": 0.04, "clamp_fraction": 0.0, "slowdown_mean": 1.0, "slowdown_max": 1.0},
           "demos": [row], "recommendation": {"grid_point": {"gx": 0.5, "gy": 0.0}, "yaw_deg": 0.0}, "robot": "panda",
           "design": {"extension_m": 0.1, "camera_side": "-y"}}
    m, table = E.metrics_from(res)
    assert m["finger_extension_m"] == 0.1 and m["camera_side"] == "-y" and len(table) == 1
    assert m["finger_mode"] is None and m["swept_key"] is None and m["flange_to_tip_m"] is None
