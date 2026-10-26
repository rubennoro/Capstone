"""A1 model builder: frames, hand attachment, finger adapters, contacts, clearance and an EGL render smoke test.

Run: cd controlSW/franka_sim && env -u PYTHONPATH MUJOCO_GL=egl $RE -m pytest -q tests/test_model.py -s
"""

from __future__ import annotations

import json
import os

import numpy as np
import pytest

mujoco = pytest.importorskip("mujoco")

from franka_deploy.station import PROFILES, load_station  # noqa: E402
from franka_sim import model as M  # noqa: E402

ROBOTS = ("panda", "fr3")


def _station(robot: str, sets=()):
    spec = {k: dict(v) for k, v in PROFILES["sim-iphone"].items()}
    spec["station"]["robot"] = robot
    return load_station(spec, sets=list(sets))


@pytest.fixture(scope="module", params=ROBOTS)
def built(request):
    st = _station(request.param)
    return st, M.build_model(st)


def _site(sm, d, name):
    return M.xform_of(d, "site", M._id(sm.model, "site", name))


# ----------------------------------------------------------------------------------------------------- frames


def test_cam_main_to_tcp_equals_C_T_TCP(built):
    st, sm = built
    d = M.forward_at(sm)
    C_T_TCP = M.inv_T(M.cam_T_cv(sm.model, d, "cam_main")) @ _site(sm, d, "tcp")
    want = M.orthonormalize(st.get("tooling.C_T_TCP"))
    assert np.abs(C_T_TCP - want).max() < 1e-9
    assert np.abs(C_T_TCP - np.array(st.get("tooling.C_T_TCP"))).max() < 1e-3   # raw YAML has 3-digit rotations


def test_cam_uw_is_main_times_extrinsic(built):
    st, sm = built
    d = M.forward_at(sm)
    main_T_uw = M.inv_T(M.cam_T_cv(sm.model, d, "cam_main")) @ M.cam_T_cv(sm.model, d, "cam_uw")
    assert np.abs(main_T_uw - M.UW_PLACEHOLDER).max() < 1e-9
    assert sm.model.cam_resolution[M._id(sm.model, "cam", "cam_uw")].tolist() == [960, 720]


def test_flange_to_tcp_equals_simmodel(built):
    _, sm = built
    d = M.forward_at(sm)
    F_T_TCP = M.inv_T(_site(sm, d, "flange")) @ _site(sm, d, "tcp")
    assert np.abs(F_T_TCP - sm.F_T_TCP).max() < 1e-9
    assert np.abs(sm.EE_T_TCP - sm.F_T_TCP).max() < 1e-12          # desk_end_effector: none -> F = EE
    fr = M.export_frames(sm)
    assert np.abs(np.array(fr["F_T_TCP"]) - sm.F_T_TCP).max() < 1e-9
    print(f"\n[{sm.robot}] F_T_TCP {fr['F_T_TCP_summary']}")


def test_flange_is_link7_plus_0107_z(built):
    _, sm = built
    d = M.forward_at(sm)
    l7 = M.xform_of(d, "body", M._id(sm.model, "body", sm.names["bodies"]["link7"]))
    assert np.abs(M.inv_T(l7) @ _site(sm, d, "flange") - M.make_T(None, [0, 0, 0.107])).max() < 1e-12


def test_hand_pose_matches_stock_panda_xml(built):
    _, sm = built
    stock = mujoco.MjModel.from_xml_path(str(M.HAND_XML.parent / "panda.xml"))
    ds = mujoco.MjData(stock)
    mujoco.mj_kinematics(stock, ds)
    want = M._body_in(stock, ds, "hand", "link7")
    assert np.abs(want - M.make_T(M.Rz(-45.0), [0, 0, 0.107])).max() < 1e-6     # stock quat has 7 digits
    d = M.forward_at(sm)
    got = M._body_in(sm.model, d, sm.names["bodies"]["hand"], sm.names["bodies"]["link7"])
    assert np.abs(got - want).max() < 1e-9
    for f in ("left_finger", "right_finger"):                                    # fingers at q = 0 too
        d0 = M.forward_at(sm, gap_m=0.0)
        mujoco.mj_resetData(stock, ds)
        mujoco.mj_kinematics(stock, ds)
        assert np.abs(M._body_in(sm.model, d0, M.HAND_PREFIX + f, sm.names["bodies"]["link7"])
                      - M._body_in(stock, ds, f, "link7")).max() < 1e-9


def test_hand_and_tcp_identical_for_panda_and_fr3():
    frames = {r: M.export_frames(M.build_model(_station(r), with_scene=False)) for r in ROBOTS}
    for k in ("F_T_hand", "F_T_TCP", "F_T_cam_main", "F_T_cam_uw", "TCP_T_tag_left"):
        assert np.abs(np.array(frames["panda"][k]) - np.array(frames["fr3"][k])).max() < 1e-9, k


# ------------------------------------------------------------------------------------------------- fingers


@pytest.mark.parametrize("gap", [0.0, 0.04, 0.08])
def test_pad_separation_is_gap_plus_pad_offset(built, gap):
    st, sm = built
    d = M.forward_at(sm, gap_m=gap)
    m = sm.model
    TCP = _site(sm, d, "tcp")
    xl = (M.inv_T(TCP) @ M.xform_of(d, "geom", M._id(m, "geom", "pad_l")))[:3, 3]
    xr = (M.inv_T(TCP) @ M.xform_of(d, "geom", M._id(m, "geom", "pad_r")))[:3, 3]
    t = st.get("tooling.finger_adapter.pad_size_m")[1]
    sep = abs(xl[0] - xr[0]) - t                        # inner-face separation along the TCP x (closing) axis
    q = sum(d.qpos[m.jnt_qposadr[M._id(m, "joint", j)]] for j in sm.names["fingers"].values())
    assert abs(q - gap) < 1e-12
    assert abs(sep - (gap + sm.limits["pad_offset_m"])) < 1e-6        # pad geometry offset, not width_map.b
    assert abs(xl[1]) < 1e-12 and abs(xr[1]) < 1e-12 and abs(xl[2]) < 1e-12   # TCP = pad-centre midpoint
    assert abs(xl[0] + xr[0]) < 1e-12


def test_stock_pads_do_not_collide(built):
    _, sm = built
    m = sm.model
    for fb in (sm.names["bodies"]["finger_l"], sm.names["bodies"]["finger_r"]):
        bid = M._id(m, "body", fb)
        colliding = [m.geom(g).name for g in np.flatnonzero(m.geom_bodyid == bid)
                     if m.geom_contype[g] or m.geom_conaffinity[g]]
        pad = "pad_l" if "left" in fb else "pad_r"
        assert colliding == [pad], colliding            # stock pads (and the stock finger mesh, see model.py) off
        assert m.geom_condim[M._id(m, "geom", pad)] == 4


# --------------------------------------------------------------------------------------------- limits/contacts


def test_mjcf_joint_ranges_vs_station(built):
    _, sm = built
    rng, lo, hi = sm.limits["mjcf_range_rad"], sm.limits["q_min_rad"], sm.limits["q_max_rad"]
    diff = np.c_[rng[:, 0] - lo, rng[:, 1] - hi]
    print(f"\n[{sm.robot}] MJCF range - station limit (min, max) per joint [rad]:\n{np.round(diff, 4)}")
    # the station (datasheet) range must be reachable in sim: MJCF hard stops at or outside it
    assert np.all(rng[:, 0] <= lo + 1e-4) and np.all(rng[:, 1] >= hi - 1e-4)
    if sm.robot == "panda":
        assert np.abs(diff).max() < 1e-4


def test_no_contacts_at_home(built):
    _, sm = built
    m = sm.model
    d = M.forward_at(sm)
    scene = {M._id(m, "body", "eraser"), M._id(m, "body", "table")}
    bad = [(m.geom(c.geom1).name or c.geom1, m.geom(c.geom2).name or c.geom2, c.dist) for c in d.contact[:d.ncon]
           if not {m.geom_bodyid[c.geom1], m.geom_bodyid[c.geom2]} <= scene]
    assert not bad, bad


def test_actuators_and_options(built):
    st, sm = built
    m = sm.model
    assert m.nu == 8
    tau = st.get("robot.joints.tau_max_nm")
    for i in range(1, 8):
        a = m.actuator(sm.names["actuators"][f"j{i}"])
        assert m.actuator_trntype[a.id] == mujoco.mjtTrn.mjTRN_JOINT
        assert m.actuator_ctrlrange[a.id].tolist() == [-tau[i - 1], tau[i - 1]]
        assert m.actuator_biastype[a.id] == 0 and m.actuator_gainprm[a.id][0] == 1.0
    h = m.actuator(sm.names["actuators"]["hand"]).id
    assert m.actuator_trntype[h] == mujoco.mjtTrn.mjTRN_TENDON
    assert m.actuator_ctrlrange[h].tolist() == [-140.0, 140.0]
    j1 = m.joint(sm.names["joints"]["j1"]).id
    assert m.dof_frictionloss[m.jnt_dofadr[j1]] == 0.0      # joint_frictionloss_scale default 0
    assert m.opt.integrator == mujoco.mjtIntegrator.mjINT_IMPLICITFAST
    assert m.opt.cone == mujoco.mjtCone.mjCONE_ELLIPTIC and m.opt.impratio == 10.0


def test_payload_reported(built):
    _, sm = built
    p = sm.info["payload"]
    assert abs(p["hand_mass_kg"] - 0.76) < 1e-9
    assert abs(p["difference_kg"] - (p["tooling_mass_kg"] - (p["payload_mass_kg"] - p["hand_mass_kg"]))) < 1e-12
    print(f"\n[{sm.robot}] payload {p}")


# --------------------------------------------------------------------------------------------------- clearance


@pytest.mark.parametrize("side", ["+y", "-y"])
def test_clearance_report_runs(side):
    for robot in ROBOTS:
        sm = M.build_model(_station(robot, [f"tooling.tcp_x_along_hand={side}"]), with_scene=False)
        rep = M.clearance_report(sm, q6_steps=5, q7_steps=5)
        assert np.isfinite(rep["min_m"]) and np.isfinite(rep["min_moving_m"])
        print(f"\n[{robot} {side}] min moving {rep['min_moving_m'] * 1e3:.1f} mm ({rep['min_moving_link']}) at "
              f"q={rep['q_at_min']}; at home {rep['at_q_base_m'] * 1e3:.1f} mm; ok(>=10mm)={rep['ok']}")


def test_tcp_side_flips_camera():
    a = M.build_model(_station("panda", ["tooling.tcp_x_along_hand=+y"]), with_scene=False)
    b = M.build_model(_station("panda", ["tooling.tcp_x_along_hand=-y"]), with_scene=False)
    ca = np.array(M.export_frames(a)["F_T_cam_main"])[:3, 3]
    cb = np.array(M.export_frames(b)["F_T_cam_main"])[:3, 3]
    assert np.allclose(ca[:2], -cb[:2]) and np.isclose(ca[2], cb[2])


# ------------------------------------------------------------------------------------------------- rendering


def test_egl_render_cam_main(built):
    if os.environ.get("MUJOCO_GL") != "egl":
        pytest.skip("set MUJOCO_GL=egl")
    _, sm = built
    m = sm.model
    d = M.forward_at(sm)
    r = mujoco.Renderer(m, 720, 960)
    try:
        r.update_scene(d, camera="cam_main")
        img = r.render()
        r.enable_segmentation_rendering()
        r.update_scene(d, camera="cam_main")
        seg = r.render()
    finally:
        r.close()
    assert img.shape == (720, 960, 3) and img.std() > 1.0
    finger_bodies = {M._id(m, "body", sm.names["bodies"][k]) for k in ("finger_l", "finger_r")}
    finger_geoms = [g for g in range(m.ngeom) if m.geom_bodyid[g] in finger_bodies]
    is_obj_geom = seg[..., 1] == mujoco.mjtObj.mjOBJ_GEOM
    bottom = np.isin(seg[360:, :, 0], finger_geoms) & is_obj_geom[360:]
    frac = bottom.mean()
    print(f"\n[{sm.robot}] finger pixel fraction in the bottom half: {frac:.4f}")
    assert frac > 0


# ----------------------------------------------------------------------------------------------- determinism


def test_build_is_deterministic(built):
    st, sm = built
    sm2 = M.build_model(st)
    assert (sm2.model.nq, sm2.model.nbody, sm2.model.ngeom) == (sm.model.nq, sm.model.nbody, sm.model.ngeom)
    assert np.array_equal(sm2.F_T_TCP, sm.F_T_TCP)
    assert sm2.station_fingerprint == sm.station_fingerprint


def test_cli_writes_json(tmp_path):
    out = tmp_path / "frames.json"
    assert M.main(["--profile", "sim-rig", "--robot", "fr3", "--no-scene", "--out", str(out)]) == 0
    fr = json.loads(out.read_text())
    assert fr["robot"] == "fr3" and len(fr["F_T_cam_main"]) == 4


@pytest.mark.parametrize("robot", ROBOTS)
@pytest.mark.parametrize("ext", (0.0, 0.1))
def test_tags_are_tcp_relative_and_near_the_camera(robot, ext):
    """Tags (and the cameras) are rigid in the TCP frame for any finger extension; the tag centres are ~98 mm from the UW
    camera along its optical axis (real: iPhUMI artag_z_distance_from_ultrawide 0.098), not at the TCP depth (~210 mm)."""
    st = _station(robot, [f"tooling.finger_adapter.extension_m={ext}"])
    sm = M.build_model(st)
    d = M.forward_at(sm)
    C_uw = M.cam_T_cv(sm.model, d, "cam_uw")
    z = []
    for side in ("l", "r"):
        P = M.inv_T(C_uw) @ M.xform_of(d, "geom", M._id(sm.model, "geom", f"tag_{side}"))
        z.append(P[2, 3])
    assert abs(np.mean(z) - 0.098) < 0.012, z
    W_tcp = _site(sm, d, "tcp")
    for side in ("l", "r"):
        T = M.inv_T(W_tcp) @ M.xform_of(d, "geom", M._id(sm.model, "geom", f"tag_{side}"))
        assert abs(T[2, 3] - M.TAG_Z_FROM_PAD_M) < 1e-6, T[:3, 3]      # same for both extensions and both arms


@pytest.mark.parametrize("robot", ROBOTS)
def test_tag_separation_is_gap_plus_real_offset(robot):
    """Real tag centre separation = finger gap + 87 mm (camera-check, 5 demos); at the home key gap the sim must match."""
    sm = M.build_model(_station(robot))
    d = M.forward_at(sm)
    gap = 2 * float(d.qpos[7])                 # both fingers at the home key position
    C = M.cam_T_cv(sm.model, d, "cam_uw")
    x = [(M.inv_T(C) @ M.xform_of(d, "geom", M._id(sm.model, "geom", f"tag_{s}")))[0, 3] for s in ("l", "r")]
    assert abs(abs(x[0] - x[1]) - (gap + 0.0869)) < 0.002, (x, gap)
