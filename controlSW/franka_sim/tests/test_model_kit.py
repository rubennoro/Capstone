"""A9c finger kit -> model: box mode byte-identical to HEAD f230a96, box-equivalent kit reproduces the box pads/tags/
contacts, kit TCP at the tip midpoint, lateral_offset_m auto, derived_tcp cache keys.

Run: cd controlSW/franka_sim && MUJOCO_GL=egl env -u PYTHONPATH $RE -m pytest -q tests/test_model_kit.py
"""

from __future__ import annotations

import hashlib

import numpy as np
import pytest

mujoco = pytest.importorskip("mujoco")

from franka_deploy.station import PROFILES, load_station  # noqa: E402
from franka_sim import finger_kit as FK  # noqa: E402
from franka_sim import model as M  # noqa: E402
from franka_sim import scene as S  # noqa: E402

ROBOTS = ("panda", "fr3")
TOL = 1e-9

# ------------------------------------------------------------------------------------- box mode == HEAD f230a96
# Recorded from HEAD f230a96 (before A9c edited model.py) with mujoco 3.3.0, twice, stable: (sha256 of spec.to_xml() with
# the controlSW path replaced by a token, sha256 of to_xml + the compiled arrays below + geom names). Box mode must not
# change by a single byte; if a later phase changes box mode on purpose, re-record these with _digest and say so.
HEAD_DIGESTS = {
    ("sim-iphone", "panda"): ("2ea1dbe0f37d74d63e555297e941e68524bfa6700b4d8d3c03fc0fab66776024",
                              "48601454e4352beb455ea233a32f4ad787d569cd3553068e7a0cdc70c330b6d9"),
    ("sim-iphone", "fr3"): ("196ffb67b70372d0bd49c355e53a5db260d0579aa159600d29e2764dcc8441aa",
                            "5c7e8460a8072808511cd3d6c827dc8e5587d8bef90a2dfd1614869045527f8c"),
    ("sim-rig", "panda"): ("b053a3f21f5b2c2eead43da4d017575b01171f1bff5c468a0f795d8a611f00cd",
                           "4f7ba2fb6502a5309a0b75ec5c6231a466cec124f07a6fcff56b92f1fabe3e62"),
    ("sim-rig", "fr3"): ("e7370ad134067ca481f7b30d1cc85227521ce7965ee259c7344ceb5521fd8ea1",
                         "1cb7bc2fc3ccb8373c5a125d8d0926227b1c7427c4a08cda5e2534bbff5c5e86"),
}
ARRAYS = ("body_pos", "body_quat", "body_mass", "body_inertia", "body_ipos", "body_iquat", "geom_type", "geom_size",
          "geom_pos", "geom_quat", "geom_contype", "geom_conaffinity", "geom_condim", "geom_friction", "geom_solref",
          "geom_solimp", "geom_rgba", "geom_group", "geom_bodyid", "geom_dataid", "mesh_vert", "site_pos", "site_quat",
          "cam_pos", "cam_quat", "key_qpos", "exclude_signature")


def _digest(sm) -> tuple[str, str]:
    m = sm.model
    xml = sm.spec.to_xml().replace(str(M.CONTROLSW), "<CONTROLSW>")
    h = hashlib.sha256(xml.encode())
    for a in ARRAYS:
        h.update(a.encode())
        h.update(np.ascontiguousarray(getattr(m, a)).tobytes())
    h.update("|".join(m.geom(i).name for i in range(m.ngeom)).encode())
    return hashlib.sha256(xml.encode()).hexdigest(), h.hexdigest()


def _spec(profile: str, robot: str) -> dict:
    spec = {k: dict(v) for k, v in PROFILES[profile].items()}
    spec["station"]["robot"] = robot
    return spec


@pytest.mark.parametrize("profile,robot", sorted(HEAD_DIGESTS))
def test_box_mode_byte_identical_to_head(profile, robot):
    if mujoco.__version__ != "3.3.0":
        pytest.skip(f"digests recorded with mujoco 3.3.0, have {mujoco.__version__}")
    st = load_station(_spec(profile, robot))
    assert st.get("tooling.finger_adapter.kit") is None
    sm = M.build_model(st)
    xml_sha, all_sha = _digest(sm)
    assert xml_sha == HEAD_DIGESTS[(profile, robot)][0], "spec.to_xml() differs from HEAD box mode"
    assert all_sha == HEAD_DIGESTS[(profile, robot)][1], "compiled model arrays differ from HEAD box mode"
    assert sm.info["tcp_at"] == "pad_centre" and "finger_kit" not in sm.info
    assert sm.info["pad_length_m"] == float(st.get("tooling.finger_adapter.pad_size_m")[2])


def test_box_mode_grasp_height_unchanged():
    st = load_station(_spec("sim-iphone", "panda"))
    sm = M.build_model(st, with_scene=False)
    pl = float(st.get("tooling.finger_adapter.pad_size_m")[2])
    want = max(float(st.get("scene.grasp_tcp_height_m", 0.01)), pl / 2 + S.PAD_TABLE_CLEARANCE_M)   # HEAD formula
    assert S.grasp_tcp_height_m(sm, st) == want == 0.02


# ------------------------------------------------------------------------------------------- box-equivalent kit


def _kit_station(spec, kit_dir, L, dy, extra=()):
    return load_station(spec, sets=["tooling.finger_adapter.mode=kit", f"tooling.finger_adapter.kit={kit_dir}",
                                    f'tooling.finger_adapter.kit_sha256="{FK.kit_digest(kit_dir)}"',
                                    f"tooling.finger_adapter.adapter_length_m={L!r}",
                                    f"tooling.finger_adapter.lateral_offset_m={dy!r}", *extra])


def _jaw_tip(sm):
    fl = sm.names["bodies"]["finger_l"]
    y_c, z0 = M.jaw_mount(sm.model, fl)
    return FK.jaw_T(y_c, z0), M._finger_tip_z(sm.model, fl)


def _rot(rx, ry, rz, t):
    from scipy.spatial.transform import Rotation

    T = np.eye(4)
    T[:3, :3] = Rotation.from_euler("xyz", [rx, ry, rz], degrees=True).as_matrix()
    T[:3, 3] = t
    return T


EQUIV = [("panda", "identity"), ("fr3", "identity"), ("panda", "rotated")]


@pytest.fixture(scope="module", params=EQUIV, ids=lambda p: f"{p[0]}-{p[1]}")
def equiv(request, tmp_path_factory):
    robot, kind = request.param
    spec = _spec("sim-iphone", robot)
    st_b = load_station(spec)
    sm_b = M.build_model(st_b)
    fa = st_b.get("tooling.finger_adapter")
    J, tip_z = _jaw_tip(sm_b)
    kw = {}
    if kind == "rotated":           # CAD frames: exercises inv(kit_T_M) and per-side STLs
        kw = {"kit_T_M_left": _rot(37, -21, 113, [0.01, -0.02, 0.03]), "kit_T_M_right": _rot(15, 0, -70, [0.0, 0.05, -0.01])}
    d = FK.make_box_kit(tmp_path_factory.mktemp(f"kit_{robot}_{kind}"), pad_size_m=fa["pad_size_m"],
                        friction=(float(fa["friction"]), 0.005, 0.0001), **kw)
    pk = FK.box_equivalent_placement(J, tip_z, fa["extension_m"], fa["pad_size_m"])
    st_k = _kit_station(spec, d, pk["L"], pk["dy"])
    return {"st_b": st_b, "sm_b": sm_b, "st_k": st_k, "sm_k": M.build_model(st_k), "kit": FK.load_kit(d), "J": J,
            "pk": pk, "fa": fa}


def _body_verts(m, g) -> np.ndarray:
    """Mesh geom vertices in its body frame (compiled model: float32 vertices)."""
    mid = m.geom_dataid[g]
    v = m.mesh_vert[m.mesh_vertadr[mid]: m.mesh_vertadr[mid] + m.mesh_vertnum[mid]].astype(float)
    R = np.zeros(9)
    mujoco.mju_quat2Mat(R, m.geom_quat[g])
    return v @ R.reshape(3, 3).T + m.geom_pos[g]


def _box_corners(m, g) -> np.ndarray:
    R = np.zeros(9)
    mujoco.mju_quat2Mat(R, m.geom_quat[g])
    s = m.geom_size[g]
    c = np.array([[sx, sy, sz] for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)]) * s
    return c @ R.reshape(3, 3).T + m.geom_pos[g]


def _sorted(v):
    return v[np.lexsort(np.round(v, 7).T[::-1])]


def test_kit_pads_reproduce_box_pads(equiv):
    sm_b, sm_k, kit, J, pk = equiv["sm_b"], equiv["sm_k"], equiv["kit"], equiv["J"], equiv["pk"]
    mb, mk = sm_b.model, sm_k.model
    fk = sm_k.info["finger_kit"]
    assert fk["side_for_finger"] == {"l": "left", "r": "right"}
    for s in ("l", "r"):
        gb, gk = M._id(mb, "geom", f"pad_{s}"), M._id(mk, "geom", f"pad_{s}")
        assert mk.geom_type[gk] == mujoco.mjtGeom.mjGEOM_MESH and mb.geom_type[gb] == mujoco.mjtGeom.mjGEOM_BOX
        assert mk.geom_bodyid[gk] == mb.geom_bodyid[gb] or mk.body(int(mk.geom_bodyid[gk])).name == \
            mb.body(int(mb.geom_bodyid[gb])).name
        corners = _sorted(_box_corners(mb, gb))
        # float64 placement (what the builder hands MjSpec): exact to 1e-9 (ASCII STL keeps float64 digits)
        side = fk["side_for_finger"][s]
        out = FK.place(kit, side, J, pk["L"], pk["dy"], 0.0)
        part = kit.parts[out["contact_part"]]
        T = out["parts"][out["contact_part"]]
        v64 = _sorted(part.vertices @ T[:3, :3].T + T[:3, 3])
        assert np.abs(v64 - corners).max() < TOL
        # compiled model (MuJoCo stores mesh vertices as float32): AABB and vertices
        vk = _sorted(_body_verts(mk, gk))
        assert vk.shape == (8, 3)
        assert np.abs(vk - corners).max() < 5e-9
        assert np.abs(vk.min(0) - corners.min(0)).max() < 5e-9 and np.abs(vk.max(0) - corners.max(0)).max() < 5e-9
        for a in ("geom_friction", "geom_solref", "geom_solimp", "geom_condim", "geom_contype", "geom_conaffinity"):
            assert np.array_equal(getattr(mk, a)[gk], getattr(mb, a)[gb]), a
    assert abs(sm_k.info["pad_offset_m"] - sm_b.info["pad_offset_m"]) < TOL
    assert abs(sm_k.limits["pad_offset_m"] - sm_b.limits["pad_offset_m"]) < TOL
    assert abs(fk["pad_length_m"] - sm_b.info["pad_length_m"]) < TOL


@pytest.mark.parametrize("gap", [0.0, 0.04])
def test_kit_tags_reproduce_box_tags(equiv, gap):
    sm_b, sm_k = equiv["sm_b"], equiv["sm_k"]
    db, dk = M.forward_at(sm_b, gap_m=gap), M.forward_at(sm_k, gap_m=gap)
    for nm in ("tag_l", "tag_r", "tag_l_ink", "tag_r_ink"):
        gb, gk = M._id(sm_b.model, "geom", nm), M._id(sm_k.model, "geom", nm)
        assert np.abs(db.geom_xpos[gb] - dk.geom_xpos[gk]).max() < TOL, nm
        assert np.abs(sm_b.model.geom_size[gb] - sm_k.model.geom_size[gk]).max() < TOL, nm
        # body-frame box corners (the right tag may be Rz(180) in kit mode: compare corners, not quaternions)
        cb = _sorted(_box_corners(sm_b.model, gb))
        ck = _sorted(_box_corners(sm_k.model, gk))
        assert np.abs(cb - ck).max() < TOL, nm
        assert sm_k.model.geom_contype[gk] == 0 and np.array_equal(sm_k.model.geom_rgba[gk], sm_b.model.geom_rgba[gb])
    if gap == 0.0:
        sep = np.linalg.norm(db.geom_xpos[M._id(sm_b.model, "geom", "tag_l")] - db.geom_xpos[M._id(sm_b.model, "geom", "tag_r")])
        assert abs(sm_k.info["finger_kit"]["tag_separation_closed_m"] - sep) < TOL


def test_kit_tcp_is_tip_midpoint_pl_half_below_box_tcp(equiv):
    """By design: box TCP = pad-centre midpoint, kit TCP = midpoint of the tips (inner distal pad edges): + pl/2 along z."""
    sm_b, sm_k, fa = equiv["sm_b"], equiv["sm_k"], equiv["fa"]
    pl = float(fa["pad_size_m"][2])
    D = M.inv_T(sm_b.F_T_TCP) @ sm_k.F_T_TCP
    assert np.abs(D[:3, :3] - np.eye(3)).max() < 1e-12
    assert np.abs(D[:3, 3] - [0, 0, pl / 2]).max() < TOL
    fk = sm_k.info["finger_kit"]
    assert np.abs(fk["tcp_residual_mm"]).max() < 1e-9
    assert abs(fk["flange_to_tip_m"] - sm_k.F_T_TCP[2, 3]) < 1e-12
    assert sm_k.info["tcp_at"] == "tip" and sm_k.info["F_T_TCP_source"] == "derived"
    assert np.abs(M.derived_tcp(equiv["st_k"]) - sm_k.F_T_TCP).max() < 1e-12
    # the cameras hang off the TCP through C_T_TCP, so they move by pl/2 too; the tags (finger-fixed) do not
    st = equiv["st_k"]
    d = M.forward_at(sm_k)
    C_T_TCP = M.inv_T(M.cam_T_cv(sm_k.model, d, "cam_main")) @ M.xform_of(d, "site", M._id(sm_k.model, "site", "tcp"))
    assert np.abs(C_T_TCP - M.orthonormalize(st.get("tooling.C_T_TCP"))).max() < 1e-9


def _contacts(sm, gap, z_shift):
    """Eraser (60 mm side along the closing axis) centred between the box-mode pad centres, fingers at gap."""
    m = sm.model
    d = M.forward_at(sm, gap_m=gap)
    sc = S.Scene(sm)
    W = M.xform_of(d, "site", M._id(m, "site", "tcp"))
    W[:3, 3] -= W[:3, 2] * z_shift
    sc.set_eraser(d, W[0, 3], W[1, 3], 0.0, z=W[2, 3], quat=S.R_to_quat(W[:3, :3]))
    mujoco.mj_forward(m, d)
    out = {}
    for i in range(d.ncon):
        c = d.contact[i]
        key = tuple(sorted((sc.names[c.geom[0]], sc.names[c.geom[1]])))
        out.setdefault(key, []).append(float(c.dist))
    e = M._id(m, "geom", "eraser")
    ft = np.zeros(6)
    dist = {s: mujoco.mj_geomDistance(m, d, M._id(m, "geom", f"pad_{s}"), e, 0.05, ft) for s in ("l", "r")}
    return out, dist, sc, d


@pytest.mark.parametrize("squeeze_mm", [-2.0, 0.5, 1.0, 3.0])
def test_kit_contacts_reproduce_box_contacts(equiv, squeeze_mm):
    """Same geometry -> same pad-eraser distances (mj_geomDistance) to 1e-9 and the same contact pairs. MuJoCo's
    contact lists differ by collider (box-box: 4 points per pad at half the overlap; mesh hull vs box with multiccd:
    several points, the deepest at the full overlap = mj_geomDistance), so the kit's deepest contact is checked against
    the box-mode geom distance."""
    sm_b, sm_k, fa = equiv["sm_b"], equiv["sm_k"], equiv["fa"]
    es = np.asarray(equiv["st_b"].get("scene.eraser.size_m"), float)
    gap = es[0] - sm_b.info["pad_offset_m"] - 1e-3 * squeeze_mm
    cb, db, scb, _ = _contacts(sm_b, gap, 0.0)
    ck, dk, sck, _ = _contacts(sm_k, gap, float(fa["pad_size_m"][2]) / 2)
    for s in ("l", "r"):
        assert abs(db[s] - dk[s]) < TOL, (s, db[s], dk[s])
        assert abs(db[s] + squeeze_mm * 1e-3 / 2) < 1e-6
    assert set(cb) == set(ck), (sorted(cb), sorted(ck))
    for key, ds in ck.items():
        if "eraser" in key and any(k.startswith("pad_") for k in key):
            s = [k for k in key if k.startswith("pad_")][0][-1]
            assert abs(min(ds) - db[s]) < TOL and len(ds) >= 1        # multiccd (kit mode): up to 4 points
    assert not any(k.startswith("finger_adapter") for key in ck for k in key)


def test_kit_geoms_named_and_stock_finger_replaced(equiv):
    sm_k = equiv["sm_k"]
    m = sm_k.model
    for fb, s in ((sm_k.names["bodies"]["finger_l"], "l"), (sm_k.names["bodies"]["finger_r"], "r")):
        bid = M._id(m, "body", fb)
        coll = sorted(m.geom(g).name for g in np.flatnonzero(m.geom_bodyid == bid) if m.geom_contype[g] or m.geom_conaffinity[g])
        assert coll == sorted([f"pad_{s}", f"finger_adapter_{s}"]), coll
        g = M._id(m, "geom", f"finger_adapter_{s}")
        L = sm_k.info["finger_kit"]["L"]
        assert np.allclose(2 * m.geom_size[g], [0.021, 0.024, L])
        assert abs(sm_k.info["finger_kit"]["masses_kg"][f"finger_adapter_{s}"] - 1240 * 0.021 * 0.024 * L) < 1e-12
        assert abs(m.geom_pos[g][2] - (sm_k.info["finger_kit"]["J_m"][s][2] + L / 2)) < 1e-12
        vis = [g for g in np.flatnonzero(m.geom_bodyid == bid) if m.geom_type[g] == mujoco.mjtGeom.mjGEOM_MESH
               and not m.geom(g).name]
        assert vis and all(m.geom_group[g] == 3 for g in vis)          # stock finger meshes not rendered
    assert sm_k.info["finger_kit"]["masses_kg"]["pad_l"] == pytest.approx(1000 * 0.02 * 0.008 * 0.03)


def test_kit_scene_grasp_height_tip_clearance(equiv):
    st, sm = equiv["st_k"], equiv["sm_k"]
    assert S.grasp_tcp_height_m(sm, st) == S.PAD_TABLE_CLEARANCE_M         # tip (lowest point) at the clearance


def test_kit_kinematics_groups_include_kit_geoms(equiv):
    from franka_sim import kinematics as K

    kin = K.Kin(equiv["sm_k"], "flange")
    m = kin.m
    names = {m.geom(g).name for g in kin.groups["hand"]}
    assert {"pad_l", "pad_r", "finger_adapter_l", "finger_adapter_r"} <= names
    assert [m.geom(g).name for g in kin.groups["pads"]] == ["pad_l", "pad_r"]


# ---------------------------------------------------------------------------------------------- auto dy, cache


def test_lateral_offset_auto_pads_touch_at_gap_zero(tmp_path):
    spec = _spec("sim-iphone", "panda")
    st_b = load_station(spec)
    sm_b = M.build_model(st_b, with_scene=False)
    J, tip_z = _jaw_tip(sm_b)
    fa = st_b.get("tooling.finger_adapter")
    d = FK.make_box_kit(tmp_path / "kit")
    pk = FK.box_equivalent_placement(J, tip_z, fa["extension_m"], fa["pad_size_m"])
    st = _kit_station(spec, d, pk["L"], "auto")
    sm = M.build_model(st, with_scene=False)
    fk = sm.info["finger_kit"]
    assert fk["dy_source"] == "auto"
    for s in ("l", "r"):
        assert abs(fk["dy"][s] - (-J[1, 3] - 0.008 / 2)) < 1e-12        # finger_kit: auto = -y_c - b/2
    assert abs(sm.info["pad_offset_m"]) < 1e-12
    m = sm.model
    gl, gr = M._id(m, "geom", "pad_l"), M._id(m, "geom", "pad_r")
    ft = np.zeros(6)
    for gap in (0.0, 0.001, 0.01):
        d_ = M.forward_at(sm, gap_m=gap)
        assert abs(mujoco.mj_geomDistance(m, d_, gl, gr, 0.05, ft) - gap) < 1e-8, gap
    # the tip midpoint and the TCP are unchanged by dy (both tips move symmetrically onto the centre line)
    assert abs(fk["tcp_residual_mm"][0]) < 1e-9 and abs(fk["tcp_residual_mm"][1]) < 1e-9


def test_derived_tcp_cache_key_changes_with_kit_sha(tmp_path):
    spec = _spec("sim-iphone", "panda")
    d = tmp_path / "kit"
    FK.make_box_kit(d)
    st_a = _kit_station(spec, d, 0.02, "auto")
    tcp_a = M.derived_tcp(st_a)
    FK.make_box_kit(d, pad_size_m=(0.02, 0.008, 0.02))            # edit the kit in place: new digest
    st_b = _kit_station(spec, d, 0.02, "auto")
    assert st_a.get("tooling.finger_adapter.kit_sha256") != st_b.get("tooling.finger_adapter.kit_sha256")
    assert st_a.fingerprint() != st_b.fingerprint()
    assert st_a.fingerprint() in M._DERIVED_CACHE and st_b.fingerprint() not in M._DERIVED_CACHE
    tcp_b = M.derived_tcp(st_b)
    assert st_b.fingerprint() in M._DERIVED_CACHE
    assert abs((tcp_a[2, 3] - tcp_b[2, 3]) - 0.01) < 1e-9          # 10 mm shorter pad -> tip 10 mm closer
    with pytest.raises(FK.KitError, match="sha256 mismatch"):     # the old pin no longer loads
        M.build_model(st_a, with_scene=False)


def test_resolve_kit_relative_to_controlsw():
    assert M.resolve_kit("franka_sim") == M.CONTROLSW / "franka_sim"
    with pytest.raises(FileNotFoundError, match="relative to"):
        M.resolve_kit("no/such/kit")


def test_kit_scripted_pick_and_lift(equiv):
    """Physical grasp with the mesh pads (multiccd: several hull contacts per pad) at the kit grasp height (tip at the
    clearance, 5 mm; with 1 contact point per pad the eraser pivots and trips the 20 N reflex, see model.py step 7):
    the A6 scripted pick (test_scene.pick_and_lift) lifts and holds the eraser, 3 seeds."""
    if equiv["sm_k"].robot != "panda" or equiv["kit"].sides["left"].kit_T_M[0, 3] != 0.0:
        pytest.skip("once (panda, identity kit)")
    pytest.importorskip("zmq")
    pytest.importorskip("robotmq")
    from test_scene import pick_and_lift

    rows = [pick_and_lift(equiv["st_k"], equiv["sm_k"], seed) for seed in range(3)]
    assert all(r["success"] and r["reflex_core"] is None for r in rows), rows
    assert all(r["in_hand_slip_m"] < 0.002 for r in rows), rows


def test_multiccd_only_in_kit_mode(equiv):
    flag = int(mujoco.mjtEnableBit.mjENBL_MULTICCD)
    assert equiv["sm_k"].model.opt.enableflags & flag and not equiv["sm_b"].model.opt.enableflags & flag


def test_side_selection_uses_camera_face_in_M(tmp_path):
    """camera_face is about the M frame: yaw 180 deg turns M's x around, so the other kit side faces the camera; the
    -y camera side swaps the sides as well."""
    spec = _spec("sim-iphone", "panda")
    d = FK.make_box_kit(tmp_path / "kit")
    sides = {}
    for key, extra in (("base", []), ("yaw180", ["tooling.finger_adapter.adapter_yaw_deg=180.0"]),
                       ("my", ["tooling.tcp_x_along_hand=-y"])):
        sm = M.build_model(_kit_station(spec, d, 0.02, "auto", extra), with_scene=False)
        sides[key] = sm.info["finger_kit"]["side_for_finger"]
    assert sides["base"] == {"l": "left", "r": "right"}
    assert sides["yaw180"] == sides["my"] == {"l": "right", "r": "left"}
    with pytest.raises(ValueError, match="no kit side faces the camera"):
        M.build_model(_kit_station(spec, d, 0.02, "auto", ["tooling.finger_adapter.adapter_yaw_deg=90.0"]),
                      with_scene=False)


# ------------------------------------------------------------------------------------- addendum: modes, variants


def test_kit_set_but_mode_box_is_ignored_and_byte_identical(tmp_path, capsys):
    if mujoco.__version__ != "3.3.0":
        pytest.skip("digest recorded with mujoco 3.3.0")
    d = FK.make_box_kit(tmp_path / "kit")
    st = load_station(_spec("sim-iphone", "panda"), sets=[f"tooling.finger_adapter.kit={d}",
                                                          f'tooling.finger_adapter.kit_sha256="{FK.kit_digest(d)}"'])
    assert st.get("tooling.finger_adapter.mode") == "box"
    sm = M.build_model(st)
    assert _digest(sm) == HEAD_DIGESTS[("sim-iphone", "panda")]
    assert sm.info["finger_kit_ignored"] == "finger_adapter.kit is set but mode=box: kit ignored"
    assert "[model] finger_adapter.kit is set but mode=box: kit ignored" in capsys.readouterr().out


def test_finger_mode_defaults():
    assert M.finger_mode({}) == "box" and M.finger_mode({"kit": None}) == "box"
    assert M.finger_mode({"kit": "x"}) == "kit" and M.finger_mode({"kit": "x", "mode": "stock"}) == "stock"
    with pytest.raises(ValueError, match="kit is null"):
        M.finger_mode({"mode": "kit", "kit": None})
    with pytest.raises(ValueError, match="must be one of"):
        M.finger_mode({"mode": "boxes"})


def test_kit_variant_other_than_full_not_supported_yet(tmp_path):
    d = FK.make_box_kit(tmp_path / "kit")
    st = _kit_station(_spec("sim-iphone", "panda"), d, 0.02, "auto", ["tooling.finger_adapter.kit_variant=finger_only"])
    with pytest.raises(ValueError, match="only 'full'"):
        M.build_model(st, with_scene=False)
    sm = M.build_model(_kit_station(_spec("sim-iphone", "panda"), d, 0.02, "auto"), with_scene=False)
    assert sm.info["finger_kit"]["variant"] == "full"


@pytest.fixture(scope="module")
def stock():
    st = load_station(_spec("sim-iphone", "panda"), sets=["tooling.finger_adapter.mode=stock"])
    return st, M.build_model(st)


def test_stock_mode_pads_named_no_tags_tcp(stock):
    st, sm = stock
    m = sm.model
    assert sm.info["tcp_at"] == "stock_pads" and sm.info["finger_mode"] == "stock"
    for s, fb in (("l", "finger_l"), ("r", "finger_r")):
        g = M._id(m, "geom", f"pad_{s}")
        assert m.body(int(m.geom_bodyid[g])).name == sm.names["bodies"][fb]
        assert m.geom_type[g] == mujoco.mjtGeom.mjGEOM_BOX and np.allclose(m.geom_size[g], [0.0085, 0.004, 0.0085])
        assert m.geom_contype[g] or m.geom_conaffinity[g]
        bid = int(m.geom_bodyid[g])
        coll = [x for x in np.flatnonzero(m.geom_bodyid == bid) if m.geom_contype[x] or m.geom_conaffinity[x]]
        assert len(coll) == 6                                      # finger_0 mesh + 5 stock pads, all on
        sol = M.grip_contact(M.sim_config(st, None))
        pads = [x for x in coll if m.geom_type[x] == mujoco.mjtGeom.mjGEOM_BOX]
        assert len(pads) == 5 and all(np.allclose(m.geom_solref[x], sol["solref"]) and
                                      np.allclose(m.geom_solimp[x], sol["solimp"]) for x in pads)
        assert np.allclose(m.geom_friction[g], [1.0, 0.005, 0.0001])              # menagerie default friction kept
    for nm in ("tag_l", "tag_r", "finger_adapter_l", "finger_holder_l"):
        with pytest.raises(KeyError):
            M._id(m, "geom", nm)
    assert sm.info["pad_length_m"] == pytest.approx(0.017)
    # Franka documents the Hand TCP at 0.1034 m from the flange; the menagerie pad centres give 0.0584 + 0.0445
    assert abs(sm.F_T_TCP[2, 3] - 0.1029) < 1e-9 and np.abs(sm.F_T_TCP[:2, 3]).max() < 1e-12
    assert abs(sm.F_T_TCP[2, 3] - 0.1034) < 0.001
    fr = M.export_frames(sm)
    assert fr["TCP_T_tag_left"] is None and fr["TCP_T_tag_right"] is None
    assert S.grasp_tcp_height_m(sm, st) == max(float(st.get("scene.grasp_tcp_height_m")),
                                               0.017 / 2 + S.PAD_TABLE_CLEARANCE_M)


def test_stock_mode_static_grasp_registers(stock):
    """Eraser's 20 mm side between the stock pads, squeezed 2 mm per side past pad_l/pad_r's inner faces."""
    st, sm = stock
    m = sm.model
    es = np.asarray(st.get("scene.eraser.size_m"), float)
    gap = es[1] - sm.info["pad_offset_m"] - 0.004
    d = M.forward_at(sm, gap_m=gap)
    sc = S.Scene(sm)
    W = M.xform_of(d, "site", M._id(m, "site", "tcp"))
    R = W[:3, :3] @ np.array([[0, 1, 0], [-1, 0, 0], [0, 0, 1.0]])   # eraser y (20 mm) along the TCP x (closing)
    sc.set_eraser(d, W[0, 3], W[1, 3], 0.0, z=W[2, 3], quat=S.R_to_quat(R))
    mujoco.mj_forward(m, d)
    cs = sc.contacts(d)
    assert sc.grasped(cs), cs


def test_stock_mode_scripted_pick_and_lift(stock):
    pytest.importorskip("zmq")
    pytest.importorskip("robotmq")
    from test_scene import pick_and_lift

    st, sm = stock
    rows = [pick_and_lift(st, sm, seed) for seed in range(2)]
    print("\nstock pick:", [(r["success"], r["reason"], r["in_hand_slip_m"]) for r in rows])
    assert all(r["reflex_core"] is None for r in rows), rows
