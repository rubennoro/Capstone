"""Franka Hand finger adapter (E3): carries the iPhUMI soft fin-ray finger, in its UMI finger holder with the ArUco
tag, on one Franka Hand finger carrier. One part per side (left = tag 0 holder, right = tag 1 holder, mirror images).

Why: the policy was trained on the handheld UMI gripper. Mounting the same holders + soft fingers (+ tags) on the Franka
Hand keeps what the wrist cameras see; this part is the only new piece between the hand and the holder.

How it fits (frames in DESIGN.md "E3: as built" and adapter.yaml):
- Carrier side: a cap over the 6.00 x 21 mm carrier tongue (Franka Hand manual R50010 v1.2 Fig. 5.5). The pocket
  ceiling sits on the tongue top (height datum); 2x M4x12 through the outer wall into the tongue's M4 holes, slotted
  +-2 mm along the hand axis (decision D4); the outer wall bears on the tongue's outer face (closing-axis datum).
- Holder side: a pad that replaces the handheld gripper's MGN9C carriage: the holder plate sits on it and is held by
  its own 4 counterbored holes (15 x 10 mm pattern) with M3x16 screws threaded into 2.5 mm pilot holes. The pad's lower
  face meets the holder rack's top face (the rack is the holder's back end, toward the hand).
- Height: the holder rack straddles the tongue, so the rack's lower face must clear the tongue top (1.0 mm). That,
  not the adapter, sets the fingertip height: TCP 235.65 mm from the flange (adapter.yaml).

Frames (mm). K = carrier frame of one finger: origin = midpoint of the two M4 hole centres on the tongue's INNER face
(the face toward the other finger); x = toward the camera side (hand +x), y = outward (away from the other finger),
z = along the hand axis toward the fingertips. Hand frame (Franka Hand = flange turned -45 deg about z, origin at the
flange, z = hand axis, fingers slide along y; camera on hand +x): right side (tag 1) at hand +y, left (tag 0) at -y.
At fingertip gap g:  right: hand = (x_K, +(Y_INNER + g/2 + y_K), Z_HOLES + z_K);  left: y_hand = -(Y_INNER + g/2 + y_K).
Holder/finger CAD (vis_assembly, "asm") -> K:  x_K = z_asm - FINGER_MID_Z,  y_K = |x_asm| - tip_x(side) - Y_INNER,
z_K = y_asm - HOLDER_BACK_Y + Z_RACK_K.

Usage: ~/envs/sim_cad/bin/python -I sim_deploy/adapter/adapter.py [--no-kit] [--no-drawing]
  writes sim_deploy/adapter/out/{adapter_left,adapter_right}.{step,stl} (hand frame, gap 0), *_print.stl (lying on
  the back face, no supports), drawing.png/.svg, adapter.yaml, and the sim kit kit_franka_v1/ (asm coords, metres).
"""

from __future__ import annotations

import argparse
import shutil
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]                                   # ~/Capstone-main-sim
OUT = HERE / "out"
KIT_DIR = HERE / "kit_franka_v1"
SRC_KIT = REPO / "controlSW/franka_sim/franka_sim/assets/finger_kits/iphumi_umi_soft_v1"
SIDES = ("left", "right")


# ------------------------------------------------------------------------------------------------- parameters (mm)

@dataclass(frozen=True)
class Franka:
    """Franka Hand finger interface. Manual = Franka Hand Product Manual R50010 v1.2 (sections 5.2.3, 5.2.4);
    menagerie = the hand meshes in controlSW/franka_sim/.../franka_emika_panda (Franka's own CAD, hand frame)."""
    tongue_thickness: float = 6.00      # manual Fig. 5.5 "6.00" (along the closing axis)
    tongue_width: float = 21.0          # manual Fig. 5.5 "21"
    hole_pitch: float = 12.0            # manual Fig. 5.5 "12" (2x M4 per finger, axes along the closing axis)
    hole_below_top: float = 7.0         # manual Fig. 5.5 "15" - "8" (same lower reference line); menagerie 6.9-7.0
    exposed_height: float = 11.40       # manual Fig. 5.5 "11.40": tongue top above the housing top
    closed_inner_gap: float = 25.0      # manual Fig. 5.4: tongue inner faces at the closed stop
    open_inner_gap: float = 105.0       # manual Fig. 5.4: fully open (stroke 80)
    screw: str = "M4x12 (ISO 4762), 4 for both fingers"   # manual 5.2.4
    screw_torque_nm: float = 2.25       # manual 5.2.4 (metal fingers; see README for the printed part)
    tested_finger_length: float = 54.0  # manual 5.2.4: "designed and tested for a finger length of 54 mm"
    housing_top_z: float = 65.9         # menagerie hand_3: top of the housing either side of the finger slot
    housing_max_z: float = 66.0         # menagerie: highest housing point (slot end caps, |y| > 67)
    slot_half_width: float = 11.5       # menagerie: finger slot in the housing top, |x| < 11.5
    slot_half_length: float = 67.0      # menagerie: |y| < 67
    slot_floor_z: float = 58.4          # menagerie: slot floor (= hand.xml finger body z 0.0584)
    finger_body_z: float = 58.4         # hand.xml: left/right_finger pos 0 0 0.0584 (joint 0 = closed)
    jaw_mount_y: float = 14.414         # franka_sim.model.jaw_mount(finger_0): J = (0, 14.414, 0.133) in the body
    jaw_mount_z: float = 0.133

    @property
    def hole_z(self) -> float:          # hand z of the M4 hole centres: 65.9 + 11.4 - 7.0 = 70.3 (menagerie 70.4)
        return self.housing_top_z + self.exposed_height - self.hole_below_top

    @property
    def y_inner(self) -> float:         # hand |y| of a tongue's inner face at gap 0
        return self.closed_inner_gap / 2


@dataclass(frozen=True)
class Holder:
    """UMI finger holder + soft finger, measured on vis_assembly.stl (= the iphumi_umi_soft_v1 kit meshes, asm mm).
    Right holder (tag 1) at asm +x; the left one is its mirror image (x -> -x)."""
    back_y: float = 44.0                # holder back end (rack teeth tips), toward the hand
    rack_top_y: float = 50.8            # rack's face toward the fingertip (the pad's lower face meets it)
    pocket_y: float = 71.2              # finger pocket wall starts (nothing of the adapter beyond this)
    rack_bottom_z: float = 0.4          # rack's lowest face (camera side is +z)
    plate_under_z: float = 8.6          # holder plate underside (sits on the pad)
    plate_top_z: float = 16.6           # holder plate top face (the tags are on it)
    cbore_floor_z: float = 13.3         # floor of the 6.5 mm counterbores of the 4 carriage holes
    hole_x: tuple = (50.0, 60.0)        # carriage holes (MGN9C pattern), right side
    hole_y: tuple = (53.5, 68.5)
    hole_d: float = 3.6
    carriage_x: tuple = (40.0, 80.0)    # free zone under the plate (between rack and pocket), right side
    x_extent: tuple = (40.0, 85.09)     # holder x range, right side
    finger_mid_z: float = -2.2          # soft finger mid-plane (finger z -15.1 .. 10.7)
    tip_x: dict = field(default_factory=lambda: {"right": 39.4269, "left": 39.9960})   # |x| of the gripping face
    tip_y: float = 201.35               # soft fingertip end (TCP point, kit tip)
    tag_centre: tuple = (60.0, 87.5, 16.6)   # |x|, y, z (kit tags; 16 mm DICT_4X4_50 marker in a 20 mm outline)
    tag_size: float = 16.0
    # E7 (D8 revised): the REAL stickers sit further out than the CAD outline centre. Gripper calibration
    # pi-2026-11-18_episode_000179, fully closed frames (fingertips touching): tag centre -> holder seam 22.0 / 22.8 mm
    # in the tags' PnP plane (CAD 20.0 / 20.6), tag x-distance 44.3 mm (umiproc min_width 0.0442915; CAD 40.58).
    # The sim kit moves each tag outward by this much so its tag distance at gap 0 is training's min_width.
    tag_outward_real: float = 1.855


@dataclass(frozen=True)
class Design:
    """Choices of this design (mm)."""
    rack_clearance: float = 1.0         # holder rack lower face above the tongue top
    housing_clearance: float = 1.0      # adapter (outside the slot) above the highest housing point
    tongue_gap_inner: float = 0.3       # pocket clearance on the tongue's inner face (outer face = datum)
    tongue_gap_x: float = 0.15          # pocket clearance per side across the tongue width
    side_wall: float = 2.5              # pocket walls across the tongue (min wall rule)
    outer_wall: float = 6.0             # screw wall: M4x12 - 0.8 washer - 6.0 = 5.2 mm thread in the 6 mm tongue
    inner_y: float = -8.5               # inner face of the part (K y); adapters 8 mm apart at gap 0
    pad_y_max: float = 18.0             # outer end of the part above the tongue (K y)
    slot_travel: float = 2.0            # D4: M4 slots +-2 mm along the hand axis
    m4_clear_d: float = 4.5             # ISO 273 medium
    washer_od: float = 9.0              # DIN 125-A M4 (0.8 mm thick)
    rack_keep_margin: float = 0.5       # clearance to the rack across x
    rack_face_gap: float = 0.1          # pad lower face to the rack top face
    pocket_gap: float = 0.4             # part top to the holder's finger pocket wall
    m3_pilot_d: float = 2.5             # M3 screws threaded into the print
    m3_pilot_depth: float = 13.0        # from the pad face (M3x16: 11.3 mm in the part)
    m3_screw_len: float = 16.0
    density_kg_m3: float = 1270.0       # PETG, solid (PLA 1240)


FR, HO, DS = Franka(), Holder(), Design()


# ------------------------------------------------------------------------------------------------- derived numbers

def derived(fr: Franka = FR, ho: Holder = HO, ds: Design = DS) -> dict:
    """All key numbers in K (mm) and the hand frame. See the module docstring for the frames."""
    top = fr.hole_below_top                                        # tongue top in K
    z_rack = top + ds.rack_clearance                               # holder back end (rack lower face... along z)
    d = {
        "tongue_top_K": top,
        "z_rack_K": z_rack,                                        # K z of asm y = back_y
        "x_of_asm_z": lambda z: z - ho.finger_mid_z,               # x_K
        "rack_x_min_K": ho.rack_bottom_z - ho.finger_mid_z,        # 2.6
        "plate_under_x_K": ho.plate_under_z - ho.finger_mid_z,     # 10.8
        "plate_top_x_K": ho.plate_top_z - ho.finger_mid_z,         # 18.8
        "cbore_floor_x_K": ho.cbore_floor_z - ho.finger_mid_z,     # 15.5
        "pad_z0_K": z_rack + (ho.rack_top_y - ho.back_y) + ds.rack_face_gap,
        "part_top_K": z_rack + (ho.pocket_y - ho.back_y) - ds.pocket_gap,
        "clamp_bottom_K": fr.housing_max_z + ds.housing_clearance - fr.hole_z,
        "lip_bottom_K": -(ds.slot_travel + ds.washer_od / 2),
        "back_x_K": -(fr.tongue_width / 2 + ds.tongue_gap_x + ds.side_wall),
        "tip_K": {s: np.array([0.0, -fr.y_inner, z_rack + ho.tip_y - ho.back_y]) for s in SIDES},
    }
    d["keep_x_K"] = d["rack_x_min_K"] - ds.rack_keep_margin
    d["holes_K"] = {s: [(hx - ho.tip_x[s] - fr.y_inner, z_rack + hy - ho.back_y) for hy in ho.hole_y for hx in ho.hole_x]
                    for s in SIDES}                              # (y_K, z_K) of the 4 holder holes
    d["tip_hand_z"] = fr.hole_z + d["tip_K"]["right"][2]
    d["holder_back_hand_z"] = fr.hole_z + z_rack
    return d


# ------------------------------------------------------------------------------------------------- CAD (cadquery)

def _box(x0, x1, y0, y1, z0, z1):
    import cadquery as cq
    return cq.Solid.makeBox(x1 - x0, y1 - y0, z1 - z0, cq.Vector(x0, y0, z0))


def _cyl(r, h, p, d):
    import cadquery as cq
    return cq.Solid.makeCylinder(r, h, cq.Vector(*p), cq.Vector(*d))


def build_K(side: str, fr: Franka = FR, ho: Holder = HO, ds: Design = DS):
    """The adapter of one side as a cadquery Solid in that side's K frame (a left-handed frame for the left side:
    the physical left part is this solid mirrored, see to_hand)."""
    import cadquery as cq

    D = derived(fr, ho, ds)
    t_top, zb, zl = D["tongue_top_K"], D["clamp_bottom_K"], D["lip_bottom_K"]
    xb, xk, xp = D["back_x_K"], D["keep_x_K"], D["plate_under_x_K"]
    zp0, ztop = D["pad_z0_K"], D["part_top_K"]
    hw = fr.tongue_width / 2 + ds.tongue_gap_x                    # pocket half width
    y_out = fr.tongue_thickness + ds.outer_wall                    # outer face of the screw wall

    # clamp: cap over the tongue (full width, both x side walls), bottom above the housing
    s = _box(xb, -xb, ds.inner_y, y_out, zb, t_top)
    # lip: the screw wall continues down into the housing's finger slot (tongue footprint, |x| <= 10.5)
    s = s.fuse(_box(-fr.tongue_width / 2, fr.tongue_width / 2, fr.tongue_thickness, y_out, zl, zb + 0.01))
    # spine (camera side limited by the rack) from the clamp up to the top, wider in y above the tongue
    s = s.fuse(_box(xb, xk, ds.inner_y, ds.pad_y_max, t_top - 0.01, ztop))
    # pad under the holder plate (replaces the MGN9C carriage), above the rack
    s = s.fuse(_box(xk - 0.01, xp, ds.inner_y, ds.pad_y_max, zp0, ztop))
    # tongue pocket, open at the bottom; its ceiling (z = tongue top) is the height datum
    s = s.cut(_box(-hw, hw, -ds.tongue_gap_inner, fr.tongue_thickness, zl - 1.0, t_top))
    # M4 slots through the screw wall (axis along y), +-slot_travel along z
    r4 = ds.m4_clear_d / 2
    for x in (-fr.hole_pitch / 2, fr.hole_pitch / 2):
        for z in (-ds.slot_travel, ds.slot_travel):
            s = s.cut(_cyl(r4, y_out + 2.0, (x, -1.0, z), (0, 1, 0)))
        s = s.cut(_box(x - r4, x + r4, -1.0, y_out + 1.0, -ds.slot_travel, ds.slot_travel))
    # 4 pilot holes for the holder's M3 screws (axis along x, from the pad face)
    for (y, z) in D["holes_K"][side]:
        s = s.cut(_cyl(ds.m3_pilot_d / 2, ds.m3_pilot_depth, (xp + 0.01, y, z), (-1, 0, 0)))
    s = s.clean()
    if not s.isValid():
        raise RuntimeError(f"adapter {side}: invalid solid")
    return s


def K_to_hand(side: str, gap: float = 0.0, fr: Franka = FR) -> np.ndarray:
    """4x4 (mm): K of `side` -> hand frame at fingertip gap `gap` (mm). det = -1 for the left side (mirror)."""
    sy = 1.0 if side == "right" else -1.0
    T = np.diag([1.0, sy, 1.0, 1.0])
    T[:3, 3] = [0.0, sy * (fr.y_inner + gap / 2), fr.hole_z]
    return T


def hand_to_asm(side: str, fr: Franka = FR, ho: Holder = HO, ds: Design = DS) -> np.ndarray:
    """4x4 (mm): hand frame (gap 0) -> the holder/finger CAD frame (asm) of that side; a proper rotation."""
    D = derived(fr, ho, ds)
    c = ho.tip_x["right"] if side == "right" else -ho.tip_x["left"]
    T = np.zeros((4, 4))
    T[0, 1] = T[1, 2] = T[2, 0] = T[3, 3] = 1.0                   # asm x = hand y, asm y = hand z, asm z = hand x
    T[:3, 3] = [c, ho.back_y - D["holder_back_hand_z"], ho.finger_mid_z]
    return T


def _transform(solid, T):
    """Rigid transform (4x4, mm) of a cadquery Shape; a det -1 matrix must be diag(1, -1, 1) (mirror in y)."""
    import cadquery as cq

    R, t = T[:3, :3], T[:3, 3]
    if np.linalg.det(R) < 0:
        if not np.allclose(R, np.diag([1.0, -1.0, 1.0])):
            raise ValueError("only the y mirror is supported")
        solid, R = solid.mirror("XZ"), np.eye(3)
    if not np.allclose(R, np.eye(3)):
        raise ValueError("K -> hand is a translation (+ the y mirror) only")
    return solid.translate(cq.Vector(*map(float, t)))


def to_hand(solid_K, side: str):
    return _transform(solid_K, K_to_hand(side))


# ------------------------------------------------------------------------------------------------- outputs

def _mesh(solid, tol=0.02, ang=0.2):
    verts, faces = solid.tessellate(tol, ang)
    return np.array([[v.x, v.y, v.z] for v in verts]), np.array(faces, int)


def _write_stl(path: Path, v, f, scale=1.0):
    import trimesh
    m = trimesh.Trimesh(v * scale, f, process=True)
    if m.volume < 0:
        m.invert()
    m.export(path)
    return m


def _py(o):
    """numpy scalars/arrays -> plain Python (for yaml.safe_dump)."""
    if isinstance(o, dict):
        return {k: _py(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_py(v) for v in o]
    if isinstance(o, np.ndarray):
        return _py(o.tolist())
    if isinstance(o, np.generic):
        return o.item()
    return o


def export_parts(parts_K: dict) -> dict:
    import cadquery as cq

    OUT.mkdir(exist_ok=True)
    info = {}
    for side, sK in parts_K.items():
        sH = to_hand(sK, side)
        cq.exporters.export(cq.Workplane().add(sH), str(OUT / f"adapter_{side}.step"))
        v, f = _mesh(sH)
        m = _write_stl(OUT / f"adapter_{side}.stl", v, f)
        # print orientation: back face (K x = min) down. hand x == K x for both sides; x -> +z
        R = np.array([[0, 0, -1.0], [0, 1, 0], [1, 0, 0]])
        vp = v @ R.T
        vp -= [vp[:, 0].mean(), vp[:, 1].mean(), vp[:, 2].min()]
        _write_stl(OUT / f"adapter_{side}_print.stl", vp, f)
        info[side] = {"volume_mm3": float(sH.Volume()), "mesh_volume_mm3": float(m.volume),
                      "bbox_hand_mm": [np.round(v.min(0), 3).tolist(), np.round(v.max(0), 3).tolist()],
                      "watertight": bool(m.is_watertight)}
    return info


def holder_parts_K(side: str):
    """(holder mesh, finger mesh) of the side, from the source kit, in that side's K frame (mm, trimesh)."""
    import trimesh

    Kh = K_to_hand(side)
    asm_T_K = hand_to_asm(side) @ Kh
    K_T_asm = np.linalg.inv(asm_T_K)
    out = []
    for nm in (f"finger_holder_{side}.stl", f"finger_{side}.stl"):
        m = trimesh.load(SRC_KIT / nm)
        m.apply_scale(1000.0)
        m.apply_transform(K_T_asm)
        if m.volume < 0:
            m.invert()
        out.append(m)
    return out


def clearance_check(parts_K: dict) -> dict:
    """Min clearance (mm) of the adapter to the holder, soft finger, tongue (as designed) and housing, in K; < 0 =
    interpenetration. Points sampled on each surface; signed distance to the adapter mesh."""
    import trimesh

    D = derived()
    res = {}
    for side, sK in parts_K.items():
        v, f = _mesh(sK, 0.01, 0.1)
        A = trimesh.Trimesh(v, f, process=True)
        if A.volume < 0:
            A.invert()
        holder, finger = holder_parts_K(side)
        hw = FR.tongue_width / 2
        r = {}
        for nm, M in (("holder", holder), ("soft_finger", finger)):
            pts = np.vstack([M.sample(60000, seed=1), M.vertices])
            r[nm] = float(-trimesh.proximity.signed_distance(A, pts).max())      # sd > 0 inside the adapter
            if nm == "holder":     # without the plate underside resting on the pad (the intended contact face)
                off = np.abs(pts[:, 0] - D["plate_under_x_K"]) > 0.05
                r["holder_except_plate_face"] = float(-trimesh.proximity.signed_distance(A, pts[off]).max())
                lo = np.array([-FR.tongue_width / 2, 0.0, -50.0])
                hi = np.array([FR.tongue_width / 2, FR.tongue_thickness, D["tongue_top_K"]])
                r["holder_to_tongue"] = float(np.linalg.norm(np.maximum(0, np.maximum(lo - pts, pts - hi)), axis=1).min())
        g = np.mgrid[-hw:hw:43j, 0:FR.tongue_thickness:13j, D["lip_bottom_K"] - 2:D["tongue_top_K"]:60j]
        r["tongue"] = float(-trimesh.proximity.signed_distance(A, g.reshape(3, -1).T).max())
        # housing: highest point z = housing_max_z everywhere outside the slot; inside the slot (|x| < 11.5) the floor
        z_house = FR.housing_max_z - FR.hole_z
        z_floor = FR.slot_floor_z - FR.hole_z
        out_slot = np.abs(A.vertices[:, 0]) > FR.slot_half_width
        r["housing_top"] = float((A.vertices[out_slot, 2] - z_house).min())
        r["slot_floor"] = float((A.vertices[:, 2] - z_floor).min())
        r["slot_side"] = float(FR.slot_half_width - np.abs(A.vertices[A.vertices[:, 2] < z_house, 0]).max())
        y_hand_max_open = FR.y_inner + 40.0 + A.vertices[A.vertices[:, 2] < z_house, 1].max()
        r["slot_end_at_open"] = float(FR.slot_half_length - y_hand_max_open)
        r["other_adapter_at_gap0"] = float(2 * (FR.y_inner + A.vertices[:, 1].min()))
        res[side] = r
    return res


def write_kit(parts_K: dict) -> dict:
    """kit_franka_v1: the iphumi_umi_soft_v1 meshes + this adapter, all in asm coordinates (m), kit_T_M = the
    menagerie jaw mount J (use tooling adapter_length_m 0, lateral_offset_m 0.0, adapter_yaw_deg 0)."""
    import yaml

    KIT_DIR.mkdir(exist_ok=True)
    for nm in ("finger_left.stl", "finger_right.stl", "finger_holder_left.stl", "finger_holder_right.stl"):
        shutil.copyfile(SRC_KIT / nm, KIT_DIR / nm)
    src = yaml.safe_load((SRC_KIT / "kit.yaml").read_text())
    for side, sK in parts_K.items():
        T = hand_to_asm(side) @ K_to_hand(side)                    # K -> asm (mm); mirror for the left side
        v, f = _mesh(sK)
        v = v @ T[:3, :3].T + T[:3, 3]
        if np.linalg.det(T[:3, :3]) < 0:
            f = f[:, ::-1]
        _write_stl(KIT_DIR / f"adapter_{side}.stl", v, f, scale=1e-3)
    sides = {}
    for side in SIDES:
        sd = dict(src["sides"][side])
        # J in the finger body = (0, jaw_mount_y, jaw_mount_z); finger body (gap 0) in the hand: kit right side on the
        # body at hand +y (body axes = hand axes for the camera-facing x), left on the body at -y (x, y flipped)
        sy = 1.0 if side == "right" else -1.0
        J_hand = np.array([0.0, sy * FR.jaw_mount_y, FR.finger_body_z + FR.jaw_mount_z])
        H = hand_to_asm(side)
        o = (H[:3, :3] @ J_hand + H[:3, 3]) * 1e-3
        Rm = np.column_stack([H[:3, :3] @ [sy, 0, 0], H[:3, :3] @ [0, sy, 0], H[:3, :3] @ [0, 0, 1.0]])
        T = np.eye(4)
        T[:3, :3], T[:3, 3] = Rm, o
        sd["kit_T_M"] = [[round(float(x), 7) for x in row] for row in T]
        sd["parts"] = [f"finger_{side}", f"finger_holder_{side}", f"adapter_{side}"]
        sx = 1.0 if side == "right" else -1.0                     # E7: real sticker position (Holder.tag_outward_real)
        sd["tags"] = [dict(t, on=f"finger_holder_{side}",
                           centre=[round(t["centre"][0] + sx * HO.tag_outward_real * 1e-3, 7)] + list(t["centre"][1:]))
                      for t in sd["tags"]]
        sides[side] = sd
    doc = {
        "schema": "finger_kit/v1", "id": "iphumi_umi_soft_franka_v1", "units": "m",
        "parts": {**{f"finger_{s}": {"mesh": f"finger_{s}.stl", "role": "contact", "material": "tpu",
                                     "collision": "hull"} for s in SIDES},
                  **{f"finger_holder_{s}": {"mesh": f"finger_holder_{s}.stl", "role": "structure",
                                            "material": "pla", "collision": "hull"} for s in SIDES},
                  **{f"adapter_{s}": {"mesh": f"adapter_{s}.stl", "role": "structure", "material": "petg",
                                      "collision": "hull"} for s in SIDES}},
        "materials": {"pla": {"density_kg_m3": 1240}, "tpu": {"density_kg_m3": 1210},
                      "petg": {"density_kg_m3": DS.density_kg_m3}},
        "sides": sides,
        "variants": {"finger_only": {"left": ["finger_left"], "right": ["finger_right"]},
                     "no_holder": {"left": ["finger_left", "adapter_left"], "right": ["finger_right", "adapter_right"]}},
        "contact": src["contact"],
        "provenance": {
            "geometry": {"status": "derived", "source": "fingers + holders: iphumi_umi_soft_v1 meshes unchanged (vis_assembly "
                         "asm coords); adapter_{left,right}.stl: sim_deploy/adapter/adapter.py (E3), placed in the same asm "
                         "coords of each side"},
            "kit_T_M": {"status": "derived", "source": "M = the menagerie jaw mount J (franka_sim.model.jaw_mount: (0, 14.414, "
                        "0.133) mm in the finger body) expressed in asm coords, so the kit is placed with tooling "
                        "adapter_length_m 0, lateral_offset_m 0.0, adapter_yaw_deg 0 (no placeholder block); the adapter "
                        "pocket then sits on the real carrier tongue (manual R50010 Fig. 5.4/5.5: inner face at |y| 12.5, "
                        "holes at hand z 70.3, top 77.3) and the soft fingertips touch at finger joint 0"},
            "tip": {"status": "derived", "source": "unchanged from iphumi_umi_soft_v1 (soft fingertip end, gripping face, "
                    "mid-plane); hand z 235.65 mm with this adapter"},
            "tags_pose": {"status": "measured", "source": "iphumi_umi_soft_v1's engraved sticker outline centre moved %.3f mm "
                          "outward (asm |x| %.3f mm): the real stickers, measured in the gripper calibration "
                          "pi-2026-11-18_episode_000179 (E7, DESIGN.md D8): tag distance at gap 0 = %.2f mm = "
                          "training's min_width" % (HO.tag_outward_real, HO.tag_centre[0] + HO.tag_outward_real,
                                                    HO.tag_centre[0] * 2 - HO.tip_x["left"] - HO.tip_x["right"]
                                                    + 2 * HO.tag_outward_real)},
            "tag_size": {"status": "derived", "source": "iPhUMI aruco_config.yaml: 0.016 m"},
            "tag_ids": {"status": "derived", "source": "left = 0, right = 1 (iPhUMI); right side = hand +y with the camera on +x"},
            "materials": {"status": "datasheet", "source": "PLA 1240 (holders), TPU 1210 (fingers), PETG 1270 (adapter), "
                          "solid volume; real prints are lighter (infill) until weighed"},
            "contact": {"status": "placeholder", "source": "TPU friction 1.0, as iphumi_umi_soft_v1"},
        },
    }
    (KIT_DIR / "kit.yaml").write_text(
        "# Generated by sim_deploy/adapter/adapter.py (E3); do not edit by hand, re-run the script.\n"
        "# Tooling for this kit: adapter_length_m 0.0, lateral_offset_m 0.0, adapter_yaw_deg 0.0 (kit_T_M = jaw mount J).\n"
        + yaml.safe_dump(_py(doc), sort_keys=False, default_flow_style=None, width=200))
    sys.path.insert(0, str(REPO / "controlSW/franka_sim"))
    from franka_sim import finger_kit as FK                        # noqa: E402  (numpy + yaml only)

    FK.write_source_md(KIT_DIR, "iphumi_umi_soft_franka_v1")
    with (KIT_DIR / "SOURCE.md").open("a") as fh:
        fh.write("\n## Origin\n\n- finger_*.stl, finger_holder_*.stl: copied unchanged from "
                 "controlSW/franka_sim/franka_sim/assets/finger_kits/iphumi_umi_soft_v1 (see its SOURCE.md).\n"
                 "- adapter_*.stl: sim_deploy/adapter/adapter.py (E3), binary STL in metres, asm coordinates of each side.\n"
                 "- Re-generate with the script (it re-pins this file); then update tooling kit_sha256.\n")
    kit = FK.load_kit(KIT_DIR)
    return {"sha256": kit.sha256, "masses_kg": {n: p.mass_kg for n, p in kit.parts.items()}}


def write_yaml(D: dict, info: dict, clr: dict, kitinfo: dict | None) -> None:
    import yaml

    ho, fr = HO, FR
    tipz = D["tip_hand_z"]
    tag_gap0 = {s: ho.tag_centre[0] - ho.tip_x[s] for s in SIDES}           # tag centre |y| from the centre plane
    tag_d0 = tag_gap0["left"] + tag_gap0["right"]
    tag_d0_real = tag_d0 + 2 * ho.tag_outward_real                           # E7: the real stickers (and the sim kit)
    mass = {s: info[s]["volume_mm3"] * 1e-9 * DS.density_kg_m3 for s in SIDES}
    doc = {
        "schema": "sim_deploy.adapter/v1",
        "part": "Franka Hand -> UMI finger holder adapter, one per side (left = tag 0, right = tag 1)",
        "units": "mm unless the key says otherwise",
        "frames": {
            "hand": "Franka Hand frame = flange (libfranka F) turned -45 deg about flange z, origin at the flange (the "
                    "hand base face); z = hand axis toward the fingers, fingers slide along y; camera side = +x "
                    "(PLAN 2, deployment contracts/frames.yaml). The right side (tag 1) is at +y.",
            "K": "carrier frame of one finger: origin = midpoint of the two M4 hole centres on the tongue inner face; x = "
                 "hand +x (camera side), y = outward (away from the other finger), z = hand +z. Left K is mirrored.",
            "K_in_hand": "right: (x, +(12.5 + g/2 + y), 70.3 + z); left: (x, -(12.5 + g/2 + y), 70.3 + z), g = gap",
            "finger_body": "menagerie left_finger / right_finger body (hand.xml): origin hand z 58.4, y outward, joint q = "
                           "g/2; K origin = (0, 12.5, 11.9) in the body (x = +-hand x)",
            "assumed": ["Hand frame origin at the flange with the hand base face on it (menagerie: hand body at link7 "
                        "z 0.107 = flange, -45 deg); the menagerie hand geometry is Franka's CAD",
                        "tongue inner faces 25.0 apart at the closed stop = gap 0 (stock fingertips touch there, so "
                        "libfranka width = our fingertip gap)",
                        "housing top 65.9 (menagerie) + 11.40 (manual) = tongue top 77.3; holes 7.0 below (manual 15 - 8)"],
        },
        "interface": {"tongue": {"thickness": fr.tongue_thickness, "width": fr.tongue_width,
                                 "hole_pitch": fr.hole_pitch, "top_hand_z": round(fr.hole_z + fr.hole_below_top, 3),
                                 "holes_hand_z": round(fr.hole_z, 3), "inner_face_hand_abs_y_at_gap0": fr.y_inner},
                      "screws": "2x M4x12 ISO 4762 + DIN 125 washer per side, slotted +-2 mm along z (D4)",
                      "holder_screws": "4x M3x16 ISO 4762 per side, into 2.5 mm pilot holes (MGN9C 15 x 10 pattern)"},
        "fingertip_point": {
            "definition": "midpoint between the two soft fingertips at their very end (gripping face, finger mid-plane)",
            "hand_m": [0.0, 0.0, round(tipz * 1e-3, 5)],
            "hand_vs_gap": "(0, 0, %.2f) for every gap g; each tip at (0, +-g/2, %.2f)" % (tipz, tipz),
            "per_carrier_K": {"tip_K": [0.0, -fr.y_inner, round(tipz - fr.hole_z, 2)],
                              "note": "each fingertip relative to its own carrier: 12.5 inward of the tongue inner face, "
                                      "%.2f beyond the hole centres (%.2f beyond the tongue top)" %
                                      (tipz - fr.hole_z, tipz - fr.hole_z - fr.hole_below_top)},
            "finger_body_m": [0.0, 0.0, round((tipz - fr.finger_body_z) * 1e-3, 5)],
        },
        "D7_tcp_z_offset_from_flange_m": round(tipz * 1e-3, 5),
        "D7_note": "F_T_TCP translation (0, 0, %.5f) m; the rotation (+45 deg about flange z) is unchanged; flange->"
                   "tongue top %.1f + rack clearance %.1f + holder back end -> fingertip %.2f. The rig design "
                   "(spec page, franka_mount_with_rig.STEP) assumed 223.9 (no tongue): the fingertips are %.1f mm "
                   "further out." % (tipz * 1e-3, fr.hole_z + fr.hole_below_top, DS.rack_clearance,
                                     ho.tip_y - ho.back_y, tipz - 223.9),
        "D8_width": {
            "tag_distance_cad_at_gap0_m": round(tag_d0 * 1e-3, 5),
            "tag_distance_real_at_gap0_m": round(tag_d0_real * 1e-3, 5),
            "tag_distance_vs_gap": "tag centre distance along the closing axis = gap + %.2f mm with the real stickers "
                                   "(the CAD outline centres give gap + %.2f); slope 1: the holders are rigid on the "
                                   "carriers; iPhUMI width = (tvec1 - tvec0) . x of tag 0" % (tag_d0_real, tag_d0),
            "tag_outward_real_mm": ho.tag_outward_real,
            "tag_centres_hand_at_gap0": {s: [round(ho.tag_centre[2] - ho.finger_mid_z, 2),
                                             round((1 if s == "right" else -1) * (tag_gap0[s] + ho.tag_outward_real), 3),
                                             round(D["holder_back_hand_z"] + ho.tag_centre[1] - ho.back_y, 2)]
                                         for s in SIDES},
            "training_min_width_m": {"PICK_TEST1": 0.0442915, "PICK_TEST2": 0.0439261},
            "training_max_width_m": {"PICK_TEST1": 0.1342073, "PICK_TEST2": 0.1354158},
            "training_width": "umiproc's width stage subtracts min_width (calibration_interpolator: closed reads 0), so "
                              "a recorded width = tag x-distance - min_width = the fingertip gap (slope 1, offset 0)",
            "recommended_offset_m": 0.0,
            "note": "E7 (DESIGN.md D8 revised): the recorded width IS the fingertip gap; gripper.width_offset_m = 0. "
                    "In the calibration's closed frames the fingertips touch and the tags are 44.3 mm apart because the "
                    "real stickers sit ~2 mm further out on each holder than the CAD outline (tag_outward_real), not "
                    "because of the tag size. Check on the robot: hand fully closed -> C2 width ~0.000.",
        },
        "masses_kg": {
            "adapter_solid": {s: round(mass[s], 4) for s in SIDES},
            "adapter_volume_cm3": {s: round(info[s]["volume_mm3"] * 1e-3, 2) for s in SIDES},
            "per_finger_assembly_estimate": "adapter + holder (33.0 cm3 PLA, 41 g solid) + soft finger (47.0 cm3 TPU, "
                                            "57 g solid) + screws (~6 g) = about %.0f g solid; printed parts weigh "
                                            "less (infill): weigh them" % (mass["right"] * 1e3 + 41 + 57 + 6),
            "kit_part_masses_kg": None if kitinfo is None else {k: round(v, 4) for k, v in kitinfo["masses_kg"].items()},
        },
        "dimensions": {
            "adapter_bbox_hand_gap0": {s: info[s]["bbox_hand_mm"] for s in SIDES},
            "holder_back_end_hand_z": round(D["holder_back_hand_z"], 2),
            "part_height_z": round(D["part_top_K"] - D["lip_bottom_K"], 2),
            "slot_length_center_to_center": 2 * DS.slot_travel, "slot_width": DS.m4_clear_d,
            "holder_holes_K_yz": {s: [[round(a, 3), round(b, 3)] for a, b in D["holes_K"][s]] for s in SIDES},
        },
        "clearances_mm": clr,
        "sim_kit": None if kitinfo is None else {
            "dir": "sim_deploy/adapter/kit_franka_v1", "kit_sha256": kitinfo["sha256"],
            "tooling": {"mode": "kit", "kit": "../sim_deploy/adapter/kit_franka_v1 (relative to controlSW/)",
                        "adapter_length_m": 0.0, "lateral_offset_m": 0.0, "adapter_yaw_deg": 0.0},
        },
        "parameters": {"franka": asdict(FR), "holder": {k: (list(v) if isinstance(v, tuple) else v)
                                                        for k, v in asdict(HO).items()}, "design": asdict(DS)},
    }
    (HERE / "adapter.yaml").write_text("# Generated by sim_deploy/adapter/adapter.py (E3); re-run the script to update.\n"
                                       + yaml.safe_dump(_py(doc), sort_keys=False, width=120, default_flow_style=None))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--no-kit", action="store_true", help="skip the sim kit")
    ap.add_argument("--no-drawing", action="store_true")
    ap.add_argument("--no-check", action="store_true", help="skip the clearance check")
    a = ap.parse_args(argv)
    if not a.no_drawing:   # pyexpat must load before cadquery/OCP pulls in an older libexpat
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot  # noqa: F401
    D = derived()
    parts_K = {s: build_K(s) for s in SIDES}
    info = export_parts(parts_K)
    clr = {} if a.no_check else clearance_check(parts_K)
    kitinfo = None if a.no_kit else write_kit(parts_K)
    write_yaml(D, info, clr, kitinfo)
    if not a.no_drawing:
        from drawing import make_drawing                         # sim_deploy/adapter/drawing.py
        make_drawing(parts_K, D, OUT)
    print(f"fingertip point hand z = {D['tip_hand_z']:.2f} mm; holder back end z = {D['holder_back_hand_z']:.2f} mm")
    for s in SIDES:
        print(f"{s}: volume {info[s]['volume_mm3'] / 1e3:.2f} cm3, bbox {info[s]['bbox_hand_mm']}, "
              f"watertight {info[s]['watertight']}, clearance {clr.get(s)}")
    if kitinfo:
        print(f"kit {KIT_DIR} sha256 {kitinfo['sha256']}")
    return 0


if __name__ == "__main__":
    sys.path.insert(0, str(HERE))
    sys.exit(main())
