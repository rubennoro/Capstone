"""STEP -> simulation rig: meshes, collision hulls, camera/IMU poses and mass properties in the Franka Hand frame.

Reads the team's SolidWorks assembly ``franka_mount_with_rig.STEP`` (mm) and writes, next to this file:
  assets/<part>.stl             visual meshes, Franka Hand frame, metres, <= --max-faces faces each
  assets/collision/<part>.stl   convex hulls for the parts that can touch the world (mount, plate, cover, pi5)
  rig.yaml                      schema sim_deploy.rig/v1: transforms, camera + IMU poses, masses, derived numbers

Frames (DESIGN.md section 2): hand = menagerie Franka Hand body frame (z along the hand toward the fingertips, 0 at the
hand-arm joint = libfranka's flange F; x out of the hand's camera-side face; fingers close along y). The hand is
-45 deg about flange z (libfranka F_T_NE / menagerie), so flange_T_hand = Rz(-45 deg), no translation.
STEP world (mm) -> hand (mm): x_h = z_S - 1214.765, y_h = x_S - 699.8003, z_h = y_S - 925.6717 (the
Generic_Camera_Mount part frame is the hand frame turned 180 deg about z, its origin at hand z = +7.5 mm, the lip
from local z -7.5 to 0 wraps the hand's ring at the joint; the camera side is the hand's +x face).
Cameras: OpenCV axes (x right, y down, z optical). Image down = toward the hand axis (the fingers show at the bottom
as in training), image right = hand +y (the fingertip point is +8.4 mm right of the IMX219, training: +8.1).

Run with the CAD venv (README.md):  <venv>/bin/python -I sim_deploy/rig/extract.py [--step PATH] [--out DIR]
"""

from __future__ import annotations

import argparse
import datetime
import hashlib
import sys
import tempfile
from pathlib import Path

import numpy as np
import trimesh
import yaml

from OCP.BRepGProp import BRepGProp
from OCP.BRepMesh import BRepMesh_IncrementalMesh
from OCP.GProp import GProp_GProps
from OCP.OCP.collections import Sequence_TDF_Label as TDF_LabelSequence
from OCP.STEPCAFControl import STEPCAFControl_Reader
from OCP.StlAPI import StlAPI_Writer
from OCP.TCollection import TCollection_ExtendedString
from OCP.TDataStd import TDataStd_Name
from OCP.TDF import TDF_Label
from OCP.TDocStd import TDocStd_Document
from OCP.TopAbs import TopAbs_SOLID
from OCP.TopExp import TopExp_Explorer
from OCP.TopLoc import TopLoc_Location
from OCP.XCAFDoc import XCAFDoc_DocumentTool

HERE = Path(__file__).resolve().parent
DEFAULT_STEP = Path.home() / "Downloads" / "franka_mount_with_rig.STEP"

# ---------------------------------------------------------------------------------------------- frames
# hand (mm) = R_HS @ step (mm) + T_HS
R_HS = np.array([[0.0, 0.0, 1.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
T_HS = np.array([-1214.765, -699.8003, -925.6717])
GCM_LOCAL_Z0_IN_HAND_MM = 7.5          # Generic_Camera_Mount origin (its placement t_y 933.1717 - 925.6717)


def Rz(deg: float) -> np.ndarray:
    a = np.deg2rad(deg)
    return np.array([[np.cos(a), -np.sin(a), 0], [np.sin(a), np.cos(a), 0], [0, 0, 1.0]])


FLANGE_R_HAND = Rz(-45.0)              # menagerie panda.xml hand quat (0.9238795, 0, 0, -0.3826834); libfranka F_T_NE
HAND_R_TCP = Rz(90.0)                  # deployment contracts/frames.yaml F_T_TCP = Rz(+45) = flange_R_hand @ Rz(+90)

# ---------------------------------------------------------------------------------------------- parts
# product name in the STEP -> group (every leaf solid below it goes into that group's mesh and mass)
GROUPS = {
    "Generic_Camera_Mount": "mount",
    "Gripper_cover_flat": "plate",
    "gripper - top_cover_modified.step": "cover",
    "B0371.STEP": "imx519_board",
    "UC-609(B0179&B0193).STEP": "imx219_board",
    "ABX00101-step.step": "imu_board",
    "raspberry_pi_5.step": "pi5",
    "ABX00110-step.step": "button_board",
    "ABX00109-step.step": "led_strip",
}
COLLISION_GROUPS = ("mount", "plate", "cover", "pi5")

PLA_DENSITY_G_CM3 = 1.24
# Printed parts: mass = solid volume x PLA density x fill factor (share of the solid that is plastic). Thin parts are
# nearly all perimeters and top/bottom skin; the 13 mm block is mostly sparse infill.
PRINTED = {
    "mount": {"fill": 0.60, "why": "13 mm block + lip: 3-4 perimeters + ~25 % infill"},
    "plate": {"fill": 0.95, "why": "3 mm plate: skin + perimeters"},
    "cover": {"fill": 0.85, "why": "2-3 mm walls with bosses: mostly perimeters"},
}
PRINTED_REL_SIGMA = 0.25
# Bought boards: typical masses (spread over each board's CAD volume for the COM/inertia)
BOARDS_G = {
    "pi5": (46.0, "Raspberry Pi 5 product brief (bare board, no cooler)"),
    "imx219_board": (8.0, "Arducam UC-609 IMX219 board ~2.5 g + M12 holder ~1.5 g + M12 fisheye lens ~4 g (typical)"),
    "imx519_board": (4.0, "Arducam B0371 IMX519 AF module on its board, typical 3-5 g"),
    "imu_board": (2.0, "Arduino Modulino Movement ABX00101, typical"),
    "button_board": (3.0, "Arduino Modulino Buttons ABX00110 with caps, typical"),
    "led_strip": (2.0, "Arduino Modulino Pixels ABX00109 (8 LEDs, APA102-compatible), typical"),
}
BOARD_REL_SIGMA = 0.3
# Not in the STEP: screws, standoffs, nuts, the two CSI ribbons, Qwiic cables, the Pi's power cable stub.
UNMODELLED = {"mass_g": 20.0, "sigma_g": 10.0,
              "where": "spread as a 2 mm shell over the plate (screws/standoffs between plate, Pi and cover)"}

# Lens optical centres: distance behind each lens's front surface (from the CAD) along the optical axis
OPTICAL_CENTRE = {
    "cam_imx219": {"behind_front_mm": 2.5, "sigma_mm": 2.5,
                   "why": "M12 fisheye (fx 449 px @1280 = f ~1.3 mm): its entrance pupil sits just behind the front "
                          "element group; the CAD barrel (8.STEP) has a flat r 5.5 window 0.2 mm behind its r 7 front "
                          "cap and the board 13.8 mm behind it, so 0-5 mm behind the front covers the plausible range"},
    "cam_imx519": {"behind_front_mm": 0.8, "sigma_mm": 0.8,
                   "why": "phone-style AF lens (f ~4.3 mm, aperture stop at the front): entrance pupil a few tenths of "
                          "a mm to ~1.5 mm behind the front glass (UC-873 r 2.75 barrel + dome); board 6 mm behind"},
}
CAM_ROT_SIGMA_DEG = 1.0                # print + board seating + lens-in-holder tilt; image roll from sensor placement
CAM_POS_LATERAL_SIGMA_MM = 0.5         # cover print / assembly; the CAD itself is consistent to 0.01 mm

# Training (handheld) camera -> fingertip point, rig_pi_v3.yaml (OpenCV ultrawide frame), for comparison
TRAINING_T_UW_TCP = np.array([[1.0, 0.0, 0.0, 0.00813],
                              [0.0, 0.965925826, -0.258819045, 0.054845479],
                              [0.0, 0.258819045, 0.965925826, 0.204889292],
                              [0.0, 0.0, 0.0, 1.0]])
TRAINING_T_IMU_MAIN = np.array([[0.0, 0.965925826, 0.258819045, -0.040924868],
                                [-1.0, 0.0, 0.0, -0.029440000],
                                [0.0, -0.258819045, 0.965925826, 0.009361928],
                                [0.0, 0.0, 0.0, 1.0]])
TRAINING_T_IMU_UW = np.array([[0.0, 1.0, 0.0, -0.014637073],
                              [-1.0, 0.0, 0.0, -0.029490000],
                              [0.0, 0.0, 1.0, 0.006500336],
                              [0.0, 0.0, 0.0, 1.0]])
OLD_KIT_TIP_Z_M = 0.2239               # fingertip point of the franka_sim kit (11-17 spec page)
CHECK_11_17 = {"cover_out_from_hand_face_mm": 38.1, "front_edge_to_tip_mm": 156.7, "tilt_deg": 0.0,
               "sideways_mm": 0.0, "hand_face_x_mm": 17.9}

RGBA = {
    "mount": [0.85, 0.48, 0.16, 1.0], "plate": [0.55, 0.55, 0.58, 1.0], "cover": [0.20, 0.36, 0.72, 1.0],
    "imx519_board": [0.08, 0.30, 0.16, 1.0], "imx219_board": [0.10, 0.10, 0.11, 1.0],
    "imu_board": [0.05, 0.45, 0.50, 1.0], "pi5": [0.10, 0.50, 0.22, 1.0], "button_board": [0.05, 0.45, 0.50, 1.0],
    "led_strip": [0.05, 0.45, 0.50, 1.0],
}


# ---------------------------------------------------------------------------------------------- STEP reading
def read_step(path: Path):
    doc = TDocStd_Document(TCollection_ExtendedString("doc"))
    r = STEPCAFControl_Reader()
    r.SetNameMode(True)
    if r.ReadFile(str(path)) != 1:
        raise RuntimeError(f"cannot read {path}")
    r.Transfer(doc)
    return XCAFDoc_DocumentTool.ShapeTool_s(doc.Main()), doc


def label_name(lab) -> str:
    a = TDataStd_Name()
    return a.Get().ToExtString() if lab.FindAttribute(TDataStd_Name.GetID_s(), a) else "?"


def leaves(st):
    """[(product name, group, shape placed in STEP world, path of product names)] for every leaf part. Groups come
    from GROUPS by the nearest ancestor."""
    out = []

    def walk(lab, loc, group, path):
        ref = TDF_Label()
        pl = ref if (st.IsReference_s(lab) and st.GetReferredShape_s(lab, ref)) else lab
        n = label_name(pl)
        group = GROUPS.get(n, group)
        path = path + [n]
        kids = TDF_LabelSequence()
        st.GetComponents_s(pl, kids)
        if kids.Length() == 0:
            out.append((n, group, st.GetShape_s(pl).Moved(loc), tuple(path)))
        for i in range(1, kids.Length() + 1):
            c = kids.Value(i)
            walk(c, loc.Multiplied(st.GetLocation_s(c)), group, path)

    roots = TDF_LabelSequence()
    st.GetFreeShapes(roots)
    for i in range(1, roots.Length() + 1):
        walk(roots.Value(i), TopLoc_Location(), None, [])
    return out


def to_hand_mm(p) -> np.ndarray:
    return np.asarray(p, float) @ R_HS.T + T_HS


def mesh_shape(shape, lin_mm: float, ang_rad: float, tmp: Path) -> trimesh.Trimesh:
    BRepMesh_IncrementalMesh(shape, lin_mm, False, ang_rad, True)
    f = tmp / "part.stl"
    w = StlAPI_Writer()
    w.ASCIIMode = False
    if not w.Write(shape, str(f)):
        raise RuntimeError("StlAPI_Writer failed")
    m = trimesh.load(str(f), force="mesh", process=True)
    m.vertices = to_hand_mm(m.vertices)
    return m


def solid_props(shape) -> list[tuple[float, np.ndarray, np.ndarray]]:
    """[(volume mm^3, COM hand mm, inertia about COM per unit density, hand axes, mm^5)] per solid."""
    out = []
    ex = TopExp_Explorer(shape, TopAbs_SOLID)
    while ex.More():
        p = GProp_GProps()
        BRepGProp.VolumeProperties_s(ex.Current(), p)
        ex.Next()
        v = p.Mass()
        c = p.CentreOfMass()
        M = p.MatrixOfInertia()
        I = np.array([[M.Value(i, j) for j in (1, 2, 3)] for i in (1, 2, 3)])
        if v < 0:                      # reversed solid
            v, I = -v, -I
        out.append((v, to_hand_mm([c.X(), c.Y(), c.Z()]), R_HS @ I @ R_HS.T))
    return out


# ---------------------------------------------------------------------------------------------- geometry
def _decimate(m: trimesh.Trimesh, target: int) -> trimesh.Trimesh:
    import fast_simplification

    v, f = np.asarray(m.vertices, np.float64), np.asarray(m.faces, np.int64)
    for _ in range(4):                 # fast_simplification can land above the target: tighten and repeat
        if len(f) <= target:
            break
        v, f = fast_simplification.simplify(v, f, target_reduction=1.0 - 0.97 * target / len(f))
    out = trimesh.Trimesh(v, f, process=True)
    if len(out.faces) > 1.5 * target:  # decimation stuck (e.g. the Pi's GPIO header): its convex hull instead
        out = m.convex_hull
    return out


def simplify(ms: list, max_faces: int) -> trimesh.Trimesh:
    """Concatenate one group's leaf meshes into <= max_faces faces: small leaves (< 200 faces) stay as they are, the
    others are decimated one by one with a budget proportional to their size (keeps each closed solid closed)."""
    total = sum(len(m.faces) for m in ms)
    if total <= max_faces:
        out = trimesh.util.concatenate(ms)
    else:
        small = sum(len(m.faces) for m in ms if len(m.faces) < 200)
        big = total - small
        k = max(max_faces - small, 0.2 * max_faces) / big
        out = trimesh.util.concatenate([m if len(m.faces) < 200 else _decimate(m, max(int(k * len(m.faces)), 60))
                                        for m in ms])
        if len(out.faces) > max_faces:
            out = _decimate(out, max_faces)
    out.merge_vertices()
    return out


def axis_cyl(shapes, radius: float, tol: float = 0.02):
    """(point hand mm, unit direction hand) of the first cylindrical face of the given radius in ``shapes``."""
    from OCP.BRepAdaptor import BRepAdaptor_Surface
    from OCP.GeomAbs import GeomAbs_Cylinder
    from OCP.TopAbs import TopAbs_FACE
    from OCP.TopoDS import TopoDS

    for shape in shapes:
        ex = TopExp_Explorer(shape, TopAbs_FACE)
        while ex.More():
            s = BRepAdaptor_Surface(TopoDS.Face(ex.Current()))
            ex.Next()
            if s.GetType() == GeomAbs_Cylinder and abs(s.Cylinder().Radius() - radius) < tol:
                ax = s.Cylinder().Axis()
                o, d = ax.Location(), ax.Direction()
                return to_hand_mm([o.X(), o.Y(), o.Z()]), R_HS @ np.array([d.X(), d.Y(), d.Z()])
    raise ValueError(f"no cylinder of radius {radius} mm")


def lens_axis(lens_mesh: trimesh.Trimesh, board_mesh: trimesh.Trimesh, p0, d, r_max: float):
    """Optical axis pointing away from the board; front = furthest vertex along it within r_max of the axis."""
    d = d / np.linalg.norm(d)
    if (board_mesh.vertices.mean(axis=0) - p0) @ d > 0:
        d = -d
    rel = lens_mesh.vertices - p0
    s = rel @ d
    r = np.linalg.norm(rel - np.outer(s, d), axis=1)
    front = float(s[r <= r_max + 0.05].max())
    return d, p0 + front * d, front - board_front(board_mesh, p0, d)


def board_front(m: trimesh.Trimesh, p0, d) -> float:
    """Axial position of the PCB's front surface: of the two largest-area planes normal to d, the one further along d."""
    par = np.abs(m.face_normals @ d) > 0.99
    s = np.round(((m.triangles_center[par] - p0) @ d) / 0.05) * 0.05
    area: dict = {}
    for si, ai in zip(s, m.area_faces[par]):
        area[si] = area.get(si, 0.0) + ai
    top2 = sorted(area, key=area.get)[-2:]
    return float(max(top2))


def cv_rotation(d_opt: np.ndarray) -> np.ndarray:
    """hand_R_cam (OpenCV): z = optical axis, y = image down = toward the hand axis (-x_h) projected, x = y x z."""
    z = d_opt / np.linalg.norm(d_opt)
    down = np.array([-1.0, 0.0, 0.0])
    y = down - (down @ z) * z
    y /= np.linalg.norm(y)
    x = np.cross(y, z)
    return np.column_stack([x, y, z])


def make_T(R, t) -> np.ndarray:
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = t
    return T


def inv_T(T) -> np.ndarray:
    return make_T(T[:3, :3].T, -T[:3, :3].T @ T[:3, 3])


def rot_angle_deg(Ra, Rb) -> float:
    return float(np.degrees(np.arccos(np.clip((np.trace(Ra.T @ Rb) - 1) / 2, -1, 1))))


def combine(items):
    """items: [(mass kg, com m (3,), I_com kg m^2 (3,3))] -> (mass, com, I about the combined com)."""
    M = sum(m for m, _, _ in items)
    c = sum(m * np.asarray(p) for m, p, _ in items) / M
    I = np.zeros((3, 3))
    for m, p, Ii in items:
        r = np.asarray(p) - c
        I += Ii + m * ((r @ r) * np.eye(3) - np.outer(r, r))
    return M, c, I


def r(a, n=6):
    return np.round(np.asarray(a, float), n).tolist()


# ---------------------------------------------------------------------------------------------- main
def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--step", type=Path, default=DEFAULT_STEP)
    ap.add_argument("--out", type=Path, default=HERE)
    ap.add_argument("--max-faces", type=int, default=20000)
    ap.add_argument("--lin-mm", type=float, default=0.05, help="BRepMesh linear deflection (mm)")
    ap.add_argument("--ang-rad", type=float, default=0.25, help="BRepMesh angular deflection (rad)")
    a = ap.parse_args(argv)

    sha = hashlib.sha256(a.step.read_bytes()).hexdigest()
    print(f"[extract] {a.step} sha256 {sha[:16]}..., reading (~15 s)", flush=True)
    st, _doc = read_step(a.step)
    parts = leaves(st)
    missing = sorted({n for n, g, _, _ in parts if g is None})
    if missing:
        raise SystemExit(f"leaf parts outside every GROUPS entry: {missing}")
    print(f"[extract] {len(parts)} leaf parts in {len(set(g for _, g, _, _ in parts))} groups", flush=True)

    out = a.out
    (out / "assets" / "collision").mkdir(parents=True, exist_ok=True)
    meshes: dict[str, list] = {g: [] for g in GROUPS.values()}
    leaf_meshes: list = []
    props: dict[str, list] = {g: [] for g in GROUPS.values()}
    with tempfile.TemporaryDirectory() as td:
        for n, g, shp, _path in parts:
            m = mesh_shape(shp, a.lin_mm, a.ang_rad, Path(td))
            meshes[g].append(m)
            leaf_meshes.append(m)
            props[g].extend(solid_props(shp))

    def node_mesh(name: str) -> trimesh.Trimesh:
        """All leaf meshes below the product ``name`` (the names used here occur once in the assembly)."""
        ms = [m for m, (_, _, _, path) in zip(leaf_meshes, parts) if name in path]
        if not ms:
            raise KeyError(f"no part named {name!r} in the STEP")
        return trimesh.util.concatenate(ms)

    def node_shapes(name: str) -> list:
        return [shp for _, _, shp, path in parts if name in path]

    # ---- visual meshes + collision hulls (hand frame, metres)
    files, mesh_info = {}, {}
    for g, ms in meshes.items():
        full = trimesh.util.concatenate(ms)
        simp = simplify(ms, a.max_faces)
        simp.apply_scale(1e-3)
        fn = f"assets/{g}.stl"
        simp.export(out / fn)
        files[g] = fn
        mesh_info[g] = {"file": fn, "faces": int(len(simp.faces)), "faces_full": int(len(full.faces)),
                        "watertight": bool(simp.is_watertight),
                        "bounds_m": r(simp.bounds, 5)}
        if g in COLLISION_GROUPS:
            hull = trimesh.util.concatenate(ms).convex_hull
            hull.apply_scale(1e-3)
            cfn = f"assets/collision/{g}.stl"
            hull.export(out / cfn)
            mesh_info[g]["collision_file"] = cfn
            mesh_info[g]["collision_faces"] = int(len(hull.faces))
        print(f"[extract] {g:13s} {len(full.faces):7d} -> {len(simp.faces):6d} faces, "
              f"watertight {simp.is_watertight}", flush=True)

    # ---- mass properties
    mass_items, mass_rows = [], {}
    for g, pr in props.items():
        vol = sum(v for v, _, _ in pr)                                   # mm^3
        if g in PRINTED:
            rho = PLA_DENSITY_G_CM3 * 1e-3 * PRINTED[g]["fill"]          # g / mm^3
            src = f"CAD volume x PLA {PLA_DENSITY_G_CM3} g/cm3 x fill {PRINTED[g]['fill']} ({PRINTED[g]['why']})"
            sig = PRINTED_REL_SIGMA
        else:
            rho = BOARDS_G[g][0] / vol
            src = f"{BOARDS_G[g][1]}; spread over the CAD volume"
            sig = BOARD_REL_SIGMA
        items = [(v * rho * 1e-3, c * 1e-3, I * rho * 1e-3 * 1e-6) for v, c, I in pr]   # kg, m, kg m^2
        m_g, c_g, I_g = combine(items)
        mass_items.append((m_g, c_g, I_g))
        mass_rows[g] = {"mass_kg": round(m_g, 5), "sigma_kg": round(m_g * sig, 5), "volume_cm3": round(vol * 1e-3, 3),
                        "com_hand_m": r(c_g), "source": src}
    # unmodelled hardware: a thin shell over the plate's outer box
    plate = node_mesh("Gripper_cover_flat")
    lo, hi = plate.bounds * 1e-3
    mu = UNMODELLED["mass_g"] * 1e-3
    c_u = (lo + hi) / 2
    ext = hi - lo
    I_u = mu / 12 * np.diag([ext[1] ** 2 + ext[2] ** 2, ext[0] ** 2 + ext[2] ** 2, ext[0] ** 2 + ext[1] ** 2])
    mass_items.append((mu, c_u, I_u))
    mass_rows["unmodelled_hardware"] = {"mass_kg": mu, "sigma_kg": UNMODELLED["sigma_g"] * 1e-3, "com_hand_m": r(c_u),
                                        "source": "estimate: screws, standoffs, nuts, 2 CSI ribbons, Qwiic cables; "
                                                  + UNMODELLED["where"]}
    M, com_h, I_h = combine(mass_items)
    sigma_M = float(np.sqrt(sum(row["sigma_kg"] ** 2 for row in mass_rows.values())))

    # ---- cameras
    # IMX219: lens barrel 8.STEP (r 7 front cap), board UC-609.STEP
    p0, d0 = axis_cyl(node_shapes("8.STEP"), 7.0)
    d219, front219, front219_to_board = lens_axis(node_mesh("8.STEP"), node_mesh("UC-609.STEP"), p0, d0, 7.0)
    p0, d0 = axis_cyl(node_shapes("UC-873.STEP"), 2.75)
    lens519 = node_mesh("UC-873.STEP")
    d519, front519, front519_to_board = lens_axis(lens519, lens519, p0, d0, 2.75)
    cams = {}
    for name, d, front, f2b in (("cam_imx219", d219, front219, front219_to_board),
                                ("cam_imx519", d519, front519, front519_to_board)):
        oc = OPTICAL_CENTRE[name]
        pos = front - oc["behind_front_mm"] * d
        R = cv_rotation(d)
        tilt = float(np.degrees(np.arctan2(-d[0], d[2])))               # toward the hand (-x_h) from +z_h
        cams[name] = {"pos_mm": pos, "R": R, "front_mm": front, "tilt_deg": tilt, "front_to_board_mm": f2b}

    # IMU: chip U1_1 on ABX00101; axes per the user (facing the fingers: x down, y right, z toward the viewer)
    chip = node_mesh("U1_1.step")
    imu_pos = (chip.bounds[0] + chip.bounds[1]) / 2
    R219 = cams["cam_imx219"]["R"]
    hand_R_imu = np.column_stack([R219[:, 1], -R219[:, 0], R219[:, 2]])    # x = image down, y = -image right, z = optical
    # chip top/bottom normal (the area-weighted dominant facet direction) vs the IMX219 axis: same 15 deg face?
    fn = chip.face_normals * np.sign(chip.face_normals @ R219[:, 2])[:, None]
    w = chip.area_faces * (np.abs(chip.face_normals @ R219[:, 2]) > 0.9)
    chip_n = (fn * w[:, None]).sum(axis=0)
    imu_vs_219_deg = float(np.degrees(np.arccos(np.clip(chip_n @ R219[:, 2] / np.linalg.norm(chip_n), -1, 1))))

    # ---- derived numbers
    cover = node_mesh("gripper - top_cover_modified.step")
    gcm = node_mesh("Generic_Camera_Mount")
    cover_base_x = float(cover.bounds[0, 0])
    cover_front_z = float(cover.bounds[1, 2])
    cover_y_centre = float((cover.bounds[0, 1] + cover.bounds[1, 1]) / 2)
    # tilt of the cover base: normal of the plate's big faces
    big = plate.face_normals[np.abs(plate.face_normals[:, 0]) > 0.5]
    cover_tilt = float(np.degrees(np.arccos(np.clip(np.abs(big[:, 0]).min(), -1, 1))))

    def cam_tip(name, z_tip_m):
        c = cams[name]
        hand_T_cam = make_T(c["R"], c["pos_mm"] * 1e-3)
        hand_T_tip = make_T(HAND_R_TCP, [0.0, 0.0, z_tip_m])
        return inv_T(hand_T_cam) @ hand_T_tip

    T_main_tcp_train = inv_T(TRAINING_T_IMU_MAIN) @ TRAINING_T_IMU_UW @ TRAINING_T_UW_TCP
    training = {"cam_imx219": TRAINING_T_UW_TCP, "cam_imx519": T_main_tcp_train}

    # ---- rig.yaml
    flange_T_hand = make_T(FLANGE_R_HAND, [0, 0, 0])
    com_F = FLANGE_R_HAND @ com_h
    I_F = FLANGE_R_HAND @ I_h @ FLANGE_R_HAND.T
    cam_yaml, tips = {}, {}
    for name, c in cams.items():
        oc = OPTICAL_CENTRE[name]
        R = c["R"]
        R_mj = R @ np.diag([1.0, -1.0, -1.0])
        T = cam_tip(name, OLD_KIT_TIP_Z_M)
        Tt = training[name]
        const = -R.T @ (c["pos_mm"] * 1e-3)
        slope = R.T @ np.array([0, 0, 1.0])
        cam_yaml[name] = {
            "sensor": {"cam_imx219": "imx219 (UC-609 board, M12 fisheye 8.STEP)",
                       "cam_imx519": "imx519 (B0371 / UC-873)"}[name],
            "pos_hand_m": r(c["pos_mm"] * 1e-3),
            "hand_R_cam_cv": r(R, 9),
            "hand_R_cam_mujoco": r(R_mj, 9),
            "pos_flange_m": r(FLANGE_R_HAND @ c["pos_mm"] * 1e-3),
            "flange_R_cam_cv": r(FLANGE_R_HAND @ R, 9),
            "optical_axis_hand": r(R[:, 2], 9),
            "tilt_toward_hand_deg": round(c["tilt_deg"], 4),
            "lens_front_hand_m": r(c["front_mm"] * 1e-3),
            "lens_front_to_board_mm": round(c["front_to_board_mm"], 3),
            "optical_centre_behind_front_mm": oc["behind_front_mm"],
            "sigma": {"along_axis_mm": oc["sigma_mm"], "lateral_mm": CAM_POS_LATERAL_SIGMA_MM,
                      "rotation_deg": CAM_ROT_SIGMA_DEG},
            "provenance": f"STEP lens geometry (axis = cylinder axis, front = lens front); optical centre: {oc['why']}; "
                          "image axes chosen as training (fingers at the bottom, fingertip to the right of the "
                          "IMX219); the sensor's real readout orientation is not in the CAD",
        }
        e_ang = rot_angle_deg(T[:3, :3], Tt[:3, :3])
        dt = T[:3, 3] - Tt[:3, 3]
        tips[name] = {
            "cam_T_tip_at_old_kit": r(T, 6),
            "t_cam_tip_m": {"const": r(const), "per_m_of_z_tip": r(slope, 9),
                            "formula": "t = const + z_tip * per_m_of_z_tip; R = hand_R_cam_cv^T @ hand_R_tcp"},
            "cam_R_tip": r(R.T @ HAND_R_TCP, 9),
            "training_T_cam_tcp": r(Tt, 6),
            "vs_training_at_old_kit": {"rotation_deg": round(e_ang, 3), "dt_cam_m": r(dt, 5),
                                       "distance_m": round(float(np.linalg.norm(dt)), 5),
                                       "range_m": round(float(np.linalg.norm(T[:3, 3])), 5),
                                       "training_range_m": round(float(np.linalg.norm(Tt[:3, 3])), 5)},
        }
    imu_T = make_T(hand_R_imu, imu_pos * 1e-3)

    # station tooling values (what franka_sim's iPhone-era station keys need, now from the CAD)
    F_T_C = {n: flange_T_hand @ make_T(c["R"], c["pos_mm"] * 1e-3) for n, c in cams.items()}
    boxes = []
    for g in COLLISION_GROUPS:
        lo, hi = np.array(mesh_info[g]["bounds_m"], float)
        Tb = make_T(np.eye(3), (lo + hi) / 2)
        boxes.append({"name": g, "size_m": r(hi - lo, 5), "hand_T_box": r(Tb, 6), "flange_T_box": r(flange_T_hand @ Tb, 6)})
    station_tooling = {
        "F_T_C_imx519": r(F_T_C["cam_imx519"], 7),
        "F_T_C_imx219": r(F_T_C["cam_imx219"], 7),
        "C_imx519_T_C_imx219": r(inv_T(F_T_C["cam_imx519"]) @ F_T_C["cam_imx219"], 7),
        "C_imx519_T_tip_at_old_kit": r(cam_tip("cam_imx519", OLD_KIT_TIP_Z_M), 7),
        "C_imx219_T_tip_at_old_kit": r(cam_tip("cam_imx219", OLD_KIT_TIP_Z_M), 7),
        "collision_boxes": boxes,
        "payload_rig": {"mass_kg": round(M, 5), "com_flange_m": r(com_F),
                        "inertia_com_flange_kgm2": r(I_F, 9),
                        "note": "rig only; the Franka Hand (0.73 kg) is the robot's configured end effector; the "
                                "fingers + adapters (E3) add to this payload"},
        "note": "C_main = IMX519 and C_uw = IMX219 in franka_sim's old names (cam_main / cam_uw); boxes are the "
                "parts' hand-frame bounding boxes (all parts are square to the hand)",
    }
    doc = {
        "schema": "sim_deploy.rig/v1",
        "generated": {"by": "sim_deploy/rig/extract.py", "date": datetime.date.today().isoformat(),
                      "args": {"max_faces": a.max_faces, "lin_mm": a.lin_mm, "ang_rad": a.ang_rad}},
        "step": {"file": a.step.name, "sha256": sha, "units": "mm"},
        "frames": {
            "hand": "menagerie Franka Hand body frame: z along the hand to the fingertips (0 at the hand-arm joint = "
                    "flange), x out of the camera-side face, fingers close along y",
            "step_to_hand_mm": {"R": R_HS.tolist(), "t": T_HS.tolist(),
                                "formula": "x_h = z_S - 1214.765, y_h = x_S - 699.8003, z_h = y_S - 925.6717"},
            "hand_T_step_m": r(make_T(R_HS, T_HS * 1e-3), 7),
            "mount_frame": "Generic_Camera_Mount local = hand turned 180 deg about z, origin at hand "
                           f"(0, 0, {GCM_LOCAL_Z0_IN_HAND_MM} mm)",
            "flange_T_hand": r(flange_T_hand, 9),
            "flange_T_hand_note": "Rz(-45 deg), no translation (menagerie panda.xml, libfranka F_T_NE)",
            "hand_R_tcp": r(HAND_R_TCP, 9),
            "hand_R_tcp_note": "training TCP axes (x finger closing = image right, y away from the camera side, z to "
                               "the tips) = hand turned +90 deg about z; flange_R_hand @ hand_R_tcp = Rz(+45) = "
                               "contracts/frames.yaml F_T_TCP rotation, so its camera side = hand +x = this rig",
            "camera_axes": "OpenCV (x right, y down, z optical); MuJoCo cameras: R_mj = R_cv @ diag(1, -1, -1)",
        },
        "cameras": cam_yaml,
        "imu": {
            "chip": "LSM6DSOX (U1_1 on ABX00101)",
            "pos_hand_m": r(imu_pos * 1e-3),
            "hand_R_imu": r(hand_R_imu, 9),
            "hand_T_imu": r(imu_T, 7),
            "flange_T_imu": r(flange_T_hand @ imu_T, 7),
            "imu_T_cam_imx219": r(inv_T(imu_T) @ make_T(cams["cam_imx219"]["R"], cams["cam_imx219"]["pos_mm"] * 1e-3), 7),
            "imu_T_cam_imx519": r(inv_T(imu_T) @ make_T(cams["cam_imx519"]["R"], cams["cam_imx519"]["pos_mm"] * 1e-3), 7),
            "axes": "the user (physical board): facing the fingers, x down (= IMX219 image down), y to the viewer's "
                    "right (= IMX219 image left), z toward the viewer (= IMX219 optical axis); not the CAD pin-1 "
                    "reading",
            "chip_normal_vs_imx219_axis_deg": round(imu_vs_219_deg, 4),
            "sigma": {"position_mm": 1.0, "rotation_deg": 1.0},
            "provenance": "position = U1_1 bounding-box centre (the die is ~0.3 mm inside the package); axes as `axes`",
        },
        "fingertip": {"old_kit_z_tip_m": OLD_KIT_TIP_Z_M,
                      "point": "(0, 0, z_tip) in the hand frame, axes hand_R_tcp", **tips},
        "mass": {
            "parts": mass_rows,
            "total_kg": round(M, 5), "sigma_kg": round(sigma_M, 5),
            "com_hand_m": r(com_h), "inertia_com_hand_kgm2": r(I_h, 9),
            "com_flange_m": r(com_F), "inertia_com_flange_kgm2": r(I_F, 9),
            "note": "rig only (excludes the Franka Hand 0.73 kg and the fingers/adapters)",
        },
        "meshes": mesh_info,
        "visual_rgba": RGBA,
        "collision_parts": list(COLLISION_GROUPS),
        "station_tooling": station_tooling,
        "check_11_17": {
            "cover_base_x_hand_mm": round(cover_base_x, 3),
            "hand_face_x_mm_used": CHECK_11_17["hand_face_x_mm"],
            "cover_out_from_hand_face_mm": round(cover_base_x - CHECK_11_17["hand_face_x_mm"], 3),
            "expected_out_mm": CHECK_11_17["cover_out_from_hand_face_mm"],
            "mount_contact_face_x_mm": round(float(gcm.bounds[0, 0]), 3),
            "cover_front_edge_z_hand_mm": round(cover_front_z, 3),
            "front_edge_to_old_kit_tip_mm": round(OLD_KIT_TIP_Z_M * 1e3 - cover_front_z, 3),
            "expected_front_edge_to_tip_mm": CHECK_11_17["front_edge_to_tip_mm"],
            "cover_tilt_deg": round(cover_tilt, 4),
            "cover_sideways_offset_mm": round(cover_y_centre, 4),
            "note": "hand face x 17.9 mm = the 11-17 spec page's reference (EVK4 body outer face x 40.0 mm = 22.1 mm "
                    "out from it); the mount's own contact faces with the hand are at x 15.0 / 20.2 mm",
        },
    }
    with open(out / "rig.yaml", "w") as f:
        f.write("# sim_deploy rig from CAD: generated by sim_deploy/rig/extract.py, do not edit by hand (README.md).\n")
        f.write("# Units: metres, kg, kg m^2 unless a key says mm. Frames: see `frames`.\n")
        yaml.safe_dump(doc, f, sort_keys=False, default_flow_style=None, width=120)
    print(f"[extract] wrote {out / 'rig.yaml'}: rig mass {M * 1e3:.1f} g (+-{sigma_M * 1e3:.0f}), "
          f"COM hand {np.round(com_h * 1e3, 1).tolist()} mm", flush=True)
    for name, c in cams.items():
        print(f"[extract] {name}: pos hand {np.round(c['pos_mm'], 2).tolist()} mm, axis "
              f"{np.round(c['R'][:, 2], 4).tolist()}, tilt {c['tilt_deg']:.3f} deg", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
