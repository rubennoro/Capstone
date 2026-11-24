"""Build the one MuJoCo model file of sim_deploy (DESIGN.md section 2): Panda + Franka Hand + E3's finger kit + E2's
rig + table + eraser + an external camera, written as ``scene.xml`` + ``meshes/`` + ``model.json``.

    python -m sim_deploy.model build --station sim_deploy/stations/sim_rig.yaml [--out DIR] [--force]

Steps: franka_sim's MjSpec builder (controlSW/franka_sim, Panda + hand + kit, scene off) -> franka_sim's iPhone housing, cradle and cameras removed -> ``add_rig`` -> sites ``flange`` and
``tcp`` (= flange x frames.yaml F_T_TCP) -> table, pedestal, floor, eraser, ``cam_external``, lights -> geom groups
(0 scene, 1 arm + hand, 2 rig, 3 fingers + adapter + tags, 5 collision-only) -> key ``home`` -> meshes copied (kit
meshes written as STL) with relative paths -> scene.xml, with the attached hand's ``hand/`` name prefix dropped
(finger_joint1/2, bodies hand, left_finger, right_finger; assets that would clash become ``hand_...``) -> reloaded from disk and checked -> model.json, written last
(the sim PC and sim Pi poll for it).
"""

from __future__ import annotations

import datetime as _dt
import hashlib
import json
import os
import shutil
import struct
import subprocess
from pathlib import Path

import numpy as np
import yaml

from sim_deploy import REPO, ROOT

SCHEMA = "sim_deploy.model/v1"
HAND_PREFIX = "hand/"
CV_TO_MJ = np.diag([1.0, -1.0, -1.0])
GROUP_SCENE, GROUP_ARM, GROUP_RIG, GROUP_FINGERS, GROUP_COLLISION = 0, 1, 2, 3, 5
FRANKA_ASSETS = REPO / "controlSW" / "franka_sim" / "franka_sim" / "assets" / "franka_emika_panda" / "assets"
SHA_METHOD = ("sha256 of the text 'sha256  relpath\\n' per file (scene.xml and meshes/*), sorted by relpath (C locale) "
              "= `cd <dir> && LC_ALL=C sha256sum scene.xml meshes/* | LC_ALL=C sort -k2 | sha256sum`")


# ------------------------------------------------------------------------------------------------------ helpers


def _quat(R) -> list:
    import mujoco

    q = np.zeros(4)
    mujoco.mju_mat2Quat(q, np.ascontiguousarray(R, float).reshape(9))
    return q.tolist()


def _T(R=None, t=None) -> np.ndarray:
    T = np.eye(4)
    if R is not None:
        T[:3, :3] = R
    if t is not None:
        T[:3, 3] = t
    return T


def Rz(deg: float) -> np.ndarray:
    a = np.deg2rad(deg)
    return np.array([[np.cos(a), -np.sin(a), 0], [np.sin(a), np.cos(a), 0], [0, 0, 1.0]])


def load_station_file(path) -> dict:
    path = Path(path).resolve()
    st = yaml.safe_load(path.read_text())
    if st.get("schema") != "sim_deploy.station/v1":
        raise ValueError(f"{path}: schema {st.get('schema')!r}, expected 'sim_deploy.station/v1'")
    st["_path"] = path
    st["_dir"] = path.parent
    return st


def _rel(st: dict, key: str) -> Path:
    return (st["_dir"] / st[key]).resolve()


def read_frames_F_T_TCP(contracts_dir: Path) -> np.ndarray:
    doc = yaml.safe_load((contracts_dir / "frames.yaml").read_text())
    T = np.array(doc["F_T_TCP"], float)
    if T.shape != (4, 4) or not np.allclose(T[:3, :3] @ T[:3, :3].T, np.eye(3), atol=1e-6):
        raise ValueError(f"{contracts_dir / 'frames.yaml'}: F_T_TCP is not a rigid 4x4")
    return T


def read_q_home(control_config: Path) -> list:
    doc = yaml.safe_load(control_config.read_text())
    q = [float(x) for x in doc["homing"]["q_home_rad"]]
    if len(q) != 7:
        raise ValueError(f"{control_config}: homing.q_home_rad must have 7 values")
    return q


def read_intrinsics(rig_file: Path, spec: dict) -> dict:
    """{cam name: (K 3x3, width, height)} from the rig file entries named in the station's ``cameras``."""
    rig = yaml.safe_load(rig_file.read_text())
    out = {}
    for cam, c in spec.items():
        node = rig
        for k in str(c["from"]).split("."):
            node = node[k]
        fx, fy, cx, cy = (float(v) for v in (node["model"]["intrinsics"] if "model" in node else node["intrinsics"]))
        w, h = (int(v) for v in node["resolution"])
        out[cam] = (np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1.0]]), w, h)
    return out


def git_commit(path: Path = REPO) -> str:
    try:
        c = subprocess.run(["git", "-C", str(path), "rev-parse", "HEAD"], capture_output=True, text=True, check=True)
        dirty = subprocess.run(["git", "-C", str(path), "status", "--porcelain"], capture_output=True, text=True)
        return c.stdout.strip() + ("-dirty" if dirty.stdout.strip() else "")
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def model_sha256(out_dir: Path) -> tuple[str, dict]:
    """(model sha, {relpath: file sha}) over scene.xml + meshes/* (SHA_METHOD)."""
    files = ["scene.xml"] + sorted(f"meshes/{p.name}" for p in (out_dir / "meshes").iterdir() if p.is_file())
    per = {f: hashlib.sha256((out_dir / f).read_bytes()).hexdigest() for f in files}
    text = "".join(f"{per[f]}  {f}\n" for f in sorted(per, key=lambda s: s.encode()))
    return hashlib.sha256(text.encode()).hexdigest(), per


def _write_stl(path: Path, verts: np.ndarray, faces: np.ndarray) -> None:
    v = np.asarray(verts, np.float32).reshape(-1, 3)
    f = np.asarray(faces, np.int64).reshape(-1, 3)
    tri = v[f]
    n = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    n /= np.maximum(np.linalg.norm(n, axis=1, keepdims=True), 1e-20)
    rec = np.zeros(len(f), dtype=[("n", "<f4", 3), ("v", "<f4", (3, 3)), ("a", "<u2")])
    rec["n"], rec["v"] = n, tri
    with open(path, "wb") as fh:
        fh.write(b"sim_deploy kit mesh".ljust(80, b"\0"))
        fh.write(struct.pack("<I", len(f)))
        fh.write(rec.tobytes())


# ------------------------------------------------------------------------------------------------------- build


def build(station_path, out_dir=None, verbose: bool = True) -> dict:
    os.environ.setdefault("MUJOCO_GL", "egl")
    import mujoco

    from franka_sim.model import build_model          # puts controlSW/franka_deploy on sys.path
    from franka_deploy.station import load_station

    from sim_deploy.rig.add_rig import add_rig

    st = load_station_file(station_path)
    out_dir = (Path(out_dir) if out_dir else ROOT / "build" / st["id"]).resolve()
    log = print if verbose else (lambda *a, **k: None)

    # pollers (sim PC, sim Pi) take model.json as "the model is complete": remove it first, write it last
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "model.json").unlink(missing_ok=True)

    fs = st["franka_sim"]
    station = load_station({"station": dict(fs["station"]), "run": dict(fs.get("run") or {})}, sets=list(fs["set"]))
    sm = build_model(station, with_scene=False)
    spec = sm.spec
    kit_info = sm.info["finger_kit"]
    F_T_TCP_kit = np.array(sm.info["F_T_TCP_derived"], float)

    for k in list(spec.keys):
        k.delete()

    # -- franka_sim's phone-era tooling: housing/cradle geoms, cam_main/cam_uw, the ee site, the menagerie spotlight
    tool = spec.body("tool")
    for g in list(tool.geoms):
        g.delete()
    for c in list(tool.cameras):
        c.delete()
    for s in list(tool.sites):
        if s.name not in ("flange", "tcp"):
            s.delete()
    for lt in list(spec.worldbody.lights):
        lt.delete()

    # -- tcp site = flange x frames.yaml F_T_TCP (D7); franka_sim's kit-derived value is kept for the report
    contracts = _rel(st, "contracts_dir")
    F_T_TCP = read_frames_F_T_TCP(contracts)
    for s in tool.sites:
        if s.name == "tcp":
            s.pos, s.quat = F_T_TCP[:3, 3].tolist(), _quat(F_T_TCP[:3, :3])
        s.group = 4
    dev_mm = float(np.linalg.norm(F_T_TCP[:3, 3] - F_T_TCP_kit[:3, 3]) * 1e3)
    dR = F_T_TCP[:3, :3].T @ F_T_TCP_kit[:3, :3]
    dev_deg = float(np.degrees(np.arccos(np.clip((np.trace(dR) - 1) / 2, -1, 1))))

    # -- the menagerie stock finger geoms (kit mode: hidden and contact-free) go; the kit's own geoms stay
    for fb in ("left_finger", "right_finger"):
        for g in list(spec.body(HAND_PREFIX + fb).geoms):
            if not g.name:
                g.delete()

    # -- rig (E2): meshes, hulls, body rig with its inertial, cam_imx519 / cam_imx219 on the hand
    intr = read_intrinsics(_rel(st, "rig_file"), st["cameras"])
    rig_info = add_rig(spec, HAND_PREFIX + "hand", _rel(st, "rig_dir"), intrinsics=intr)

    # -- scene
    sc = st["scene"]
    _add_scene(spec, sc, log)
    W, H = (int(v) for v in st.get("offscreen_px", [1600, 1600]))
    spec.visual.global_.offwidth, spec.visual.global_.offheight = W, H
    spec.visual.map.znear = 0.0005                     # x extent: ~1 mm, the IMX219 sees rig parts up close
    spec.visual.quality.shadowsize = 4096

    # -- geom groups (DESIGN section 2)
    _set_groups(spec)

    # -- E7: the real look (white fin-ray fingers, blue holders, ArUco stickers, wood table) and the fin-ray's
    #    compliance on the pad-eraser contacts (DESIGN.md "E7: integration")
    tex_tmp = out_dir / ".textures"
    look = _appearance(spec, st.get("appearance") or {}, kit_info, tex_tmp, base_dir=st["_dir"])
    contact = _finger_contact(spec, st.get("finger_contact"))

    # -- W0: the printed fingers' stroke (finger_max_gap_m, the fingertip gap 0..max; limits.yaml
    #    gripper_width_max_m): each finger joint 0..max/2 (menagerie: 0..0.04 each). The geometry at a given gap is
    #    unchanged (the fingertips touch at q = 0); only the reachable range shrinks.
    max_gap = st.get("finger_max_gap_m")
    if max_gap is not None:
        for jn in ("finger_joint1", "finger_joint2"):
            j = spec.joint(HAND_PREFIX + jn)
            j.range = [0.0, float(max_gap) / 2]
            j.limited = mujoco.mjtLimited.mjLIMITED_TRUE

    # -- key "home": q_home of the copied franka_control config, hand fully open, eraser at its default pose
    q_home = read_q_home(_rel(st, "control_config"))
    gap = float(st.get("hand_home_gap_m", 0.08))
    if max_gap is not None and gap > float(max_gap) + 1e-12:
        raise ValueError(f"hand_home_gap_m {gap} is above finger_max_gap_m {max_gap}")
    er_qpos = eraser_qpos(sc, *sc["eraser"]["default_pose"]["xy_m"], sc["eraser"]["default_pose"]["yaw_deg"])
    key = spec.add_key()
    key.name = "home"
    key.qpos = q_home + [gap / 2, gap / 2] + er_qpos
    key.ctrl = [0.0] * len(spec.actuators)

    m_mem = spec.compile()

    # -- meshes -> out_dir/meshes (relative paths), scene.xml
    mesh_dir = out_dir / "meshes"
    if mesh_dir.exists():
        shutil.rmtree(mesh_dir)
    mesh_dir.mkdir()
    used = set()
    for me in spec.meshes:
        if me.file:
            src = Path(me.file)
            src = src if src.is_absolute() else FRANKA_ASSETS / src
            if not src.is_file():
                raise FileNotFoundError(f"mesh {me.name}: {src}")
            stem = (src.name if src.parent == tex_tmp else
                    ("rig_" if str(src).startswith(str(ROOT / "rig")) else "panda_") + src.name)
            if stem in used:
                stem = f"{me.name.replace('/', '_')}_{src.name}"
            shutil.copy2(src, mesh_dir / stem)
        else:                                          # kit meshes are inline vertices (+ faces) in franka_sim
            v = np.array(me.uservert, float).reshape(-1, 3)
            f = np.array(me.userface, int).reshape(-1, 3) if len(me.userface) else _hull_faces(v)
            stem = f"kit_{me.name}.stl"
            _write_stl(mesh_dir / stem, v, f)
            me.uservert, me.userface = [], []
        used.add(stem)
        me.file = str(mesh_dir / stem)                   # absolute for this compile (to_xml writes the compiled
    for tx in spec.textures:                             # E7: texture PNGs next to the meshes (in the model sha)
        if tx.file:
            shutil.copy2(tx.file, mesh_dir / Path(tx.file).name)
            tx.file = str(mesh_dir / Path(tx.file).name)
    spec.compile()                                       # state); scene.xml gets "meshes/" + the bare file names
    shutil.rmtree(tex_tmp, ignore_errors=True)
    xml = strip_hand_prefix(spec.to_xml(), meshdir="meshes/")
    header = (f"<!-- sim_deploy model {st['id']}: generated by `python -m sim_deploy.model build --station "
              f"{os.path.relpath(st['_path'], REPO)}` (do not edit; see model.json) -->\n")
    (out_dir / "scene.xml").write_text(header + xml)

    # -- reload from disk (what the sim PC's mj_loadXML and the sim Pi's MjModel.from_xml_path see) and check
    # MuJoCo's in-process asset cache can serve stale mesh data for files it just compiled under the same path, so
    # the reload runs in a fresh interpreter (exactly what another process sees) and returns describe() + checks.
    disk = inspect_in_subprocess(out_dir / "scene.xml")
    _check_same(m_mem, disk)
    sha, per_file = model_sha256(out_dir)

    info = disk["describe"]
    W_F = np.array(disk["O_T_flange"])
    W_TCP = np.array(disk["O_T_tcp"])
    er = sc["eraser"]
    doc = {
        "schema": SCHEMA,
        "station_id": st["id"],
        "station_file": os.path.relpath(st["_path"], REPO),
        "built_utc": _dt.datetime.now(_dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "git_commit": git_commit(),
        "mujoco_version": mujoco.__version__,
        "scene_xml": "scene.xml",
        "model_sha256": sha,
        "sha256_method": SHA_METHOD,
        "files": per_file,
        **info,
        "home": {"key": "home", "q_arm_rad": q_home, "finger_qpos_m": [gap / 2, gap / 2], "finger_gap_m": gap,
                 "qpos": disk["qpos_home"], "O_T_flange": W_F.tolist(), "O_T_tcp": W_TCP.tolist(),
                 "source": os.path.relpath(_rel(st, "control_config"), REPO) + " homing.q_home_rad"},
        "F_T_TCP": F_T_TCP.tolist(),
        "F_T_TCP_source": os.path.relpath(contracts / "frames.yaml", REPO),
        "F_T_TCP_kit_derived": F_T_TCP_kit.tolist(),
        "F_T_TCP_vs_kit": {"mm": dev_mm, "deg": dev_deg},
        "finger_kit": {"id": kit_info["id"], "sha256": kit_info["sha"], "tag_ids": kit_info["tag_ids"],
                       "finger_body_mass_kg": kit_info["finger_body_mass_kg"],
                       "tag_separation_closed_m": kit_info["tag_separation_closed_m"]},
        "rig": {"mass_kg": rig_info["mass_kg"], "com_hand_m": rig_info["com_hand_m"],
                "rig_yaml_sha256": rig_info["rig_yaml_sha256"]},
        "table": {"top_z_m": float(sc["table"]["top_z_m"]), "x_m": sc["table"]["x_m"], "y_m": sc["table"]["y_m"]},
        "eraser": {"size_m": er["size_m"], "resting": er.get("resting", "flat"),
                   "box_m": eraser_box(sc).tolist(), "mass_kg": info["bodies"]["eraser"]["mass_kg"],
                   "friction": er["friction"], "condim": int(er["condim"]),
                   "default_qpos": er_qpos, "free_joint": "eraser",
                   "qposadr": info["joints"]["eraser"]["qposadr"]},
        "camera_intrinsics_note": "cam_imx519: rig file cameras.main (nominal pinhole); cam_imx219: the policy's "
                                  "virtual_ultrawide pinhole as a stand-in (the sim Pi renders the real fisheye)",
        "geom_groups": {"0": "scene", "1": "arm + hand", "2": "rig parts", "3": "fingers + adapter + tags",
                        "4": "sites", "5": "collision-only (hidden)"},
        "appearance": look,
        "finger_contact": contact,
    }
    tmp = out_dir / ".model.json.tmp"
    tmp.write_text(json.dumps(doc, indent=1))
    tmp.replace(out_dir / "model.json")
    log(f"[model] wrote {out_dir}/scene.xml, meshes/ ({len(per_file) - 1} files), model.json; sha {sha[:16]}…")
    return doc


_ASSET_TAGS = {"mesh", "material", "texture", "hfield", "skin"}
_ASSET_REF_ATTRS = {"mesh", "material", "texture", "hfield", "skin"}


def strip_hand_prefix(xml: str, meshdir: str | None = None) -> str:
    """Drop the ``hand/`` prefix MjSpec.attach_body put on the hand's names, in every name and reference of the XML.
    Non-asset names (bodies, joints, geoms, sites, tendons, actuators, equalities) lose the prefix and must stay
    unique; asset names (mesh, material, ...) and default classes clash with the arm's (e.g. material white), so
    they become ``hand_<name>``."""
    import xml.etree.ElementTree as ET

    root = ET.fromstring(xml)
    if meshdir is not None:
        for c in root.iter("compiler"):
            c.set("meshdir", meshdir)
            c.attrib.pop("texturedir", None)
        for el in root.iter("mesh"):
            if el.get("file"):
                el.set("file", Path(el.get("file")).name)
        for el in root.iter("texture"):                  # E7: no texturedir: relative to scene.xml
            if el.get("file"):
                el.set("file", meshdir + Path(el.get("file")).name)
    asset = {}
    klass = {}
    for el in root.iter():
        n = el.get("name")
        if el.tag in _ASSET_TAGS and n and n.startswith(HAND_PREFIX):
            asset[n] = "hand_" + n[len(HAND_PREFIX):]
        if el.tag == "default" and el.get("class", "").startswith(HAND_PREFIX):
            klass[el.get("class")] = "hand_" + el.get("class")[len(HAND_PREFIX):]
    names = set()
    for el in root.iter():
        for k, v in list(el.attrib.items()):
            if not v.startswith(HAND_PREFIX):
                continue
            if k in ("class", "childclass"):
                el.set(k, klass.get(v, v))
            elif (k == "name" and el.tag in _ASSET_TAGS) or k in _ASSET_REF_ATTRS:
                el.set(k, asset.get(v, v))
            else:
                el.set(k, v[len(HAND_PREFIX):])
        if el.tag not in _ASSET_TAGS and el.tag != "default" and el.get("name"):
            key = (el.tag, el.get("name"))
            if key in names and el.tag in ("body", "joint", "geom", "site", "camera", "tendon", "fixed", "motor",
                                           "general", "position"):
                raise ValueError(f"name clash after dropping {HAND_PREFIX!r}: <{el.tag} name={el.get('name')!r}>")
            names.add(key)
    ET.indent(root, space="  ")
    return ET.tostring(root, encoding="unicode") + "\n"


RESTING = {                       # scene.eraser.resting -> which of size_m [length, width, thickness] is along body x, y, z
    "flat": (0, 1, 2),            # length x width face down, thickness up
    "on_long_edge": (0, 2, 1),    # length x thickness face down, width up
    "on_end": (2, 1, 0),          # thickness x width face down, length up
}


def eraser_box(sc: dict) -> np.ndarray:
    """Full box extents along the eraser body's x, y, z (z up when resting) from size_m and resting."""
    er = sc["eraser"]
    size = [float(v) for v in er["size_m"]]
    rest = str(er.get("resting", "flat"))
    if rest not in RESTING:
        raise ValueError(f"scene.eraser.resting {rest!r}: one of {sorted(RESTING)}")
    return np.array([size[i] for i in RESTING[rest]])


def eraser_qpos(sc: dict, x: float, y: float, yaw_deg: float) -> list:
    """Free-joint qpos [x, y, z, qw, qx, qy, qz] of the eraser resting on the table top at (x, y), yaw about O z
    (yaw 0 = body x, its length for flat / on_long_edge, along O x)."""
    z = float(sc["table"]["top_z_m"]) + float(eraser_box(sc)[2]) / 2
    a = np.deg2rad(yaw_deg) / 2
    return [float(x), float(y), z, float(np.cos(a)), 0.0, 0.0, float(np.sin(a))]


def _hull_faces(v: np.ndarray) -> np.ndarray:
    from scipy.spatial import ConvexHull

    return ConvexHull(v).simplices


def _site_T(d, i) -> np.ndarray:
    return _T(d.site_xmat[i].reshape(3, 3), d.site_xpos[i])


def _add_scene(spec, sc: dict, log) -> None:
    import mujoco

    wb = spec.worldbody
    t = sc["table"]
    top, fz = float(t["top_z_m"]), float(t["floor_z_m"])
    (x0, x1), (y0, y1) = t["x_m"], t["y_m"]
    tb = wb.add_body()
    tb.name = "table"
    tb.pos = [(x0 + x1) / 2, (y0 + y1) / 2, top]                 # body frame: centre of the table top
    g = tb.add_geom()
    g.name, g.type = "table", mujoco.mjtGeom.mjGEOM_BOX
    g.size = [(x1 - x0) / 2, (y1 - y0) / 2, (top - fz) / 2]
    g.pos = [0, 0, -(top - fz) / 2]
    g.rgba, g.friction, g.group = list(t["rgba"]), list(t["friction"]), GROUP_SCENE
    pd = sc.get("pedestal")                                      # P3/P4: optional (none: the base is on the table)
    if pd:
        g = wb.add_geom()
        g.name, g.type = "pedestal", mujoco.mjtGeom.mjGEOM_BOX
        (px0, px1), (py0, py1) = pd["x_m"], pd["y_m"]
        g.size = [(px1 - px0) / 2, (py1 - py0) / 2, -fz / 2]
        g.pos = [(px0 + px1) / 2, (py0 + py1) / 2, fz / 2]
        g.rgba, g.contype, g.conaffinity, g.group = list(pd["rgba"]), 0, 0, GROUP_SCENE
    g = wb.add_geom()
    g.name, g.type, g.size, g.pos = "floor", mujoco.mjtGeom.mjGEOM_PLANE, [2.0, 2.0, 0.1], [0.4, 0.0, fz]
    g.rgba, g.contype, g.conaffinity, g.group = list(sc["floor_rgba"]), 0, 0, GROUP_SCENE
    for b in ("link0", "link1"):                                 # the base sits on the table (P3/P4; was beside it)
        ex = spec.add_exclude()
        ex.bodyname1, ex.bodyname2 = "table", b

    er = sc["eraser"]
    eb = wb.add_body()
    eb.name = "eraser"
    j = eb.add_freejoint()
    j.name = "eraser"
    g = eb.add_geom()
    g.name, g.type = "eraser", mujoco.mjtGeom.mjGEOM_BOX
    g.size = (eraser_box(sc) / 2).tolist()
    g.mass = float(er["mass_kg"])
    g.friction, g.condim = list(er["friction"]), int(er["condim"])
    g.rgba, g.group = list(er["rgba"]), GROUP_SCENE
    from franka_sim.model import grip_contact
    sol = grip_contact({"timestep_s": float(spec.option.timestep)})
    g.solref, g.solimp = sol["solref"], sol["solimp"]

    ec = sc["external_camera"]
    pos, tgt = np.asarray(ec["pos_m"], float), np.asarray(ec["target_m"], float)
    f = (tgt - pos) / np.linalg.norm(tgt - pos)
    x = np.cross(f, [0, 0, 1.0])
    x /= np.linalg.norm(x)
    y = np.cross(x, f)
    c = wb.add_camera()
    c.name, c.pos = "cam_external", pos.tolist()
    c.quat = _quat(np.column_stack([x, y, -f]))
    c.fovy = float(ec["fovy_deg"])
    c.resolution = [int(v) for v in ec["resolution_px"]]

    sky = sc.get("skybox")
    if sky:
        tx = spec.add_texture()
        tx.name, tx.type, tx.builtin = "skybox", mujoco.mjtTexture.mjTEXTURE_SKYBOX, mujoco.mjtBuiltin.mjBUILTIN_GRADIENT
        tx.rgb1, tx.rgb2, tx.width, tx.height = list(sky["rgb_top"]), list(sky["rgb_bottom"]), 256, 1536
    for i, lt in enumerate(sc["lights"]):
        L = wb.add_light()
        L.name = f"light{i}"
        L.directional = True
        dvec = np.asarray(lt["dir"], float)
        L.dir = (dvec / np.linalg.norm(dvec)).tolist()
        L.pos = (np.array([0.45, 0, 0.05]) - 2.0 * L.dir).tolist()
        L.diffuse, L.specular = list(lt["diffuse"]), list(lt["specular"])
        L.ambient = [0.0, 0.0, 0.0]
        L.castshadow = bool(lt["castshadow"])
    hl = sc["headlight"]
    spec.visual.headlight.ambient = list(hl["ambient"])
    spec.visual.headlight.diffuse = list(hl["diffuse"])
    spec.visual.headlight.specular = list(hl["specular"])


# ------------------------------------------------------------------------------------------- E7: look + contact

STICKER_M, MARKER_M = 0.020, 0.016            # real finger tags: 16 mm DICT_4X4_50 marker on a 20 mm white sticker


def tag_sticker_png(path: Path, tag_id: int, px: int = 500) -> None:
    """The real sticker (same pattern as sim_pi.handheld_check.tag_texture): marker centred on white."""
    import cv2

    inner = int(round(px * MARKER_M / STICKER_M))
    marker = cv2.aruco.generateImageMarker(cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50), int(tag_id), inner)
    img = np.full((px, px), 255, np.uint8)
    o = (px - inner) // 2
    img[o:o + inner, o:o + inner] = marker
    cv2.imwrite(str(path), img)


def wood_png(path: Path, px: int = 1024, seed: int = 7) -> None:
    """A light, procedural wood grain (the lab table is light beech/birch): stripes along u, warped by noise."""
    import cv2

    rng = np.random.default_rng(seed)
    u, v = np.meshgrid(np.linspace(0, 1, px, endpoint=False), np.linspace(0, 1, px, endpoint=False))
    warp = sum(a * np.sin(2 * np.pi * (f * u + rng.uniform()) + rng.uniform(0, 6)) for a, f in
               ((0.020, 1), (0.010, 3), (0.004, 7)))
    ring = 0.5 + 0.5 * np.sin(2 * np.pi * (v + warp) * 40 + 3 * np.sin(2 * np.pi * (v * 3 + u * 0.5)))
    fine = cv2.GaussianBlur(rng.normal(0, 1, (px, px)).astype(np.float32), (0, 0), sigmaX=8, sigmaY=0.8)
    t = np.clip(0.55 * ring ** 6 + 0.25 + 0.12 * fine, 0, 1)[..., None]
    light, dark = np.array([226, 190, 138], float), np.array([190, 142, 88], float)      # RGB
    img = (light * (1 - t) + dark * t).clip(0, 255).astype(np.uint8)
    cv2.imwrite(str(path), img[:, :, ::-1])


def _tag_slab_obj(path: Path, half: float, thick: float) -> None:
    """A thin box (tag frame: x = normal, y = in-plane, z = up), its +x face carrying the whole texture: u along +y,
    v along +z (marker top = the tag's up, toward the fingertips, as on the real holders)."""
    lines = ["# sim_deploy E7 tag sticker slab"]
    vs = [(x, y, z) for x in (0.0, thick) for y in (-half, half) for z in (-half, half)]
    for v in vs:
        lines.append("v %.6f %.6f %.6f" % v)
    for v in vs:
        lines.append("vt %.4f %.4f" % ((v[1] / half + 1) / 2, (v[2] / half + 1) / 2))
    quads = [(4, 6, 7, 5), (0, 1, 3, 2), (0, 4, 5, 1), (2, 3, 7, 6), (0, 2, 6, 4), (1, 5, 7, 3)]
    for q in quads:
        a, b, c, d = (i + 1 for i in q)
        lines += [f"f {a}/{a} {b}/{b} {c}/{c}", f"f {a}/{a} {c}/{c} {d}/{d}"]
    path.write_text("\n".join(lines) + "\n")


def _appearance(spec, ap: dict, kit_info: dict, tex_dir: Path, base_dir=None) -> dict:
    """Pads, holders and adapters coloured as the real parts; tag_{l,r} become the real ArUco stickers (textured
    slabs, the ink boxes go); the table gets a wood texture. Changes nothing physical (visual geoms, rgba)."""
    import mujoco

    if not ap:
        return {}
    tex_dir.mkdir(parents=True, exist_ok=True)
    rep = {"tags": {}}
    col = {"pad": ap.get("pads_rgba"), "holder": ap.get("holders_rgba"), "adapter": ap.get("adapters_rgba")}
    for b in spec.bodies:
        if _body_kind(b.name) != "finger":
            continue
        for g in list(b.geoms):
            n = g.name.split("/")[-1]
            if g.rgba[3] <= 0:
                continue
            if n.startswith("pad_") and col["pad"]:
                g.rgba = list(col["pad"])
            elif n.startswith("finger_holder_") and n.endswith("_1") and col["adapter"]:
                g.rgba = list(col["adapter"])
            elif n.startswith("finger_holder_") and col["holder"]:
                g.rgba = list(col["holder"])
            elif n.startswith("tag_") and n.endswith("_ink") and ap.get("aruco_tags", True):
                g.delete()
            elif n.startswith("tag_") and ap.get("aruco_tags", True):
                tid = int(kit_info["tag_ids"][n])
                png = tex_dir / f"tag_aruco{tid}.png"
                tag_sticker_png(png, tid)
                obj = tex_dir / "tag_sticker_slab.obj"
                if not obj.exists():
                    _tag_slab_obj(obj, STICKER_M / 2, 0.0004)
                    me = spec.add_mesh()
                    me.name, me.file = "tag_sticker", str(obj)
                tx = spec.add_texture()
                tx.name, tx.file, tx.type = f"tag_aruco{tid}", str(png), mujoco.mjtTexture.mjTEXTURE_2D
                mt = spec.add_material()
                mt.name = f"tag_aruco{tid}"
                mt.textures[mujoco.mjtTextureRole.mjTEXROLE_RGB] = f"tag_aruco{tid}"
                mt.specular, mt.shininess = 0.2, 0.2
                # the kit's white box: x = normal, 1 mm thick, centred 0.5 mm off the holder -> slab from its back
                pos = np.asarray(g.pos, float)
                R = np.zeros(9)
                mujoco.mju_quat2Mat(R, np.asarray(g.quat, float))
                R = R.reshape(3, 3)
                g.type, g.meshname, g.material = mujoco.mjtGeom.mjGEOM_MESH, "tag_sticker", mt.name
                g.size = [0, 0, 0]
                g.pos = (pos - 0.0005 * R[:, 0]).tolist()
                g.rgba = [1, 1, 1, 1]
                rep["tags"][n] = {"aruco_id": tid, "sticker_m": STICKER_M, "marker_m": MARKER_M}
    if ap.get("table_texture") == "wood":
        png = tex_dir / "table_wood.png"
        wood_png(png)
        tx = spec.add_texture()
        tx.name, tx.file, tx.type = "table_wood", str(png), mujoco.mjtTexture.mjTEXTURE_2D
        mt = spec.add_material()
        mt.name = "table_wood"
        mt.textures[mujoco.mjtTextureRole.mjTEXROLE_RGB] = "table_wood"
        mt.texrepeat = [float(x) for x in ap.get("table_texrepeat", [1.5, 1.5])]
        mt.texuniform = True
        mt.specular, mt.shininess = 0.15, 0.3
        g = spec.geom("table")
        g.material, g.rgba = "table_wood", [1, 1, 1, 1]
        rep["table"] = {"texture": "wood (procedural)", "texrepeat_per_m": [float(x) for x in mt.texrepeat]}
    elif ap.get("table_texture") == "file":                  # W2 look: the real table photographed (opt-in)
        rep["table"] = _table_texture_file(spec, ap, base_dir, tex_dir)
    if ap.get("background_mesh"):                            # W2 look: the lab around the table (opt-in)
        bgs = ap["background_mesh"]
        if isinstance(bgs, dict):
            rep["background_mesh"] = _background_mesh(spec, bgs, base_dir, tex_dir)
        else:                                                # a list: background, background_1, ...
            rep["background_mesh"] = [_background_mesh(spec, b, base_dir, tex_dir,
                                                       name="background" + (f"_{i}" if i else ""))
                                      for i, b in enumerate(bgs)]
    rep.update({k: v for k, v in ap.items() if k.endswith("_rgba")})
    return rep


def _file_sha(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _table_top_slab_obj(path: Path, x: tuple, y: tuple, z_top: float, thick: float, origin: tuple,
                        size: tuple) -> None:
    """W2 look: a thin box over the table top (world coordinates), its top face at ``z_top``; every vertex's
    texcoord is its O (x, y) in texture periods: u = (x - origin x) / size x, v = (y - origin y) / size y (v up =
    image top toward O +y), so the image lands at its real place and repeats beyond it."""
    (x0, x1), (y0, y1) = x, y
    vs = [(xx, yy, zz) for xx in (x0, x1) for yy in (y0, y1) for zz in (z_top - thick, z_top)]
    lines = ["# sim_deploy W2 look: table-top texture slab (world frame)"]
    for v in vs:
        lines.append("v %.6f %.6f %.6f" % v)
    for v in vs:
        lines.append("vt %.6f %.6f" % ((v[0] - origin[0]) / size[0], (v[1] - origin[1]) / size[1]))
    # vertex k = 4 ix + 2 iy + iz (+1 in OBJ); outward faces, counter-clockwise seen from outside
    quads = [(2, 6, 8, 4), (1, 3, 7, 5), (1, 5, 6, 2), (3, 4, 8, 7), (1, 2, 4, 3), (5, 7, 8, 6)]
    for a, b, c, d in quads:
        lines += [f"f {a}/{a} {b}/{b} {c}/{c}", f"f {a}/{a} {c}/{c} {d}/{d}"]
    path.write_text("\n".join(lines) + "\n")


def _table_texture_file(spec, ap: dict, base_dir, tex_dir: Path) -> dict:
    """W2 look, ``table_texture: file``: the image ``table_texture_file`` (relative to the station file) at real
    scale on the table top, ``table_texture_size_m`` [x, y] per image, its bottom-left corner at O
    ``table_texture_origin_O_m``, repeated beyond (seamless tiles), on a visual-only slab ``table_texture_lift_m``
    (0.3 mm) above the table box (a world geom: no contact, no mass), over the box's extent or the optional
    ``table_visual_x_m`` / ``table_visual_y_m`` (a real table smaller than the colliding box), ``table_texture_slab_m``
    thick (1 mm). The box keeps its collision role and gets ``table_rgba`` (its sides; alpha 0 hides it) when given."""
    import mujoco

    src = (Path(base_dir) / ap["table_texture_file"]).resolve()
    png = tex_dir / src.name
    shutil.copy2(src, png)
    size = [float(v) for v in ap["table_texture_size_m"]]
    origin = [float(v) for v in ap["table_texture_origin_O_m"]]
    tb, g = spec.body("table"), spec.geom("table")
    cx, cy, top = (float(v) for v in tb.pos)
    hx, hy = float(g.size[0]), float(g.size[1])
    lift = float(ap.get("table_texture_lift_m", 0.0003))
    xr = [float(v) for v in ap.get("table_visual_x_m", (cx - hx, cx + hx))]   # optional: the real table's extent
    yr = [float(v) for v in ap.get("table_visual_y_m", (cy - hy, cy + hy))]   # (default: the colliding box's)
    thick = float(ap.get("table_texture_slab_m", 0.001))
    obj = tex_dir / "table_top_slab.obj"
    _table_top_slab_obj(obj, tuple(xr), tuple(yr), top + lift, thick, origin, size)
    me = spec.add_mesh()
    me.name, me.file = "table_top", str(obj)
    tx = spec.add_texture()
    tx.name, tx.file, tx.type = "table_real", str(png), mujoco.mjtTexture.mjTEXTURE_2D
    mt = spec.add_material()
    mt.name = "table_real"
    mt.textures[mujoco.mjtTextureRole.mjTEXROLE_RGB] = "table_real"
    mt.specular = float(ap.get("table_specular", 0.0))
    mt.shininess = float(ap.get("table_shininess", 0.0))
    mt.reflectance = 0.0
    top_g = spec.worldbody.add_geom()
    top_g.name, top_g.type, top_g.meshname = "table_top", mujoco.mjtGeom.mjGEOM_MESH, "table_top"
    top_g.material, top_g.rgba = "table_real", [1, 1, 1, 1]
    top_g.contype, top_g.conaffinity, top_g.group = 0, 0, GROUP_SCENE
    top_g.density = 0.0
    if ap.get("table_rgba"):
        g.rgba = [float(v) for v in ap["table_rgba"]]
    return {"texture": "file", "file": os.path.relpath(src, REPO), "sha256": _file_sha(src),
            "size_m": size, "origin_O_m": origin, "slab_top_z_m": top + lift, "geom": "table_top (visual only)",
            "slab_x_m": xr, "slab_y_m": yr, "slab_thickness_m": thick,
            "specular": mt.specular, "shininess": mt.shininess}


def _background_mesh(spec, bg: dict, base_dir, tex_dir: Path, name: str = "background") -> dict:
    """W2 look, ``background_mesh: {obj, texture, pos, quat, emission}``: a textured mesh (e.g. the workspace
    reconstruction) as a visual-only world geom (contype/conaffinity 0, group 0, no mass). Its material is nearly
    unlit: emission e with rgb k = 1 / (e + 1), so the photographed colours come out as they are under a light with
    ambient + diffuse = 1 (a face lit by the ambient a only renders (e + a) / (e + 1) of its texture)."""
    import mujoco

    obj_src = (Path(base_dir) / bg["obj"]).resolve()
    png_src = (Path(base_dir) / bg["texture"]).resolve()
    obj, png = tex_dir / obj_src.name, tex_dir / png_src.name
    shutil.copy2(obj_src, obj)
    shutil.copy2(png_src, png)
    me = spec.add_mesh()
    me.name, me.file = name, str(obj)
    me.inertia = mujoco.mjtMeshInertia.mjMESH_INERTIA_SHELL
    tx = spec.add_texture()
    tx.name, tx.file, tx.type = name, str(png), mujoco.mjtTexture.mjTEXTURE_2D
    e = float(bg.get("emission", 3.0))
    k = 1.0 / (e + 1.0)
    mt = spec.add_material()
    mt.name = name
    mt.textures[mujoco.mjtTextureRole.mjTEXROLE_RGB] = name
    mt.specular, mt.shininess, mt.reflectance, mt.emission = 0.0, 0.0, 0.0, e
    mt.rgba = [k, k, k, 1.0]
    g = spec.worldbody.add_geom()
    g.name, g.type, g.meshname, g.material = name, mujoco.mjtGeom.mjGEOM_MESH, name, name
    g.pos = [float(v) for v in bg.get("pos", [0.0, 0.0, 0.0])]
    g.quat = [float(v) for v in bg.get("quat", [1.0, 0.0, 0.0, 0.0])]
    g.rgba = [1, 1, 1, 1]
    g.contype, g.conaffinity, g.group = 0, 0, GROUP_SCENE
    g.density = 0.0
    return {"obj": os.path.relpath(obj_src, REPO), "obj_sha256": _file_sha(obj_src),
            "texture": os.path.relpath(png_src, REPO), "texture_sha256": _file_sha(png_src),
            "pos": list(g.pos), "quat": list(g.quat), "emission": e, "material_rgb": k,
            "geom": f"{name} (visual only, world)"}


def _finger_contact(spec, fc: dict | None) -> dict:
    """The fin-ray fingers' compliance where they hold the eraser: explicit pad-eraser contact pairs (they replace
    the dynamic pad-eraser contacts) with a margin = a soft layer that starts pushing that far from the rigid pad
    surface, and a direct spring-damper solref (negative values: stiffness, damping). Holders, adapters and the
    table keep the stiff contacts."""
    if not fc:
        return {}
    out = {"pairs": []}
    for s in ("l", "r"):
        p = spec.add_pair()
        p.name = f"pad_{s}_eraser"
        p.geomname1, p.geomname2 = f"pad_{s}", "eraser"
        p.condim = int(fc.get("condim", 4))
        p.margin, p.gap = float(fc["margin_m"]), 0.0
        p.solref = [float(x) for x in fc["solref"]]
        p.solimp = [float(x) for x in fc["solimp"]]
        p.friction = [float(x) for x in fc["friction"]]
        if fc.get("solreffriction"):                     # stiff friction: no creep out of the soft grip
            p.solreffriction = [float(x) for x in fc["solreffriction"]]
        out["pairs"].append(f"pad_{s}_eraser")
    out.update({k: fc[k] for k in ("margin_m", "solref", "solimp", "solreffriction", "friction") if k in fc})
    return out


def _body_kind(name: str) -> str:
    name = name[len(HAND_PREFIX):] if name.startswith(HAND_PREFIX) else name
    if name in ("left_finger", "right_finger"):
        return "finger"
    if name == "rig":
        return "rig"
    if name in ("world", "table", "eraser"):
        return "scene"
    return "arm"


def _set_groups(spec) -> None:
    for b in spec.bodies:
        kind = _body_kind(b.name)
        for g in b.geoms:
            collides = g.contype != 0 or g.conaffinity != 0
            visible = g.rgba[3] > 0
            if kind == "scene":
                g.group = GROUP_SCENE
            elif kind == "rig":
                continue                                          # E2: visuals 2, hulls 5
            elif kind == "finger":                                # kit: pads (contact, shown), holders, adapters, tags
                g.group = GROUP_FINGERS if visible else GROUP_COLLISION
            else:                                                 # menagerie arm + hand: class visual / collision
                g.group = GROUP_COLLISION if collides else GROUP_ARM
    for g in spec.worldbody.geoms:
        g.group = GROUP_SCENE


def inspect_in_subprocess(scene_xml: Path) -> dict:
    import sys

    r = subprocess.run([sys.executable, "-m", "sim_deploy.model.inspect", str(scene_xml)], capture_output=True,
                       text=True, env={**os.environ, "MUJOCO_GL": os.environ.get("MUJOCO_GL", "egl")})
    if r.returncode != 0:
        raise RuntimeError(f"loading {scene_xml} in a fresh process failed:\n{r.stderr[-3000:]}")
    return json.loads(r.stdout)


def checks_of(m) -> dict:
    """Numbers that must agree between the in-memory build and the scene.xml reload."""
    import mujoco

    d = mujoco.MjData(m)
    mujoco.mj_resetDataKeyframe(m, d, 0)
    mujoco.mj_forward(m, d)
    counts = {k: int(getattr(m, k)) for k in ("nq", "nv", "nu", "nbody", "njnt", "ngeom", "nsite", "ncam", "nmesh",
                                                 "neq", "ntendon", "nkey")}
    return {"counts": counts, "xpos": d.xpos.tolist(), "cam_xpos": d.cam_xpos.tolist(),
            "body_mass": m.body_mass.tolist(), "geom_size": m.geom_size.tolist(), "geom_rbound": m.geom_rbound.tolist()}


def _check_same(m_mem, disk: dict) -> None:
    a, b = checks_of(m_mem), disk["checks"]
    if a["counts"] != b["counts"]:
        raise RuntimeError(f"scene.xml reload differs from the built spec: {a['counts']} vs {b['counts']}")
    for k in ("xpos", "cam_xpos", "body_mass", "geom_size", "geom_rbound"):
        err = float(np.abs(np.array(a[k]) - np.array(b[k])).max()) if a[k] else 0.0
        if err > 1e-5:
            raise RuntimeError(f"scene.xml reload differs from the built spec: {k} by {err:.2e}")


def describe(m, d) -> dict:
    """Counts, names and addresses for model.json."""
    import mujoco

    J = {mujoco.mjtJoint.mjJNT_FREE: "free", mujoco.mjtJoint.mjJNT_BALL: "ball", mujoco.mjtJoint.mjJNT_SLIDE: "slide",
         mujoco.mjtJoint.mjJNT_HINGE: "hinge"}
    joints = {m.joint(i).name: {"type": J[m.jnt_type[i]], "qposadr": int(m.jnt_qposadr[i]),
                                "dofadr": int(m.jnt_dofadr[i]), "body": m.body(m.jnt_bodyid[i]).name,
                                "range": m.jnt_range[i].tolist() if m.jnt_limited[i] else None}
              for i in range(m.njnt)}
    cams = {}
    for i in range(m.ncam):
        c = m.camera(i)
        w, h = (int(v) for v in m.cam_resolution[i])
        entry = {"body": m.body(m.cam_bodyid[i]).name, "resolution_px": [w, h]}
        if m.cam_sensorsize[i].any():
            sx, sy = m.cam_sensorsize[i] / np.array([w, h], float)          # metres per pixel
            fx, fy = m.cam_intrinsic[i][0] / sx, m.cam_intrinsic[i][1] / sy   # intrinsic is in sensor units
            ppx, ppy = m.cam_intrinsic[i][2] / sx, m.cam_intrinsic[i][3] / sy
            entry["K_opencv"] = [[float(fx), 0, float(w / 2 - ppx - 0.5)], [0, float(fy), float(ppy + h / 2 - 0.5)],
                                 [0, 0, 1]]
        else:
            entry["fovy_deg"] = float(m.cam_fovy[i])
        T = _T(d.cam_xmat[i].reshape(3, 3) @ CV_TO_MJ, d.cam_xpos[i])
        entry["O_T_cam_cv_home"] = T.tolist()
        cams[c.name] = entry
    groups = {}
    for g in range(m.ngeom):
        groups.setdefault(str(int(m.geom_group[g])), 0)
        groups[str(int(m.geom_group[g]))] += 1
    return {
        "nq": int(m.nq), "nv": int(m.nv), "nu": int(m.nu), "nbody": int(m.nbody), "ngeom": int(m.ngeom),
        "timestep_s": float(m.opt.timestep),
        "qpos_layout": {"arm": [0, 7], "fingers": [7, 9], "eraser": [m.nq - 7, m.nq]},
        "arm_joints": [f"joint{i}" for i in range(1, 8)],
        "finger_joints": ["finger_joint1", "finger_joint2"],
        "joints": joints,
        "actuators": {m.actuator(i).name: {"target": (m.joint(m.actuator_trnid[i, 0]).name
                                                      if m.actuator_trntype[i] == mujoco.mjtTrn.mjTRN_JOINT
                                                      else m.tendon(m.actuator_trnid[i, 0]).name),
                                           "ctrlrange": m.actuator_ctrlrange[i].tolist()} for i in range(m.nu)},
        "tendons": [m.tendon(i).name for i in range(m.ntendon)],
        "equalities": [m.eq(i).name or f"eq{i}" for i in range(m.neq)],
        "sites": [m.site(i).name for i in range(m.nsite)],
        "cameras": cams,
        "bodies": {m.body(i).name: {"mass_kg": float(m.body_mass[i]), "parent": m.body(m.body_parentid[i]).name}
                   for i in range(m.nbody)},
        "named_geoms": {m.geom(g).name: {"body": m.body(m.geom_bodyid[g]).name, "group": int(m.geom_group[g]),
                                         "collides": bool(m.geom_contype[g] or m.geom_conaffinity[g])}
                        for g in range(m.ngeom) if m.geom(g).name},
        "geom_group_counts": groups,
        "robot_mass_kg": float(m.body_subtreemass[m.body("link0").id]),     # arm + hand + kit + rig (static table excluded)
        "hand_subtree_mass_kg": float(m.body_subtreemass[m.body("hand").id]),
    }
