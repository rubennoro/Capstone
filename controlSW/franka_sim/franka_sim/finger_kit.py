"""Finger kit (``finger_kit/v1``): the printed fingers as data (STL meshes + named reference frames), so the model
builder can mount the real CAD fingers on the Franka Hand instead of the two placeholder boxes.

Why: the box pads, their 8 mm offset and the fitted tag offsets are stand-ins (model.py constants). A kit carries the
geometry, the mount frame, the fingertip and the ArUco tags of each side in the CAD's own coordinates; this module
validates it, converts units to metres and computes the poses in the Franka finger body frame. No mujoco here: the
loader also runs in the tools and tests, and the station validator must not need the assets.

Frame chain per finger (A9.md; all frames z = toward the fingertip, y = outward, away from the other finger):
    finger body ─▶ J (jaw mount, from the stock finger mesh) ─▶ J_T_M = Trans(0, dy, L) · Rz(yaw) ─▶ M (kit mount frame)
Meshes, ``tip`` and the tags are in kit (CAD) coordinates (one assembly frame, as the CAD exports them) and posed by
``inv(kit_T_M)``: ``body_T_kit = J · J_T_M · inv(kit_T_M)``. Only ``camera_face`` is about the side's M axes. A tag ``centre`` is the point on the part's
surface where the tag is stuck (its back face); ``normal`` points out of the tag face toward the camera.

kit sha256 (``kit_digest``) = sha256 over the ``kit.yaml`` bytes followed by the ASCII sha256 hex of every file kit.yaml
lists (meshes and collision meshes), in sorted relative-path order.

Usage: kit = load_kit(dir, expect_sha=tooling["finger_adapter"]["kit_sha256"])
       out = place(kit, "left", J, L, dy, yaw_deg); dy = auto_dy(kit, "left", J, L, yaw_deg)
       make_box_kit(tmp) + box_equivalent_placement(J, tip_z, extension, pad_size) = today's box pads (A9c test)
"""

from __future__ import annotations

import hashlib
import struct
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath

import numpy as np
import yaml

from franka_sim.tools.check_assets import parse_source, sha256_file

SCHEMA = "finger_kit/v1"
KIT_YAML = "kit.yaml"
UNITS = {"mm": 1e-3, "m": 1.0}
ROLES = ("contact", "structure")
PROV_STATUS = ("measured", "derived", "placeholder", "decision", "datasheet")
TOP_KEYS = ("schema", "id", "units", "parts", "materials", "sides", "contact", "provenance")
TAG_LIFT_M = 0.0005          # white tag box centre above the surface (half its 1 mm thickness): model.py step 3
ROT_TOL = 1e-3               # kit_T_M rotation must be orthonormal to this (CAD exports carry few digits)


class KitError(ValueError):
    """A finger kit that cannot be used; the message names the kit and the offending key."""


@dataclass
class Part:
    name: str
    mesh: Path                       # absolute path of the visual/contact STL
    role: str                        # contact | structure
    material: str
    collision: str | list[Path]      # "hull" or absolute paths of convex pieces
    vertices: np.ndarray             # (n, 3) unique vertices, metres, kit coords
    faces: np.ndarray                # (m, 3) int indices into vertices
    collision_vertices: list[np.ndarray] = field(default_factory=list)   # per piece (metres, kit coords); [] = hull
    mesh_scale: float = 1.0          # file units -> m (MjSpec mesh.scale)
    volume_m3: float = 0.0           # signed-tetra volume of the closed mesh
    mass_kg: float = 0.0             # measured mass_kg if given, else density * volume
    mass_source: str = "density"


@dataclass
class Tag:
    id: int
    centre: np.ndarray               # m, kit coords (on the part surface)
    normal: np.ndarray               # unit, kit coords, out of the tag face
    up: np.ndarray                   # unit, kit coords, orthogonal to normal
    size_m: float


@dataclass
class Side:
    name: str
    parts: tuple[str, ...]
    contact_part: str
    kit_T_M: np.ndarray              # 4x4, metres, proper rotation
    camera_face: float               # +1 / -1: the M-frame x sign the camera looks at
    tip: np.ndarray                  # m, kit coords
    tags: list[Tag]


@dataclass
class Kit:
    id: str
    path: Path
    sha256: str
    units: str
    parts: dict[str, Part]
    sides: dict[str, Side]
    materials: dict
    contact: dict                    # friction (3 floats), solref / solimp (None -> model.grip_contact())
    provenance: dict


# --------------------------------------------------------------------------------------------------- small math
# (local copies of model.py's helpers: importing model would pull in the station loader)


def _T(R=None, t=None) -> np.ndarray:
    T = np.eye(4)
    if R is not None:
        T[:3, :3] = R
    if t is not None:
        T[:3, 3] = t
    return T


def _inv(T: np.ndarray) -> np.ndarray:
    R, t = T[:3, :3], T[:3, 3]
    return _T(R.T, -R.T @ t)


def _Rz(deg: float) -> np.ndarray:
    a = np.deg2rad(deg)
    return np.array([[np.cos(a), -np.sin(a), 0], [np.sin(a), np.cos(a), 0], [0, 0, 1.0]])


def _apply(T: np.ndarray, p: np.ndarray) -> np.ndarray:
    return np.asarray(p, float) @ T[:3, :3].T + T[:3, 3]


def jaw_T(y_c: float, z_base: float) -> np.ndarray:
    """J in the finger body frame: origin (0, y_c, z_base), finger body axes (A9.md frame chain)."""
    return _T(None, [0.0, y_c, z_base])


def J_T_M(L: float, dy: float, yaw_deg: float) -> np.ndarray:
    return _T(_Rz(yaw_deg), [0.0, dy, L])


# --------------------------------------------------------------------------------------------------- STL


def read_stl(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """(unique vertices (n, 3), faces (m, 3)) in file units, binary or ASCII STL, plain numpy."""
    path = Path(path)
    try:
        data = path.read_bytes()
    except OSError as e:
        raise KitError(f"cannot read STL {path}: {e}") from None
    if len(data) >= 84:
        n = struct.unpack_from("<I", data, 80)[0]
        if len(data) == 84 + 50 * n:      # binary (a binary header may also start with "solid": size decides)
            rec = np.dtype([("n", "<f4", 3), ("v", "<f4", (3, 3)), ("a", "<u2")])
            tri = np.frombuffer(data, rec, count=n, offset=84)["v"].astype(float)
            return _index(tri, path)
    head = data[:512].lstrip().lower()
    if not head.startswith(b"solid"):
        raise KitError(f"unreadable STL {path}: neither binary (size != 84 + 50 n) nor ASCII ('solid ...')")
    try:
        text = data.decode("ascii")
    except UnicodeDecodeError:
        raise KitError(f"unreadable STL {path}: ASCII STL with non-ASCII bytes") from None
    vs = []
    for ln in text.splitlines():
        tok = ln.split()
        if tok and tok[0].lower() == "vertex":
            if len(tok) != 4:
                raise KitError(f"unreadable STL {path}: bad vertex line {ln.strip()!r}")
            try:
                vs.append([float(x) for x in tok[1:]])
            except ValueError:
                raise KitError(f"unreadable STL {path}: bad vertex line {ln.strip()!r}") from None
    if not vs or len(vs) % 3:
        raise KitError(f"unreadable STL {path}: {len(vs)} vertices (need a positive multiple of 3)")
    return _index(np.array(vs, float).reshape(-1, 3, 3), path)


def _index(tri: np.ndarray, path: Path) -> tuple[np.ndarray, np.ndarray]:
    if len(tri) == 0:
        raise KitError(f"unreadable STL {path}: no triangles")
    if not np.all(np.isfinite(tri)):
        raise KitError(f"unreadable STL {path}: non-finite coordinates")
    v, inv = np.unique(tri.reshape(-1, 3), axis=0, return_inverse=True)
    return v, inv.reshape(-1, 3)


def write_stl(path: Path, vertices, faces, *, ascii: bool = False) -> None:
    """Write a triangle mesh (normals from the winding)."""
    v = np.asarray(vertices, float)
    tri = v[np.asarray(faces, int)]
    nrm = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    nrm /= np.maximum(np.linalg.norm(nrm, axis=1, keepdims=True), 1e-300)
    if ascii:
        lines = ["solid kit"]
        for n_, t in zip(nrm, tri):
            lines += [f"  facet normal {n_[0]:.9e} {n_[1]:.9e} {n_[2]:.9e}", "    outer loop"]
            lines += [f"      vertex {p[0]:.17g} {p[1]:.17g} {p[2]:.17g}" for p in t]
            lines += ["    endloop", "  endfacet"]
        Path(path).write_text("\n".join(lines + ["endsolid kit"]) + "\n")
        return
    rec = np.zeros(len(tri), np.dtype([("n", "<f4", 3), ("v", "<f4", (3, 3)), ("a", "<u2")]))
    rec["n"], rec["v"] = nrm, tri
    Path(path).write_bytes(b"binary kit".ljust(80, b" ") + struct.pack("<I", len(tri)) + rec.tobytes())


def _volume(v: np.ndarray, f: np.ndarray) -> float:
    t = v[f]
    return float(abs(np.einsum("ij,ij->i", t[:, 0], np.cross(t[:, 1], t[:, 2])).sum()) / 6.0)


# --------------------------------------------------------------------------------------------------- digest


def _read_yaml(d: Path) -> dict:
    p = d / KIT_YAML
    if not p.is_file():
        raise KitError(f"finger kit {d}: no {KIT_YAML}")
    try:
        doc = yaml.safe_load(p.read_text())
    except yaml.YAMLError as e:
        raise KitError(f"finger kit {d}: {KIT_YAML} is not valid YAML: {e}") from None
    if not isinstance(doc, dict):
        raise KitError(f"finger kit {d}: {KIT_YAML} must be a mapping")
    return doc


def _rel(d: Path, s, where: str) -> str:
    """A kit-relative posix path that stays inside the kit dir."""
    if not isinstance(s, str) or not s:
        raise KitError(f"finger kit {d}: {where} must be a relative file path, got {s!r}")
    p = PurePosixPath(s)
    if p.is_absolute() or ".." in p.parts:
        raise KitError(f"finger kit {d}: {where} {s!r} must be relative and inside the kit directory")
    return str(p)


def listed_files(d: Path, doc: dict | None = None) -> list[str]:
    """Sorted kit-relative paths of every file kit.yaml lists (part meshes + collision pieces)."""
    d = Path(d)
    doc = _read_yaml(d) if doc is None else doc
    parts = doc.get("parts")
    if not isinstance(parts, dict) or not parts:
        raise KitError(f"finger kit {d}: 'parts' must be a non-empty mapping")
    out = set()
    for name, p in parts.items():
        if not isinstance(p, dict):
            raise KitError(f"finger kit {d}: parts.{name} must be a mapping")
        out.add(_rel(d, p.get("mesh"), f"parts.{name}.mesh"))
        col = p.get("collision", "hull")
        if isinstance(col, list):
            out.update(_rel(d, c, f"parts.{name}.collision") for c in col)
    return sorted(out)


def kit_digest(d) -> str:
    """sha256 over kit.yaml bytes + the sha256 hex of each listed file, sorted by relative path."""
    d = Path(d)
    h = hashlib.sha256((d / KIT_YAML).read_bytes() if (d / KIT_YAML).is_file() else b"")
    for rel in listed_files(d):
        if not (d / rel).is_file():
            raise KitError(f"finger kit {d}: listed file {rel} does not exist")
        h.update(sha256_file(d / rel).encode())
    return h.hexdigest()


def write_source_md(d, title: str | None = None) -> Path:
    """SOURCE.md table (relative path | bytes | sha256) over kit.yaml and the listed files (assets/SOURCE.md format)."""
    d = Path(d)
    rows = [KIT_YAML] + listed_files(d)
    lines = [f"# Finger kit {title or d.name}", "", f"- kit sha256: {kit_digest(d)}", "",
             "| relative path | bytes | sha256 |", "|---|---|---|"]
    lines += [f"| {r} | {(d / r).stat().st_size} | {sha256_file(d / r)} |" for r in rows]
    p = d / "SOURCE.md"
    p.write_text("\n".join(lines) + "\n")
    return p


# --------------------------------------------------------------------------------------------------- loader


def _vec(d, x, n: int, where: str) -> np.ndarray:
    try:
        a = np.asarray(x, float)
    except (TypeError, ValueError):
        raise KitError(f"finger kit {d}: {where} must be {n} numbers, got {x!r}") from None
    if a.shape != (n,) or not np.all(np.isfinite(a)):
        raise KitError(f"finger kit {d}: {where} must be {n} finite numbers, got {x!r}")
    return a


def _unit(d, x, where: str) -> np.ndarray:
    a = _vec(d, x, 3, where)
    n = np.linalg.norm(a)
    if n < 1e-9:
        raise KitError(f"finger kit {d}: {where} must be a non-zero vector")
    return a / n


def _pose(d, x, s: float, where: str) -> np.ndarray:
    try:
        T = np.asarray(x, float)
    except (TypeError, ValueError):
        raise KitError(f"finger kit {d}: {where} must be a 4x4 matrix") from None
    if T.shape != (4, 4) or not np.all(np.isfinite(T)) or not np.allclose(T[3], [0, 0, 0, 1]):
        raise KitError(f"finger kit {d}: {where} must be a 4x4 rigid transform with last row [0, 0, 0, 1]")
    R = T[:3, :3]
    if np.abs(R.T @ R - np.eye(3)).max() > ROT_TOL:
        raise KitError(f"finger kit {d}: {where} rotation is not orthonormal (tol {ROT_TOL})")
    if np.linalg.det(R) < 0:
        raise KitError(f"finger kit {d}: {where} is a reflection (det < 0); list a mirrored STL instead "
                       f"(the code never mirrors)")
    U, _, Vt = np.linalg.svd(R)
    return _T(U @ Vt, T[:3, 3] * s)


def load_kit(path, expect_sha: str | None = None) -> Kit:
    """Load and validate a finger kit directory; lengths -> metres; raises KitError with the offending key."""
    d = Path(path).resolve()
    if not d.is_dir():
        raise KitError(f"finger kit {path}: not a directory")
    doc = _read_yaml(d)
    missing = [k for k in TOP_KEYS if k not in doc]
    if missing:
        raise KitError(f"finger kit {d}: missing top-level keys {missing}")
    unknown = sorted(set(doc) - set(TOP_KEYS))
    if unknown:
        raise KitError(f"finger kit {d}: unknown top-level keys {unknown}")
    if doc["schema"] != SCHEMA:
        raise KitError(f"finger kit {d}: schema must be {SCHEMA!r}, got {doc['schema']!r}")
    if doc["units"] not in UNITS:
        raise KitError(f"finger kit {d}: units must be one of {list(UNITS)}, got {doc['units']!r}")
    s = UNITS[doc["units"]]
    if not isinstance(doc["id"], str) or not doc["id"]:
        raise KitError(f"finger kit {d}: id must be a non-empty string")

    files = listed_files(d, doc)
    for rel in files:
        if not (d / rel).is_file():
            raise KitError(f"finger kit {d}: listed file {rel} does not exist")
    sha = kit_digest(d)
    if expect_sha is not None and sha != expect_sha:
        raise KitError(f"finger kit {d}: sha256 mismatch: kit_digest = {sha}, expected (tooling kit_sha256) "
                       f"{expect_sha}; re-pin the kit or update tooling.finger_adapter.kit_sha256")
    src = d / "SOURCE.md"
    if src.is_file():
        rows = parse_source(src)
        for rel in [KIT_YAML] + files:
            if rel not in rows:
                raise KitError(f"finger kit {d}: SOURCE.md does not list {rel}; re-pin the kit")
            if rows[rel] != ((d / rel).stat().st_size, sha256_file(d / rel)):
                raise KitError(f"finger kit {d}: {rel} differs from SOURCE.md (bytes/sha256); re-pin the kit")

    mats = doc["materials"]
    if not isinstance(mats, dict):
        raise KitError(f"finger kit {d}: materials must be a mapping")
    for mn, m in mats.items():
        rho = m.get("density_kg_m3") if isinstance(m, dict) else None
        if not isinstance(rho, (int, float)) or isinstance(rho, bool) or rho <= 0:
            raise KitError(f"finger kit {d}: materials.{mn}.density_kg_m3 must be a positive number")

    parts = {}
    for name, p in doc["parts"].items():
        for k in ("mesh", "role", "material"):
            if k not in p:
                raise KitError(f"finger kit {d}: parts.{name} is missing '{k}'")
        extra = sorted(set(p) - {"mesh", "role", "material", "collision", "mass_kg"})
        if extra:
            raise KitError(f"finger kit {d}: parts.{name} has unknown keys {extra}")
        if p["role"] not in ROLES:
            raise KitError(f"finger kit {d}: parts.{name}.role must be one of {list(ROLES)}, got {p['role']!r}")
        if p["material"] not in mats:
            raise KitError(f"finger kit {d}: parts.{name}.material {p['material']!r} is not in materials")
        v, f = read_stl(d / p["mesh"])
        v = v * s
        col = p.get("collision", "hull")
        if col == "hull":
            col_paths, col_v = "hull", []
        elif isinstance(col, list) and col:
            col_paths = [d / c for c in col]
            col_v = [read_stl(c)[0] * s for c in col_paths]
        else:
            raise KitError(f"finger kit {d}: parts.{name}.collision must be 'hull' or a non-empty list of STLs")
        vol = _volume(v, f)
        if "mass_kg" in p:
            mk = p["mass_kg"]
            if not isinstance(mk, (int, float)) or isinstance(mk, bool) or mk <= 0:
                raise KitError(f"finger kit {d}: parts.{name}.mass_kg must be a positive number")
            mass, msrc = float(mk), "mass_kg"
        else:
            mass, msrc = float(mats[p["material"]]["density_kg_m3"]) * vol, "density"
        parts[name] = Part(name, d / p["mesh"], p["role"], p["material"], col_paths, v, f, col_v, s, vol, mass, msrc)

    sides_doc = doc["sides"]
    if not isinstance(sides_doc, dict) or set(sides_doc) != {"left", "right"}:
        raise KitError(f"finger kit {d}: sides must have exactly 'left' and 'right', got "
                       f"{sorted(sides_doc) if isinstance(sides_doc, dict) else sides_doc!r}")
    sides, used = {}, set()
    for sn, sd in sides_doc.items():
        w = f"sides.{sn}"
        if not isinstance(sd, dict):
            raise KitError(f"finger kit {d}: {w} must be a mapping")
        for k in ("parts", "kit_T_M", "camera_face", "tip", "tags"):
            if k not in sd:
                raise KitError(f"finger kit {d}: {w} is missing '{k}'")
        extra = sorted(set(sd) - {"parts", "kit_T_M", "camera_face", "tip", "tags"})
        if extra:
            raise KitError(f"finger kit {d}: {w} has unknown keys {extra}")
        pl = sd["parts"]
        if not isinstance(pl, list) or not pl or len(set(pl)) != len(pl):
            raise KitError(f"finger kit {d}: {w}.parts must be a non-empty list without duplicates")
        for pn in pl:
            if pn not in parts:
                raise KitError(f"finger kit {d}: {w}.parts names unknown part {pn!r} (parts: {sorted(parts)})")
        contact = [pn for pn in pl if parts[pn].role == "contact"]
        if len(contact) != 1:
            raise KitError(f"finger kit {d}: {w} must have exactly one role: contact part, got {contact}")
        used.update(pl)
        if sd["camera_face"] not in ("+x", "-x"):
            raise KitError(f"finger kit {d}: {w}.camera_face must be '+x' or '-x', got {sd['camera_face']!r}")
        tags = []
        if not isinstance(sd["tags"], list):
            raise KitError(f"finger kit {d}: {w}.tags must be a list")
        for i, t in enumerate(sd["tags"]):
            wt = f"{w}.tags[{i}]"
            if not isinstance(t, dict) or set(t) != {"id", "centre", "normal", "up", "size"}:
                raise KitError(f"finger kit {d}: {wt} must have exactly id, centre, normal, up, size")
            if not isinstance(t["id"], int) or isinstance(t["id"], bool) or t["id"] < 0:
                raise KitError(f"finger kit {d}: {wt}.id must be a non-negative int")
            if not isinstance(t["size"], (int, float)) or isinstance(t["size"], bool) or t["size"] <= 0:
                raise KitError(f"finger kit {d}: {wt}.size must be a positive number ({doc['units']})")
            n_, u_ = _unit(d, t["normal"], f"{wt}.normal"), _unit(d, t["up"], f"{wt}.up")
            if abs(n_ @ u_) > 1e-3:
                raise KitError(f"finger kit {d}: {wt}.up must be orthogonal to normal (|cos| = {abs(n_ @ u_):.3g})")
            u_ = u_ - (n_ @ u_) * n_
            tags.append(Tag(t["id"], _vec(d, t["centre"], 3, f"{wt}.centre") * s, n_, u_ / np.linalg.norm(u_),
                            float(t["size"]) * s))
        sides[sn] = Side(sn, tuple(pl), contact[0], _pose(d, sd["kit_T_M"], s, f"{w}.kit_T_M"),
                         1.0 if sd["camera_face"] == "+x" else -1.0, _vec(d, sd["tip"], 3, f"{w}.tip") * s, tags)
    ids = [t.id for sd in sides.values() for t in sd.tags]
    if len(ids) != len(set(ids)):
        raise KitError(f"finger kit {d}: tag ids must be unique across sides, got {ids}")
    if sides["left"].camera_face == sides["right"].camera_face:
        raise KitError(f"finger kit {d}: the two sides must have opposite camera_face (one '+x', one '-x')")
    unused = sorted(set(parts) - used)
    if unused:
        raise KitError(f"finger kit {d}: parts {unused} are not used by any side")

    c = doc["contact"]
    if not isinstance(c, dict) or "friction" not in c:
        raise KitError(f"finger kit {d}: contact must be a mapping with 'friction'")
    extra = sorted(set(c) - {"friction", "solref", "solimp"})
    if extra:
        raise KitError(f"finger kit {d}: contact has unknown keys {extra}")
    fr = np.atleast_1d(np.asarray(c["friction"], float))
    if fr.ndim != 1 or not 1 <= len(fr) <= 3 or np.any(fr < 0):
        raise KitError(f"finger kit {d}: contact.friction must be 1-3 non-negative numbers")
    fr = np.r_[fr, [0.005, 0.0001][len(fr) - 1:]][:3] if len(fr) < 3 else fr   # same padding as model._box
    sr, si = c.get("solref"), c.get("solimp")
    if sr is not None and len(sr) != 2:
        raise KitError(f"finger kit {d}: contact.solref must be null or 2 numbers")
    if si is not None and len(si) not in (3, 5):
        raise KitError(f"finger kit {d}: contact.solimp must be null or 3/5 numbers")
    contact = {"friction": [float(x) for x in fr], "solref": None if sr is None else [float(x) for x in sr],
               "solimp": None if si is None else [float(x) for x in si]}

    prov = doc["provenance"]
    if not isinstance(prov, dict):
        raise KitError(f"finger kit {d}: provenance must be a mapping")
    for k, v in prov.items():
        if not isinstance(v, dict) or v.get("status") not in PROV_STATUS or not isinstance(v.get("source"), str):
            raise KitError(f"finger kit {d}: provenance.{k} must be {{status: {'|'.join(PROV_STATUS)}, source: str}}")
    return Kit(doc["id"], d, sha, doc["units"], parts, sides, dict(mats), contact, dict(prov))


# --------------------------------------------------------------------------------------------------- placement


def _side(kit: Kit, side: str) -> Side:
    if side not in kit.sides:
        raise KitError(f"finger kit {kit.id}: unknown side {side!r} (sides: {sorted(kit.sides)})")
    return kit.sides[side]


def place(kit: Kit, side: str, J, L: float, dy: float, yaw_deg: float = 0.0) -> dict:
    """Poses of one kit side in the Franka finger body frame (J = jaw mount 4x4 in that frame).

    Returns {body_T_M, body_T_kit, parts: {name: body_T_part}, contact_part, camera_face, tip (3,),
    tags: [{id, size_m, centre, normal, up, body_T_tag}]}; body_T_tag has x = normal, z = up, origin = centre (on the
    surface; model.py's white tag box goes at centre + 0.5 mm * normal)."""
    sd = _side(kit, side)
    bM = np.asarray(J, float) @ J_T_M(L, dy, yaw_deg)
    bK = bM @ _inv(sd.kit_T_M)
    tags = []
    for t in sd.tags:
        n_, u_ = bK[:3, :3] @ t.normal, bK[:3, :3] @ t.up
        c_ = _apply(bK, t.centre)
        tags.append({"id": t.id, "size_m": t.size_m, "centre": c_, "normal": n_, "up": u_,
                     "body_T_tag": _T(np.column_stack([n_, np.cross(u_, n_), u_]), c_)})
    return {"body_T_M": bM, "body_T_kit": bK, "parts": {pn: bK.copy() for pn in sd.parts},
            "contact_part": sd.contact_part, "camera_face": sd.camera_face, "tip": _apply(bK, sd.tip), "tags": tags}


def auto_dy(kit: Kit, side: str, J, L: float, yaw_deg: float = 0.0) -> float:
    """dy such that the contact part's inner-most vertex sits at finger-body y = 0 (the two contact parts just touch
    at Franka gap 0)."""
    sd = _side(kit, side)
    J = np.asarray(J, float)
    ymin = float(_apply(place(kit, side, J, L, 0.0, yaw_deg)["body_T_kit"], kit.parts[sd.contact_part].vertices)[:, 1].min())
    if abs(J[1, 1]) < 1e-9:
        raise KitError(f"finger kit {kit.id}: J's y axis has no body-y component; dy cannot move the part laterally")
    return -ymin / float(J[1, 1])


# --------------------------------------------------------------------------------------------------- box kit


def _box_mesh(lo, hi) -> tuple[np.ndarray, np.ndarray]:
    lo, hi = np.asarray(lo, float), np.asarray(hi, float)
    v = np.array([[(hi if (i >> k) & 1 else lo)[k] for k in range(3)] for i in range(8)])
    f = np.array([[0, 2, 3], [0, 3, 1], [4, 5, 7], [4, 7, 6], [0, 1, 5], [0, 5, 4],
                  [2, 6, 7], [2, 7, 3], [0, 4, 6], [0, 6, 2], [1, 3, 7], [1, 7, 5]])   # outward winding
    return v, f


def make_box_kit(d, pad_size_m=(0.02, 0.008, 0.03), pad_offset_m: float = 0.008, tag_y_extra_m: float = 0.0312,
                 tag_z_from_pad_m: float = -0.108, tag_size_m: float = 0.016, *, friction=(1.0, 0.005, 0.0001),
                 kit_T_M_left=None, kit_T_M_right=None, units: str = "m", ascii: bool = True,
                 density_kg_m3: float = 1000.0, kit_id: str = "box_equivalent") -> Path:
    """Write a kit that reproduces model.py's box pads + tags (defaults = model.py constants and the tooling YAMLs).

    M frame of each side (body axes): origin at the pad's proximal face on the gripper centre plane, i.e. the pad
    (px, pt, pl) occupies x in [-px/2, px/2], y in [b/2, b/2 + pt], z in [0, pl]; b = pad_offset_m.
      - pad centre in M = (0, b/2 + pt/2, pl/2)
      - tip in M = (0, b/2, pl) (inner face, distal edge): kit TCP = box TCP + pl/2 along z
      - tag centre in M = (s (px/2), b/2 + tag_size/2 + tag_y_extra, pl/2 + tag_z_from_pad), normal (s, 0, 0),
        up (0, 0, 1), s = +1 for side left (camera_face +x), -1 for right. Box mode puts the white tag box at
        centre + 0.5 mm * normal and the ink box 0.6 mm further out (model.py step 3).
    Placement that reproduces box mode (see ``box_equivalent_placement``), with J = (0, y_c, z_base) from the mesh:
        dy = -y_c,   L = tip_z + extension_m - pl - z_base,   yaw = 0
    (``lateral_offset_m: auto`` gives dy = -y_c - b/2 instead: auto closes the 8 mm placeholder pad offset.)
    kit_T_M_left/right (4x4, metres): put the side's meshes, tip and tags at kit_T_M . (M-frame values) in kit coords
    (the M-frame values above equal the kit coords when kit_T_M = I, the default), so the placed
    geometry is the same for any rigid kit_T_M (exercises the inv(kit_T_M) path). Both sides share one pad STL when
    their kit_T_M are equal, else each side gets its own. Mass: density 1000 = MuJoCo's default geom density (box mode
    pads have no explicit mass). ascii=True (default) keeps float64 vertices (17 digits); binary STL is float32
    (~1e-9 m rounding at 30 mm), too coarse for a 1e-9 equivalence test."""
    d = Path(d)
    d.mkdir(parents=True, exist_ok=True)
    s = 1.0 / UNITS[units]
    px, pt, pl = (float(x) for x in pad_size_m)
    b = float(pad_offset_m)
    Ts = {"left": _T() if kit_T_M_left is None else np.asarray(kit_T_M_left, float),
          "right": _T() if kit_T_M_right is None else np.asarray(kit_T_M_right, float)}
    v, f = _box_mesh([-px / 2, b / 2, 0.0], [px / 2, b / 2 + pt, pl])
    same = np.array_equal(Ts["left"], Ts["right"])
    parts, sides = {}, {}
    for sn, cf in (("left", 1.0), ("right", -1.0)):
        pn = "pad" if same else f"pad_{sn}"
        if pn not in parts:
            write_stl(d / f"{pn}.stl", _apply(Ts[sn], v) * s, f, ascii=ascii)
            parts[pn] = {"mesh": f"{pn}.stl", "role": "contact", "material": "pad", "collision": "hull"}
        Tk = Ts[sn].copy()
        Tk[:3, 3] *= s
        Rk = Ts[sn][:3, :3]
        tip = _apply(Ts[sn], [0.0, b / 2, pl]) * s
        centre = _apply(Ts[sn], [cf * px / 2, b / 2 + tag_size_m / 2 + tag_y_extra_m, pl / 2 + tag_z_from_pad_m]) * s
        sides[sn] = {"parts": [pn], "kit_T_M": Tk.tolist(), "camera_face": "+x" if cf > 0 else "-x",
                     "tip": [float(x) for x in tip],
                     "tags": [{"id": 0 if sn == "left" else 1, "centre": [float(x) for x in centre],
                               "normal": [float(x) for x in Rk @ [cf, 0.0, 0.0]],
                               "up": [float(x) for x in Rk @ [0.0, 0.0, 1.0]], "size": tag_size_m * s}]}
    doc = {"schema": SCHEMA, "id": kit_id, "units": units, "parts": parts,
           "materials": {"pad": {"density_kg_m3": float(density_kg_m3)}}, "sides": sides,
           "contact": {"friction": [float(x) for x in np.atleast_1d(friction)], "solref": None, "solimp": None},
           "provenance": {"geometry": {"status": "placeholder", "source": "make_box_kit: model.py box pads"},
                          "tags": {"status": "derived", "source": "model.py TAG_Y_EXTRA_M / TAG_Z_FROM_PAD_M"}}}
    (d / KIT_YAML).write_text(yaml.safe_dump(doc, sort_keys=False, default_flow_style=None))
    write_source_md(d, kit_id)
    return d


def box_equivalent_placement(J, tip_z_m: float, extension_m: float, pad_size_m) -> dict:
    """(L, dy, yaw) that places a ``make_box_kit`` side where model.py puts the box pad: M origin at finger-body
    (0, 0, tip_z + extension - pl)."""
    J = np.asarray(J, float)
    pl = float(pad_size_m[2])
    q = _apply(_inv(J), [0.0, 0.0, tip_z_m + extension_m - pl])     # = (0, dy, L) in J
    if abs(q[0]) > 1e-12:
        raise KitError(f"J has an x offset ({q[0]:.3g} m in J): no (L, dy) reproduces the box pad")
    return {"L": float(q[2]), "dy": float(q[1]), "yaw_deg": 0.0}
