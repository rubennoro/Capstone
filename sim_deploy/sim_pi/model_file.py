"""The shared model file (DESIGN.md section 2): ``<build dir>/scene.xml`` + ``meshes/`` + ``model.json``.

``model.json`` is written last by the builder (E6), so its presence means the model is complete; ``wait_for`` polls
for it. The sha is recomputed here with the builder's method (``model.json sha256_method``: sha256 over the lines
``"<file sha256>  <relpath>\\n"`` of scene.xml and meshes/*, sorted by relpath) and must equal ``model_sha256``,
which the state export also carries.

``load_render_model`` compiles a private copy for rendering: lights and offscreen size per render.prepare_spec; the
kinematic tree, qpos layout and names are unchanged (checked: same nq and joint addresses as the file).
"""

from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass
from pathlib import Path

import mujoco
import numpy as np

from sim_deploy.sim_pi.render import Lighting, finish_model, prepare_spec

CAMERAS = {"main": "cam_imx519", "ultrawide": "cam_imx219"}
#: Each camera's own board + lens barrel (E2's rig geoms): not drawn in that camera's views (the optical centre
#: sits inside the barrel; a real lens never images its own barrel).
OWN_BOARD = {"main": "rig_imx519_board", "ultrawide": "rig_imx219_board"}


def model_sha256(build_dir: Path) -> str:
    build_dir = Path(build_dir)
    files = ["scene.xml"]
    if (build_dir / "meshes").is_dir():
        files += sorted(f"meshes/{p.name}" for p in (build_dir / "meshes").iterdir() if p.is_file())
    per = {f: hashlib.sha256((build_dir / f).read_bytes()).hexdigest() for f in files}
    text = "".join(f"{per[f]}  {f}\n" for f in sorted(per, key=lambda s: s.encode()))
    return hashlib.sha256(text.encode()).hexdigest()


def wait_for(scene_xml: Path, timeout_s: float, log=print) -> None:
    """Poll for ``model.json`` next to ``scene_xml`` (the builder writes it last)."""
    mj = Path(scene_xml).with_name("model.json")
    t_end = time.monotonic() + timeout_s
    said = False
    while not mj.is_file():
        if time.monotonic() > t_end:
            raise FileNotFoundError(f"{mj}: not there after {timeout_s:.0f} s (build the model: python -m "
                                    "sim_deploy.model build --station sim_deploy/stations/sim_rig.yaml)")
        if not said:
            log(f"[sim_pi] waiting for {mj} ...")
            said = True
        time.sleep(0.5)


@dataclass
class RenderModel:
    scene_xml: Path
    doc: dict
    sha256: str
    model: "mujoco.MjModel"
    data: "mujoco.MjData"
    cam_id: dict            # role -> camera id
    own_board: dict         # role -> geom id or None
    home_qpos: np.ndarray

    def set_qpos(self, q: np.ndarray) -> None:
        self.data.qpos[:] = q
        mujoco.mj_kinematics(self.model, self.data)
        mujoco.mj_camlight(self.model, self.data)

    def camera_pose(self, role: str) -> tuple[np.ndarray, np.ndarray]:
        """(R_world_cam, p_world_cam) in OpenCV axes (MuJoCo looks along -z with +y up: R_cv = R_mj diag(1,-1,-1))."""
        c = self.cam_id[role]
        return self.data.cam_xmat[c].reshape(3, 3) @ np.diag([1.0, -1.0, -1.0]), self.data.cam_xpos[c].copy()


def read_doc(scene_xml: Path) -> tuple[dict, str]:
    scene_xml = Path(scene_xml)
    doc = json.loads(scene_xml.with_name("model.json").read_text())
    sha = model_sha256(scene_xml.parent)
    if doc.get("model_sha256") != sha:
        raise ValueError(f"{scene_xml.parent}: files hash to {sha[:16]}..., model.json says "
                         f"{str(doc.get('model_sha256'))[:16]}... (rebuild, or a build is in progress)")
    return doc, sha


def load_render_model(scene_xml: Path, lighting: Lighting, max_w: int, max_h: int, *,
                      keep_model_lights: bool = True) -> RenderModel:
    scene_xml = Path(scene_xml)
    doc, sha = read_doc(scene_xml)
    ref = mujoco.MjModel.from_xml_path(str(scene_xml))
    spec = mujoco.MjSpec.from_file(str(scene_xml))
    prepare_spec(spec, lighting, max_w, max_h, keep_model_lights=keep_model_lights)
    m = spec.compile()
    finish_model(m, lighting)
    if m.nq != ref.nq or not np.array_equal(m.jnt_qposadr, ref.jnt_qposadr):
        raise RuntimeError("render model's qpos layout differs from the file's (prepare_spec must not change it)")
    if "nq" in doc and int(doc["nq"]) != m.nq:
        raise ValueError(f"model.json nq {doc['nq']} != the compiled model's {m.nq}")
    cam_id = {}
    for role, name in CAMERAS.items():
        cid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_CAMERA, name)
        if cid < 0:
            raise ValueError(f"{scene_xml}: no camera {name!r} (DESIGN.md section 2)")
        cam_id[role] = cid
    own = {}
    for role, gname in OWN_BOARD.items():
        gid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, gname)
        own[role] = gid if gid >= 0 else None
    home = doc.get("home", {}).get("qpos")
    if home is None:
        k = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_KEY, "home")
        home = m.key_qpos[k] if k >= 0 else m.qpos0
    d = mujoco.MjData(m)
    rm = RenderModel(scene_xml, doc, sha, m, d, cam_id, own, np.asarray(home, float))
    rm.set_qpos(rm.home_qpos)
    return rm
