"""Renderer (A7, I-RENDER): the images each sensor puts on the wire, rendered from the full sim ``qpos``.

Why: Stage 3 closes the loop through images. The render sender (``franka_sim.senders.render_sender``) and the
camera-view check (``camera_check.py``) need one place that turns a sim state into exactly what the wire carries:
- iPhone (v1): main + ultrawide, BGR uint8, 960x720, at the station sensor K (``sensor.streams.*.K``);
- rig (v2): the rig's *virtual-iPhone* images. The rig remaps both physical cameras to the same virtual K at
  960x720, so the plain render at the station K is the wire image (``rig1_wifi`` carries the virtual K).
  ``rig=<rig yaml>`` (fidelity mode) also runs the rig's own resampling: virtual -> physical
  (``RigConfig.inverse_remap``) -> virtual (``RigConfig.remap``), as ``obs_equivalence.rig_images`` does through
  ``RigFrameSource``. That reproduces the remap blur and any uncovered border (e.g. the B0179 fisheye UW).

The cameras are the model's ``cam_main`` / ``cam_uw`` (placed from ``C_T_TCP``; intrinsics set in
``model._set_cam_intrinsics`` with the +0.5 px OpenCV/MuJoCo principal-point fix). Other intrinsics (e.g. the
1920x1440 / 640x480 training K) need a second ``SimModel`` built from a station with ``sensor.streams.<s>.K`` /
``width_px`` / ``height_px`` overridden; ``sizes`` only allows a uniform rescale of the model camera (K is then
scaled with the half-pixel convention c' = s (c + 0.5) - 0.5 and reported in ``Renderer.K``).

Rendering never steps physics: a private ``MjData`` and model copy, ``mj_kinematics`` + ``mj_camlight`` only.
Lighting and colours are keyword arguments (``DEFAULT_LIGHTING``, ``colors``); there is no station/run group for
them yet (``run/sim/render.yaml`` is Track B; see the A7 report).

Headless: set ``MUJOCO_GL=egl`` before importing mujoco, and call ``close()`` explicitly.

    r = Renderer(sm, station)                        # streams main + ultrawide at the sensor K
    imgs = r.render(qpos)                            # {"main": (720, 960, 3) BGR uint8, "ultrawide": ...}
    seg = r.segmentation(qpos)                       # {"main": (720, 960) int32 geom id, -1 = background}
    r.geom_groups()                                  # {"fingers": [...], "tags": [...], "hand_body": [...], ...}
    r.close()
"""

from __future__ import annotations

import copy
from pathlib import Path

import numpy as np

import mujoco

CAMS = {"main": "cam_main", "ultrawide": "cam_uw"}
RIG_NAMES = {"main": "main", "ultrawide": "ultrawide"}       # stream -> rig virtual camera name
# Documented defaults (one place). Headlight = MuJoCo's camera-attached light; the scene's directional light (straight
# down, diffuse 0.7, from model.build_model) adds to it. Chosen once on physical grounds before any comparison with
# real frames (not tuned to the camera-check numbers): total diffuse+ambient ~1.25 x the geom colour, so the
# 0.6/0.5/0.4 table does not saturate; no hard shadows, since one directional light straight down casts a sharp arm
# shadow that diffuse lab ceiling light does not.
DEFAULT_LIGHTING = {
    "headlight_ambient": [0.25, 0.25, 0.25],
    "headlight_diffuse": [0.3, 0.3, 0.3],
    "headlight_specular": [0.1, 0.1, 0.1],
    "shadows": False,            # mjRND_SHADOW
    "skybox": False,             # mjRND_SKYBOX (the model defines no skybox texture: the background is black)
}
# Cheap appearance defaults (same principle: chosen from the real frames' look, not fitted to the camera-check numbers):
# the real table is a mid grey (not tan), the eraser yellow-green, and past the table edge the real frames show a grey floor
# / wall, not black. The background (model.py "backdrop": a visual-only floor plane below the table) takes DEFAULT_COLORS["backdrop"].
DEFAULT_COLORS = {"table": [0.30, 0.30, 0.29, 1.0], "eraser": [0.55, 0.66, 0.04, 1.0],
                  "backdrop": [0.20, 0.20, 0.21, 1.0]}
TAG_GEOMS = ("tag_l", "tag_r", "tag_l_ink", "tag_r_ink")


def scale_K(K, s: float) -> np.ndarray:
    """K for an image scaled by s (half-pixel convention, as camera_model.scale_intrinsics)."""
    K = np.array(K, float)
    out = K.copy()
    out[0, 0], out[1, 1] = K[0, 0] * s, K[1, 1] * s
    out[0, 2], out[1, 2] = s * (K[0, 2] + 0.5) - 0.5, s * (K[1, 2] + 0.5) - 0.5
    return out


class Renderer:
    def __init__(self, sm, station, *, streams=("main", "ultrawide"), sizes: dict | None = None, rig: str | None = None,
                 lighting: dict | None = None, colors: dict | None = None):
        """sm: model.SimModel; station: the Station it was built from (sensor K). sizes: {stream: (w, h)} (uniform
        rescale of the model camera only). rig: a sensor_rig rig yaml (fidelity mode). lighting: overrides of
        DEFAULT_LIGHTING. colors: {geom name or name prefix ending in '*': rgba}."""
        self.sm = sm
        self.streams = tuple(streams)
        for s in self.streams:
            if s not in CAMS:
                raise ValueError(f"unknown stream {s!r} (known: {', '.join(CAMS)})")
        m = self.m = copy.deepcopy(sm.model)          # private copy: lighting/colours/offscreen size never leak
        self.d = mujoco.MjData(m)
        self.cam_id = {s: mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_CAMERA, CAMS[s]) for s in self.streams}
        self.size, self.K = {}, {}
        for s in self.streams:
            st = station.get(f"sensor.streams.{s}")
            K0 = np.array(st["K"], float)
            res = tuple(int(v) for v in m.cam_resolution[self.cam_id[s]])
            if res != (int(st["width_px"]), int(st["height_px"])):
                raise ValueError(f"model camera {CAMS[s]} is {res}, station stream {s} is "
                                 f"{st['width_px']}x{st['height_px']}: build the model from this station")
            w, h = (sizes or {}).get(s, res)
            sx, sy = w / res[0], h / res[1]
            if abs(sx - sy) > 1e-9:
                raise ValueError(f"sizes[{s}] {w}x{h} is not a uniform scale of {res[0]}x{res[1]}")
            self.size[s] = (int(w), int(h))
            self.K[s] = scale_K(K0, sx) if sx != 1.0 else K0
        W = max(w for w, _ in self.size.values())
        H = max(h for _, h in self.size.values())
        m.vis.global_.offwidth = max(int(m.vis.global_.offwidth), W)
        m.vis.global_.offheight = max(int(m.vis.global_.offheight), H)
        self.lighting = {**DEFAULT_LIGHTING, **(lighting or {})}
        lt = self.lighting
        m.vis.headlight.ambient[:] = lt["headlight_ambient"]
        m.vis.headlight.diffuse[:] = lt["headlight_diffuse"]
        m.vis.headlight.specular[:] = lt["headlight_specular"]
        self.colors = {**DEFAULT_COLORS, **(colors or {})}
        for key, rgba in self.colors.items():
            for g in range(m.ngeom):
                nm = m.geom(g).name
                if (key.endswith("*") and nm.startswith(key[:-1])) or nm == key:
                    m.geom_rgba[g] = rgba
        self._renderers: dict = {}
        for s in self.streams:
            if self.size[s] not in self._renderers:
                w, h = self.size[s]
                self._renderers[self.size[s]] = mujoco.Renderer(m, h, w)
        self.opt = mujoco.MjvOption()
        self._groups = self._build_groups()
        self._key_qpos = np.array(m.key_qpos[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_KEY, "home")], float)
        # -- rig fidelity mode
        self.rig_path, self.rig, self._remaps, self.coverage = None, None, {}, None
        if rig is not None:
            from sensor_rig.camera_model import VirtualPinhole
            from sensor_rig.rig_config import load_rig

            self.rig_path = str(rig)
            self.rig = load_rig(rig)
            self.coverage = self.rig.coverage_report()
            for s in self.streams:
                name = RIG_NAMES[s]
                virt = VirtualPinhole.from_K(self.K[s], self.size[s])
                fwd = self.rig.remap(name, self.size[s])
                vK = self.rig.virtual_camera(name, self.size[s]).K
                if np.abs(np.asarray(vK) - self.K[s]).max() > 1e-3:
                    raise ValueError(f"rig {Path(rig).name}: virtual {name} K at {self.size[s]} {np.round(vK, 4).tolist()} "
                                     f"!= the station K {np.round(self.K[s], 4).tolist()}")
                self._remaps[s] = (self.rig.inverse_remap(name, virt), fwd)
            self.coverage = {**self.coverage, **{f"{s}_deploy": float(self._remaps[s][1].coverage) for s in self.streams}}

    # ------------------------------------------------------------------------------------------------ state
    def _set(self, qpos_full) -> None:
        q = np.asarray(qpos_full, float).ravel()
        nq = self.m.nq
        if q.size == nq:
            self.d.qpos[:] = q
        elif q.size in (7, 9):                     # convenience for tests: arm (+ fingers), rest from the home key
            self.d.qpos[:] = self._key_qpos
            self.d.qpos[:q.size] = q
        else:
            raise ValueError(f"qpos has {q.size} values; the model has nq = {nq} (I-SIMSTATE qpos is the full vector)")
        mujoco.mj_kinematics(self.m, self.d)
        mujoco.mj_camlight(self.m, self.d)

    def _flags(self, r) -> None:
        r.scene.flags[mujoco.mjtRndFlag.mjRND_SHADOW] = bool(self.lighting["shadows"])
        r.scene.flags[mujoco.mjtRndFlag.mjRND_SKYBOX] = bool(self.lighting["skybox"])

    # ------------------------------------------------------------------------------------------------ API
    def render(self, qpos_full) -> dict:
        """{stream: BGR uint8 (H, W, 3)} at the stream size (rig mode: after the rig's inverse + forward remap)."""
        self._set(qpos_full)
        out = {}
        for s in self.streams:
            r = self._renderers[self.size[s]]
            r.update_scene(self.d, camera=self.cam_id[s], scene_option=self.opt)
            self._flags(r)
            bgr = np.ascontiguousarray(r.render()[:, :, ::-1])
            if self.rig is not None:
                inv, fwd = self._remaps[s]
                bgr = fwd.apply(np.ascontiguousarray(inv.apply(bgr)))
            out[s] = bgr
        return out

    def segmentation(self, qpos_full) -> dict:
        """{stream: int32 (H, W) geom id per pixel, -1 = background or a non-geom object}. Always the plain virtual
        camera (no rig remap)."""
        self._set(qpos_full)
        out = {}
        for s in self.streams:
            r = self._renderers[self.size[s]]
            r.enable_segmentation_rendering()
            try:
                r.update_scene(self.d, camera=self.cam_id[s], scene_option=self.opt)
                seg = r.render()
            finally:
                r.disable_segmentation_rendering()
            ids = seg[..., 0].astype(np.int32)
            ids[seg[..., 1] != int(mujoco.mjtObj.mjOBJ_GEOM)] = -1
            out[s] = ids
        return out

    def _build_groups(self) -> dict:
        m, nm = self.m, self.sm.names
        bid = lambda n: mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, n)  # noqa: E731
        fingers_b = {bid(nm["bodies"]["finger_l"]), bid(nm["bodies"]["finger_r"])}
        hand_b = bid(nm["bodies"]["hand"])
        tool_names = set(nm["geoms"].get("tooling", []))
        scene_names = {nm["geoms"].get("table"), nm["geoms"].get("object")} - {None}
        g = {"fingers": [], "tags": [], "hand_body": [], "tooling": [], "scene": [], "arm": []}
        for i in range(m.ngeom):
            name, b = m.geom(i).name, int(m.geom_bodyid[i])
            if name in TAG_GEOMS:
                g["tags"].append(i)
            elif b in fingers_b:
                g["fingers"].append(i)
            elif b == hand_b:
                g["hand_body"].append(i)
            elif name in tool_names:
                g["tooling"].append(i)
            elif name in scene_names:
                g["scene"].append(i)
            elif b != 0:
                g["arm"].append(i)
            else:
                g["scene"].append(i)               # world-body geoms (none today)
        return g

    def geom_groups(self) -> dict:
        """{"fingers" (incl. pads), "tags", "hand_body", "tooling" (housing + cradle), "scene" (table, eraser), "arm"}."""
        return {k: list(v) for k, v in self._groups.items()}

    def mask(self, seg: np.ndarray, groups) -> np.ndarray:
        """bool mask of the pixels whose geom is in any of ``groups``."""
        ids = sorted({i for gname in groups for i in self._groups[gname]})
        return np.isin(seg, ids)

    def info(self) -> dict:
        return {"streams": list(self.streams), "size": {s: list(v) for s, v in self.size.items()},
                "K": {s: np.round(self.K[s], 6).tolist() for s in self.streams}, "rig": self.rig_path,
                "coverage": self.coverage, "lighting": self.lighting, "colors": self.colors}

    def close(self) -> None:
        for r in self._renderers.values():
            try:
                r.close()
            except Exception:  # noqa: BLE001
                pass
        self._renderers = {}

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
