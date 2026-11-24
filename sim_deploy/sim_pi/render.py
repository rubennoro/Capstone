"""Offscreen MuJoCo rendering of arbitrary pinhole views (one GL context, one MjrContext, one MjvScene).

A ``View`` is a pinhole in OpenCV axes (x right, y down, z forward) given by its pose in the world and a pixel
grid: ``u = f * x / z + cx``, ``v = f * y / z + cy`` for ``u`` in [0, w), ``v`` in [0, h) (pixel centres at integer
positions, as camera_model.py). Its asymmetric GL frustum is set directly on the scene's ``mjvGLCamera`` (both stereo
eyes), so any sub-rectangle of a cube face, or an off-centre pinhole, renders exactly. ``render_views`` updates the
scene once per state and renders every view from it (shadows are recomputed per view by ``mjr_render``).

Lighting (``prepare_spec``): the model's own lights (made directional: no spot cone), or one directional light from
above, plus a view-independent ambient term (the headlight's ambient only: its diffuse/specular follow each view's
axis and would make seams between cube faces). The directional shadow map is sized to cover the whole workspace
(``finish_model``), so no lit/unlit boundary shows on the table.

Headless: ``MUJOCO_GL=egl`` must be set before ``mujoco`` is imported (``__main__`` does it).
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field

#: The NVIDIA EGL vendor file. A shell may point glvnd elsewhere (here: Anaconda's __EGL_VENDOR_LIBRARY_DIRS, Mesa
#: only), and EGL then renders on the Intel iGPU, about 3x slower. SIM_PI_EGL_VENDOR=default keeps the shell's choice;
#: any other value is used as the vendor file.
NVIDIA_EGL_VENDOR = "/usr/share/glvnd/egl_vendor.d/10_nvidia.json"


def _select_egl_vendor() -> None:
    choice = os.environ.get("SIM_PI_EGL_VENDOR", NVIDIA_EGL_VENDOR)
    if choice == "default" or "__EGL_VENDOR_LIBRARY_FILENAMES" in os.environ:
        return
    if os.path.isfile(choice):
        os.environ["__EGL_VENDOR_LIBRARY_FILENAMES"] = choice


os.environ.setdefault("MUJOCO_GL", "egl")
_select_egl_vendor()

import mujoco  # noqa: E402
import numpy as np  # noqa: E402

#: Geom groups drawn (DESIGN.md section 2: 0 scene, 1 arm + hand, 2 rig parts, 3 fingers + adapter + tags;
#: 4 unused, 5 collision-only).
VISIBLE_GROUPS = (0, 1, 2, 3)
NEAR_M = 0.002          # near clip plane (the lens front sits ~0.8-2.5 mm in front of the optical centres, E2)
FAR_M = 20.0


@dataclass(frozen=True)
class Lighting:
    """The renderer's light set (requirement: neutral, from above, shadows on, no spotlight cone)."""
    direction: tuple[float, float, float] = (0.15, 0.10, -1.0)   # world, from above (slight tilt: soft-edged shadows
                                                                 # fall a little to the side, as under a lab ceiling)
    diffuse: tuple[float, float, float] = (0.62, 0.62, 0.60)
    specular: tuple[float, float, float] = (0.10, 0.10, 0.10)
    ambient: tuple[float, float, float] = (0.36, 0.36, 0.37)     # view-independent (headlight ambient only)
    shadows: bool = True
    shadow_size: int = 4096
    shadow_extent_m: float = 3.0      # side of the square the directional light's shadow map covers (whole table)
    centre: tuple[float, float, float] = (0.45, 0.0, 0.0)        # world point the light (and its shadow map) centre on
    offsamples: int = 4               # MSAA samples of the offscreen buffer (a real lens is never aliased)


def prepare_spec(spec: "mujoco.MjSpec", lighting: Lighting, max_w: int, max_h: int, *,
                 keep_model_lights: bool = True) -> None:
    """Rendering-only edits of a model spec before it is compiled (qpos layout and physics are untouched).

    ``keep_model_lights`` (default): the model's own lights stay (E6's scene has a key light with shadows + a fill,
    both directional); any spot light is made directional (no cone). Otherwise they are replaced by ``lighting``'s one
    overhead light. Either way the headlight keeps only an ambient term (its diffuse/specular would follow each
    cube face's axis and draw seams), and the offscreen buffer is sized for the largest view."""
    hl = spec.visual.headlight
    model_ambient = np.asarray(hl.ambient, float) + 0.5 * np.asarray(hl.diffuse, float)
    lights = list(spec.lights)
    if keep_model_lights and lights:
        for light in lights:
            light.directional = True
        ambient = model_ambient
    else:
        for light in lights:
            light.delete()
        d = np.asarray(lighting.direction, float)
        d = d / np.linalg.norm(d)
        c = np.asarray(lighting.centre, float)
        lt = spec.worldbody.add_light()
        lt.name = "sim_pi_overhead"
        lt.directional = True
        lt.castshadow = bool(lighting.shadows)
        lt.dir = d
        lt.pos = c - 2.0 * d                  # the shadow map of a directional light is centred on its position
        lt.diffuse = list(lighting.diffuse)
        lt.specular = list(lighting.specular)
        lt.ambient = [0.0, 0.0, 0.0]
        ambient = np.asarray(lighting.ambient, float)
    g = spec.visual.global_
    g.offwidth = max(int(g.offwidth), int(max_w))
    g.offheight = max(int(g.offheight), int(max_h))
    q = spec.visual.quality
    q.shadowsize = int(lighting.shadow_size)
    q.offsamples = int(lighting.offsamples)
    hl.active = 1
    hl.ambient = [float(v) for v in np.clip(ambient, 0, 1)]
    hl.diffuse = [0.0, 0.0, 0.0]
    hl.specular = [0.0, 0.0, 0.0]


def finish_model(model: "mujoco.MjModel", lighting: Lighting) -> None:
    """Settings that live on the compiled model only: the directional light's shadow-map extent."""
    extent = float(model.stat.extent)
    if extent > 0:
        model.vis.map.shadowscale = float(lighting.shadow_extent_m) / extent
    model.vis.map.znear = NEAR_M / max(extent, 1e-9)
    model.vis.map.zfar = FAR_M / max(extent, 1e-9)


@dataclass
class View:
    """One pinhole view, OpenCV axes. ``R_cam_view`` rotates view axes into the carrying camera's axes (identity for
    the camera itself, a cube-face rotation for the ultrawide). ``hide`` names geoms not drawn in this view (the
    camera's own board and lens barrel)."""
    name: str
    w: int
    h: int
    f: float
    cx: float
    cy: float
    R_cam_view: np.ndarray = field(default_factory=lambda: np.eye(3))
    hide: tuple[int, ...] = ()

    def frustum(self, near: float) -> tuple[float, float, float, float]:
        """(left, right, bottom, top) at the near plane, GL convention (y up)."""
        left = near * (-0.5 - self.cx) / self.f
        right = near * (self.w - 0.5 - self.cx) / self.f
        top = near * (self.cy + 0.5) / self.f
        bottom = -near * (self.h - 0.5 - self.cy) / self.f
        return left, right, bottom, top


class GLRenderer:
    """One EGL context sized for the largest view; renders Views of one model from one MjData."""

    def __init__(self, model: "mujoco.MjModel", max_w: int, max_h: int):
        self.m = model
        self.max_w, self.max_h = int(max_w), int(max_h)
        if model.vis.global_.offwidth < self.max_w or model.vis.global_.offheight < self.max_h:
            raise ValueError(f"model offscreen buffer {model.vis.global_.offwidth}x{model.vis.global_.offheight} is "
                             f"smaller than the largest view {self.max_w}x{self.max_h} (prepare_spec sizes it)")
        self.gl = mujoco.GLContext(self.max_w, self.max_h)
        self.gl.make_current()
        try:
            from OpenGL import GL
            self.gl_renderer = GL.glGetString(GL.GL_RENDERER).decode(errors="replace")
        except Exception:  # noqa: BLE001
            self.gl_renderer = "unknown"
        self.con = mujoco.MjrContext(model, mujoco.mjtFontScale.mjFONTSCALE_100)
        mujoco.mjr_setBuffer(mujoco.mjtFramebuffer.mjFB_OFFSCREEN, self.con)
        self.scn = mujoco.MjvScene(model, maxgeom=20000)
        self.opt = mujoco.MjvOption()
        for i in range(len(self.opt.geomgroup)):
            self.opt.geomgroup[i] = 1 if i in VISIBLE_GROUPS else 0
        for i in range(len(self.opt.sitegroup)):       # sites (flange, tcp, ...) are frames, not things a camera sees
            self.opt.sitegroup[i] = 0
        self.pert = mujoco.MjvPerturb()
        self.cam = mujoco.MjvCamera()
        self.cam.type = mujoco.mjtCamera.mjCAMERA_FREE
        self._bufs: dict[tuple[int, int], np.ndarray] = {}

    def set_shadows(self, on: bool) -> None:
        self.scn.flags[mujoco.mjtRndFlag.mjRND_SHADOW] = bool(on)

    def update(self, data: "mujoco.MjData") -> None:
        """Rebuild the scene's geoms from ``data`` (kinematics must be current). Once per state."""
        mujoco.mjv_updateScene(self.m, data, self.opt, self.pert, self.cam, mujoco.mjtCatBit.mjCAT_ALL, self.scn)
        self._geom_objid = np.array([self.scn.geoms[i].objid if self.scn.geoms[i].objtype == mujoco.mjtObj.mjOBJ_GEOM
                                     else -1 for i in range(self.scn.ngeom)], int)

    def render(self, view: View, R_world_cam: np.ndarray, p_world_cam: np.ndarray) -> np.ndarray:
        """RGB uint8 (h, w, 3), top row first (a copy)."""
        return np.flipud(self.render_raw(view, R_world_cam, p_world_cam)).copy()

    def render_raw(self, view: View, R_world_cam: np.ndarray, p_world_cam: np.ndarray) -> np.ndarray:
        """RGB uint8 (h, w, 3) of ``view`` carried by a camera at (R_world_cam, p_world_cam), OpenCV axes, as GL reads
        it back: BOTTOM row first, in a buffer reused by the next call of the same size. ``update`` must have been
        called for the current state."""
        if view.w > self.max_w or view.h > self.max_h:
            raise ValueError(f"view {view.name} {view.w}x{view.h} exceeds the GL buffer {self.max_w}x{self.max_h}")
        R = np.asarray(R_world_cam, float) @ view.R_cam_view
        fwd, up = R[:, 2], -R[:, 1]
        left, right, bottom, top = view.frustum(NEAR_M)
        for eye in range(2):
            c = self.scn.camera[eye]
            c.pos[:] = p_world_cam
            c.forward[:] = fwd
            c.up[:] = up
            c.frustum_center = 0.5 * (left + right)
            c.frustum_width = 0.5 * (right - left)
            c.frustum_bottom = bottom
            c.frustum_top = top
            c.frustum_near = NEAR_M
            c.frustum_far = FAR_M
            c.orthographic = 0
        hidden = []
        for gid in view.hide:
            for i in np.flatnonzero(self._geom_objid == gid):
                hidden.append((int(i), float(self.scn.geoms[i].rgba[3])))
                self.scn.geoms[i].rgba[3] = 0.0
        rect = mujoco.MjrRect(0, 0, view.w, view.h)
        mujoco.mjr_render(rect, self.scn, self.con)
        for i, a in hidden:
            self.scn.geoms[i].rgba[3] = a
        buf = self._bufs.get((view.w, view.h))
        if buf is None:
            buf = self._bufs[(view.w, view.h)] = np.empty((view.h, view.w, 3), np.uint8)
        mujoco.mjr_readPixels(buf, None, rect, self.con)
        return buf

    def close(self) -> None:
        if getattr(self, "con", None) is not None:
            try:
                self.con.free()
            except Exception:  # noqa: BLE001
                pass
            self.con = None
        if getattr(self, "gl", None) is not None:
            try:
                self.gl.free()
            except Exception:  # noqa: BLE001
                pass
            self.gl = None
