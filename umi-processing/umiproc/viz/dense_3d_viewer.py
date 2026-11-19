"""EXPERIMENTAL TRIAL (may be removed if it does not work out): the HTML viewer of :mod:`umiproc.viz.dense_3d`.

One self-contained page: three.js (pinned, from jsdelivr; needs internet once) draws the dense cloud as coloured
points, the mapping clip's camera path (colour = time), the keyframes as small frustums, the table tag as a square
and a 10 cm grid in the tag's plane. The scene is W turned so that gravity (the map export's ``up_W``, else the tag's
normal) is +z, with the tag's centre at the origin. The points are embedded as base64: uint16 positions quantised
over the cloud's box (< 0.1 mm steps at these sizes) and uint8 colours. ``#view=top|front|side|persp`` in the URL
picks the first view (for headless screenshots); ``&size=N`` the point size in pixels.
"""

from __future__ import annotations

import base64
import json
from pathlib import Path

import numpy as np

__all__ = ["scene_transform", "write_page", "THREE_VERSION"]

THREE_VERSION = "0.160.0"
THREE_URL = f"https://cdn.jsdelivr.net/npm/three@{THREE_VERSION}/build/three.module.js"
ADDONS_URL = f"https://cdn.jsdelivr.net/npm/three@{THREE_VERSION}/examples/jsm/"


def scene_transform(up_W, tag: dict | None) -> np.ndarray:
    """4x4 W -> scene: gravity up is +z, the tag's centre the origin (W's origin without a tag)."""
    from umiproc.viz.slam_3d import scene_rotation

    T_tag = np.asarray(tag["tx_slam_tag"], np.float64) if tag else None
    if up_W is None and T_tag is not None:
        up_W = tag.get("up_world") or (-T_tag[:3, 2]).tolist()
    T = np.eye(4)
    T[:3, :3] = scene_rotation(up_W)
    if T_tag is not None:
        T[:3, 3] = -T[:3, :3] @ T_tag[:3, 3]
    return T


def _apply(T: np.ndarray, X: np.ndarray) -> np.ndarray:
    return np.asarray(X, np.float64).reshape(-1, 3) @ T[:3, :3].T + T[:3, 3]


def _b64(a: np.ndarray) -> str:
    return base64.b64encode(np.ascontiguousarray(a).tobytes()).decode("ascii")


def _frustum_segments(T_W_cam: np.ndarray, size: float) -> np.ndarray:
    """(8 segments x 2 vertices, 3) per pose: apex to the 4 corners and the image rectangle (x right, y down)."""
    c = np.array([[-0.67, -0.5, 1.0], [0.67, -0.5, 1.0], [0.67, 0.5, 1.0], [-0.67, 0.5, 1.0]]) * size
    out = []
    for M in T_W_cam:
        o = M[:3, 3]
        w = c @ M[:3, :3].T + o
        out += [o, w[0], o, w[1], o, w[2], o, w[3], w[0], w[1], w[1], w[2], w[2], w[3], w[3], w[0]]
    return np.asarray(out).reshape(-1, 3)


def _tag_corners(tag: dict | None) -> list | None:
    if not tag:
        return None
    T = np.asarray(tag["tx_slam_tag"], np.float64)
    h = float(tag.get("tag", {}).get("size_m", 0.16)) / 2
    return (np.array([[-h, -h, 0], [h, -h, 0], [h, h, 0], [-h, h, 0]]) @ T[:3, :3].T + T[:3, 3]).tolist()


def payload(inp, frames, X: np.ndarray, C: np.ndarray, summary: dict) -> dict:
    """The page's data, in the scene frame (module docstring)."""
    T = scene_transform(inp.export.meta.get("up_W"), inp.tag)
    P = _apply(T, X).astype(np.float32)
    lo, hi = (P.min(axis=0), P.max(axis=0)) if len(P) else (np.zeros(3), np.ones(3))
    q = np.rint((P - lo) / np.maximum(hi - lo, 1e-9) * 65535).astype("<u2") if len(P) else np.zeros((0, 3), "<u2")
    traj = inp.traj
    k = np.flatnonzero(traj.tracked)
    path = _apply(T, traj.T_W_cam[k, :3, 3])
    t_s = (traj.t_ns[k] - traj.t_ns[0]) / 1e9
    tag = _tag_corners(inp.tag)
    return {
        "title": f"{inp.folder}: dense cloud of the mapping clip (experimental)",
        "n": int(len(P)), "lo": lo.tolist(), "hi": hi.tolist(),
        "pos": _b64(q), "rgb": _b64(np.asarray(C, np.uint8)),
        "path": np.round(path, 4).tolist(), "path_t": np.round(t_s, 3).tolist(),
        "kf": np.round(_apply(T, _frustum_segments(inp.export.T_W_kf, 0.03)), 4).tolist(),
        "used": np.round(_apply(T, np.array([f.T_W_cam[:3, 3] for f in frames])), 4).tolist(),
        "tag": np.round(_apply(T, np.array(tag)), 4).tolist() if tag else None,
        "T_scene_W": np.round(T, 6).tolist(),
        "summary": {k2: summary[k2] for k2 in ("frames", "pairs", "points", "sparse_to_dense") if k2 in summary},
    }


PAGE = r"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>Dense map cloud</title>
<style>
:root { --bg: #f4f4f2; --panel: rgba(255,255,255,0.92); --text: #1d1d1f; --muted: #5c5c60; --line: #d6d6d2;
  --accent: #1f5fbf; --canvas: #e9e9e6; }
@media (prefers-color-scheme: dark) { :root:not([data-theme="light"]) { --bg: #141416; --panel: rgba(28,28,32,0.9);
  --text: #ececee; --muted: #a0a0a8; --line: #3a3a40; --accent: #6ea8ff; --canvas: #1b1b1f; } }
:root[data-theme="dark"] { --bg: #141416; --panel: rgba(28,28,32,0.9); --text: #ececee; --muted: #a0a0a8;
  --line: #3a3a40; --accent: #6ea8ff; --canvas: #1b1b1f; }
html, body { margin: 0; height: 100%; background: var(--bg); color: var(--text);
  font: 13px/1.4 system-ui, -apple-system, "Segoe UI", sans-serif; overflow: hidden; }
#view { position: fixed; inset: 0; }
#view canvas { display: block; width: 100% !important; height: 100% !important; }
#panel { position: fixed; top: 12px; left: 12px; width: min(300px, calc(100vw - 32px)); background: var(--panel);
  border: 1px solid var(--line); border-radius: 8px; padding: 10px 12px; box-sizing: border-box;
  max-height: calc(100vh - 24px); overflow: auto; }
#panel h1 { font-size: 14px; margin: 0 0 4px; font-weight: 600; }
#panel .muted { color: var(--muted); }
#panel label { display: flex; align-items: center; gap: 8px; margin: 4px 0; }
#panel input[type=range] { flex: 1; }
#panel .row { display: flex; gap: 6px; flex-wrap: wrap; margin: 6px 0; }
#panel button { font: inherit; padding: 3px 8px; border: 1px solid var(--line); border-radius: 5px;
  background: transparent; color: var(--text); cursor: pointer; }
#panel button:hover { border-color: var(--accent); }
#stats { font: 11px/1.45 ui-monospace, monospace; color: var(--muted); white-space: pre-wrap; margin-top: 6px; }
#scale { position: fixed; right: 16px; bottom: 16px; background: var(--panel); border: 1px solid var(--line);
  border-radius: 6px; padding: 6px 10px; text-align: center; }
#bar { height: 4px; background: var(--text); margin: 0 auto 3px; }
#err { position: fixed; inset: 0; display: none; align-items: center; justify-content: center; padding: 16px;
  text-align: center; }
details summary { cursor: pointer; color: var(--muted); }
</style>
<script type="importmap">{"imports": {"three": "__THREE__", "three/addons/": "__ADDONS__"}}</script>
</head><body>
<div id="view"></div>
<div id="panel">
 <h1 id="title"></h1>
 <div class="muted">Gravity up, the table tag at the origin; the grid has 10 cm squares. Drag to orbit, right-drag
  to pan, scroll to zoom.</div>
 <label>point size <input id="size" type="range" min="0.5" max="6" step="0.25" value="1.5"><span id="sizev"></span></label>
 <label><input id="showPath" type="checkbox" checked> camera path (colour = time)</label>
 <label><input id="showKf" type="checkbox" checked> keyframes</label>
 <label><input id="showUsed" type="checkbox"> frames used for depth</label>
 <label><input id="showTag" type="checkbox" checked> table tag</label>
 <label><input id="showGrid" type="checkbox" checked> 10 cm grid</label>
 <div class="row"><button data-v="persp">3/4 view</button><button data-v="top">top</button>
  <button data-v="front">front</button><button data-v="side">side</button><button data-v="cam">first camera</button></div>
 <details><summary>numbers</summary><div id="stats"></div></details>
</div>
<div id="scale"><div id="bar"></div><span id="barv"></span></div>
<div id="err">The viewer needs three.js from cdn.jsdelivr.net (internet once).</div>
<script>window.D = __DATA__;</script>
<script type="module">
import * as THREE from "three";
import { OrbitControls } from "three/addons/controls/OrbitControls.js";
const D = window.D, $ = id => document.getElementById(id);
$("title").textContent = D.title;
$("stats").textContent = JSON.stringify(D.summary, null, 1);
const css = getComputedStyle(document.documentElement);
const renderer = new THREE.WebGLRenderer({antialias: true, preserveDrawingBuffer: true});
renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
renderer.setSize(innerWidth, innerHeight);
// the colours are the camera's sRGB bytes: no colour management (it would wash them out)
THREE.ColorManagement.enabled = false;
renderer.outputColorSpace = THREE.LinearSRGBColorSpace;
renderer.setClearColor(new THREE.Color(css.getPropertyValue("--canvas").trim() || "#1b1b1f"));
$("view").appendChild(renderer.domElement);
const scene = new THREE.Scene();
const camera = new THREE.PerspectiveCamera(50, innerWidth / innerHeight, 0.005, 50);
camera.up.set(0, 0, 1);
const controls = new OrbitControls(camera, renderer.domElement);
controls.enableDamping = true;

async function bytes(b64) { return new Uint8Array(await (await fetch("data:application/octet-stream;base64," + b64)).arrayBuffer()); }
const q = new Uint16Array((await bytes(D.pos)).buffer), rgb = await bytes(D.rgb);
const pos = new Float32Array(D.n * 3);
for (let i = 0; i < D.n; i++) for (let k = 0; k < 3; k++)
  pos[3 * i + k] = D.lo[k] + q[3 * i + k] / 65535 * (D.hi[k] - D.lo[k]);
const geo = new THREE.BufferGeometry();
geo.setAttribute("position", new THREE.BufferAttribute(pos, 3));
geo.setAttribute("color", new THREE.BufferAttribute(rgb, 3, true));
const mat = new THREE.PointsMaterial({size: 1.5, sizeAttenuation: false, vertexColors: true});
const cloud = new THREE.Points(geo, mat);
scene.add(cloud);

function ramp(t) {  // viridis-like
  const s = [[0.27, 0.0, 0.33], [0.23, 0.32, 0.55], [0.13, 0.57, 0.55], [0.37, 0.79, 0.38], [0.99, 0.91, 0.14]];
  const x = Math.min(0.9999, Math.max(0, t)) * (s.length - 1), i = Math.floor(x), f = x - i;
  return s[i].map((v, k) => v + f * (s[i + 1][k] - v));
}
function lineObj(pts, colours, segments) {
  const g = new THREE.BufferGeometry();
  g.setAttribute("position", new THREE.Float32BufferAttribute(pts.flat(), 3));
  if (colours) g.setAttribute("color", new THREE.Float32BufferAttribute(colours.flat(), 3));
  const m = new THREE.LineBasicMaterial(colours ? {vertexColors: true} : {color: 0x3b82f6});
  return segments ? new THREE.LineSegments(g, m) : new THREE.Line(g, m);
}
const T1 = D.path_t[D.path_t.length - 1] || 1;
const path = lineObj(D.path, D.path_t.map(t => ramp(t / T1)), false);
scene.add(path);
const kf = lineObj(D.kf, null, true); kf.material.color.set(0xff8c1a); scene.add(kf);
const ug = new THREE.BufferGeometry(); ug.setAttribute("position", new THREE.Float32BufferAttribute(D.used.flat(), 3));
const used = new THREE.Points(ug, new THREE.PointsMaterial({size: 4, sizeAttenuation: false, color: 0xe0201b}));
used.visible = false; scene.add(used);
const tagGroup = new THREE.Group();
if (D.tag) {
  const loop = lineObj([...D.tag, D.tag[0]], null, false); loop.material.color.set(0xff2d55); tagGroup.add(loop);
  const fg = new THREE.BufferGeometry();
  fg.setAttribute("position", new THREE.Float32BufferAttribute([D.tag[0], D.tag[1], D.tag[2], D.tag[0], D.tag[2], D.tag[3]].flat(), 3));
  tagGroup.add(new THREE.Mesh(fg, new THREE.MeshBasicMaterial({color: 0xff2d55, transparent: true, opacity: 0.25, side: THREE.DoubleSide})));
}
scene.add(tagGroup);
const grid = new THREE.GridHelper(2, 20, 0x888888, 0x888888);
grid.rotation.x = Math.PI / 2; grid.material.transparent = true; grid.material.opacity = 0.35;
scene.add(grid);

// the view: the central 90 % of the cloud
function pct(axis, p) { const n = Math.min(D.n, 20000), a = new Float32Array(n);
  for (let i = 0; i < n; i++) a[i] = pos[3 * Math.floor(i * D.n / n) + axis]; a.sort(); return a[Math.floor(p * (n - 1))] || 0; }
const lo = [0, 1, 2].map(k => pct(k, 0.05)), hi = [0, 1, 2].map(k => pct(k, 0.95));
const ctr = new THREE.Vector3((lo[0] + hi[0]) / 2, (lo[1] + hi[1]) / 2, (lo[2] + hi[2]) / 2);
const R = Math.max(0.3, Math.hypot(hi[0] - lo[0], hi[1] - lo[1], hi[2] - lo[2]) / 2);
function setView(v) {
  controls.target.copy(ctr);
  const d = R * 2.2;
  const dir = {persp: [1, -1.2, 0.9], top: [0, -0.001, 1], front: [0, -1, 0.35], side: [1, 0, 0.35]}[v];
  if (v === "cam" && D.path.length) {
    const a = D.path[0];
    camera.position.set(...a); controls.target.set(...ctr.toArray());
  } else {
    const u = new THREE.Vector3(...(dir || [1, -1.2, 0.9])).normalize();
    camera.position.copy(ctr).addScaledVector(u, d);
  }
  controls.update();
}
document.querySelectorAll("button[data-v]").forEach(b => b.onclick = () => setView(b.dataset.v));
const hash = new URLSearchParams(location.hash.slice(1));
setView(hash.get("view") || "persp");
if (hash.get("size")) $("size").value = hash.get("size");
function sizes() { mat.size = +$("size").value; $("sizev").textContent = $("size").value + " px"; }
$("size").oninput = sizes; sizes();
for (const [id, o] of [["showPath", path], ["showKf", kf], ["showUsed", used], ["showTag", tagGroup], ["showGrid", grid]])
  $(id).onchange = () => { o.visible = $(id).checked; };
addEventListener("resize", () => { camera.aspect = innerWidth / innerHeight; camera.updateProjectionMatrix();
  renderer.setSize(innerWidth, innerHeight); });
function scaleBar() {   // a round length that is ~120 px at the orbit target's distance
  const dist = camera.position.distanceTo(controls.target);
  const pxPerM = innerHeight / (2 * dist * Math.tan(THREE.MathUtils.degToRad(camera.fov) / 2));
  const want = 120 / pxPerM, steps = [0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1, 2];
  const L = steps.reduce((b, s) => Math.abs(s - want) < Math.abs(b - want) ? s : b, steps[0]);
  $("bar").style.width = (L * pxPerM).toFixed(0) + "px";
  $("barv").textContent = (L < 1 ? (L * 100).toFixed(0) + " cm" : L + " m") + " at the orbit centre";
}
function loop() { controls.update(); scaleBar(); renderer.render(scene, camera); requestAnimationFrame(loop); }
loop();
document.title = "Dense map cloud";
window.READY = true;
</script>
<script>setTimeout(() => { if (!window.READY) document.getElementById("err").style.display = "flex"; }, 15000);</script>
</body></html>
"""


def write_page(path: Path, inp, frames, X: np.ndarray, C: np.ndarray, summary: dict) -> Path:
    data = json.dumps(payload(inp, frames, X, C, summary), separators=(",", ":"))
    html = PAGE.replace("__THREE__", THREE_URL).replace("__ADDONS__", ADDONS_URL).replace("__DATA__", data)
    Path(path).write_text(html)
    return Path(path)
