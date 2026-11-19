"""EXPERIMENTAL TRIAL (may be removed if it does not work out): the HTML viewer of :mod:`umiproc.viz.mesh_3d`.

One self-contained page: three.js (pinned, from jsdelivr; needs internet once) draws the textured mesh (or its
vertex colours, or plain shaded geometry), the mapping clip's camera path, the keyframes, the table tag and a 10 cm
grid, in the scene frame (gravity +z, the tag at the origin; as :mod:`umiproc.viz.dense_3d_viewer`). The mesh is
embedded as base64 (uint16 vertex positions quantised over its box, uint32 faces, uint8 vertex colours), the atlas
as a JPEG data URI; each face's texture coordinates follow from its index (one atlas cell per face).

"Look through" a keyframe puts the camera at that frame's pose with the pinhole view's field of view, and the
frame's real (undistorted) photo can be laid over it. URL hash: ``#view=persp|top|front|side|cam``,
``&cam=N`` (keyframe number), ``&photo=0..1`` (photo opacity), ``&mode=tex|vcol|shade``, ``&ui=0`` (hide the panel).
"""

from __future__ import annotations

import base64
import json
from pathlib import Path

import numpy as np

from umiproc.viz.dense_3d_viewer import ADDONS_URL, THREE_URL, _apply, _b64, _frustum_segments, _tag_corners

__all__ = ["payload", "write_page"]


def _jpeg_uri(rgb: np.ndarray, quality: int = 80) -> str:
    import cv2

    ok, buf = cv2.imencode(".jpg", cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, quality])
    return "data:image/jpeg;base64," + base64.b64encode(buf.tobytes()).decode("ascii")


def payload(inp, views, photos: dict, T_SW: np.ndarray, V: np.ndarray, F: np.ndarray, vcol: np.ndarray,
            jpeg: bytes, cell: int, cols: int, W: int, H: int, summary: dict, dp) -> dict:
    lo, hi = V.min(axis=0), V.max(axis=0)
    q = np.rint((V - lo) / np.maximum(hi - lo, 1e-9) * 65535).astype("<u2")
    traj = inp.traj
    k = np.flatnonzero(traj.tracked)
    path = _apply(T_SW, traj.T_W_cam[k, :3, 3])
    t_s = (traj.t_ns[k] - traj.t_ns[0]) / 1e9
    tag = _tag_corners(inp.tag)
    K = dp.K
    cams = [v for v in views if v.keyframe and v.fed in photos]
    if views and views[0].fed not in {v.fed for v in cams} and views[0].fed in photos:
        cams.insert(0, views[0])
    return {
        "title": f"{inp.folder}: textured mesh of the mapping clip (experimental)",
        "nv": int(len(V)), "nf": int(len(F)), "lo": lo.tolist(), "hi": hi.tolist(),
        "pos": _b64(q), "faces": _b64(F.astype("<u4")), "vcol": _b64(np.asarray(vcol, np.uint8)),
        "atlas": "data:image/jpeg;base64," + base64.b64encode(jpeg).decode("ascii"),
        "cell": int(cell), "cols": int(cols), "W": int(W), "H": int(H),
        "path": np.round(path, 4).tolist(), "path_t": np.round(t_s, 3).tolist(),
        "kf": np.round(_apply(T_SW, _frustum_segments(inp.export.T_W_kf, 0.03)), 4).tolist(),
        "tag": np.round(_apply(T_SW, np.array(tag)), 4).tolist() if tag else None,
        "fov_v": float(np.degrees(2 * np.arctan((dp.height / 2) / K[1, 1]))), "aspect": dp.width / dp.height,
        "cams": [{"fed": int(v.fed), "M": np.round(v.T_S_cam[:3], 6).tolist(), "photo": _jpeg_uri(photos[v.fed])}
                 for v in cams],
        "summary": {k2: summary[k2] for k2 in ("frames", "box", "mesh", "texture", "photometry") if k2 in summary},
    }


PAGE = r"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>Textured map mesh</title>
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
#photo { position: fixed; top: 0; left: 50%; height: 100vh; transform: translateX(-50%); pointer-events: none;
  display: none; }
#panel { position: fixed; top: 12px; left: 12px; width: min(310px, calc(100vw - 32px)); background: var(--panel);
  border: 1px solid var(--line); border-radius: 8px; padding: 10px 12px; box-sizing: border-box;
  max-height: calc(100vh - 24px); overflow: auto; }
#panel h1 { font-size: 14px; margin: 0 0 4px; font-weight: 600; }
#panel .muted { color: var(--muted); }
#panel label { display: flex; align-items: center; gap: 8px; margin: 4px 0; }
#panel input[type=range] { flex: 1; }
#panel .row { display: flex; gap: 6px; flex-wrap: wrap; margin: 6px 0; }
#panel button { font: inherit; padding: 3px 8px; border: 1px solid var(--line); border-radius: 5px;
  background: transparent; color: var(--text); cursor: pointer; }
#panel button:hover, #panel button.on { border-color: var(--accent); }
#stats { font: 11px/1.45 ui-monospace, monospace; color: var(--muted); white-space: pre-wrap; margin-top: 6px; }
#err { position: fixed; inset: 0; display: none; align-items: center; justify-content: center; padding: 16px;
  text-align: center; }
details summary { cursor: pointer; color: var(--muted); }
</style>
<script type="importmap">{"imports": {"three": "__THREE__", "three/addons/": "__ADDONS__"}}</script>
</head><body>
<div id="view"></div>
<img id="photo" alt="">
<div id="panel">
 <h1 id="title"></h1>
 <div class="muted">Gravity up, the table tag at the origin; the grid has 10 cm squares. Drag to orbit, right-drag
  to pan, scroll to zoom.</div>
 <div class="row"><button data-m="tex">texture</button><button data-m="vcol">vertex colours</button>
  <button data-m="shade">shaded</button></div>
 <label><input id="showPath" type="checkbox" checked> camera path (colour = time)</label>
 <label><input id="showKf" type="checkbox"> keyframes</label>
 <label><input id="showTag" type="checkbox"> table tag</label>
 <label><input id="showGrid" type="checkbox"> 10 cm grid</label>
 <div class="row"><button data-v="persp">3/4 view</button><button data-v="top">top</button>
  <button data-v="front">front</button><button data-v="side">side</button></div>
 <label>look through <input id="cam" type="range" min="0" max="0" step="1" value="0"><span id="camv"></span></label>
 <label>real photo <input id="photoA" type="range" min="0" max="1" step="0.05" value="0"></label>
 <details><summary>numbers</summary><div id="stats"></div></details>
</div>
<div id="err">The viewer needs three.js from cdn.jsdelivr.net (internet once).</div>
<script>
function showErr(m) { const e = document.getElementById("err"); e.textContent = "Viewer error: " + m; e.style.display = "flex"; }
addEventListener("error", ev => showErr(ev.message)); addEventListener("unhandledrejection", ev => showErr(ev.reason));
</script>
<script>window.D = __DATA__;</script>
<script type="module">
import * as THREE from "three";
import { OrbitControls } from "three/addons/controls/OrbitControls.js";
window.THREE_OK = true;
const D = window.D, $ = id => document.getElementById(id);
$("title").textContent = D.title;
$("stats").textContent = JSON.stringify(D.summary, null, 1);
const css = getComputedStyle(document.documentElement);
const renderer = new THREE.WebGLRenderer({antialias: true, preserveDrawingBuffer: true});
renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
renderer.setSize(innerWidth, innerHeight);
THREE.ColorManagement.enabled = false;       // texture and colours are the camera's sRGB bytes
renderer.outputColorSpace = THREE.LinearSRGBColorSpace;
renderer.setClearColor(new THREE.Color(css.getPropertyValue("--canvas").trim() || "#1b1b1f"));
$("view").appendChild(renderer.domElement);
const scene = new THREE.Scene();
const camera = new THREE.PerspectiveCamera(50, innerWidth / innerHeight, 0.005, 50);
camera.up.set(0, 0, 1);
const controls = new OrbitControls(camera, renderer.domElement);
controls.enableDamping = true;

async function bytes(b64) { return new Uint8Array(await (await fetch("data:application/octet-stream;base64," + b64)).arrayBuffer()); }
const q = new Uint16Array((await bytes(D.pos)).buffer), faces = new Uint32Array((await bytes(D.faces)).buffer);
const vcol = await bytes(D.vcol);
const pos = new Float32Array(D.nv * 3);
for (let i = 0; i < D.nv; i++) for (let k = 0; k < 3; k++)
  pos[3 * i + k] = D.lo[k] + q[3 * i + k] / 65535 * (D.hi[k] - D.lo[k]);
let geo = new THREE.BufferGeometry();
geo.setAttribute("position", new THREE.BufferAttribute(pos, 3));
geo.setAttribute("color", new THREE.BufferAttribute(vcol, 3, true));
geo.setIndex(new THREE.BufferAttribute(faces, 1));
geo.computeVertexNormals();
geo = geo.toNonIndexed();
const uv = new Float32Array(D.nf * 6), c = D.cell;
const corner = [[0.5, 0.5], [c - 0.5, 0.5], [0.5, c - 0.5]];
for (let f = 0; f < D.nf; f++) {
  const x0 = (f % D.cols) * c, y0 = Math.floor(f / D.cols) * c;
  for (let k = 0; k < 3; k++) { uv[6 * f + 2 * k] = (x0 + corner[k][0]) / D.W; uv[6 * f + 2 * k + 1] = (y0 + corner[k][1]) / D.H; }
}
geo.setAttribute("uv", new THREE.BufferAttribute(uv, 2));
const img = new Image(); img.src = D.atlas; await img.decode();
const tex = new THREE.Texture(img);
tex.flipY = false; tex.generateMipmaps = false; tex.minFilter = THREE.LinearFilter; tex.magFilter = THREE.LinearFilter;
tex.needsUpdate = true;
const mats = {
  tex: new THREE.MeshBasicMaterial({map: tex, side: THREE.DoubleSide}),
  vcol: new THREE.MeshBasicMaterial({vertexColors: true, side: THREE.DoubleSide}),
  shade: new THREE.MeshLambertMaterial({color: 0xc8c8c8, side: THREE.DoubleSide}),
};
const mesh = new THREE.Mesh(geo, mats.tex);
scene.add(mesh);
scene.add(new THREE.HemisphereLight(0xffffff, 0x444444, 0.9));
const sun = new THREE.DirectionalLight(0xffffff, 1.6); sun.position.set(0.5, -0.8, 1.5); scene.add(sun);
function setMode(m) { mesh.material = mats[m] || mats.tex;
  document.querySelectorAll("button[data-m]").forEach(b => b.classList.toggle("on", b.dataset.m === m)); }
document.querySelectorAll("button[data-m]").forEach(b => b.onclick = () => setMode(b.dataset.m));

function ramp(t) {
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
const path = lineObj(D.path, D.path_t.map(t => ramp(t / T1)), false); scene.add(path);
const kf = lineObj(D.kf, null, true); kf.material.color.set(0xff8c1a); kf.visible = false; scene.add(kf);
const tagGroup = new THREE.Group();
if (D.tag) { const loop = lineObj([...D.tag, D.tag[0]], null, false); loop.material.color.set(0xff2d55); tagGroup.add(loop); }
tagGroup.visible = false; scene.add(tagGroup);
const grid = new THREE.GridHelper(2, 20, 0x888888, 0x888888);
grid.rotation.x = Math.PI / 2; grid.material.transparent = true; grid.material.opacity = 0.35; grid.visible = false;
scene.add(grid);

// orbit views: the central 90 % of the vertices
function pct(axis, p) { const n = Math.min(D.nv, 20000), a = new Float32Array(n);
  for (let i = 0; i < n; i++) a[i] = pos[3 * Math.floor(i * D.nv / n) + axis]; a.sort(); return a[Math.floor(p * (n - 1))] || 0; }
const lo = [0, 1, 2].map(k => pct(k, 0.05)), hi = [0, 1, 2].map(k => pct(k, 0.95));
const ctr = new THREE.Vector3((lo[0] + hi[0]) / 2, (lo[1] + hi[1]) / 2, (lo[2] + hi[2]) / 2);
const R = Math.max(0.3, Math.hypot(hi[0] - lo[0], hi[1] - lo[1], hi[2] - lo[2]) / 2);
let locked = false;   // looking through a keyframe: the orbit controls must not turn the camera
function setView(v) {
  locked = false; $("photo").style.display = "none";
  camera.fov = 50; camera.up.set(0, 0, 1); camera.updateProjectionMatrix();
  controls.target.copy(ctr);
  const dir = {persp: [1, -1.2, 0.9], top: [0, -0.001, 1], front: [0, -1, 0.35], side: [1, 0, 0.35]}[v] || [1, -1.2, 0.9];
  camera.position.copy(ctr).addScaledVector(new THREE.Vector3(...dir).normalize(), R * 2.2);
  controls.update();
}
function lookThrough(i) {
  const C = D.cams[i]; if (!C) return;
  const M = C.M, m = new THREE.Matrix4();
  // camera x right, y down, z forward -> three.js x right, y up, looking down -z
  m.set(M[0][0], -M[0][1], -M[0][2], M[0][3], M[1][0], -M[1][1], -M[1][2], M[1][3],
        M[2][0], -M[2][1], -M[2][2], M[2][3], 0, 0, 0, 1);
  camera.position.setFromMatrixPosition(m); camera.quaternion.setFromRotationMatrix(m);
  camera.fov = D.fov_v; camera.updateProjectionMatrix();
  const fwd = new THREE.Vector3(M[0][2], M[1][2], M[2][2]);
  controls.target.copy(camera.position).addScaledVector(fwd, 0.4);
  locked = true;
  $("camv").textContent = (i + 1) + "/" + D.cams.length + " (frame " + C.fed + ")";
  $("photo").src = C.photo; $("photo").style.display = "block"; photoA();
}
function photoA() { $("photo").style.opacity = $("photoA").value; }
renderer.domElement.addEventListener("pointerdown", () => { if (locked) { locked = false; camera.up.set(0, 0, 1);
  $("photo").style.display = "none"; } });
document.querySelectorAll("button[data-v]").forEach(b => b.onclick = () => setView(b.dataset.v));
$("cam").max = Math.max(0, D.cams.length - 1);
$("cam").oninput = () => lookThrough(+$("cam").value);
$("photoA").oninput = photoA;
for (const [id, o] of [["showPath", path], ["showKf", kf], ["showTag", tagGroup], ["showGrid", grid]]) {
  o.visible = $(id).checked; $(id).onchange = () => { o.visible = $(id).checked; }; }
const hash = new URLSearchParams(location.hash.slice(1));
setMode(hash.get("mode") || "tex");
if (hash.get("photo")) $("photoA").value = hash.get("photo");
if (hash.get("ui") === "0") $("panel").style.display = "none";
if (hash.get("path") === "0") { $("showPath").checked = false; path.visible = false; }
if ((hash.get("view") || "persp") === "cam") { $("cam").value = +(hash.get("cam") || 0); lookThrough(+$("cam").value); }
else setView(hash.get("view") || "persp");
addEventListener("resize", () => { camera.aspect = innerWidth / innerHeight; camera.updateProjectionMatrix();
  renderer.setSize(innerWidth, innerHeight); });
function loop() { if (!locked) controls.update(); renderer.render(scene, camera); requestAnimationFrame(loop); }
loop();
window.viewer = {setView, lookThrough, setMode, photo: a => { $("photoA").value = a; photoA(); },
  ui: on => { $("panel").style.display = on ? "" : "none"; }};   // for scripted screenshots
window.READY = true;
</script>
<script>setTimeout(() => { if (!window.THREE_OK) document.getElementById("err").style.display = "flex"; }, 20000);</script>
</body></html>
"""


def write_page(path: Path, inp, views, photos: dict, T_SW, V, F, vcol, jpeg, cell, cols, W, H, summary, dp) -> Path:
    data = json.dumps(payload(inp, views, photos, T_SW, V, F, vcol, jpeg, cell, cols, W, H, summary, dp),
                      separators=(",", ":"))
    html = PAGE.replace("__THREE__", THREE_URL).replace("__ADDONS__", ADDONS_URL).replace("__DATA__", data)
    Path(path).write_text(html)
    return Path(path)
