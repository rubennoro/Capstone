"""Live data for the UI (DESIGN.md sections 3 and 5): camera JPEGs, the policy's view, the external render, C2, ground
truth and the policy timing. Everything is read only: SUB sockets to C0 / state / C2, a REQ for ``ground_truth`` at a
few Hz, and a tail of the policy_deploy session's ``manager/steps.jsonl``.

One thread per source; each reconnects by itself (ZeroMQ reconnects SUB sockets; the REQ is reset after a timeout;
the model and its GL context are reloaded on a model change) and never blocks the caller: ``latest_frames()``,
``frame_seq()`` and ``telemetry()`` only copy what the threads stored, under a lock.

Channels (binary frames, DESIGN section 3): 1 external (``cam_external``, 640x480 render of the mirrored state),
2 main (IMX519 JPEG as streamed), 3 ultrawide (IMX219 JPEG as streamed), 4 policy (448x448 mosaic of the four 224x224
images the policy gets: top main T-50 ms / T, bottom ultrawide T-100 ms / T, oldest left).

The policy view runs the copied ``policy_deploy`` (installed editable in ``~/envs/sim_deploy``): the rig file loader,
``ImagePrep``, ``PairRule``, ``BufferedPacket`` and ``PacketSelector`` with the spacings and tolerance of
``config/frames_process.yaml``. Simplified against the frames process: packet times are the Pi clock's main
mid-exposure (no clock map: only differences matter for the pick), no camera-health gate, and only the picked
packets are decoded.

GL: ``MUJOCO_GL=egl`` and the NVIDIA EGL vendor file are set when this module is imported (E5's fix in
``sim_pi/render.py``: the user's shell points ``__EGL_VENDOR_LIBRARY_DIRS`` at Anaconda's Mesa, which puts EGL on the
Intel iGPU), so import this module before anything imports ``mujoco``. ``SIM_PI_EGL_VENDOR=default`` keeps the shell's
choice. The external renderer's offscreen buffer is cut to the image size and its shadow map to 2048 (small VRAM: the
policy shares the 4 GB GPU).
"""

from __future__ import annotations

import json
import os
import re
import threading
import time
from collections import OrderedDict
from pathlib import Path
from typing import Any, Callable, Mapping

NVIDIA_EGL_VENDOR = "/usr/share/glvnd/egl_vendor.d/10_nvidia.json"     # as sim_deploy/sim_pi/render.py


def _select_egl_vendor() -> None:
    os.environ.setdefault("MUJOCO_GL", "egl")
    choice = os.environ.get("SIM_PI_EGL_VENDOR", NVIDIA_EGL_VENDOR)
    if choice == "default" or "__EGL_VENDOR_LIBRARY_FILENAMES" in os.environ:
        return
    if os.path.isfile(choice):
        os.environ["__EGL_VENDOR_LIBRARY_FILENAMES"] = choice


_select_egl_vendor()

import cv2  # noqa: E402
import numpy as np  # noqa: E402
import zmq  # noqa: E402

UI_DIR = Path(__file__).resolve().parent
REPO = UI_DIR.parents[1]                                              # ~/Capstone-main-sim

CH_EXTERNAL, CH_MAIN, CH_ULTRAWIDE, CH_POLICY = 1, 2, 3, 4
CHANNELS = (CH_EXTERNAL, CH_MAIN, CH_ULTRAWIDE, CH_POLICY)

#: Every key Live reads, with its default. Endpoints are the sim chain's (sim_deploy/DESIGN.md section 1).
DEFAULTS: dict[str, Any] = {
    "c0_endpoint": "tcp://127.0.0.1:5555",          # sim Pi pi_stream/v1 PUB
    "state_endpoint": "tcp://127.0.0.1:5650",       # sim PC sim.state/v1 PUB
    "c2_endpoint": "tcp://127.0.0.1:5620",          # sim PC C2 PUB
    "scene_endpoint": "tcp://127.0.0.1:5651",       # sim PC scene REP (shared with the eval driver)
    "scene_xml": str(REPO / "sim_deploy" / "build" / "sim_rig" / "scene.xml"),
    "rig_file": str(REPO / "deployment" / "contracts" / "rigs" / "rig_pi_v3_640.yaml"),
    "frames_config": str(REPO / "deployment" / "policy_deploy" / "config" / "frames_process.yaml"),
    "camera_fps": 10.0,              # channels 2, 3 (pass-through, decimated)
    "policy_fps": 5.0,               # channel 4 (each mosaic costs ~50 ms CPU: 4 decodes + ImagePrep)
    "external_fps": 10.0,            # channel 1
    "external_size": [640, 480],     # (width, height)
    "external_camera": "cam_external",
    "external_shadows": True,
    "external_shadow_size": 2048,
    "jpeg_quality": 85,              # channels 1 and 4 (2 and 3 are passed through as sent)
    "ground_truth_hz": 5.0,
    "scene_timeout_s": 0.25,
    "scene_retry_s": 1.0,            # after a timeout, wait this long before the next request
    "c2_hz": 20.0,                   # C2 is 100 Hz; the newest message is decoded at most this often
    "steps_poll_s": 0.2,
    "stale_s": 1.0,                  # a source older than this is reported with ok false and its fields as null
    "table_top_z_m": 0.0,            # the table top in O (P3/P4: the base is on the table; was 0.050)
}

PAD = re.compile(r"^(pad|finger_holder)_(l|r)")       # the eval driver's finger-contact rule (eval/driver.py)


#: U1's ui/settings.py names -> ours (``external_width`` / ``external_height`` are handled below).
_ALIASES = {"rig_yaml": "rig_file"}


def _resolve_settings(settings: Any) -> dict[str, Any]:
    """DEFAULTS overridden by ``settings``: a mapping or an object with attributes (U1's frozen ``Settings``), then
    its ``extra`` mapping (config.yaml keys U1 does not know, e.g. ``policy_fps``), then a ``live`` mapping if any.
    U1's ``external_width`` / ``external_height`` set ``external_size``, ``rig_yaml`` sets ``rig_file``, and its
    ``frame_fps`` caps every channel's rate. Unknown keys are ignored."""
    out = dict(DEFAULTS)
    if settings is None:
        return out

    def get(src: Any, k: str) -> Any:
        return src.get(k) if isinstance(src, Mapping) else getattr(src, k, None)

    sources = [settings]
    for sub_key in ("extra", "live"):
        sub = get(settings, sub_key)
        if isinstance(sub, Mapping):
            sources.append(sub)
    for src in sources:
        for k in DEFAULTS:
            if get(src, k) is not None:
                out[k] = get(src, k)
        for theirs, ours in _ALIASES.items():
            if get(src, theirs) is not None:
                out[ours] = get(src, theirs)
        w, h = get(src, "external_width"), get(src, "external_height")
        if w is not None or h is not None:
            out["external_size"] = [int(w or out["external_size"][0]), int(h or out["external_size"][1])]
    cap = get(settings, "frame_fps")
    if cap is not None:
        for k in ("camera_fps", "policy_fps", "external_fps"):
            out[k] = min(float(out[k]), float(cap))
    return out


class _Source:
    """Per-source bookkeeping shown under telemetry ``sources``."""

    def __init__(self) -> None:
        self.n = 0
        self.t_last = 0.0             # time.monotonic() of the last message
        self.error: str | None = None
        self._times: list[float] = []

    def mark(self) -> None:
        now = time.monotonic()
        self.n += 1
        self.t_last = now
        self._times.append(now)
        if len(self._times) > 64:
            del self._times[:32]

    def report(self, stale_s: float) -> dict:
        now = time.monotonic()
        age = None if self.n == 0 else now - self.t_last
        recent = [t for t in self._times if now - t <= 2.0]
        rate = (len(recent) - 1) / (recent[-1] - recent[0]) if len(recent) >= 2 and recent[-1] > recent[0] else 0.0
        return {"ok": age is not None and age <= stale_s, "age_s": None if age is None else round(age, 3),
                "rate_hz": round(rate, 1), "n": self.n, "error": self.error}


class Live:
    """The UI's live data. ``start()`` once, ``stop()`` at shutdown; the getters never block."""

    def __init__(self, settings: Any = None) -> None:
        self.s = _resolve_settings(settings)
        self._lock = threading.Lock()
        self._halt = threading.Event()
        self._threads: list[threading.Thread] = []
        self._frames: dict[int, bytes] = {}
        self._seq: dict[int, int] = {ch: 0 for ch in CHANNELS}
        self._src = {name: _Source() for name in ("c0", "state", "external", "policy_view", "c2", "scene", "steps")}
        self._c2: dict | None = None
        self._gt: dict | None = None
        self._policy: dict | None = None
        self._session_dir: Path | None = None
        self._session_gen = 0
        self._policy_reason = "no packets"
        # C0 -> policy view (filled by the C0 thread, read by the policy thread)
        self._c0_new = threading.Event()
        self._pv = None                   # _PolicyView, built in start()

    # ------------------------------------------------------------------------------------------------ public API

    def start(self) -> None:
        if self._threads:
            return
        self._halt.clear()
        try:
            self._pv = _PolicyView(self.s)
        except Exception as e:                            # noqa: BLE001 (no policy view is not fatal)
            self._pv = None
            self._src["policy_view"].error = f"policy view unavailable: {type(e).__name__}: {e}"
        for name, body in (("live-c0", self._run_c0), ("live-policy", self._run_policy),
                           ("live-external", self._run_external), ("live-c2", self._run_c2),
                           ("live-scene", self._run_scene), ("live-steps", self._run_steps)):
            t = threading.Thread(target=self._guard(name, body), name=name, daemon=True)
            t.start()
            self._threads.append(t)

    def stop(self) -> None:
        self._halt.set()
        self._c0_new.set()
        for t in self._threads:
            t.join(timeout=3.0)
        self._threads = []

    def latest_frames(self) -> dict[int, bytes]:
        """Channel id -> the newest JPEG (channels with no frame yet are absent)."""
        with self._lock:
            return dict(self._frames)

    def frame_seq(self) -> dict[int, int]:
        """Channel id -> how many frames that channel has produced (send a frame when its number changed)."""
        with self._lock:
            return dict(self._seq)

    def set_session_dir(self, path: str | os.PathLike | None) -> None:
        """The running policy_deploy session folder (its ``manager/steps.jsonl`` is tailed), or None. Also accepts
        the ``manager`` folder or the ``steps.jsonl`` file itself."""
        with self._lock:
            self._session_dir = None if path is None else Path(path).expanduser()
            self._session_gen += 1
            self._policy = None

    def telemetry(self) -> dict:
        """DESIGN.md section 3 ``telemetry`` (without ``type``). Fields of a source that is stale are null;
        ``sources`` says, per source, whether it is live (``ok``), its age, rate and last error."""
        stale = float(self.s["stale_s"])
        with self._lock:
            c2, gt, pol = self._c2, self._gt, self._policy
            reports = {k: v.report(stale) for k, v in self._src.items()}
            reason = self._policy_reason
        reports["policy_view"]["reason"] = reason
        c2_ok, gt_ok = reports["c2"]["ok"], reports["scene"]["ok"]
        out = {"t_ns": time.monotonic_ns(),
               "mode": c2["mode"] if c2 and c2_ok else None,
               "tcp": c2["tcp"] if c2 and c2_ok else None,
               "width_m": c2["width_m"] if c2 and c2_ok else None,
               "c2_error": c2["error"] if c2 and c2_ok else None,
               "eraser": gt["eraser"] if gt and gt_ok else None,
               "contacts": gt["contacts"] if gt and gt_ok else None,
               "finger_gap_m": gt["finger_gap_m"] if gt and gt_ok else None,
               "policy": pol,
               "sources": reports}
        return out

    # ------------------------------------------------------------------------------------------------ helpers

    def _guard(self, name: str, body: Callable[[], None]) -> Callable[[], None]:
        def run() -> None:
            while not self._halt.is_set():
                try:
                    body()
                    return
                except Exception as e:                    # noqa: BLE001 (a source never takes the UI down)
                    src = self._src.get(name.split("-", 1)[1].replace("policy", "policy_view"))
                    if src is not None:
                        src.error = f"{type(e).__name__}: {e}"
                    self._halt.wait(1.0)                  # restart the source
        return run

    def _put_frame(self, ch: int, jpeg: bytes) -> None:
        with self._lock:
            self._frames[ch] = jpeg
            self._seq[ch] += 1

    def _sub(self, endpoint: str, conflate: bool, hwm: int = 10) -> zmq.Socket:
        sock = zmq.Context.instance().socket(zmq.SUB)
        sock.setsockopt(zmq.LINGER, 0)
        if conflate:
            sock.setsockopt(zmq.CONFLATE, 1)              # single-part messages only: keep the newest
        else:
            sock.setsockopt(zmq.RCVHWM, hwm)
        sock.setsockopt(zmq.SUBSCRIBE, b"")
        sock.connect(endpoint)
        return sock

    # ------------------------------------------------------------------------------------------------ C0

    def _run_c0(self) -> None:
        from policy_deploy.contracts.pi_stream import PiStreamPacket
        from deploy_contracts import ContractError

        sock = self._sub(self.s["c0_endpoint"], conflate=False, hwm=60)
        src = self._src["c0"]
        period = 1.0 / max(0.1, float(self.s["camera_fps"]))
        next_pub = 0.0
        try:
            while not self._halt.is_set():
                if not sock.poll(100, zmq.POLLIN):
                    continue
                while True:                               # drain: every packet feeds the policy view buffer
                    try:
                        parts = sock.recv_multipart(zmq.NOBLOCK)
                    except zmq.Again:
                        break
                    try:
                        pkt = PiStreamPacket.from_parts(parts)
                    except ContractError as e:
                        src.error = f"bad packet: {e}"
                        continue
                    src.mark()
                    if self._pv is not None:
                        self._pv.add(pkt)
                        self._c0_new.set()
                    now = time.monotonic()
                    if now >= next_pub:
                        next_pub = max(next_pub + period, now)
                        with self._lock:
                            self._frames[CH_MAIN] = pkt.main_jpeg
                            self._frames[CH_ULTRAWIDE] = pkt.ultrawide_jpeg
                            self._seq[CH_MAIN] += 1
                            self._seq[CH_ULTRAWIDE] += 1
        finally:
            sock.close(linger=0)

    def _run_policy(self) -> None:
        if self._pv is None:
            return
        period = 1.0 / max(0.1, float(self.s["policy_fps"]))
        q = int(self.s["jpeg_quality"])
        src = self._src["policy_view"]
        last_newest = None
        while not self._halt.is_set():
            t0 = time.monotonic()
            self._c0_new.wait(0.5)
            self._c0_new.clear()
            if self._halt.is_set():
                break
            newest = self._pv.newest_seq()
            if newest is None or newest == last_newest:
                continue
            last_newest = newest
            mosaic, reason = self._pv.mosaic()
            with self._lock:
                self._policy_reason = reason
            if mosaic is not None:
                ok, buf = cv2.imencode(".jpg", cv2.cvtColor(mosaic, cv2.COLOR_RGB2BGR),
                                       [cv2.IMWRITE_JPEG_QUALITY, q])
                if ok:
                    self._put_frame(CH_POLICY, buf.tobytes())
                    src.mark()
            dt = time.monotonic() - t0
            self._halt.wait(max(0.0, period - dt))

    # ------------------------------------------------------------------------------------------------ state + render

    def _run_external(self) -> None:
        import mujoco

        sock = self._sub(self.s["state_endpoint"], conflate=True)
        src_state, src_ext = self._src["state"], self._src["external"]
        period = 1.0 / max(0.1, float(self.s["external_fps"]))
        w, h = (int(v) for v in self.s["external_size"])
        q = int(self.s["jpeg_quality"])
        scene_xml = Path(self.s["scene_xml"]).expanduser()
        m = d = renderer = opt = None
        model_sha = None
        retry_at = 0.0
        last_step = None
        next_render = 0.0
        try:
            while not self._halt.is_set():
                if not sock.poll(100, zmq.POLLIN):
                    continue
                try:
                    msg = json.loads(sock.recv(zmq.NOBLOCK))
                except (zmq.Again, ValueError):
                    continue
                src_state.mark()
                now = time.monotonic()
                if now < next_render:
                    self._halt.wait(min(next_render - now, 0.1))
                    continue
                sha = msg.get("model_sha256")
                if renderer is not None and sha and model_sha and sha != model_sha:
                    renderer.close()                       # the model was rebuilt: load it again
                    m = d = renderer = None
                    src_ext.error = "model changed, reloading"
                if renderer is None:
                    if now < retry_at:
                        continue
                    retry_at = now + 5.0
                    try:
                        scene_xml = _scene_for_sha(Path(self.s["scene_xml"]).expanduser(), sha)   # the run's look
                        model_sha = _model_sha(scene_xml)
                        if sha and model_sha and sha != model_sha:
                            src_ext.error = (f"state model {sha[:12]}... != {scene_xml.parent.name}/model.json "
                                             f"{model_sha[:12]}... (rebuild in progress?)")
                            continue
                        m = mujoco.MjModel.from_xml_path(str(scene_xml))
                        m.vis.global_.offwidth, m.vis.global_.offheight = w, h       # small VRAM
                        m.vis.quality.shadowsize = int(self.s["external_shadow_size"])
                        d = mujoco.MjData(m)
                        renderer = mujoco.Renderer(m, h, w)
                        opt = mujoco.MjvOption()
                        for g in range(6):
                            opt.geomgroup[g] = 1 if g in (0, 1, 2, 3) else 0
                        src_ext.error = None
                    except Exception as e:                # noqa: BLE001
                        src_ext.error = f"renderer: {type(e).__name__}: {e}"
                        m = d = renderer = None
                        continue
                qpos = msg.get("qpos")
                if qpos is None or len(qpos) != m.nq:
                    src_ext.error = f"qpos has {None if qpos is None else len(qpos)} entries, model nq {m.nq}"
                    continue
                if msg.get("step") == last_step:
                    continue
                last_step = msg.get("step")
                next_render = max(next_render + period, now)
                d.qpos[:] = qpos
                mujoco.mj_forward(m, d)
                renderer.update_scene(d, camera=self.s["external_camera"], scene_option=opt)
                if not self.s["external_shadows"]:
                    renderer.scene.flags[mujoco.mjtRndFlag.mjRND_SHADOW] = 0
                img = renderer.render()
                ok, buf = cv2.imencode(".jpg", cv2.cvtColor(img, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, q])
                if ok:
                    self._put_frame(CH_EXTERNAL, buf.tobytes())
                    src_ext.mark()
        finally:
            if renderer is not None:
                renderer.close()
            sock.close(linger=0)

    # ------------------------------------------------------------------------------------------------ C2

    def _run_c2(self) -> None:
        from deploy_contracts import ArmState, ContractError, decode

        sock = self._sub(self.s["c2_endpoint"], conflate=True)
        src = self._src["c2"]
        period = 1.0 / max(0.1, float(self.s["c2_hz"]))
        try:
            while not self._halt.is_set():
                if not sock.poll(100, zmq.POLLIN):
                    continue
                try:
                    msg = decode(sock.recv(zmq.NOBLOCK), ArmState)
                except zmq.Again:
                    continue
                except ContractError as e:
                    src.error = f"C2 decode: {e}"
                    continue
                if not isinstance(msg, ArmState):
                    continue
                T = np.asarray(msg.O_T_TCP, float)
                val = {"mode": msg.mode, "tcp": [round(float(v), 5) for v in T[:3, 3]],
                       "width_m": round(float(msg.gripper_width_m), 5), "error": msg.error or None,
                       "t_ns": int(msg.t_ns)}
                with self._lock:
                    self._c2 = val
                src.mark()
                self._halt.wait(period)
        finally:
            sock.close(linger=0)

    # ------------------------------------------------------------------------------------------------ ground truth

    def _run_scene(self) -> None:
        ctx = zmq.Context.instance()
        timeout_ms = int(float(self.s["scene_timeout_s"]) * 1000)
        period = 1.0 / max(0.1, float(self.s["ground_truth_hz"]))
        retry_s = float(self.s["scene_retry_s"])
        table_z = float(self.s["table_top_z_m"])
        src = self._src["scene"]
        req = json.dumps({"cmd": "ground_truth"}).encode()

        def open_sock() -> zmq.Socket:
            s = ctx.socket(zmq.REQ)
            s.setsockopt(zmq.LINGER, 0)
            s.setsockopt(zmq.RCVTIMEO, timeout_ms)
            s.setsockopt(zmq.SNDTIMEO, timeout_ms)
            s.connect(self.s["scene_endpoint"])
            return s

        sock = open_sock()
        try:
            while not self._halt.is_set():
                t0 = time.monotonic()
                try:
                    sock.send(req)
                    rep = json.loads(sock.recv())
                except zmq.Again:
                    sock.close(linger=0)                  # a REQ socket is stuck after a timeout: start over
                    sock = open_sock()
                    src.error = f"no reply within {timeout_ms} ms"
                    self._halt.wait(retry_s)
                    continue
                except ValueError as e:
                    src.error = f"bad reply: {e}"
                    self._halt.wait(retry_s)
                    continue
                if not rep.get("ok", False):
                    src.error = str(rep.get("error", "ok false"))
                    self._halt.wait(retry_s)
                    continue
                val = _ground_truth_view(rep, table_z)
                with self._lock:
                    self._gt = val
                src.mark()
                src.error = None
                self._halt.wait(max(0.0, period - (time.monotonic() - t0)))
        finally:
            sock.close(linger=0)

    # ------------------------------------------------------------------------------------------------ steps.jsonl

    def _run_steps(self) -> None:
        src = self._src["steps"]
        poll = float(self.s["steps_poll_s"])
        gen = -1
        f = None
        path: Path | None = None
        ino = None
        buf = ""
        per_rollout: dict[int, dict] = {}
        try:
            while not self._halt.is_set():
                with self._lock:
                    cur_gen, sdir = self._session_gen, self._session_dir
                if cur_gen != gen:                        # a new session folder (or none)
                    gen = cur_gen
                    if f is not None:
                        f.close()
                    f, ino, buf, per_rollout = None, None, "", {}
                    path = None if sdir is None else _steps_path(sdir)
                if path is None:
                    self._halt.wait(poll)
                    continue
                if f is None:
                    try:
                        f = open(path, "r", encoding="utf-8")
                        ino = os.fstat(f.fileno()).st_ino
                    except OSError:
                        self._halt.wait(poll)
                        continue
                try:                                      # replaced or truncated: read it again from the start
                    st = os.stat(path)
                    if st.st_ino != ino or st.st_size < f.tell():
                        f.close()
                        f, buf, per_rollout = None, "", {}
                        continue
                except OSError:
                    pass
                chunk = f.read()
                if chunk:
                    buf += chunk
                    *lines, buf = buf.split("\n")
                    last = None
                    for line in lines:
                        if not line.strip():
                            continue
                        try:
                            rec = json.loads(line)
                        except ValueError:
                            continue
                        ep = per_rollout.setdefault(int(rec.get("episode_id", -1)), {"steps": 0, "skipped": 0})
                        ep["steps"] += 1
                        if not rec.get("sent", True):
                            ep["skipped"] += 1
                        last = rec
                    if last is not None:
                        ep = per_rollout[int(last.get("episode_id", -1))]
                        val = {"infer_ms": _r(last.get("policy_ms")), "margin_ms": _r(last.get("margin_ms")),
                               "skipped": ep["skipped"], "steps": ep["steps"],
                               "last_skip_reason": last.get("skip_reason") or None,
                               "cycle_ms": _r(last.get("cycle_ms")), "image_age_ms": _r(last.get("image_age_ms")),
                               "frames_ms": _r(last.get("frames_ms")), "episode_id": last.get("episode_id"),
                               "chunk_id": last.get("chunk_id")}
                        with self._lock:
                            if self._session_gen == gen:
                                self._policy = val
                        src.mark()
                self._halt.wait(poll)
        finally:
            if f is not None:
                f.close()


# ---------------------------------------------------------------------------------------------------- helpers


def _r(v, nd: int = 1):
    return None if v is None else round(float(v), nd)


def _steps_path(p: Path) -> Path:
    if p.name == "steps.jsonl":
        return p
    if p.name == "manager":
        return p / "steps.jsonl"
    return p / "manager" / "steps.jsonl"


def _scene_for_sha(default: Path, sha: str | None) -> Path:
    """The scene.xml whose model.json has this sha: the default (sim_rig), else any sibling build (run.sh --look runs
    the sim on build/<look station id>/), else the default (the caller reports the mismatch)."""
    if not sha or _model_sha(default) == sha:
        return default
    for mj in sorted(default.parent.parent.glob("*/model.json")):
        if _model_sha(mj.with_name("scene.xml")) == sha:
            return mj.with_name("scene.xml")
    return default


def _model_sha(scene_xml: Path) -> str | None:
    try:
        return json.loads((scene_xml.parent / "model.json").read_text()).get("model_sha256")
    except (OSError, ValueError):
        return None


def _ground_truth_view(rep: dict, table_z: float) -> dict:
    """The telemetry part of a scene ``ground_truth`` reply."""
    er = (rep.get("objects") or {}).get("eraser") or {}
    pose = er.get("pose")
    eraser = None
    if pose is not None:
        eraser = {"pose": [round(float(v), 5) for v in pose],
                  "height_m": round(float(pose[2]) - table_z, 5)}     # centre above the table top
    contacts = rep.get("contacts") or []
    sides = set()
    table = False
    for c in contacts:
        a, b = str(c.get("a", "")), str(c.get("b", ""))
        for n, o in ((a, b), (b, a)):
            mm = PAD.match(n)
            if o == "eraser" and mm:
                sides.add(mm.group(2))
        if "table" in (a, b) and "eraser" not in (a, b):
            table = True                                  # anything but the eraser on the table
    return {"eraser": eraser,
            "contacts": {"finger_eraser": bool(sides), "both_fingers": sides == {"l", "r"}, "table": table},
            "finger_gap_m": rep.get("finger_gap_m")}


class _PolicyView:
    """The copy's picking rule and ImagePrep over the C0 packets (see the module docstring)."""

    MAX_DECODED = 12

    def __init__(self, s: Mapping[str, Any]) -> None:
        import yaml
        from policy_deploy.contracts import OBS_STEPS
        from policy_deploy.frames._dataproc.imageprep import ImagePrep
        from policy_deploy.frames.packet_buffer import BufferedPacket, PacketBuffer
        from policy_deploy.frames.packet_selector import PacketSelector
        from policy_deploy.frames.pair_rule import OK, PairRule
        from policy_deploy.frames.rig_file import load_live_rig

        cfg = yaml.safe_load(Path(s["frames_config"]).expanduser().read_text())
        sel, keep_s = cfg["select"], float(cfg["buffer"]["keep_s"])
        rig = load_live_rig(Path(s["rig_file"]).expanduser())
        self.prep = ImagePrep(rig)
        self.selector = PacketSelector(fps={c: rig.cameras[c].fps for c in ("main", "ultrawide")},
                                       spacing_ns={"main": int(float(sel["main_spacing_ms"]) * 1e6),
                                                   "ultrawide": int(float(sel["ultrawide_spacing_ms"]) * 1e6)},
                                       count=OBS_STEPS, tolerance_frames=float(sel["tolerance_frames"]))
        self._pair_args = (float(sel["pair_max_age_ms"]), float(sel["pair_max_lead_ms"]), int(sel["pair_max_reuse"]))
        self._PairRule, self._OK, self._BP, self._PB = PairRule, OK, BufferedPacket, PacketBuffer
        self._keep_s = keep_s
        self._lock = threading.Lock()
        self._mark = np.zeros((1, 1, 3), np.uint8)     # stands in for a decoded image: only picked packets decode
        self._reset()
        self._decoded: OrderedDict = OrderedDict()     # (packet_seq, camera) -> 224x224 RGB, the last few

    def _reset(self) -> None:
        self.pair_rule = self._PairRule(*self._pair_args)
        self.buffer = self._PB(self._keep_s)
        self._jpegs: dict[int, tuple[bytes, bytes]] = {}
        self._last_no = -1

    def add(self, pkt) -> None:
        h = pkt.header
        t = int(h.main.t_mid_ns)
        with self._lock:
            newest = self.buffer.newest()
            if h.packet_no <= self._last_no or (newest is not None and t <= newest.t_ns):
                self._reset()                              # a Pi (sim Pi) restart
            self._last_no = h.packet_no
            verdict = self.pair_rule.check(h.ultrawide.frame_no, h.packet_no, t, int(h.ultrawide.t_mid_ns))
            ok = verdict == self._OK
            bp = self._BP(packet_seq=int(h.packet_no), t_ns=t, t_pi_mid_ns=t, arrival_ns=time.monotonic_ns(),
                          main_seq=int(h.main.frame_no), main_t_pi_ns=int(h.main.t_sensor_ns), main_rgb=self._mark,
                          pair_seq=int(h.ultrawide.frame_no),
                          pair_t_pi_ns=int(h.ultrawide.t_sensor_ns) if ok else None,
                          pair_t_pi_mid_ns=int(h.ultrawide.t_mid_ns) if ok else None,
                          ultrawide_rgb=self._mark if ok else None)
            self.buffer.push(bp)
            self._jpegs[bp.packet_seq] = (pkt.main_jpeg, pkt.ultrawide_jpeg)
            oldest = self.buffer.snapshot()[0].packet_seq
            for k in [k for k in self._jpegs if k < oldest]:
                del self._jpegs[k]

    def newest_seq(self) -> int | None:
        with self._lock:
            n = self.buffer.newest()
            return None if n is None else n.packet_seq

    def _prepped(self, seq: int, cam: str, jpeg: bytes, prep) -> np.ndarray:
        """The 224x224 RGB image of that packet's camera (kept for the next mosaics that pick the same packet)."""
        key = (seq, cam)
        img = self._decoded.get(key)
        if img is None:
            bgr = cv2.imdecode(np.frombuffer(jpeg, np.uint8), cv2.IMREAD_COLOR)   # as the copy's JpegDecoder
            if bgr is None:
                raise ValueError(f"packet {seq}: {cam} JPEG does not decode")
            img = prep(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
            self._decoded[key] = img
            while len(self._decoded) > self.MAX_DECODED:
                self._decoded.popitem(last=False)
        return img

    def mosaic(self) -> tuple[np.ndarray | None, str]:
        """(448, 448, 3) RGB: top main oldest..newest, bottom ultrawide oldest..newest; or (None, why not)."""
        with self._lock:
            packets = self.buffer.snapshot()
            jpegs = dict(self._jpegs)
        picks = self.selector.pick(packets)
        bad = [f"{c}: {p.reason}" for c, p in picks.items() if not p.ok]
        if bad:
            return None, "; ".join(bad)
        rows = []
        for cam, idx, prep in (("main", 0, self.prep.prep_main), ("ultrawide", 1, self.prep.prep_ultrawide)):
            row = []
            for p in picks[cam].packets:
                jp = jpegs.get(p.packet_seq)
                if jp is None:
                    return None, f"packet {p.packet_seq} dropped from the buffer"
                row.append(self._prepped(p.packet_seq, cam, jp[idx], prep))
            rows.append(np.concatenate(row, axis=1))
        return np.concatenate(rows, axis=0), ""
