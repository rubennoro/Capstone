"""The eval driver's links to the sim (DESIGN.md sections 1, 4, 5): scene control REQ -> :5651, C2 SUB :5620 (read
only, decoded with the copy's deploy_contracts) and the state export SUB :5650 (mirrored into the driver's own MuJoCo
model to render the external-camera video)."""

from __future__ import annotations

import json
import os
import threading
import time
from collections import Counter, deque
from pathlib import Path

import numpy as np
import zmq


class SceneError(RuntimeError):
    pass


class SceneClient:
    """REQ socket to the sim PC's scene control (JSON request -> JSON reply with ok/error)."""

    def __init__(self, endpoint: str = "tcp://127.0.0.1:5651", timeout_s: float = 2.0) -> None:
        self.endpoint = endpoint
        self.timeout_ms = int(timeout_s * 1000)
        self._ctx = zmq.Context.instance()
        self._sock = None
        self._open()

    def _open(self) -> None:
        if self._sock is not None:
            self._sock.close(linger=0)
        self._sock = self._ctx.socket(zmq.REQ)
        self._sock.setsockopt(zmq.LINGER, 0)
        self._sock.setsockopt(zmq.RCVTIMEO, self.timeout_ms)
        self._sock.setsockopt(zmq.SNDTIMEO, self.timeout_ms)
        self._sock.connect(self.endpoint)

    def call(self, req: dict) -> dict:
        try:
            self._sock.send(json.dumps(req).encode())
            rep = json.loads(self._sock.recv())
        except zmq.Again as e:
            self._open()                                  # a REQ socket is stuck after a timeout: start over
            raise SceneError(f"scene control {self.endpoint}: no reply to {req.get('cmd')} within "
                             f"{self.timeout_ms} ms (is the sim PC running?)") from e
        if not rep.get("ok", False):
            raise SceneError(f"scene control {req.get('cmd')}: {rep.get('error', rep)}")
        return rep

    def hello(self) -> dict:
        return self.call({"cmd": "hello"})

    def set_object(self, pose, name: str = "eraser") -> dict:
        return self.call({"cmd": "set_object", "name": name, "pose": [float(v) for v in pose]})

    def ground_truth(self) -> dict:
        return self.call({"cmd": "ground_truth"})

    def set_arm(self, q) -> dict:
        return self.call({"cmd": "set_arm", "q": [float(v) for v in q]})

    def key(self, key: str) -> dict:
        """P1's sim-only scene key: the PC keyboard key (h, p, g, s) through franka_control's own key queue. The
        reply carries the supervisor's line about it (``log``); a refused key raises SceneError with its reason."""
        return self.call({"cmd": "key", "key": key})

    def close(self) -> None:
        if self._sock is not None:
            self._sock.close(linger=0)
            self._sock = None


class C2Watcher(threading.Thread):
    """Keeps the newest C2 state and counts modes / errors over a window the driver opens per rollout."""

    def __init__(self, endpoint: str = "tcp://127.0.0.1:5620") -> None:
        super().__init__(name="c2-watch", daemon=True)
        from deploy_contracts import ArmState
        from deploy_contracts.pubsub import Subscriber

        self._sub = Subscriber("PC", endpoint, ArmState, 1000)
        self._lock = threading.Lock()
        self._halt = threading.Event()
        self.latest = None
        self.t_latest = 0.0
        self.n = 0
        self.decode_errors = 0
        self._window: dict | None = None
        self.mode_changes: deque = deque(maxlen=500)      # P2: (monotonic t, mode) at every change

    def run(self) -> None:
        from deploy_contracts import ContractError

        while not self._halt.is_set():
            try:
                msg = self._sub.receive(0.1)
            except ContractError:
                self.decode_errors += 1
                continue
            if msg is None:
                continue
            with self._lock:
                prev = self.latest.mode if self.latest is not None else None
                self.latest, self.t_latest = msg, time.monotonic()
                self.n += 1
                if msg.mode != prev:
                    self.mode_changes.append((self.t_latest, msg.mode))
                w = self._window
                if w is not None:
                    w["modes"][msg.mode] += 1
                    if msg.error and msg.error not in w["errors"]:
                        w["errors"].append(msg.error)
                    if msg.mode in ("fault", "stopped") and w["first_bad"] is None:
                        w["first_bad"] = {"mode": msg.mode, "error": msg.error, "t": time.monotonic()}
                    w["last"] = msg

    def open_window(self) -> None:
        with self._lock:
            self._window = {"modes": Counter(), "errors": [], "first_bad": None, "last": None,
                            "first": self.latest}

    def close_window(self) -> dict:
        with self._lock:
            w, self._window = self._window, None
        if w is None:
            return {}
        out = {"modes": dict(w["modes"]), "errors": w["errors"], "first_bad": w["first_bad"]}
        a, b = w["first"], w["last"]
        if a is not None and b is not None:
            out["gate"] = {k: int(getattr(b.gate, k)) - int(getattr(a.gate, k))
                           for k in ("accepted", "shortened", "pulled_back", "rejected")}
            out["loop_late"] = int(b.loop_late) - int(a.loop_late)
        return out

    def mode(self) -> str | None:
        with self._lock:
            return None if self.latest is None else self.latest.mode

    def modes_since(self, t: float) -> list[str]:
        """The modes C2 changed to after monotonic time t (P2: did the key's move show `homing`)."""
        with self._lock:
            return [m for (tm, m) in self.mode_changes if tm >= t]

    def stop(self) -> None:
        self._halt.set()
        self.join(timeout=1.0)
        self._sub.close()


class StateMirror(threading.Thread):
    """SUB to the state export (sim.state/v1 JSON): the newest qpos."""

    def __init__(self, endpoint: str = "tcp://127.0.0.1:5650", model_sha256: str | None = None) -> None:
        super().__init__(name="state-mirror", daemon=True)
        self._sock = zmq.Context.instance().socket(zmq.SUB)
        self._sock.setsockopt(zmq.LINGER, 0)
        self._sock.setsockopt(zmq.RCVHWM, 10)
        self._sock.setsockopt(zmq.SUBSCRIBE, b"")
        self._sock.connect(endpoint)
        self._halt = threading.Event()
        self._lock = threading.Lock()
        self.model_sha256 = model_sha256
        self.qpos: np.ndarray | None = None
        self.t_ns = 0
        self.n = 0
        self.sha_mismatch = 0

    def run(self) -> None:
        while not self._halt.is_set():
            if not self._sock.poll(100, zmq.POLLIN):
                continue
            try:
                m = json.loads(self._sock.recv())
            except (ValueError, zmq.ZMQError):
                continue
            if self.model_sha256 and m.get("model_sha256") not in (None, self.model_sha256):
                self.sha_mismatch += 1
                continue
            with self._lock:
                self.qpos = np.asarray(m["qpos"], float)
                self.t_ns = int(m.get("t_ns", 0))
                self.n += 1

    def newest(self) -> np.ndarray | None:
        with self._lock:
            return None if self.qpos is None else self.qpos.copy()

    def stop(self) -> None:
        self._halt.set()
        self.join(timeout=1.0)
        self._sock.close(linger=0)


class VideoRecorder(threading.Thread):
    """Renders ``cam_external`` from the mirrored qpos at ``fps`` while recording; writes an mp4 per rollout.
    The MuJoCo model and its GL context live in this thread only."""

    def __init__(self, scene_xml: Path, mirror: StateMirror, fps: float = 10.0, camera: str = "cam_external",
                 size=(640, 480)) -> None:
        super().__init__(name="video", daemon=True)
        self.scene_xml, self.mirror, self.fps, self.camera, self.size = Path(scene_xml), mirror, fps, camera, size
        self._halt = threading.Event()
        self._rec = threading.Event()
        self._frames: deque = deque()
        self._lock = threading.Lock()
        self.error: str | None = None

    def run(self) -> None:
        os.environ.setdefault("MUJOCO_GL", "egl")
        try:
            import mujoco

            m = mujoco.MjModel.from_xml_path(str(self.scene_xml))
            d = mujoco.MjData(m)
            # the robot look's photo backdrop (visual-only mesh "background") is a shell around the rig camera's
            # view; cam_external sits outside it and would film only its back: hide it in this copy only
            bg = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, "background")
            if bg >= 0 and not (m.geom_contype[bg] or m.geom_conaffinity[bg]):
                m.geom_rgba[bg, 3] = 0.0
                m.geom_matid[bg] = -1
            w, h = self.size
            r = mujoco.Renderer(m, h, w)
            opt = mujoco.MjvOption()
            for g in range(6):
                opt.geomgroup[g] = 1 if g in (0, 1, 2, 3) else 0
        except Exception as e:                            # noqa: BLE001 (no video is not fatal for an eval)
            self.error = f"video renderer: {e}"
            return
        period = 1.0 / self.fps
        nxt = time.monotonic()
        while not self._halt.is_set():
            now = time.monotonic()
            if now < nxt:
                time.sleep(min(nxt - now, 0.05))
                continue
            nxt += period
            if not self._rec.is_set():
                nxt = time.monotonic() + period
                continue
            q = self.mirror.newest()
            if q is None or len(q) != m.nq:
                continue
            d.qpos[:] = q
            mujoco.mj_forward(m, d)
            r.update_scene(d, camera=self.camera, scene_option=opt)
            img = r.render()
            with self._lock:
                self._frames.append(img.copy())

    def start_rollout(self) -> None:
        with self._lock:
            self._frames.clear()
        self._t_rec = time.monotonic()
        self._rec.set()

    def stop_rollout(self, path: Path) -> int:
        self._rec.clear()
        elapsed = time.monotonic() - getattr(self, "_t_rec", time.monotonic())
        time.sleep(0.15)
        with self._lock:
            frames = list(self._frames)
            self._frames.clear()
        if not frames:
            return 0
        import cv2

        h, w = frames[0].shape[:2]
        vw = None
        # mp4v (MPEG-4 part 2): what opencv-python-headless's bundled ffmpeg can encode (no H.264 encoder); VLC and
        # ffmpeg play it, some browsers do not
        # played back at the rate actually rendered (rendering may fall below `fps`), so the video runs in real time
        rate = min(self.fps, max(1.0, len(frames) / elapsed)) if elapsed > 0 else self.fps
        self.last_rate = rate
        vw = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), rate, (w, h))
        for f in frames:
            vw.write(f[:, :, ::-1])
        vw.release()
        return len(frames)

    def stop(self) -> None:
        self._halt.set()
        self.join(timeout=2.0)
