"""Builds raw teleop_collect task and episode folders with a known trajectory (PLAN.md §11). Shared by every test.

The files follow teleop_collect's formats (its PLAN §6; ``teleop/episode/recorders.py`` STATE_COLUMNS,
``frames/recorder.py`` FRAME_COLUMNS, ``frames/mkv_writer.py``, ``teleop/episode/meta.py``), written here
independently: teleop_train never imports teleop_collect. Wave 3 checks the reader against a real ``--mock`` episode.

The default trajectory, in seconds from the episode start (laptop clock ``T0_NS``):

    [0, still_start)              still at HOME_POS, TCP pointing down (rotation vector norm ~pi)
    + 2.0 s                       translate +x at 0.05 m/s (0.10 m)
    + 1.5 s                       rotate about the base z axis at 10 deg/s (15 deg), position fixed
    + 1.0 s                       gripper width 0.08 -> 0.02 m (6 cm/s), pose fixed
    + still_end                   still

``Truth`` gives the exact target pose and width at any time; the measured pose is the target shifted by
MEASURED_OFFSET_M in x, so a test can tell the two pose sources apart.

Frames are 640x480 JPEGs: the top half holds INDEX_BITS vertical bands (white = 1) encoding the frame's own
video_index, the bottom half a fixed gradient. ``decode_index`` reads the index back from a decoded frame.
"""
from __future__ import annotations

import datetime
import hashlib
import json
from dataclasses import dataclass, field
from fractions import Fraction
from pathlib import Path
from typing import Any, Callable

import av
import cv2
import numpy as np
import yaml
from scipy.spatial.transform import Rotation

FIXTURES = Path(__file__).resolve().parent / "fixtures"
RIG_FILE = FIXTURES / "rig_pi_v3_640.yaml"

T0_NS = 1_000_000_000_000            # episode start on the laptop clock
PC_OFFSET_NS = 123_456_789           # t_pc_ns = t_ns + PC_OFFSET_NS
PI_OFFSET_NS = -987_654_321          # t_*_pi_ns = t_ns + PI_OFFSET_NS
HOME_POS = np.array([0.5, 0.0, 0.4])
HOME_ROT = Rotation.from_euler("x", 180.0, degrees=True)     # TCP pointing down
MEASURED_OFFSET_M = np.array([0.001, 0.0, 0.0])
# the deployment TCP, deployment/contracts/frames.yaml as of final-deployment ea9f7fb (+45 deg about flange z, 150 mm)
F_T_TCP = np.array([[0.7071067811865476, -0.7071067811865476, 0.0, 0.0], [0.7071067811865476, 0.7071067811865476, 0.0, 0.0],
                    [0.0, 0.0, 1.0, 0.150], [0.0, 0.0, 0.0, 1.0]])
INDEX_BITS = 12
WIDTH, HEIGHT = 640, 480

STATE_COLUMNS = (
    "t_pc_ns", "t_ns", "arrival_ns", "seq", "mode",
    *[f"O_T_TCP_{r}{c}" for r in range(4) for c in range(4)],
    *[f"O_T_TCP_target_{r}{c}" for r in range(4) for c in range(4)],
    *(f"q_{i}" for i in range(7)), *(f"dq_{i}" for i in range(7)),
    "gripper_width_m", "gate_accepted", "gate_shortened", "gate_pulled_back", "gate_rejected", "loop_late", "error",
)
FRAME_COLUMNS = ("video_index", "frame_no", "t_sensor_pi_ns", "exposure_us", "t_mid_pi_ns", "t_mid_ns", "packet_no",
                 "arrival_ns", "bytes")
HEADER_ONLY = {
    "packets.csv": "packet_no,t_send_pi_ns,arrival_ns,main_frame_no,ultrawide_frame_no,uw_reused,pair_used,pair_gap_ms",
    "commands.csv": "t_send_ns,t_reply_ns,request,chunk_id,n",
    "keys.csv": "t_ns,held,slow,focus",
    "clock_pi.csv": "t_ns,offset_ns,drift_ppm,samples,used",
    "clock_pc.csv": "kind,t_send_ns,t_recv_ns,t_pc_ns,t_ns,offset_ns,drift_ppm,samples,used,rtt_min_ms",
}


@dataclass(frozen=True)
class Truth:
    """The exact trajectory. Times in ns on the laptop clock."""
    t0_ns: int
    still_start_s: float
    still_end_s: float
    move_s: float = 2.0
    rotate_s: float = 1.5
    grip_s: float = 1.0
    v_mps: float = 0.05
    w_deg_s: float = 10.0
    width_open: float = 0.08
    width_closed: float = 0.02

    @property
    def duration_s(self) -> float:
        return self.still_start_s + self.move_s + self.rotate_s + self.grip_s + self.still_end_s

    @property
    def motion_start_ns(self) -> int:
        return self.t0_ns + round(self.still_start_s * 1e9)

    @property
    def motion_end_ns(self) -> int:
        return self.t0_ns + round((self.still_start_s + self.move_s + self.rotate_s + self.grip_s) * 1e9)

    def _phases(self, t_ns) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        s = (np.asarray(t_ns, dtype=np.float64) - self.t0_ns) / 1e9 - self.still_start_s
        a = np.clip(s, 0.0, self.move_s)
        b = np.clip(s - self.move_s, 0.0, self.rotate_s)
        c = np.clip(s - self.move_s - self.rotate_s, 0.0, self.grip_s)
        return a, b, c

    def position(self, t_ns) -> np.ndarray:
        a, _, _ = self._phases(t_ns)
        out = np.tile(HOME_POS, (np.size(a), 1))
        out[:, 0] += self.v_mps * np.atleast_1d(a)
        return out

    def rotation(self, t_ns) -> Rotation:
        _, b, _ = self._phases(t_ns)
        return Rotation.from_euler("z", self.w_deg_s * np.atleast_1d(b), degrees=True) * HOME_ROT

    def width(self, t_ns) -> np.ndarray:
        _, _, c = self._phases(t_ns)
        return self.width_open + (self.width_closed - self.width_open) * np.atleast_1d(c) / self.grip_s

    def target(self, t_ns) -> np.ndarray:
        """(n, 4, 4) target poses."""
        p, r = self.position(t_ns), self.rotation(t_ns)
        T = np.tile(np.eye(4), (len(p), 1, 1))
        T[:, :3, :3] = r.as_matrix()
        T[:, :3, 3] = p
        return T

    def measured(self, t_ns) -> np.ndarray:
        T = self.target(t_ns)
        T[:, :3, 3] += MEASURED_OFFSET_M
        return T


@dataclass(eq=False)
class SyntheticEpisode:
    episode_dir: Path
    truth: Truth
    t_state_ns: np.ndarray       # the t_ns written (before any blanking)
    main_t_ns: np.ndarray        # t_mid_ns per written main frame (row i = video frame i)
    uw_t_ns: np.ndarray
    meta: dict[str, Any] = field(default_factory=dict)


def frame_image(index: int, width: int = WIDTH, height: int = HEIGHT) -> np.ndarray:
    """RGB uint8 (h, w, 3): the index as INDEX_BITS bands in the top half, a gradient below."""
    img = np.zeros((height, width, 3), np.uint8)
    band = width // INDEX_BITS
    for b in range(INDEX_BITS):
        if (index >> b) & 1:
            img[: height // 2, b * band:(b + 1) * band] = 255
    grad = np.linspace(0, 255, width, dtype=np.float64)
    img[height // 2:, :, 0] = grad.astype(np.uint8)
    img[height // 2:, :, 1] = grad[::-1].astype(np.uint8)
    img[height // 2:, :, 2] = 128
    return img


def decode_index(rgb: np.ndarray) -> int:
    """The index frame_image encoded, read back from a decoded full-size frame."""
    h, w = rgb.shape[:2]
    band = w // INDEX_BITS
    value = 0
    for b in range(INDEX_BITS):
        if rgb[h // 8: 3 * h // 8, b * band + band // 4:(b + 1) * band - band // 4].mean() > 128:
            value |= 1 << b
    return value


def jpeg_bytes(rgb: np.ndarray, quality: int = 90) -> bytes:
    ok, buf = cv2.imencode(".jpg", cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, quality])
    assert ok
    return buf.tobytes()


def write_mkv(path: Path, jpegs: list[bytes], width: int, height: int, fps: int = 30) -> None:
    """As teleop_collect's MkvWriter: each JPEG muxed as one mjpeg packet, never re-encoded."""
    tb = Fraction(1, 1000)
    with av.open(str(path), "w", format="matroska") as container:
        stream = container.add_stream("mjpeg", rate=fps)
        stream.width, stream.height, stream.pix_fmt = width, height, "yuvj420p"
        stream.time_base = tb
        container.start_encoding()
        for i, data in enumerate(jpegs):
            pkt = av.Packet(data)
            pkt.stream = stream
            pkt.pts = pkt.dts = i * 1000 // fps
            pkt.time_base = tb
            pkt.is_keyframe = True
            container.mux(pkt)


def _fmt(v: Any) -> str:
    if v is None:
        return ""
    if isinstance(v, (bool, np.bool_)):
        return "1" if v else "0"
    if isinstance(v, (int, np.integer)):
        return str(int(v))
    if isinstance(v, (float, np.floating)):
        return repr(float(v))
    return str(v)


def _write_csv(path: Path, columns, rows) -> None:
    with open(path, "w", encoding="utf-8", newline="") as fh:
        fh.write(",".join(columns) + "\n")
        for row in rows:
            fh.write(",".join(_fmt(v) for v in row) + "\n")


def load_rig() -> dict[str, Any]:
    return yaml.safe_load(RIG_FILE.read_text(encoding="utf-8"))


def make_task(raw_root: Path, task: str = "pick_test", description: str = "pick up the block") -> Path:
    """<raw_root>/<task>/task.yaml as teleop_collect's TaskStore.create writes it."""
    task_dir = Path(raw_root) / task
    task_dir.mkdir(parents=True, exist_ok=True)
    data = {"name": task, "description": description,
            "created": datetime.datetime(2026, 11, 19, 12, 0, tzinfo=datetime.timezone.utc).isoformat(),
            "next_episode": 0, "settings": {"motion": {"v_max_m_s": 0.05, "w_max_deg_s": 10.0}}}
    (task_dir / "task.yaml").write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")
    return task_dir


def make_episode(task_dir: Path, number: int = 0, *, outcome: str = "success", still_start_s: float = 2.0,
                 still_end_s: float = 2.0, state_hz: float = 100.0, main_fps: float = 30.0, uw_fps: float = 30.0,
                 lead_s: float = 0.2, frame_size: tuple[int, int] = (WIDTH, HEIGHT), state_gap: tuple[float, float] | None = None,
                 main_gap: tuple[float, float] | None = None, uw_gap: tuple[float, float] | None = None,
                 blank_state_t_ns: int = 0, blank_main_t_ns: int = 0, complete: bool = True,
                 rig: dict[str, Any] | None | str = "default", meta_edit: Callable[[dict], None] | None = None,
                 extra_main_packets: int = 0, width_fn: Callable[[np.ndarray], np.ndarray] | None = None,
                 jpeg_quality: int = 90) -> SyntheticEpisode:
    """One episode folder ``episode_%03d`` with every file teleop_collect writes.

    Streams start ``lead_s`` before the episode start and end ``lead_s`` after its end (states), the cameras start at
    small offsets. ``*_gap=(start_s, duration_s)`` leaves out the samples of that stream in that window (seconds from
    T0). ``blank_*_t_ns=k`` writes an empty mapped time in the first k rows. ``rig=None`` writes ``rig: null`` with a
    rig_note. ``extra_main_packets`` muxes that many extra frames into main.mkv without CSV rows. ``meta_edit`` may
    change the meta dict before it is written. ``width_fn`` replaces the true width (t_ns -> widths).
    """
    truth = Truth(T0_NS, still_start_s, still_end_s)
    name = f"episode_{number:03d}"
    ep = Path(task_dir) / name
    ep.mkdir(parents=True)
    end_ns = T0_NS + round(truth.duration_s * 1e9)

    def times(rate: float, offset_ns: int, gap) -> np.ndarray:
        step = 1e9 / rate
        n = int(((end_ns + lead_s * 1e9) - (T0_NS - lead_s * 1e9)) / step) + 1
        t = (T0_NS - round(lead_s * 1e9) + offset_ns + np.round(np.arange(n) * step)).astype(np.int64)
        if gap is not None:
            g0, g1 = T0_NS + round(gap[0] * 1e9), T0_NS + round((gap[0] + gap[1]) * 1e9)
            t = t[(t < g0) | (t >= g1)]
        return t

    t_state = times(state_hz, 0, state_gap)
    t_main = times(main_fps, 3_000_000, main_gap)
    t_uw = times(uw_fps, 7_000_000, uw_gap)

    # robot_states.csv
    T_t, T_m = truth.target(t_state), truth.measured(t_state)
    widths = width_fn(t_state) if width_fn is not None else truth.width(t_state)
    rows = []
    for i, t in enumerate(t_state):
        rows.append([int(t) + PC_OFFSET_NS, None if i < blank_state_t_ns else int(t), int(t) + 400_000, 1000 + i,
                     "following", *T_m[i].reshape(-1), *T_t[i].reshape(-1), *([0.0] * 7), *([0.0] * 7),
                     float(widths[i]), i, 0, 0, 0, 0, ""])
    _write_csv(ep / "robot_states.csv", STATE_COLUMNS, rows)

    # frames: CSVs and MKVs
    def frames(cam: str, t: np.ndarray, blank: int, extra: int) -> None:
        jpegs = [jpeg_bytes(frame_image(i, *frame_size), jpeg_quality) for i in range(len(t) + extra)]
        write_mkv(ep / f"{cam}.mkv", jpegs, *frame_size)
        rows = []
        for i, tm in enumerate(t):
            rows.append([i, 5000 + i, int(tm) + PI_OFFSET_NS - 4_000_000, 8000, int(tm) + PI_OFFSET_NS,
                         None if i < blank else int(tm), 7000 + i, int(tm) + 30_000_000, len(jpegs[i])])
        _write_csv(ep / f"{cam}_frames.csv", FRAME_COLUMNS, rows)

    frames("main", t_main, blank_main_t_ns, extra_main_packets)
    frames("ultrawide", t_uw, 0, 0)

    for fname, header in HEADER_ONLY.items():
        (ep / fname).write_text(header + "\n", encoding="utf-8")
    (ep / "events.jsonl").write_text(json.dumps({"t_ns": T0_NS, "event": "start"}) + "\n", encoding="utf-8")

    rig_content = load_rig() if rig == "default" else rig
    meta = {
        "meta_version": 1,
        "task": Path(task_dir).name,
        "task_description": "pick up the block",
        "episode": name,
        "episode_number": number,
        "outcome": outcome,
        "times": {"start_ns": T0_NS, "stop_ns": end_ns, "start_pc_ns": T0_NS + PC_OFFSET_NS,
                  "stop_pc_ns": end_ns + PC_OFFSET_NS},
        "duration_s": (end_ns - T0_NS) / 1e9,
        "counts": {"frames_main": len(t_main), "frames_ultrawide": len(t_uw), "c2_messages": len(t_state)},
        "frames": {},
        "pc_hello": {"commit": "abc1234", "mock": True, "F_T_TCP": F_T_TCP.tolist(), "max_lead_s": 0.5},
        "laptop": {"commit": "def5678", "dirty": False},
        "rig_file": str(RIG_FILE),
        "rig": rig_content,
        "rig_note": "" if rig_content is not None else f"{RIG_FILE}: FileNotFoundError",
        "teleop": {"v_max_m_s": 0.05, "w_max_deg_s": 10.0, "slow_factor": 0.25},
        "clock": {"pi": {}, "pc": {}},
    }
    if meta_edit is not None:
        meta_edit(meta)
    (ep / "meta.yaml").write_text(yaml.safe_dump(meta, sort_keys=False), encoding="utf-8")
    if complete:
        (ep / "COMPLETE").write_text("", encoding="utf-8")
    return SyntheticEpisode(ep, truth, t_state, t_main, t_uw, meta)


def tree_hash(root: Path) -> str:
    """sha256 over every file's relative path and bytes under root (to prove a folder was not changed)."""
    h = hashlib.sha256()
    for p in sorted(Path(root).rglob("*")):
        if p.is_file():
            h.update(str(p.relative_to(root)).encode())
            h.update(p.read_bytes())
    return h.hexdigest()
