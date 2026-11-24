# new in rnoro/main-sim (sim_deploy E1, check G6); not in rnoro/final-deployment @ f632576.
# E7 (integration): the cursor follows the time since the episode's first infer, not exec_h per infer (the manager
# asks for a new chunk at every frame, ~33 ms, and replaces the old one, so h steps per infer ran the recording ~12x
# too fast and the arm moved ~1/12 of the recorded path).
# 2026-11-23: axis-angle -> matrix in numpy (_matrix_from_rotvec, Rodrigues) instead of scipy's Rotation: the
# policy process may only import numpy, yaml, zmq and torch (tests/test_import_rules.py).
# W0 (re-base on f632576): the widths go through policy.gripper_map (to_robot) when it is set, as DiffusionPolicy's do:
# the recording is in training units (pick1: ~0.051 closed .. 0.090 open), the robot opens 0..0.040 m
# (limits.yaml gripper_width_max_m), so unmapped every replayed width would clip to 0.04 and never close.
# The relative-action math is behavior_prompting's UmiTaskDataset (pose_repr relative, umi_task_dataset.py) and its
# sampler's action slice (common/sampler.py: data[i : i + (16-1)*3 + 1 : 3], the last one repeated past the end).
"""ReplayPolicy: plays back one recorded training episode through C4, as a trained policy's chunks. No torch.

Settings: policy.replay_episode.dataset (a packaged <name>.zarr.zip or .zarr folder, as umiproc writes it) and
policy.replay_episode.index (the episode, 0-based in meta/episode_ends), policy.training_timing.obs_down_sample_steps
(d = 3: the data is 60 Hz, one C4 step is 50 ms = d data steps), policy.key_map.obs (the data keys: eef_pos ->
data/gripper_right_eef_pos, eef_rot6d -> data/gripper_right_eef_rot_axis_angle (stored as an axis-angle, 3 values),
gripper_width_m -> data/gripper_right_gripper_width), policy.gripper_map (training <-> robot widths, gripper_map.py;
null: the recorded widths unchanged) and the station's manager.exec_h (h).

The math. The episode is data rows [s, e) (s = episode_ends[index-1] or 0, e = episode_ends[index]). Row j has the
recorded TCP pose P_j = [R_j p_j; 0 1] (R_j = rotation of axis-angle rot_axis_angle_j, p_j = eef_pos_j, the frame of
the training data) and the width w_j. A cursor c is the data row of "now":

- reset(): c = s (the episode's first step).
- infer(obs): for k = 0..15, j_k = min(c + d*k, e - 1) and

      T_k = inv(P_c) @ P_{j_k}          (the recorded pose k steps ahead, in the frame of the recorded pose at c)
      eef_pos[k]   = T_k[:3, 3]
      eef_rot6d[k] = the first two ROWS of T_k[:3, :3] (pose_math.rot6d_from_matrix, behavior_prompting's mat_to_rot6d)
      width[k]     = gripper_map.to_robot(w_{j_k})   (absolute; the recorded width is in training units, the map
                                         turns it into the robot's opening; gripper_map null: w_{j_k} as recorded;
                                         make_chunk then clips it into [0, gripper_width_max_m])

  This is exactly the action a training sample at row c has (relative to the current pose, action_padding repeats
  the last row), so step 0 is the identity. The cursor of an infer is c = s + round(t * d / 0.05 s), t = the time
  since the episode's first infer (E7: the manager replans at every frame and replaces the chunk, so the replay
  follows the clock; with an exec_h-per-infer cursor it ran ~12x too fast).
- The manager turns step k into the waypoint O_T_TCP = T_anchor @ T_k at t_anchor + k*50 ms and sends the first h
  of them, so the robot's TCP repeats the recorded motion from wherever it is (the observation is ignored: open loop).
- Past the end (c = e - 1) every step is the identity with width w_{e-1}: the robot holds the last pose.

A skipped or slow step does not shift the replay: the cursor is the recording's time, the waypoints are relative
to the manager's anchor (its previous target), so the robot follows the recorded path in recorded time. Reading the zarr needs no zarr package: the three low-dim
arrays and meta/episode_ends are uncompressed (compressor null, no filters, C order) in umiproc's packages; anything
else is refused with a PolicyLoadError.
"""

from __future__ import annotations

import json
import math
import time
import zipfile
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
from deploy_contracts import ContractLimits

from policy_deploy.contracts import ACTION_STEPS, ActionChunk, Observation
from policy_deploy.policy.base_policy import Policy
from policy_deploy.policy.errors import PolicyLoadError
from policy_deploy.policy.gripper_map import GripperMap

if TYPE_CHECKING:
    from policy_deploy.policy.settings import Settings

STEP_S = 0.05                                      # one C4 step (training: 60 Hz data, d = 3)
DATASET_HINT = "check policy.replay_episode.dataset (a packaged .zarr.zip or .zarr folder)"


class _ZarrReader:
    """Reads uncompressed zarr v2 arrays from a .zarr.zip or a .zarr folder (read-only)."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self._zip: zipfile.ZipFile | None = None
        if path.is_dir():
            return
        try:
            self._zip = zipfile.ZipFile(path)
        except (OSError, zipfile.BadZipFile) as e:
            raise PolicyLoadError(f"cannot open {path}: {e}", DATASET_HINT) from e

    def _read(self, name: str) -> bytes | None:
        try:
            if self._zip is not None:
                return self._zip.read(name)
            return (self.path / name).read_bytes()
        except (KeyError, FileNotFoundError):
            return None

    def array(self, key: str) -> np.ndarray:
        raw = self._read(f"{key}/.zarray")
        if raw is None:
            raise PolicyLoadError(f"{self.path.name}: no array {key}", DATASET_HINT + " and policy.key_map.obs")
        meta = json.loads(raw)
        if meta.get("zarr_format") != 2 or meta.get("compressor") is not None or meta.get("filters") \
                or meta.get("order", "C") != "C":
            raise PolicyLoadError(f"{self.path.name}: {key} is compressed or filtered "
                                  f"(compressor {meta.get('compressor')!r}, filters {meta.get('filters')!r})",
                                  "the replay reads only uncompressed low-dim arrays, as umiproc packages them")
        dtype = np.dtype(meta["dtype"])
        shape, chunks = list(meta["shape"]), list(meta["chunks"])
        sep = meta.get("dimension_separator", ".")
        if any(s != c for s, c in zip(shape[1:], chunks[1:])):
            raise PolicyLoadError(f"{self.path.name}: {key} is chunked along a non-time axis", DATASET_HINT)
        out = np.full(shape, meta.get("fill_value") or 0, dtype=dtype)
        n = shape[0]
        for i in range(math.ceil(n / chunks[0]) if n else 0):
            name = f"{key}/" + sep.join([str(i)] + ["0"] * (len(shape) - 1))
            data = self._read(name)
            if data is None:
                continue                                      # a missing chunk is fill_value
            block = np.frombuffer(data, dtype=dtype).reshape([chunks[0]] + shape[1:])
            lo, hi = i * chunks[0], min(n, (i + 1) * chunks[0])
            out[lo:hi] = block[:hi - lo]
        return out

    def close(self) -> None:
        if self._zip is not None:
            self._zip.close()


def _matrix_from_rotvec(rotvec: np.ndarray) -> np.ndarray:
    """(N, 3) axis-angles -> (N, 3, 3) float64 rotations: Rodrigues, R = I + a K + b K^2 (K = [v]x, a = sin t / t,
    b = (1 - cos t) / t^2, t = |v|), with the Taylor series below t = 1e-3 as scipy's Rotation.from_rotvec does."""
    v = np.asarray(rotvec, dtype=np.float64).reshape(-1, 3)
    t2 = np.sum(v * v, axis=1)
    t = np.sqrt(t2)
    small = t < 1e-3
    ts = np.where(small, 1.0, t)                         # no 0/0 in the unused branch
    a = np.where(small, 1.0 - t2 / 6.0 + t2 * t2 / 120.0, np.sin(ts) / ts)
    b = np.where(small, 0.5 - t2 / 24.0 + t2 * t2 / 720.0, (1.0 - np.cos(ts)) / (ts * ts))
    K = np.zeros((len(v), 3, 3))
    K[:, 0, 1], K[:, 0, 2], K[:, 1, 2] = -v[:, 2], v[:, 1], -v[:, 0]
    K[:, 1, 0], K[:, 2, 0], K[:, 2, 1] = v[:, 2], -v[:, 1], v[:, 0]
    return np.eye(3) + a[:, None, None] * K + b[:, None, None] * (K @ K)


def _pose_matrices(pos: np.ndarray, rotvec: np.ndarray) -> np.ndarray:
    """(N, 3) positions + (N, 3) axis-angles -> (N, 4, 4) float64 poses (behavior_prompting pose_to_mat)."""
    T = np.zeros((len(pos), 4, 4))
    T[:, :3, :3] = _matrix_from_rotvec(rotvec)
    T[:, :3, 3] = np.asarray(pos, dtype=np.float64)
    T[:, 3, 3] = 1.0
    return T


def _invert(T: np.ndarray) -> np.ndarray:
    out = np.eye(4)
    R = T[:3, :3]
    out[:3, :3] = R.T
    out[:3, 3] = -R.T @ T[:3, 3]
    return out


class ReplayPolicy(Policy):
    KIND = "replay"

    def __init__(self, poses: np.ndarray, widths: np.ndarray, data_steps: int, exec_h: int,
                 gripper_width_max_m: float, info: dict[str, Any] | None = None,
                 gripper_map: GripperMap | None = None) -> None:
        """poses (n, 4, 4) and widths (n,; training units) of ONE episode; data_steps d (data rows per C4 step);
        exec_h h; gripper_map: training -> robot widths on the way out (None: unchanged), checked as DiffusionPolicy."""
        super().__init__(gripper_width_max_m)
        if gripper_map is not None and gripper_map.robot_open_m > gripper_width_max_m:
            raise PolicyLoadError(f"policy.gripper_map.robot_open_m {gripper_map.robot_open_m} is above "
                                  f"gripper_width_max_m {gripper_width_max_m}",
                                  "set robot_open_m to at most contracts/limits.yaml gripper_width_max_m")
        self._gripper_map = gripper_map
        if len(poses) == 0 or len(poses) != len(widths):
            raise PolicyLoadError(f"episode has {len(poses)} poses and {len(widths)} widths", DATASET_HINT)
        self._poses = np.asarray(poses, dtype=np.float64)
        self._widths = np.asarray(widths, dtype=np.float64).reshape(-1)
        self._d = int(data_steps)
        self._h = int(exec_h)
        self._info = dict(info or {})
        self._cursor = 0                           # episode row of "now" (0 = the episode's first step)
        self._t0 = None

    @classmethod
    def from_settings(cls, settings: "Settings") -> "ReplayPolicy":
        path = settings.replay_dataset
        if path is None:
            raise PolicyLoadError("policy.replay_episode.dataset is not set", DATASET_HINT)
        if not path.exists():
            raise PolicyLoadError(f"replay dataset not found: {path}", DATASET_HINT)
        names = settings.key_map["obs"]
        reader = _ZarrReader(path)
        try:
            ends = reader.array("meta/episode_ends").astype(np.int64)
            pos = reader.array(f"data/{names['eef_pos']}")
            rot = reader.array(f"data/{names['eef_rot6d']}")
            width = reader.array(f"data/{names['gripper_width_m']}")
        finally:
            reader.close()
        i = settings.replay_index
        if not 0 <= i < len(ends):
            raise PolicyLoadError(f"policy.replay_episode.index {i}: {path.name} has {len(ends)} episodes "
                                  f"(0-{len(ends) - 1})", "pick another index")
        s, e = (int(ends[i - 1]) if i > 0 else 0), int(ends[i])
        if rot.shape[1:] != (3,) or pos.shape[1:] != (3,):
            raise PolicyLoadError(f"{path.name}: expected (N, 3) eef_pos and axis-angle arrays, got {pos.shape} and "
                                  f"{rot.shape}", "check policy.key_map.obs")
        d = int(settings.training_timing.get("obs_down_sample_steps", 0))
        if d < 1:
            raise PolicyLoadError(f"policy.training_timing.obs_down_sample_steps {d}: expected >= 1",
                                  "the data rows per 50 ms step (3 for 60 Hz data)")
        info = {"dataset": str(path), "index": i, "rows": [s, e], "steps_50ms": math.ceil((e - s) / d),
                "data_steps_per_step": d, "exec_h": settings.exec_h}
        return cls(_pose_matrices(pos[s:e], rot[s:e]), width[s:e, 0], d, settings.exec_h,
                   ContractLimits.load().gripper_width_max_m, info, gripper_map=settings.gripper_map)

    def inputs(self) -> tuple[str, ...]:
        return ()                                  # open loop: the observation is not read

    def reset(self) -> None:
        self._cursor = 0
        self._t0 = None                            # E7: monotonic time of the episode's first infer

    def chunk_at(self, cursor: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """(eef_pos (16, 3), eef_rot6d (16, 6), width (16, 1)) float32 for the episode row `cursor` (no clipping;
        width in training units, before policy.gripper_map)."""
        last = len(self._poses) - 1
        c = min(cursor, last)
        rows = np.minimum(c + self._d * np.arange(ACTION_STEPS), last)
        rel = _invert(self._poses[c]) @ self._poses[rows]
        eef_pos = rel[:, :3, 3]
        eef_rot6d = rel[:, :2, :3].reshape(ACTION_STEPS, 6)          # the first two rows (pose_math.rot6d_from_matrix)
        width = self._widths[rows].reshape(ACTION_STEPS, 1)
        return eef_pos.astype(np.float32), eef_rot6d.astype(np.float32), width.astype(np.float32)

    def infer(self, obs: Observation) -> ActionChunk:
        now = time.monotonic()
        if self._t0 is None:
            self._t0 = now
        rows_per_s = self._d / STEP_S
        self._cursor = min(int(round((now - self._t0) * rows_per_s)), len(self._poses) - 1)
        eef_pos, eef_rot6d, width = self.chunk_at(self._cursor)
        if self._gripper_map is not None:
            width = self._gripper_map.to_robot(width)
        return self.make_chunk(eef_pos, eef_rot6d, width)

    @property
    def cursor(self) -> int:
        return self._cursor

    def describe(self) -> dict[str, Any]:
        return {"kind": self.KIND, **self._info,
                "gripper_map": self._gripper_map.as_dict() if self._gripper_map is not None else None}
