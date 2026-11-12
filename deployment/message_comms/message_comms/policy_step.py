"""PolicyStep: one policy step, frames to waypoints, on the worker thread (STRUCTURE.md 4.5, 5.4 steps 1-8).

The order of the steps follows real_env/agents/policy_agent.py predict_actions (relative obs, infer, absolute from
the last obs pose, times from the last image time). Nothing is copied.
"""
from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

from contracts import ActionChunk, ContractError, HealthReport, Observation, SetWaypoints
from message_comms import pose_math
from message_comms.frames_client import FramesClient
from message_comms.policy_client import PolicyClient
from message_comms.pose_converter import Anchor, PoseConverter
from message_comms.problems import Problem, ProblemCode
from message_comms.settings import Settings
from message_comms.state_history import LookupMiss, StateHistory

NS_PER_MS = 1_000_000


@dataclass(frozen=True)
class StepJob:
    episode_id: int
    chunk_id: int                 # unique in the session, increasing
    start_pose: np.ndarray        # (4, 4): the episode start pose, for wrt_start
    last_anchor_t_ns: int | None  # the previous step's anchor time in this episode


@dataclass(frozen=True)
class StepTiming:
    t_begin_ns: int
    t_frames_ns: int
    t_policy_ns: int
    t_end_ns: int
    image_age_ms: float           # t_frames - anchor time


@dataclass(frozen=True)
class StepDone:
    job: StepJob
    waypoints: SetWaypoints
    anchor: Anchor
    health: HealthReport
    timing: StepTiming
    anchor_dev_m: float           # |p_target - p_measured| at the anchor time
    observation: Observation      # what was sent and received (for the tap)
    chunk: ActionChunk
    main_t_ns: np.ndarray         # the C1 image times (for the tap)
    ultrawide_t_ns: np.ndarray


@dataclass(frozen=True)
class StepSkipped:
    job: StepJob
    problem: Problem
    health: HealthReport | None
    width_clipped: int = 0        # the chunk's width_clipped (an odd chunk); 0 when no chunk arrived


@dataclass(frozen=True)
class NoNewFrame:
    job: StepJob
    health: HealthReport


class PolicyStep:
    def __init__(self, frames: FramesClient, policy: PolicyClient, history: StateHistory,
                 converter: PoseConverter, settings: Settings, clock: Callable[[], int]) -> None:
        self._frames = frames
        self._policy = policy
        self._history = history
        self._converter = converter
        self._s = settings
        self._clock = clock
        self.last_call: tuple[str, str] = ("process_frames", "c1.get_frames/v1")   # for the worker's error report

    def run(self, job: StepJob) -> StepDone | StepSkipped | NoNewFrame:
        """Detects problems (StepSkipped); client exceptions propagate to the worker."""
        s = self._s
        t_begin = self._clock()
        self.last_call = ("process_frames", "c1.get_frames/v1")
        frames = self._frames.get_frames()
        t_frames = self._clock()
        health = frames.health

        if frames.main is None:                                   # 2: a camera not ok -> no images
            return StepSkipped(job, self._camera_problem(health), health)

        t_a = int(frames.main_t_ns[-1])                            # 3
        if job.last_anchor_t_ns is not None and t_a <= job.last_anchor_t_ns:
            return NoNewFrame(job, health)

        age_ms = (t_frames - t_a) / NS_PER_MS                      # 4
        if age_ms > s.max_image_age_ms:
            return StepSkipped(job, Problem(ProblemCode.IMAGE_OLD, "process_frames",
                                            f"newest main image {age_ms:.0f} ms old"), health)

        looked_up = self._look_up(frames.main_t_ns, t_a, t_frames)  # 5
        if isinstance(looked_up, Problem):
            return StepSkipped(job, looked_up, health)
        poses, widths, source, dev_m = looked_up

        self.last_call = ("message_comms", "c4.infer/v1")          # 6: our own message
        try:
            obs, anchor = self._converter.to_observation(poses, widths, job.start_pose, frames, t_a, source)
        except ContractError as exc:
            raise RuntimeError(f"building the observation: {exc}") from exc

        self.last_call = ("policy", "c4.infer/v1")                 # 7
        chunk = self._policy.infer(obs)
        t_policy = self._clock()

        odd = self._odd_chunk(chunk)                               # 7b
        if odd is not None:
            return StepSkipped(job, odd, health, int(chunk.width_clipped))

        self.last_call = ("message_comms", "c3.set_waypoints/v1")  # 8
        try:
            waypoints = self._converter.to_waypoints(chunk, anchor, job.chunk_id)
        except ContractError as exc:
            raise RuntimeError(f"building the waypoints: {exc}") from exc
        t_end = self._clock()
        timing = StepTiming(t_begin, t_frames, t_policy, t_end, age_ms)
        return StepDone(job, waypoints, anchor, health, timing, dev_m, obs, chunk,
                        np.array(frames.main_t_ns), np.array(frames.ultrawide_t_ns))

    @staticmethod
    def _camera_problem(health: HealthReport) -> Problem:
        for code, status in ((ProblemCode.CAMERA_DOWN, "down"), (ProblemCode.CAMERA_STALE, "stale")):
            for name in ("main", "ultrawide"):
                cam = getattr(health, name)
                if cam.status == status:
                    detail = f"{name} camera {status}"
                    if status == "stale":
                        detail += f", last frame {cam.last_frame_age_ms:.0f} ms ago"
                    return Problem(code, "process_frames", detail)
        return Problem(ProblemCode.CAMERA_STALE, "process_frames", "no images although both cameras are ok")

    def _look_up(self, main_t_ns: np.ndarray, t_a: int, now_ns: int):
        """-> (poses (2,4,4), widths (2,), source, anchor_dev_m) or a Problem."""
        s = self._s
        latest = self._history.latest()
        if latest is None:
            return Problem(ProblemCode.STATE_OLD, "franka_control", "no arm state yet")
        state_age_ms = (now_ns - latest.t_ns) / NS_PER_MS
        if state_age_ms > s.max_state_age_ms:
            return Problem(ProblemCode.STATE_OLD, "franka_control", f"newest arm state {state_age_ms:.0f} ms old")
        target = self._history.pose_at(t_a, "target")
        measured = self._history.pose_at(t_a, "measured")
        for found in (target, measured):
            if isinstance(found, LookupMiss):
                return self._miss(found)
        dev_m = pose_math.position_distance_m(target.T, measured.T)
        source = s.pose_source
        if source == "target" and dev_m > s.anchor_fallback_m:
            source = "measured"
        poses, widths = [], []
        for t in main_t_ns:
            found = self._history.pose_at(int(t), source)
            if isinstance(found, LookupMiss):
                return self._miss(found)
            poses.append(found.T)
            widths.append(found.width_m)
        return np.stack(poses), np.array(widths), source, dev_m

    @staticmethod
    def _miss(miss: LookupMiss) -> Problem:
        code = ProblemCode.STATE_GAP if miss is LookupMiss.SEQ_GAP else ProblemCode.STATE_OLD
        return Problem(code, "franka_control", f"arm state at the image time: {miss.value}")

    def _odd_chunk(self, chunk: ActionChunk) -> Problem | None:
        s = self._s
        pos_m = float(np.linalg.norm(np.asarray(chunk.eef_pos[0], dtype=np.float64)))
        rot_deg = pose_math.rotation_angle_deg(pose_math.matrix_from_rot6d(chunk.eef_rot6d[0]))
        if pos_m > s.odd_chunk_max_pos_m or rot_deg > s.odd_chunk_max_rot_deg:
            return Problem(ProblemCode.ODD_CHUNK, "policy",
                           f"odd chunk: step 0 is {pos_m * 1000:.1f} mm / {rot_deg:.1f} deg from the anchor")
        return None
