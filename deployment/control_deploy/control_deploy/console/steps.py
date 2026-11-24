"""The console's steps (control_deploy/STRUCTURE.md 6.5): HoldStep, MoveStep, RotateStep, GripperStep, WatchdogStep,
OutOfBoxStep. Each step drives the PC through the Console's helpers and is judged by a pure evaluate_* function on the
recorded C2 samples and C3 replies, so the judging is tested on synthetic sequences."""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import numpy as np
from deploy_contracts import Ack, ArmState, SetWaypointsReply

from control_deploy.console import waypoints as wpts

if TYPE_CHECKING:
    from control_deploy.console.console import Console

MM = 1e-3
DEG = math.pi / 180.0
HOLD_DRIFT_M = 2 * MM
HOLD_DRIFT_RAD = 1.0 * DEG
HOLDING_WITHIN_S = 0.1
PLAN_TOL_M = 0.5 * MM
PLAN_TOL_RAD = 0.05 * DEG
TRACK_PASS_M, TRACK_WARN_M = 3 * MM, 10 * MM
TRACK_PASS_RAD, TRACK_WARN_RAD = 1.0 * DEG, 3.0 * DEG
GRIPPER_TOL_M = 7 * MM
GRIPPER_HOLD_S = 1.5
MOVE_HOLD_S = 0.5             # the pause between out and back
WATCHDOG_EXTRA_S = 0.060
NEAR_FACE_M = 0.05


@dataclass(frozen=True)
class Sample:
    """One C2 message and when it arrived (console clock = PC CLOCK_MONOTONIC: the console runs on the PC)."""

    arrival_ns: int
    state: ArmState


@dataclass(frozen=True)
class Sent:
    """One set_waypoints: when it was sent and answered, the waypoints, and the reply."""

    send_ns: int
    reply_ns: int
    points: list[wpts.PlanPoint]
    reply: Any   # SetWaypointsReply, or ErrorReply / an exception text


@dataclass
class StepResult:
    name: str
    status: str                        # pass | warn | fail
    detail: str
    numbers: dict[str, Any] = field(default_factory=dict)

    def line(self) -> str:
        return f"{self.name}: {self.status.upper()}: {self.detail}"


def _pos(state: ArmState, key: str = "O_T_TCP") -> np.ndarray:
    return np.asarray(getattr(state, key))[:3, 3]


def _worst(parts: list[tuple[str, str]]) -> tuple[str, str]:
    order = {"pass": 0, "warn": 1, "fail": 2}
    status = max((s for s, _ in parts), key=lambda s: order[s]) if parts else "pass"
    shown = [t for s, t in parts if s != "pass"] or [t for _, t in parts]
    return status, "; ".join(shown)


def drift(samples: list[Sample]) -> tuple[float, float]:
    """The largest distance (m) and rotation (rad) of the measured TCP from the first sample."""
    if not samples:
        return 0.0, 0.0
    first = np.asarray(samples[0].state.O_T_TCP)
    d = max(float(np.linalg.norm(_pos(s.state) - first[:3, 3])) for s in samples)
    a = max(wpts.rotation_angle(first, np.asarray(s.state.O_T_TCP)) for s in samples)
    return d, a


def check_replies(sent: list[Sent]) -> list[tuple[str, str]]:
    """Every reply accounts for each waypoint, nothing rejected, nothing shortened (the moves are slow)."""
    out: list[tuple[str, str]] = []
    for s in sent:
        r = s.reply
        n = len(s.points)
        if not isinstance(r, SetWaypointsReply):
            out.append(("fail", f"set_waypoints got {r!r}"))
            continue
        if r.accepted + r.skipped + len(r.rejected) != n:
            out.append(("fail", f"reply to chunk {r.chunk_id} accounts for {r.accepted + r.skipped + len(r.rejected)} "
                                f"of {n} waypoints"))
        if r.rejected:
            out.append(("fail", f"chunk {r.chunk_id}: rejected: {r.rejected[0].reason}"))
        if r.shortened:
            out.append(("fail", f"chunk {r.chunk_id}: {r.shortened} shortened (the console's moves are far below the "
                                f"gate's speed limits)"))
    return out


# ---- judging, pure ----

def evaluate_hold(ack: Any, ack_ns: int, samples: list[Sample], hold_samples: list[Sample]) -> StepResult:
    parts: list[tuple[str, str]] = []
    nums: dict[str, Any] = {}
    if not (isinstance(ack, Ack) and ack.ok):
        return StepResult("hold", "fail", f"start was not accepted: {getattr(ack, 'reason', ack)}")
    holding = next((s for s in samples if s.arrival_ns >= ack_ns and s.state.mode == "holding"), None)
    if holding is None:
        parts.append(("fail", "C2 never showed holding"))
    else:
        dt = (holding.arrival_ns - ack_ns) * 1e-9
        nums["holding_after_s"] = round(dt, 4)
        parts.append(("pass" if dt <= HOLDING_WITHIN_S else "fail", f"C2 holding {dt * 1e3:.0f} ms after the ack"))
    d, a = drift(hold_samples)
    nums.update(drift_mm=round(d / MM, 3), drift_deg=round(a / DEG, 3))
    ok = d <= HOLD_DRIFT_M and a <= HOLD_DRIFT_RAD
    parts.append(("pass" if ok else "fail", f"drift {d / MM:.2f} mm, {a / DEG:.2f} deg over the hold"))
    if hold_samples:
        late = int(hold_samples[-1].state.loop_late) - int(hold_samples[0].state.loop_late)
        nums["loop_late_new"] = late
        if late:
            parts.append(("warn", f"{late} new late 1 kHz cycles"))
    status, detail = _worst(parts)
    return StepResult("hold", status, detail, nums)


def evaluate_follow(name: str, rotation: bool, sent: list[Sent], plan: wpts.PlanModel, samples: list[Sample],
                    t_from_ns: int, t_to_ns: int) -> StepResult:
    """move (rotation False) or rotate: the plan check, the tracking, the modes, the replies."""
    parts = check_replies(sent)
    nums: dict[str, Any] = {}
    window = [s for s in samples if t_from_ns <= s.state.t_ns <= t_to_ns]
    if not window:
        return StepResult(name, "fail", "no C2 during the move", nums)
    plan_p = plan_a = track_p = track_a = 0.0
    for s in window:
        want, _ = plan.at(int(s.state.t_ns))
        target = np.asarray(s.state.O_T_TCP_target)
        measured = np.asarray(s.state.O_T_TCP)
        plan_p = max(plan_p, float(np.linalg.norm(target[:3, 3] - want[:3, 3])))
        plan_a = max(plan_a, wpts.rotation_angle(target, want))
        track_p = max(track_p, float(np.linalg.norm(measured[:3, 3] - target[:3, 3])))
        track_a = max(track_a, wpts.rotation_angle(measured, target))
    nums.update(plan_err_mm=round(plan_p / MM, 4), plan_err_deg=round(plan_a / DEG, 4),
                track_err_mm=round(track_p / MM, 3), track_err_deg=round(track_a / DEG, 3), samples=len(window))
    if rotation:
        parts.append(("pass" if plan_a <= PLAN_TOL_RAD else "fail", f"target vs sent path {plan_a / DEG:.3f} deg"))
        st = "pass" if track_a <= TRACK_PASS_RAD else "warn" if track_a <= TRACK_WARN_RAD else "fail"
        parts.append((st, f"tracking {track_a / DEG:.2f} deg"))
    else:
        parts.append(("pass" if plan_p <= PLAN_TOL_M else "fail", f"target vs sent path {plan_p / MM:.3f} mm"))
        st = "pass" if track_p <= TRACK_PASS_M else "warn" if track_p <= TRACK_WARN_M else "fail"
        parts.append((st, f"tracking {track_p / MM:.2f} mm"))
    modes = [s.state.mode for s in samples if s.state.t_ns >= t_from_ns]
    if "running" not in modes:
        parts.append(("fail", "C2 never showed running"))
    if not modes or modes[-1] != "holding":
        parts.append(("fail", f"C2 ends in {modes[-1] if modes else 'nothing'}, expected holding"))
    status, detail = _worst(parts)
    return StepResult(name, status, detail, nums)


def evaluate_gripper(sent: list[Sent], samples: list[Sample], commanded_m: float, t_check_ns: int) -> StepResult:
    parts = check_replies(sent)
    at = [s for s in samples if s.state.t_ns <= t_check_ns]
    if not at:
        return StepResult("gripper", "fail", "no C2 before the end of the hold")
    width = float(at[-1].state.gripper_width_m)
    err = abs(width - commanded_m)
    nums = {"commanded_m": round(commanded_m, 4), "measured_m": round(width, 4), "error_mm": round(err / MM, 2)}
    parts.append(("pass" if err <= GRIPPER_TOL_M else "warn",
                  f"width {width:.3f} m for {commanded_m:.3f} m commanded ({err / MM:.1f} mm"
                  + ("" if err <= GRIPPER_TOL_M else "; gripper.width_offset_m unmeasured?") + ")"))
    status, detail = _worst(parts)
    return StepResult("gripper", status, detail, nums)


def evaluate_watchdog(samples: list[Sample], last_request_ns: int, limit_s: float, restart: Any,
                      restarted: list[Sample], restart_ns: int) -> StepResult:
    parts: list[tuple[str, str]] = []
    nums: dict[str, Any] = {}
    stopped = next((s for s in samples if s.arrival_ns >= last_request_ns and s.state.mode == "stopped"), None)
    if stopped is None:
        return StepResult("watchdog", "fail", f"C2 never showed stopped within {limit_s + 1:.1f} s of the last request")
    elapsed = (stopped.arrival_ns - last_request_ns) * 1e-9
    nums["stopped_after_s"] = round(elapsed, 4)
    ok = limit_s <= elapsed <= limit_s + WATCHDOG_EXTRA_S
    parts.append(("pass" if ok else "fail", f"stopped {elapsed:.3f} s after the last request (limit {limit_s:g} s)"))
    d, _ = drift([s for s in samples if s.arrival_ns >= last_request_ns])
    nums["drift_mm"] = round(d / MM, 3)
    parts.append(("pass" if d <= HOLD_DRIFT_M else "fail", f"drift {d / MM:.2f} mm"))
    if not (isinstance(restart, Ack) and restart.ok):
        parts.append(("fail", f"start after the watchdog was refused: {getattr(restart, 'reason', restart)}"))
    elif not any(s.arrival_ns >= restart_ns and s.state.mode == "holding" for s in restarted):
        parts.append(("fail", "C2 not holding after the new start"))
    status, detail = _worst(parts)
    return StepResult("watchdog", status, detail, nums)


def evaluate_out_of_box(reply: Any, samples: list[Sample]) -> StepResult:
    parts: list[tuple[str, str]] = []
    nums: dict[str, Any] = {}
    if not isinstance(reply, SetWaypointsReply):
        return StepResult("out_of_box", "fail", f"set_waypoints got {reply!r}")
    nums.update(accepted=int(reply.accepted), rejected=len(reply.rejected))
    if reply.accepted:
        return StepResult("out_of_box", "fail", f"FAIL: THE GATE ACCEPTED {reply.accepted} WAYPOINT(S) OUTSIDE THE "
                                                f"WORKSPACE BOX: stop sent; check safety.workspace_* and the gate",
                          nums)
    ok = len(reply.rejected) == 1 and "outside workspace box" in reply.rejected[0].reason
    parts.append(("pass" if ok else "fail", f"rejected: {reply.rejected[0].reason if reply.rejected else 'nothing'}"))
    fault = next((s for s in samples if s.state.mode == "fault"), None)
    if fault is None or "safety gate rejected" not in fault.state.error:
        parts.append(("fail", "C2 did not show fault with \"safety gate rejected\""))
    else:
        parts.append(("pass", f"C2 fault: {fault.state.error}"))
    d, _ = drift(samples)
    nums["drift_mm"] = round(d / MM, 3)
    parts.append(("pass" if d <= HOLD_DRIFT_M else "fail", f"drift {d / MM:.2f} mm over 1 s"))
    status, detail = _worst(parts)
    return StepResult("out_of_box", status, detail, nums)


# ---- the steps ----

class Step:
    name = ""
    moves = True

    def describe(self, c: Console) -> str:
        return self.name

    def run(self, c: Console) -> StepResult:
        raise NotImplementedError


class HoldStep(Step):
    name = "hold"
    moves = False

    def describe(self, c: Console) -> str:
        return f"start, then hold for {c.settings.hold_s:g} s (nothing should move)"

    def run(self, c: Console) -> StepResult:
        c.drain()
        ack, ack_ns = c.call_start()
        c.run_for(0.15)
        t_hold = c.now()
        c.run_for(c.settings.hold_s)
        hold = [s for s in c.samples if s.arrival_ns >= t_hold]
        return evaluate_hold(ack, ack_ns, c.samples, hold)


class _FollowStep(Step):
    rotation = False

    def path_at(self, c: Console, start: np.ndarray, width: float) -> tuple[float, Any]:
        raise NotImplementedError

    def run(self, c: Console) -> StepResult:
        c.ensure_started()
        start, width = c.current_target()
        total, at = self.path_at(c, start, width)
        sent, plan, path = c.stream(total, at, start, width)
        t_from = sent[0].reply_ns if sent else c.now()
        return evaluate_follow(self.name, self.rotation, sent, plan, c.samples, t_from, path.end_ns)


class MoveStep(_FollowStep):
    name = "move"

    def describe(self, c: Console) -> str:
        return f"the TCP moves {c.settings.move_m * 100:.1f} cm up (base z) and back, {c.settings.segment_s:g} s each way"

    def path_at(self, c: Console, start: np.ndarray, width: float):
        total, s = wpts.out_hold_back(c.settings.segment_s, MOVE_HOLD_S)
        d = np.array([0.0, 0.0, c.settings.move_m])
        return total, lambda t: (wpts.translate(start, d * s(t)), width)


class RotateStep(_FollowStep):
    name = "rotate"
    rotation = True

    def describe(self, c: Console) -> str:
        return f"the hand turns {c.settings.rotate_deg:g} deg about the TCP z axis and back"

    def path_at(self, c: Console, start: np.ndarray, width: float):
        total, s = wpts.out_hold_back(c.settings.segment_s, MOVE_HOLD_S)
        angle = c.settings.rotate_deg * DEG
        return total, lambda t: (wpts.rotate_about_tcp_z(start, angle * s(t)), width)


class GripperStep(Step):
    name = "gripper"

    def commanded(self, c: Console, width: float) -> float:
        delta = c.settings.gripper_delta_m
        target = width + delta if width < 0.03 else width - delta
        return min(max(target, 0.0), c.limits.gripper_width_max_m)

    def describe(self, c: Console) -> str:
        return f"the fingers move by {c.settings.gripper_delta_m * 1000:.0f} mm, hold {GRIPPER_HOLD_S:g} s, and back"

    def run(self, c: Console) -> StepResult:
        c.ensure_started()
        start, _ = c.current_target()
        width = min(max(c.latest().gripper_width_m, 0.0), c.limits.gripper_width_max_m)
        target = self.commanded(c, width)
        seg = c.settings.segment_s
        total, s = wpts.out_hold_back(seg, GRIPPER_HOLD_S)
        sent, _, path = c.stream(total, lambda t: (start, width + (target - width) * s(t)), start, width)
        t_check = path.t0_ns + int((seg + GRIPPER_HOLD_S) * 1e9)
        return evaluate_gripper(sent, c.samples, target, t_check)


class WatchdogStep(Step):
    name = "watchdog"
    moves = False

    def describe(self, c: Console) -> str:
        return f"no request for {c.link.watchdog_limit_s:g} s: the PC must stop and hold, then start again"

    def run(self, c: Console) -> StepResult:
        c.ensure_started()
        c.heartbeat()
        last = c.last_request_ns
        c.listen(c.link.watchdog_limit_s + 0.6)   # C2 only: nothing is sent
        window = [s for s in c.samples if s.arrival_ns >= last]
        restart, restart_ns = c.call_start()
        c.run_for(0.2)
        return evaluate_watchdog(window, last, c.link.watchdog_limit_s, restart,
                                 [s for s in c.samples if s.arrival_ns >= restart_ns], restart_ns)


class OutOfBoxStep(Step):
    name = "out_of_box"

    def describe(self, c: Console) -> str:
        return (f"one waypoint {c.settings.out_of_box_m * 100:.0f} cm above the workspace box (z "
                f"{c.box_max[2] + c.settings.out_of_box_m:.3f} m): the gate must reject it; franka_control ends in "
                f"fault by design")

    def run(self, c: Console) -> StepResult:
        c.ensure_started()
        start, width = c.current_target()
        pose = np.array(start, copy=True)
        pose[2, 3] = c.box_max[2] + c.settings.out_of_box_m
        t = c.now() + 1_000_000_000
        t_sent = c.now()
        reply = c.send_waypoints([wpts.PlanPoint(t, pose, width)], chunk_id=900)
        if isinstance(reply, SetWaypointsReply) and reply.accepted:
            c.stop()
        c.run_for(1.0)
        return evaluate_out_of_box(reply, [s for s in c.samples if s.arrival_ns >= t_sent])


ORDER = ("hold", "move", "rotate", "gripper", "watchdog", "out_of_box")
STEPS: dict[str, type[Step]] = {"hold": HoldStep, "move": MoveStep, "rotate": RotateStep, "gripper": GripperStep,
                                "watchdog": WatchdogStep, "out_of_box": OutOfBoxStep}
