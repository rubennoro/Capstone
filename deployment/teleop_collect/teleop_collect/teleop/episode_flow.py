"""EpisodeFlow: tasks and the episode lifecycle on teleop's main thread (STRUCTURE.md 6.4; PLAN 4.3, 6, 7).

Start: a new episode folder (E4's Task.new_episode), its EpisodeRecording, R1 start_recording, C3 start, the motion
controller begun at the arm's width, the listener's state recorder on, phase RECORDING. Start is allowed only at the
start pose (StartPoseReference); the PC's answer to the C3 start confirms it, or (`refused: not at the start pose`)
drops it. Stop: phase STOPPING, motion off, C3 stop, the state recorder off, R1 stop_recording (the summary), the
recorders closed, then discard or meta.yaml + COMPLETE (E4's build_meta and EpisodeDir.finish), phase IDLE.

No episode is ever deleted or refused a save automatically: only the operator's Discard or delete removes one.
- abort (an EPISODE problem: the PC stopped the arm, or Stop arm): the same stop, then the episode is kept unsaved
  (meta.yaml with outcome `unsaved`, no COMPLETE; the reason in its `warnings`); the episode list offers Save success,
  Save fail and delete (label()).
- abort_open (conclude, Ctrl-C, SIGTERM; never raises): the same, with short waits; the next run lists it.
- A failure found at Stop (frames' summary, a recorder or disk error) is listed in meta.yaml `warnings` and the
  episode is saved as asked. Only a meta.yaml that cannot be written leaves the folder without it: unsaved, delete only.

Data problems (WARN: a camera down or stale, the recording failed) never stop the arm or end the episode: warn() shows
each once on the page and lists it in meta.yaml `warnings`.

The storage folder and whole tasks (only while no episode is open, after start-up):
- set_root (u1.set_data_root): a full path, never the other mode's folder (real vs --mock), made if missing and
  written to; a new TaskStore there, no task selected, its unsaved episodes named as at start-up; remembered in
  storage.local.yaml for later runs (contracts/station.py).
- delete_task (u1.delete_task): the task folder and every episode in it (TaskStore.delete_task).

It uses the loop's clients through TeleopLoop._rpc (so a failure names whom it went to); every request is made on the
main thread.
"""
from __future__ import annotations

import logging
from collections.abc import Callable
from pathlib import Path
from typing import TYPE_CHECKING, Any

from deploy_contracts import Ack, ContractError, HelloReply, RemoteError, SetWaypoints, TransportTimeout

from teleop_collect.contracts import CAMERAS, UNSAVED, CameraSummary, NotSynced, RecordingSummary, StationError
from teleop_collect.contracts.station import remember_data_root, roots_overlap, same_folder
from teleop_collect.teleop.episode.meta import build_meta
from teleop_collect.teleop.episode.recorders import EpisodeRecording
from teleop_collect.teleop.episode.store import (EpisodeDir, EpisodeInfo, Task, TaskError, TaskStore, episode_number,
                                                 probe_writable)
from teleop_collect.teleop.failure_monitor import COUNTED_CODES
from teleop_collect.teleop.motion.keys import HeldKeys
from teleop_collect.teleop.pc_clock import HelloSample
from teleop_collect.teleop.problems import Problem, ProblemCode, Verdict
from teleop_collect.teleop.startup_checks import split_commit

if TYPE_CHECKING:
    from teleop_collect.teleop.teleop_loop import TeleopLoop

log = logging.getLogger("teleop.episode_flow")
C = ProblemCode
NS_PER_S = 1_000_000_000
PC = "PC"
#: the PC's C3 start refusal away from its start pose (control_deploy supervisor onStart, once q_start_rad is set)
PC_NOT_AT_START_POSE = "refused: not at the start pose"
START_REFUSED_HERE = "the PC refused the start here (not the start pose): press p on the PC"
START_POSE_UNKNOWN = "put the arm at the start pose: h, then p on the PC"
NOT_AT_START_POSE = "not at the start pose: h, then p on the PC"
ABORT_STOP_TIMEOUT_S = 1.0       # R1 stop_recording on the quit / conclude paths (SIGTERM < 5 s)
CAMERA_LATCHED = "{name} camera down: check the Pi (after a Pi restart: Ctrl-C and run teleop_collect again)"
WARN_GOES_ON = "{episode}: {text}; the episode goes on (noted in its meta.yaml)"
WARN_FAILED = "{episode}: {text}; the episode goes on (data may be missing; noted in its meta.yaml)"
KEPT_UNSAVED = ("{episode} ended: {reason}. Kept unsaved: in its row under Episodes, Save ✓ (success), "
                "Save ✗ (fail) or the trash icon")
KEPT_UNSAVED_NEXT_RUN = ("{episode} ended: {reason}. Kept unsaved: at the next run, select its task, then in its "
                         "row under Episodes Save ✓ (success), Save ✗ (fail) or the trash icon")
#: The summary in meta.yaml when frames gave none (R1 stop failed): zero counts; the failure is in `warnings`.
_EMPTY_CAMERA = CameraSummary(frames=0, frame_no_gaps=0, bytes=0, first_t_mid_ns=None, last_t_mid_ns=None)
NO_SUMMARY = RecordingSummary(main=_EMPTY_CAMERA, ultrawide=_EMPTY_CAMERA, packets=0, packet_no_gaps=0, uw_reused=0,
                              pair_refused=0, error="no summary from frames", pi_clock_offset_ns=None,
                              pi_clock_drift_ppm=None)
ENDED_CODE = "episode_ended"     # meta.yaml warnings code: why an unsaved episode ended (quit, conclude)
FULL_PATH = "give a full path: /... or ~/..."
OTHER_ROOT = "{root} is the storage folder of {other} runs: --mock episodes are kept apart from the real ones"
NESTED_ROOT = ("{root} and the storage folder of {other} runs ({other_root}) are inside one another: --mock episodes "
               "are kept apart from the real ones")
UNSAVED_THERE = "{n} unsaved episode(s) in {root}: select their task to save or delete them"
NOT_REMEMBERED = "storage folder {root} is used for this run only: it could not be remembered ({why})"


class EpisodeFlow:
    def __init__(self, loop: TeleopLoop, store: TaskStore) -> None:
        self.lp = loop
        self.store = store
        self.task: Task | None = None
        self.episode: EpisodeDir | None = None
        self.recording: EpisodeRecording | None = None
        self._r1_started = False
        self._t_start_ns = 0
        self._start_pc_ns: int | None = None
        self._tasks: list[str] | None = None            # cached: never re-read from disk while recording
        self._episodes: list[EpisodeInfo] = []
        self._write_error = ""                           # an OSError of a record_* write: RECORDING_FAILED
        self._warnings: list[dict[str, Any]] = []        # the episode's WARN problems, for meta.yaml `warnings`
        self._warned: set[str] = set()                   # their keys: each shown and listed once per episode

    # ---- the lists the page shows --------------------------------------------------------------------------------

    # The lists are read from disk only at start-up and after a task change (select, create); a stop, a discard or
    # a delete updates them from what was just written. Reading every meta.yaml on the main loop (status at 10 Hz)
    # would stall the 20 ms cycle and trip the PC's watchdog.

    def _reload_tasks(self) -> None:
        self._tasks = self.store.tasks()

    def _reload_episodes(self) -> None:
        self._episodes = self.task.episodes() if self.task is not None else []

    def tasks(self) -> list[str]:
        if self._tasks is None:                          # the first status before start-up step 1 (tests)
            self._reload_tasks()
        return list(self._tasks)

    def episodes(self) -> list[EpisodeInfo]:
        open_name = self.episode.name if self.episode is not None else None
        return [e for e in self._episodes if e.name != open_name]

    def recording_name(self) -> str:
        return self.episode.name if self.episode is not None else ""

    def episode_s(self, now: int) -> float:
        return max(0.0, (now - self._t_start_ns) / NS_PER_S) if self.recording is not None else 0.0

    def report_unsaved(self) -> list[Path]:
        """Start-up step 1: every episode folder without COMPLETE is kept and named on the page (label or delete it)."""
        found = self.store.unsaved()
        self._reload_tasks()
        for path in found:
            self.lp._safe_event("unsaved_kept", str(path))
            self.lp._messages.add("warn", f"unsaved episode kept from an earlier run: {path.parent.name}/{path.name} "
                                          "(select its task: Save ✓, Save ✗ or the trash icon in its row)")
        return found

    # ---- tasks ---------------------------------------------------------------------------------------------------

    def _busy(self) -> str:
        from teleop_collect.teleop.teleop_loop import Phase
        if self.lp._phase in (Phase.RECORDING, Phase.STOPPING):
            return "an episode is recording: stop it first"
        return ""

    def _not_idle(self) -> str:
        """Why the storage folder or a whole task cannot change now ("" when it can)."""
        from teleop_collect.teleop.teleop_loop import Phase
        busy = self._busy()
        if busy:
            return busy
        if self.lp._phase is Phase.STARTING:
            return "starting up: wait until teleop is ready"
        if self.lp._phase is Phase.DONE:
            return "teleop has stopped"
        return ""

    def set_root(self, path: str) -> Ack:
        """The storage folder (u1.set_data_root): checked, made, written to, then used and remembered."""
        refusal = self._not_idle()
        if refusal:
            return Ack(ok=False, reason=refusal)
        text = path.strip()
        try:
            root = Path(text).expanduser()               # RuntimeError: ~name of an unknown user
        except RuntimeError:
            return Ack(ok=False, reason=FULL_PATH)
        if not text.startswith(("/", "~")) or not root.is_absolute():
            return Ack(ok=False, reason=FULL_PATH)
        s = self.lp._s
        try:
            root = root.resolve()
            other = "real" if s.mock else "--mock"
            if same_folder(root, s.other_data_root):
                return Ack(ok=False, reason=OTHER_ROOT.format(root=root, other=other))
            if roots_overlap(root, s.other_data_root):
                return Ack(ok=False, reason=NESTED_ROOT.format(root=root, other=other,
                                                               other_root=s.other_data_root.resolve()))
            probe_writable(root)
        except (OSError, RuntimeError, ValueError) as exc:  # not writable, a symlink loop, a NUL in the path
            return Ack(ok=False, reason=f"cannot use {root}: {exc}")
        if not same_folder(root, self.store.root):
            self.store = TaskStore(root)
            self.task = None
            self._episodes = []
            self.lp._safe_event("data_root", f"storage folder: {root}")
            self.lp._messages.add("info", f"storage folder: {root}")
            found = self.store.unsaved()                 # one line for all (start-up names each)
            self._reload_tasks()
            for p in found:
                self.lp._safe_event("unsaved_kept", str(p))
            if found:
                self.lp._messages.add("warn", UNSAVED_THERE.format(n=len(found), root=root))
        try:
            remember_data_root(s.station_file, s.mock, root)
        except StationError as exc:
            self.lp._safe_event("data_root", f"not remembered: {exc}")
            self.lp._messages.add("warn", NOT_REMEMBERED.format(root=root, why=exc))
        return Ack(ok=True, reason="")

    def delete_task(self, name: str) -> Ack:
        """A whole task (u1.delete_task): its folder and every episode in it."""
        refusal = self._not_idle()                       # an rmtree must never run with the watchdog armed
        if refusal:
            return Ack(ok=False, reason=refusal)
        try:
            count = self.store.delete_task(name)
        except TaskError as exc:
            return Ack(ok=False, reason=str(exc))
        except OSError as exc:                           # part of it may be gone: deselected, the lists re-read
            self._forget_task(name)
            self.lp._safe_event("task_delete_failed", f"{name}: {exc}")
            return Ack(ok=False, reason=f"task {name} only partly deleted: {exc}")
        self._forget_task(name)
        self.lp._safe_event("task_deleted", f"{name} ({count} episodes)")
        self.lp._messages.add("info", f"task {name} deleted ({count} episodes)")
        return Ack(ok=True, reason="")

    def _forget_task(self, name: str) -> None:
        """After deleting a task (or part of it): deselected if it was selected; the task list re-read."""
        if self.task is not None and self.task.name == name:
            self.task = None
            self._episodes = []
        self._reload_tasks()

    def select(self, name: str) -> Ack:
        busy = self._busy()
        if busy:
            return Ack(ok=False, reason=busy)
        try:
            self.task = self.store.open(name)
        except TaskError as exc:
            return Ack(ok=False, reason=str(exc))
        self._reload_tasks()
        self._reload_episodes()
        self.lp._safe_event("task", f"selected {name}")
        return Ack(ok=True, reason="")

    def create(self, name: str, description: str) -> Ack:
        busy = self._busy()
        if busy:
            return Ack(ok=False, reason=busy)
        try:
            self.task = self.store.create(name, description, self.lp._s.task_settings())
        except (TaskError, OSError) as exc:
            return Ack(ok=False, reason=str(exc))
        self._reload_tasks()
        self._episodes = []
        self.lp._safe_event("task", f"created {name}: {description}")
        self.lp._messages.add("info", f"task {name} created")
        return Ack(ok=True, reason="")

    def delete(self, episode: str) -> Ack:
        if self._busy():                                 # an rmtree must never run with the watchdog armed
            return Ack(ok=False, reason="stop the episode first")
        if self.task is None:
            return Ack(ok=False, reason="select a task first")
        if self.episode is not None and episode == self.episode.name:
            return Ack(ok=False, reason=f"{episode} is being recorded")
        try:
            self.task.delete(episode)
        except (TaskError, OSError) as exc:
            return Ack(ok=False, reason=str(exc))
        self._episodes = [e for e in self._episodes if e.name != episode]
        self.lp._safe_event("episode_deleted", f"{self.task.name}/{episode}")
        self.lp._messages.add("info", f"{episode} deleted")
        return Ack(ok=True, reason="")

    def label(self, episode: str, outcome: str) -> Ack:
        """Save an unsaved episode as success or fail."""
        if self._busy():
            return Ack(ok=False, reason="stop the episode first")
        if self.task is None:
            return Ack(ok=False, reason="select a task first")
        if self.episode is not None and episode == self.episode.name:
            return Ack(ok=False, reason=f"{episode} is being recorded")
        try:
            info = self.task.label(episode, outcome)
        except (TaskError, OSError) as exc:
            return Ack(ok=False, reason=str(exc))
        self._put_episode(info)
        self.lp._safe_event("episode_labeled", f"{self.task.name}/{episode}: {outcome}")
        self.lp._messages.add("info", f"{episode} saved: {outcome}")
        return Ack(ok=True, reason="")

    # ---- start -----------------------------------------------------------------------------------------------------

    def start_refusal(self, now: int) -> str:
        """"" when Start is allowed, else why not: the first rule that applies (STRUCTURE.md 6.4)."""
        from teleop_collect.teleop.teleop_loop import Phase
        lp = self.lp
        phase = lp._phase
        if phase is Phase.STARTING:
            return "starting up"
        if phase in (Phase.RECORDING, Phase.STOPPING):
            return "recording"
        if phase is Phase.DONE:
            return "teleop has stopped"
        if self.task is None:
            return "select or create a task first"
        latest = lp._history.latest()
        if latest is None:
            return "no C2 state from the PC yet"
        if latest.mode not in ("idle", "stopped"):
            return f"the PC is {latest.mode}"
        ref = lp.start_pose
        if not ref.known:
            if ref.at_refused_pose(latest.q, latest.O_T_TCP_target):
                return START_REFUSED_HERE
            return START_POSE_UNKNOWN
        if not ref.at_start(latest.q, latest.O_T_TCP_target):
            return NOT_AT_START_POSE
        health = lp._last_health
        if health is None:
            return "waiting for the cameras"
        for name in CAMERAS:
            status = getattr(health.health, name).status
            if status == "down" and lp.camera_down_s(name, now) > lp._s.camera_down_s:
                return CAMERA_LATCHED.format(name=name)
            if status != "ok":
                return f"{name} camera {status}"
        if not lp.clock_ready(now):
            return "waiting for the PC clock map"
        return ""

    def start(self) -> Ack:
        from teleop_collect.teleop.teleop_loop import Phase
        lp = self.lp
        refusal = self.start_refusal(lp._clock())
        if refusal:
            return Ack(ok=False, reason=refusal)
        try:
            episode = self.task.new_episode()                                         # 1
        except (TaskError, OSError) as exc:
            return Ack(ok=False, reason=f"cannot make the episode folder: {exc}")
        try:
            recording = EpisodeRecording(episode, lp._s.motion.waypoints)             # 2
        except OSError as exc:
            episode.discard()
            return Ack(ok=False, reason=f"cannot open the episode files: {exc}")
        self.episode, self.recording = episode, recording
        ack = lp._rpc("frames", "r1.start_recording/v1", lp._frames.start_recording, str(episode.path))   # 3
        if not ack.ok:
            self._drop(f"frames refused to record: {ack.reason}")
            return Ack(ok=False, reason=f"frames refused to record: {ack.reason}")
        self._r1_started = True
        t_send = lp._clock()
        ack = lp._rpc(PC, "c3.start/v1", lp._pc.start)                               # 4
        t_reply = lp._clock()
        lp._note_request("c3.start/v1")
        self._write_error = ""
        self.record_command(t_send, t_reply, "c3.start/v1", None, ack, None, False)
        if not ack.ok:
            if ack.reason.startswith(PC_NOT_AT_START_POSE):
                self._refused_here()
            self._stop_frames()
            self._drop(f"the PC refused start: {ack.reason}")
            return Ack(ok=False, reason=f"the PC refused start: {ack.reason}")
        if not lp.start_pose.confirmed:
            lp.start_pose.confirm()
            lp._safe_event("start_pose", "confirmed: the PC accepted a start here; fixed for this session")
        lp._stop_sent = False
        self._t_start_ns = t_reply
        self._start_pc_ns = self._to_pc(t_reply)
        lp._start_ack_pc_ns = self._start_pc_ns
        latest = lp._history.latest()
        width = float(latest.gripper_width_m)                                         # 5
        width_max = lp._s.gripper_width_max_m
        clipped = min(max(width, 0.0), width_max)        # an open hand can read 0.081: the gate refuses > max
        lp._motion.begin(clipped)
        if clipped != width:
            text = f"start width {width:.4f} m clipped to {clipped:.4f} m (limits.yaml gripper_width_max_m)"
            lp._safe_event("width_clipped", text)
            lp._messages.add("info", text)
        lp._keys.changes()                                    # keys.csv starts with the episode
        lp._monitor.reset_episode_counts()
        lp._mismatch_count = 0
        lp._next_fit_ns = 0
        lp._listener.set_recorder(recording.states)
        lp._phase = Phase.RECORDING                                                   # 6
        self._write(lambda: recording.events.write(t_reply, "start", {"task": self.task.name,
                                                                      "episode": episode.name,
                                                                      "start_pc_ns": self._start_pc_ns}))
        lp._safe_event("episode_start", f"{self.task.name}/{episode.name}")
        lp._messages.add("info", f"REC {episode.name}: wait about a second before the first key")
        return Ack(ok=True, reason="")

    def _refused_here(self) -> None:
        """The PC refused the start away from its start pose: the learned pose is dropped (while not confirmed); the
        next arm move of the PC (h, then p) gives the new one."""
        lp = self.lp
        latest = lp._history.latest()
        if latest is None:
            return
        lp.start_pose.refused_here(latest.q, latest.O_T_TCP_target)
        lp._safe_event("start_pose", "the PC refused a start here (not its start pose): "
                                     + ("kept (confirmed)" if lp.start_pose.confirmed else
                                        "dropped; the pose after the PC's next h, then p is the start pose"))

    # ---- stop, abort ---------------------------------------------------------------------------------------------

    def stop(self, outcome: str) -> Ack:
        """Stop episode, then success / fail (saved) or discard (deleted). Never refused for a data problem: a failure
        found here is listed in meta.yaml `warnings` and the episode is saved as asked."""
        from teleop_collect.teleop.teleop_loop import Phase
        lp = self.lp
        if lp._phase is not Phase.RECORDING or self.recording is None:
            return Ack(ok=False, reason="no episode recording")
        lp._phase = Phase.STOPPING                                                    # 1
        lp._motion.stop()                                                             # 2
        t_send = lp._clock()
        lp._send_stop()                                                               # 3
        t_stop = lp._clock()
        self.record_command(t_send, t_stop, "c3.stop/v1", None, Ack(ok=True, reason=""), None, False)
        text = self._end(outcome, t_stop, None)                                       # 4-6
        lp._phase = Phase.IDLE                                                        # 7
        lp._messages.add("info", text)
        return Ack(ok=True, reason="")

    def _end(self, outcome: str, t_stop: int, r1_timeout_s: float | None) -> str:
        """After the C3 stop: the state recorder off, R1 stop_recording, the files closed, then the episode discarded,
        saved (success, fail) or kept unsaved. Every failure on the way is a meta.yaml warning, never a deletion.
        Returns the page line."""
        lp, recording, episode, task = self.lp, self.recording, self.episode, self.task
        stop_pc = self._to_pc(t_stop)
        lp._listener.set_recorder(None)
        self._flush_keys()
        summary = None
        if self._r1_started:                                                          # 4
            self._r1_started = False
            try:
                if r1_timeout_s is None:
                    summary = lp._rpc("frames", "r1.stop_recording/v1", lp._frames.stop_recording)
                else:
                    summary = lp._rpc("frames", "r1.stop_recording/v1", lp._frames.stop_recording, r1_timeout_s)
            except RemoteError as exc:
                self._failure(f"frames was not recording: {exc.reply.error}")
            except (TransportTimeout, ContractError) as exc:   # the C3 stop went out first: the watchdog is not armed
                self._failure(f"r1.stop_recording: {exc}")
        if summary is not None and summary.error:
            self._failure(summary.error)
        error = self.recorder_failure(lp._listener.recorder_error())
        if error:
            self._failure(error)
        self._write(lambda: recording.events.write(t_stop, "stop", {"outcome": outcome, "stop_pc_ns": stop_pc}))
        counts = self._counts(recording)
        try:
            recording.close()                                                         # 5
        except OSError as exc:
            self._failure(f"closing the episode files: {exc}")
        summary = summary if summary is not None else NO_SUMMARY
        name = episode.name
        if outcome == "discard":                                                      # 6
            episode.discard()
            text = f"{name} discarded (deleted)"
        else:
            try:
                meta = self._meta(task, episode, outcome, t_stop, stop_pc, summary, counts)
                if outcome == UNSAVED:
                    episode.keep_unsaved(meta)
                else:
                    episode.finish(meta)
            except Exception as exc:                          # the folder stays: listed unsaved, delete only
                log.error("%s: meta.yaml / COMPLETE: %s", episode.path, exc)
                info = EpisodeInfo(name, UNSAVED, 0.0, 0)
                text = (f"{name} kept unsaved, but its meta.yaml could not be written ({exc}): it can only be "
                        "deleted")
            else:
                info = EpisodeInfo(name, outcome, float(meta["duration_s"]), int(summary.main.frames))
                text = (f"{name} saved: {outcome}, {meta['duration_s']:.1f} s, {summary.main.frames} main frames"
                        if outcome != UNSAVED else f"{name} kept unsaved")
                if self._warnings:
                    text += f" ({len(self._warnings)} warning(s) in its meta.yaml)"
            self._put_episode(info)
        self._clear()
        lp._safe_event("episode_stop", f"{task.name}/{text}")
        return text

    def _failure(self, detail: str) -> None:
        """A recording failure found at the end: listed in meta.yaml `warnings` (each text once)."""
        log.warning("episode %s: %s", self.episode.name if self.episode else "-", detail)
        self._add_warning(C.RECORDING_FAILED.value, detail, f"{C.RECORDING_FAILED.value}:{detail}")

    def _add_warning(self, code: str, what: str, key: str) -> bool:
        if key in self._warned:
            return False
        self._warned.add(key)
        self._warnings.append({"code": code, "what": what, "t_s": round(self.episode_s(self.lp._clock()), 3)})
        return True

    def _put_episode(self, info: EpisodeInfo) -> None:
        eps = [e for e in self._episodes if e.name != info.name] + [info]
        self._episodes = sorted(eps, key=lambda e: episode_number(e.name))

    def warn(self, verdict: Verdict) -> None:
        """A WARN problem while recording (a camera down or stale, the recording failed): the arm and the episode go
        on. Each is shown on the page and listed in meta.yaml `warnings` once per episode (per code and camera; the
        first recording failure)."""
        if self.recording is None or self.episode is None:
            return
        problem = verdict.problem
        if problem.code is C.RECORDING_FAILED:
            key = problem.code.value
        else:
            key = f"{problem.code.value}:{problem.detail.split(',', 1)[0]}"
        if not self._add_warning(problem.code.value, verdict.what_failed, key):
            return
        lp, name = self.lp, self.episode.name
        text = f"{verdict.what_failed}: {verdict.what_to_check}"
        if problem.code is C.RECORDING_FAILED:
            page = WARN_FAILED.format(episode=name, text=text)
        else:
            page = WARN_GOES_ON.format(episode=name, text=text)
        lp._safe_event("episode_warning", f"{name}: {text}")
        self.event("warning", {"code": problem.code.value, "what": verdict.what_failed})
        lp._messages.add("warn", page)

    def abort(self, verdict: Verdict) -> None:
        """An EPISODE problem while recording (the PC stopped the arm, Stop arm): C3 stop, the recording closed, the
        episode kept unsaved with the reason in its `warnings`; back to IDLE. The list offers Save and delete."""
        from teleop_collect.teleop.teleop_loop import Phase
        lp = self.lp
        if self.episode is None:
            return
        lp._phase = Phase.STOPPING
        lp._motion.stop()
        if not lp._stop_sent and lp._beating:
            lp._send_stop()
        t_stop = lp._clock()
        problem = verdict.problem
        self._add_warning(problem.code.value, verdict.what_failed, f"ended:{problem.code.value}")
        self.event("ended", {"code": problem.code.value, "what": verdict.what_failed})
        reason = f"{verdict.what_failed}: {verdict.what_to_check}"
        name = self.episode.name
        self._end(UNSAVED, t_stop, None)
        lp._phase = Phase.IDLE
        lp._safe_event("episode_ended", f"{name}: {reason}")
        lp._messages.add("warn", KEPT_UNSAVED.format(episode=name, reason=reason))

    def abort_open(self, reason: str, frames_ok: bool = True) -> None:
        """Conclude, Ctrl-C, SIGTERM: whatever episode is open is kept unsaved (the next run lists it; short waits).
        Never raises (the loop is ending); on an error the folder is left as it is, never deleted."""
        lp = self.lp
        if self.episode is None:
            return
        name = self.episode.name
        try:
            lp._motion.stop()
        except Exception as exc:
            log.error("abort: %s", exc)
        if not frames_ok:
            self._r1_started = False                     # frames failed: no R1 request
        try:
            self._add_warning(ENDED_CODE, reason, f"ended:{ENDED_CODE}")
            self.event("ended", {"code": ENDED_CODE, "what": reason})
            self._end(UNSAVED, lp._clock(), ABORT_STOP_TIMEOUT_S)
        except Exception as exc:
            log.error("abort: keeping %s: %s", name, exc)
            try:
                if self.recording is not None:
                    self.recording.close()
            except Exception:
                pass
            self._clear()
        lp._safe_event("episode_ended", f"{name}: {reason}")
        lp._messages.add("warn", KEPT_UNSAVED_NEXT_RUN.format(episode=name, reason=reason))

    # ---- the recorders -------------------------------------------------------------------------------------------

    def _write(self, write: Callable[[], None]) -> None:
        """One write to the episode's files on the main thread. An OSError (disk full, ...) never reaches the loop's
        guard (that would conclude): it is kept, and the next cycle's step 2 aborts the episode with RECORDING_FAILED
        (PLAN 7); later writes are skipped."""
        if self.recording is None or self._write_error:
            return
        try:
            write()
        except OSError as exc:
            self._write_error = f"{type(exc).__name__}: {exc}"
            log.error("episode file write failed: %s", self._write_error)

    def event(self, kind: str, detail: Any) -> None:
        rec = self.recording
        if rec is not None:
            self._write(lambda: rec.events.write(self.lp._clock(), kind, detail))

    def record_command(self, t_send_ns: int, t_reply_ns: int | None, request: str, msg: SetWaypoints | None,
                       reply, twist, slow: bool) -> None:
        rec = self.recording
        if rec is not None:
            self._write(lambda: rec.commands.write(t_send_ns, t_reply_ns, request, msg, reply, twist, slow))

    def record_hello(self, sample: HelloSample | None, reply: HelloReply) -> None:
        rec = self.recording
        if rec is None or sample is None:
            return
        self._write(lambda: rec.pc_clock.hello(sample.t_send_ns, sample.t_recv_ns, sample.t_pc_ns))
        self._write(lambda: rec.commands.write(sample.t_send_ns, sample.t_recv_ns, "c3.hello/v1", None, reply, None,
                                               False))

    def record_keys(self, t_ns: int, keys: HeldKeys, focus: bool) -> None:
        rec = self.recording
        if rec is not None:
            self._write(lambda: rec.keys.write(t_ns, keys, focus))

    def record_fit(self, now: int) -> None:
        rec = self.recording
        if rec is None:
            return
        st = self.lp._pc_clock.stats(now)
        self._write(lambda: rec.pc_clock.fit(now, st.offset_ns, st.drift_ppm, st.samples, st.used, st.rtt_min_ms))

    def recorder_failure(self, listener_error: BaseException | None) -> str:
        """"" or why the recording failed: a main-thread write (record_*), E4's state recorder (it keeps its own
        write error), or any other exception the listener's hook kept."""
        if self.recording is None:
            return ""
        if self._write_error:
            return self._write_error
        if self.recording.error:
            return self.recording.error
        if listener_error is not None:
            return f"robot_states.csv: {type(listener_error).__name__}: {listener_error}"
        return ""

    # ---- helpers -------------------------------------------------------------------------------------------------

    def _to_pc(self, t_ns: int) -> int | None:
        try:
            return self.lp._pc_clock.snapshot().to_pc_ns(t_ns)
        except NotSynced:
            return None

    def _flush_keys(self) -> None:
        for t_ns, keys, focus in self.lp._keys.changes():
            self.record_keys(t_ns, keys, focus)

    def _stop_frames(self) -> RecordingSummary | None:
        """R1 stop_recording if it was started (a refused start: the folder is deleted next). Not recording, a
        timeout or a broken reply is logged, never raised: the C3 stop went out first, so the PC's watchdog is not armed."""
        if not self._r1_started:
            return None
        self._r1_started = False
        lp = self.lp
        try:
            return lp._rpc("frames", "r1.stop_recording/v1", lp._frames.stop_recording, ABORT_STOP_TIMEOUT_S)
        except (RemoteError, TransportTimeout, ContractError) as exc:
            log.warning("R1 stop_recording: %s", exc)       # the folder is deleted anyway
            self.lp._safe_event("r1_stop_failed", str(exc))
            return None

    def _drop(self, reason: str) -> None:
        """Start failed: the folder is deleted, no message beyond the refusal."""
        self._discard(reason)
        self.lp._messages.add("warn", f"start refused: {reason}")

    def _discard(self, reason: str) -> None:
        recording, episode = self.recording, self.episode
        if recording is not None:
            try:
                recording.events.write(self.lp._clock(), "aborted", reason)
            except Exception:
                pass
            try:
                recording.close()
            except OSError as exc:                       # the folder is deleted anyway
                log.error("closing %s: %s", episode, exc)
        if episode is not None:
            episode.discard()
        self._clear()

    def _clear(self) -> None:
        self.episode = None
        self.recording = None
        self._r1_started = False
        self._write_error = ""
        self._warnings = []
        self._warned = set()
        self.lp._start_ack_pc_ns = None

    def _counts(self, recording: EpisodeRecording) -> dict[str, int]:
        counts = dict(recording.counts())
        mon = self.lp._monitor.counts()
        for code in COUNTED_CODES:
            counts.setdefault(code.value, int(mon.get(code, 0)))
        return counts

    def _meta(self, task: Task, episode: EpisodeDir, outcome: str, t_stop: int, stop_pc: int | None,
              summary: RecordingSummary, counts: dict) -> dict:
        lp = self.lp
        commit, dirty = split_commit(lp._checks.commit)
        hello = lp._heartbeats.last_hello() or lp._hello
        return build_meta(task=task, episode=episode, outcome=outcome,
                          times={"start_ns": int(self._t_start_ns), "stop_ns": int(t_stop),
                                 "start_pc_ns": self._start_pc_ns, "stop_pc_ns": stop_pc},
                          frames=summary, counts=counts, warnings=self._warnings, hello=hello,
                          laptop_commit=commit, laptop_dirty=dirty,
                          rig_file=lp._s.rig_file, motion=lp._s.motion,
                          pi_fit={"offset_ns": summary.pi_clock_offset_ns, "drift_ppm": summary.pi_clock_drift_ppm},
                          pc_fit=lp.clock_row(t_stop))
