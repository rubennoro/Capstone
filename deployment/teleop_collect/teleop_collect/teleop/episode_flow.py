"""EpisodeFlow: tasks and the episode lifecycle on teleop's main thread (STRUCTURE.md 6.4; PLAN 4.3, 6, 7).

Start: a new episode folder (E4's Task.new_episode), its EpisodeRecording, R1 start_recording, C3 start, the motion
controller begun at the arm's width, the listener's state recorder on, phase RECORDING. Stop: phase STOPPING, motion
off, C3 stop, the state recorder off, R1 stop_recording (the summary), the recorders closed, then discard or
meta.yaml + COMPLETE (E4's build_meta and EpisodeDir.finish), phase IDLE. Abort (an EPISODE problem): the same, the
folder discarded, the reason on the page and in the session log only. abort_open is the conclude / Ctrl-C path and
never raises.

It uses the loop's clients through TeleopLoop._rpc (so a failure names whom it went to); every request is made on the
main thread.
"""
from __future__ import annotations

import logging
from collections.abc import Callable
from pathlib import Path
from typing import TYPE_CHECKING, Any

from deploy_contracts import Ack, ContractError, HelloReply, RemoteError, SetWaypoints, TransportTimeout

from teleop_collect.contracts import CAMERAS, NotSynced, RecordingSummary
from teleop_collect.teleop.episode.meta import build_meta
from teleop_collect.teleop.episode.recorders import EpisodeRecording
from teleop_collect.teleop.episode.store import EpisodeDir, EpisodeInfo, Task, TaskError, TaskStore
from teleop_collect.teleop.failure_monitor import SLOW_CODES
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
HOME_FIRST = "home the arm first (Home, or h on the PC)"
ABORT_STOP_TIMEOUT_S = 1.0       # R1 stop_recording on the abort paths: the folder is deleted anyway (SIGTERM < 5 s)
CAMERA_LATCHED = "{name} camera down: check the Pi (after a Pi restart: Ctrl-C and run teleop_collect again)"


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
        return list(self._episodes)

    def recording_name(self) -> str:
        return self.episode.name if self.episode is not None else ""

    def episode_s(self, now: int) -> float:
        return max(0.0, (now - self._t_start_ns) / NS_PER_S) if self.recording is not None else 0.0

    def clean_incomplete(self) -> list[Path]:
        """Start-up step 1: every episode folder without COMPLETE is deleted (a crash mid-episode), each logged."""
        deleted = self.store.clean_incomplete()
        self._reload_tasks()
        for path in deleted:
            self.lp._safe_event("incomplete_deleted", str(path))
            self.lp._messages.add("warn", f"deleted an incomplete episode (a crash mid-episode): {path}")
        return deleted

    # ---- tasks ---------------------------------------------------------------------------------------------------

    def _busy(self) -> str:
        from teleop_collect.teleop.teleop_loop import Phase
        if self.lp._phase in (Phase.RECORDING, Phase.STOPPING):
            return "an episode is recording: stop it first"
        return ""

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
        if phase is Phase.HOMING or lp._home_requested:
            return "homing"
        if phase is Phase.DONE:
            return "teleop has stopped"
        if self.task is None:
            return "select or create a task first"
        latest = lp._history.latest()
        if latest is None:
            return "no C2 state from the PC yet"
        if latest.mode not in ("idle", "stopped"):
            return f"the PC is {latest.mode}"
        if not lp.at_home(latest):
            return HOME_FIRST
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
            self._stop_frames()
            self._drop(f"the PC refused start: {ack.reason}")
            return Ack(ok=False, reason=f"the PC refused start: {ack.reason}")
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

    # ---- stop, abort ---------------------------------------------------------------------------------------------

    def stop(self, outcome: str) -> Ack:
        from teleop_collect.teleop.teleop_loop import Phase
        lp = self.lp
        if lp._phase is not Phase.RECORDING or self.recording is None:
            return Ack(ok=False, reason="no episode recording")
        lp._phase = Phase.STOPPING                                                    # 1
        lp._motion.stop()                                                             # 2
        recording, episode, task = self.recording, self.episode, self.task
        t_send = lp._clock()
        lp._send_stop()                                                               # 3
        t_stop = lp._clock()
        self.record_command(t_send, t_stop, "c3.stop/v1", None, Ack(ok=True, reason=""), None, False)
        stop_pc = self._to_pc(t_stop)
        lp._listener.set_recorder(None)
        self._flush_keys()
        try:
            summary = lp._rpc("frames", "r1.stop_recording/v1", lp._frames.stop_recording)   # 4
        except RemoteError as exc:
            self._r1_started = False
            return self._abort_on_stop(f"frames was not recording: {exc.reply.error}")
        except (TransportTimeout, ContractError) as exc:    # the C3 stop went out first: the watchdog is not armed
            self._r1_started = False
            return self._abort_on_stop(f"r1.stop_recording: {exc}")
        self._r1_started = False
        if summary.error:
            return self._abort_on_stop(summary.error)
        error = self.recorder_failure(lp._listener.recorder_error())
        if error:
            return self._abort_on_stop(error)
        self._write(lambda: recording.events.write(t_stop, "stop", {"outcome": outcome, "stop_pc_ns": stop_pc}))
        counts = self._counts(recording)
        try:
            recording.close()                                                         # 5
        except OSError as exc:
            self._write_error = self._write_error or f"closing the episode files: {exc}"
        if self._write_error:
            return self._abort_on_stop(self._write_error)
        if outcome == "discard":                                                      # 6
            episode.discard()
            text = f"{episode.name} discarded (deleted)"
        else:
            meta = self._meta(task, episode, outcome, t_stop, stop_pc, summary, counts)
            try:
                episode.finish(meta)
            except OSError as exc:
                return self._abort_on_stop(f"{episode.path}: meta.yaml / COMPLETE: {exc}")
            self._episodes.append(EpisodeInfo(episode.name, outcome, float(meta["duration_s"]),
                                              int(summary.main.frames)))
            text = (f"{episode.name} saved: {outcome}, {meta['duration_s']:.1f} s, {summary.main.frames} main "
                    f"frames")
        self._clear()
        lp._phase = Phase.IDLE                                                        # 7
        lp._safe_event("episode_stop", f"{task.name}/{text}")
        lp._messages.add("info", text)
        return Ack(ok=True, reason="")

    def _abort_on_stop(self, detail: str) -> Ack:
        verdict = self.lp._monitor.classify(Problem(C.RECORDING_FAILED, "frames", detail))
        self.abort(verdict)
        return Ack(ok=False, reason=f"{verdict.what_failed}: the episode was deleted")

    def abort(self, verdict: Verdict) -> None:
        """An EPISODE problem while recording: C3 stop, the recording closed, the folder deleted; back to IDLE."""
        from teleop_collect.teleop.teleop_loop import Phase
        lp = self.lp
        if self.episode is None:
            return
        lp._phase = Phase.STOPPING
        lp._motion.stop()
        if not lp._stop_sent and lp._beating:
            lp._send_stop()
        lp._listener.set_recorder(None)
        self._stop_frames()
        reason = f"{verdict.what_failed}: {verdict.what_to_check}"
        name = self.episode.name
        self._discard(reason)
        lp._phase = Phase.IDLE
        lp._safe_event("episode_aborted", f"{name}: {reason}")
        lp._messages.add("error", f"{name} aborted and deleted: {reason}")

    def abort_open(self, reason: str, frames_ok: bool = True) -> None:
        """Conclude, Ctrl-C, SIGTERM: whatever episode is open is deleted. Never raises (the loop is ending)."""
        lp = self.lp
        if self.episode is None:
            return
        name = self.episode.name
        try:
            lp._motion.stop()
            lp._listener.set_recorder(None)
        except Exception as exc:
            log.error("abort: %s", exc)
        if frames_ok:
            try:
                self._stop_frames()
            except Exception as exc:
                log.error("abort: R1 stop_recording: %s", exc)
        self._r1_started = False
        try:
            self._discard(reason)
        except Exception as exc:
            log.error("abort: discard %s: %s", name, exc)
        lp._safe_event("episode_aborted", f"{name}: {reason}")
        lp._messages.add("error", f"{name} aborted and deleted: {reason}")

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
        """R1 stop_recording if it was started (abort paths: the folder is deleted next). Not recording, a timeout or a
        broken reply is logged, never raised: the C3 stop went out first, so the PC's watchdog is not armed."""
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
        self.lp._start_ack_pc_ns = None

    def _counts(self, recording: EpisodeRecording) -> dict[str, int]:
        counts = dict(recording.counts())
        mon = self.lp._monitor.counts()
        for code in SLOW_CODES:
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
                          frames=summary, counts=counts, hello=hello, laptop_commit=commit, laptop_dirty=dirty,
                          rig_file=lp._s.rig_file, motion=lp._s.motion,
                          pi_fit={"offset_ns": summary.pi_clock_offset_ns, "drift_ppm": summary.pi_clock_drift_ppm},
                          pc_fit=lp.clock_row(t_stop))
