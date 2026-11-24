"""W1 deploy path: a recorded training episode through the copied laptop pipeline, exactly as on the robot, in-process,
on the real images. Light env (PyAV; no torch): ``~/envs/sim_policy_deploy/bin/python``.

    cd ~/Capstone-main-sim && unset PYTHONPATH
    ~/envs/sim_policy_deploy/bin/python -m sim_deploy.offline.deploy_path --dataset pick1-v1 --index 14 \
        --rows 6 9 36 72 [--station FILE] [--jpeg-quality 90] [--states rows|100]

writes ``sim_evals/_pipeline/W1/obs/<dataset>/ep<ii>_r<row>.deploy.npz`` + ``.deploy.json`` (build_obs.py merges them
with the direct path). Every piece of the pipeline is the copy's own code, imported (nothing re-implemented):

frames (C0 -> C1), as the frames process does it (frames/app.py ``FramesApp._start`` wiring):
  ``PiRecording.load(staging, sizes=camera_sizes(rig_pi_v3_640), jpeg_quality=q)`` (PyAV rgb24 decode, INTER_AREA to
  640x480, one JPEG per frame, as the Pi streams) -> ``PiRecording.packets()`` (``PacketPairer``: nearest ultrawide
  frame per main frame, every ultrawide frame sent once) -> ``Receiver.receive_one`` (``JpegDecoder``, ``PairRule``,
  ``UltrawideStore``, ``PacketBuffer``, ``CameraHealth``; the frames settings' keep_s / pair limits / default
  exposure) fed one packet at a time through a one-packet ``PacketSource`` -> at the decision time the real
  ``FramesServer.get_frames`` (an instance made without ``__init__``, so no RpcServer is bound) =
  ``PacketSelector.pick`` (main T-50/T, ultrawide T-100/T, anchored on the newest packet) + ``ImagePrep``
  (prep_main / prep_ultrawide) -> ``FramesReply``, sent through the C1 codec (``encode`` / ``decode``).
manager (C2 + C1 -> C4), as manager/__main__.py wires it: ``StateHistory(history_s)`` filled with ``ArmState``
  messages built from the package's recorded TCP poses + widths at their row times, then the real
  ``PolicyStep.run`` (image age check, one clock snapshot, ``_look_up`` with the pose_source / anchor_fallback rules,
  ``PoseConverter.to_observation``, odd-chunk check, ``to_waypoints``) with a stand-in frames client (returns the
  reply above) and a stand-in policy client (C4 ``encode`` / ``decode`` of the Observation, which is captured, and
  a hold chunk back). ``PoseConverter`` gets no workspace box (the recorded poses are in the table-tag frame, the
  box is the robot base frame); exec_h / lookahead from the station's manager settings.

Clocks: Pi clock = laptop clock = PC clock (identity maps; the PC clock map is a stand-in that is always ready). A
packet arrives SEND_DELAY_NS (45.8 ms, the live Pi, fake_pi_sender.py) after its main frame's t_sensor; "now" for a
decision at row T is 1 us after the arrival of the packet whose IMX519 frame is training's main frame at row T
(umiproc's row -> frame map: the nearest main frame by mid-exposure to the row's grid time, rebuilt by
camera_study/reference.py ``_Episode``), so the buffer holds exactly the packets that had arrived by then and that
packet is the newest. C2 states: the 60 Hz package rows themselves (``--states rows``, default: the pose at a row time
is then exactly the recorded one, and the pose at an image time is one StateHistory interpolation between two
recorded rows; a 100 Hz resampling would add a second interpolation the data does not have) or ``--states 100``
(100 Hz like the PC, interpolated from the rows with the copy's pose_math.interpolate). A state is in the history
when its t_ns <= now. target = measured = the recorded pose (so anchor_dev 0, source "target"). The episode start
pose = the state at row 0 (episode_loop: the newest C2 target when the episode starts). C2 widths: if the policy
settings have ``policy.gripper_map`` (policy/gripper_map.py), ``GripperMap.to_robot(recorded)`` (what the robot would
report for that opening; the policy maps it back with ``to_model`` in ``infer``), else the recorded width.

Widths in BOTH conventions (coordinator addendum, until W0 settles it): the C2 states carry the recorded width
(convention A, the sim's D8: training width = what the robot reports, pass-through). Convention B (upstream
f632576 ``policy.gripper_map``: C2 robot width = ``GripperMap.to_robot(training width)``) is stored next to it as
``deploy_gripper_width_m_B`` = ``to_robot`` of the A widths (to_robot is affine, so mapping the states before the
StateHistory interpolation or the looked-up widths after it is the same up to float32 rounding). The GripperMap
class is the copy's ``policy/gripper_map.py`` (or, if the copy has none, the f632576 file extracted to scratch);
its numbers are the copy's ``policy.gripper_map`` config (or f632576's config/policy.yaml defaults).

Target vs measured (known deploy-vs-training difference): the manager's anchor and low-dim come from C2
``O_T_TCP_target`` by default (``manager.pose_source: target``; ``PolicyStep._look_up`` falls back to measured when
|target - measured| > anchor_fallback_m), training's low-dim is the tracked (measured) pose. Offline target =
measured = recorded; ``--target-lead-ms L`` makes the target the recorded pose L ms later (a commanded target leads
the measured pose) to quantify the effect.
"""

from __future__ import annotations

import argparse
import dataclasses
import inspect
import sys
import time
from pathlib import Path

import numpy as np

from sim_deploy.offline.common import (OBS_CACHE, SCRATCH, STATION_TEMPLATE, deploy_version, set_cpu_threads,
                                       write_json)
from sim_deploy.offline.episodes import EpisodeRef, episode, pi_staging

ARRIVAL_EPS_NS = 1_000          # "now" = newest packet's arrival + 1 us


class IdentityClockMap:
    """Pi clock = laptop clock (the brief's identity map), always fresh. Only what Receiver calls."""

    def is_fresh(self, at_local_ns: int) -> bool:
        return True

    def to_local_ns(self, t_remote_ns: int, at_local_ns: int | None = None) -> int:
        return int(t_remote_ns)


class _IdentityFit:
    def to_pc_ns(self, t_ns: int) -> int:
        return int(t_ns)


class IdentityPcClock:
    """Laptop clock = PC clock; the map is always ready. Only what PolicyStep calls."""

    def ready(self, now_ns: int) -> bool:
        return True

    def not_ready_reason(self, now_ns: int) -> str:
        return ""

    def snapshot(self) -> _IdentityFit:
        return _IdentityFit()


class SimClock:
    def __init__(self, t: int = 0) -> None:
        self.now = int(t)

    def __call__(self) -> int:
        return self.now


def _station_copy(station: Path | None) -> Path:
    """A private copy of the station (never the original): the sim station for the frames / manager / policy
    defaults (frames {}, manager {}: config/*.yaml defaults)."""
    if station is not None:
        return Path(station)
    dst = SCRATCH / "stations" / "deploy_path_station.yaml"
    dst.parent.mkdir(parents=True, exist_ok=True)
    text = STATION_TEMPLATE.read_text()
    if not dst.is_file() or dst.read_text() != text:
        dst.write_text(text)
    return dst


class DeployEpisode:
    """One package episode on the deploy path. Decodes the episode's videos once (PiRecording.load); ``samples(rows)``
    walks the packets in order and returns one record per decision row (rows sorted ascending)."""

    def __init__(self, ref: EpisodeRef, station: Path | None = None, jpeg_quality: int = 90,
                 states: str = "rows", target_lead_ms: float = 0.0) -> None:
        from policy_deploy.frames.pi_recording import PiRecording
        from policy_deploy.frames.rig_file import camera_sizes, load_live_rig
        from policy_deploy.frames.settings import Settings as FramesSettings
        from policy_deploy.manager.settings import Settings as ManagerSettings
        from policy_deploy.policy.settings import Settings as PolicySettings
        from sim_deploy.camera_study.reference import _Episode, _train_rig
        from sim_deploy.scene.zarr_lowdim import load_episode

        set_cpu_threads()
        self.ref = ref
        self.station = _station_copy(station)
        self.frames_settings = FramesSettings.load(self.station, mock=True)
        self.manager_settings = ManagerSettings.load(self.station, mock=True)
        self.policy_settings = PolicySettings.load(self.station, mock=True)
        self.gripper_map, self.gripper_map_source = gripper_map_for_b(self.policy_settings)
        self.target_lead_ms = float(target_lead_ms)
        self.rig = load_live_rig(self.frames_settings.rig_file)
        self.jpeg_quality = int(jpeg_quality)
        if states not in ("rows", "100"):
            raise ValueError("states: rows | 100")
        self.states = states
        # training's row -> frame map (umiproc), from the slot's grid / piece / raw csv with training's rig
        self.train_rig = _train_rig()
        self.rowmap = _Episode(ref.folder, ref.slot, ref.pi_name, ref.n_steps, self.train_rig)
        lowdim = load_episode(Path(ref.dataset_path), ref.index)
        assert lowdim["name"] in ("", ref.pi_name), (lowdim["name"], ref.pi_name)
        self.poses = lowdim["poses"]                       # (n, 4, 4) recorded TCP poses, table-tag frame
        self.widths = lowdim["widths"]                     # (n,) recorded widths (training's, model space)
        self.t_rows = np.asarray(self.rowmap.t_grid, dtype=np.int64)
        assert len(self.t_rows) == len(self.poses) == ref.n_steps, (len(self.t_rows), len(self.poses), ref.n_steps)
        staging, self.staging_info = pi_staging(ref)
        t0 = time.monotonic()
        self.recording = PiRecording.load(staging, max_seconds=1e6, jpeg_quality=self.jpeg_quality,
                                          sizes=camera_sizes(self.rig))
        self.packets = self.recording.packets()
        self.load_s = round(time.monotonic() - t0, 2)
        # the recording's frame numbering is the raw csv's: check the times against umiproc's camera times
        main_t = np.array([f.info.t_pi_ns for f in self.recording.cameras["main"].frames], np.int64)
        mid = self.train_rig.camera_time_ns("main", main_t, np.array(
            [f.info.exposure_ns for f in self.recording.cameras["main"].frames], np.int64))
        assert np.array_equal(mid, self.rowmap.t_cam["main"]), "recording frames != umiproc main frames"

    # ------------------------------------------------------------------------------------------- C2 states

    def _state_times_and_poses(self):
        """(t_ns, T (n,4,4), width (n,)) of the C2 states: the rows, or 100 Hz interpolated from them."""
        if self.states == "rows":
            return self.t_rows, self.poses, self.widths
        from policy_deploy.manager import pose_math
        t = np.arange(self.t_rows[0], self.t_rows[-1] + 1, 10_000_000, dtype=np.int64)
        T, w = np.zeros((len(t), 4, 4)), np.zeros(len(t))
        j = np.clip(np.searchsorted(self.t_rows, t, side="right") - 1, 0, len(self.t_rows) - 2)
        for k, (tk, jk) in enumerate(zip(t, j)):
            a = (tk - self.t_rows[jk]) / (self.t_rows[jk + 1] - self.t_rows[jk])
            T[k] = pose_math.interpolate(self.poses[jk], self.poses[jk + 1], float(a))
            w[k] = (1 - a) * self.widths[jk] + a * self.widths[jk + 1]
        return t, T, w

    def _c2_width(self, w: float) -> float:
        """Convention A: the C2 width is the recorded (training) width; B is derived in _record."""
        return float(w)

    def _target_pose(self, t_ns: int, T_measured: np.ndarray) -> np.ndarray:
        """O_T_TCP_target: the recorded pose target_lead_ms later (interpolated between rows), or the same pose."""
        if self.target_lead_ms == 0.0:
            return T_measured.copy()
        from policy_deploy.manager import pose_math
        t = int(t_ns + self.target_lead_ms * 1e6)
        n = len(self.t_rows)
        j = int(np.clip(np.searchsorted(self.t_rows, t, side="right") - 1, 0, n - 2))
        a = float(np.clip((t - self.t_rows[j]) / (self.t_rows[j + 1] - self.t_rows[j]), 0.0, 1.0))
        return pose_math.interpolate(self.poses[j], self.poses[j + 1], a)

    # ------------------------------------------------------------------------------------------- the walk

    def samples(self, rows) -> list[dict]:
        from deploy_contracts import ArmState, decode, encode
        from deploy_contracts.c2_state import GateCounters
        from policy_deploy.contracts import (OBS_STEPS, ActionChunk, FramesReply, GetFramesRequest, Observation,
                                             Packet)
        from policy_deploy.frames.camera_health import CameraHealth
        from policy_deploy.frames.fake_pi_sender import SEND_DELAY_NS
        from policy_deploy.frames.frame_decoder import make_decoder
        from policy_deploy.frames.frames_server import FramesServer
        from policy_deploy.frames._dataproc.imageprep import ImagePrep
        from policy_deploy.frames.packet_buffer import PacketBuffer
        from policy_deploy.frames.packet_selector import PacketSelector
        from policy_deploy.frames.packet_source import PacketSource
        from policy_deploy.frames.pair_rule import PairRule
        from policy_deploy.frames.receiver import CAMERAS, Receiver
        from policy_deploy.manager.policy_step import PolicyStep, StepDone, StepJob
        from policy_deploy.manager.pose_converter import PoseConverter
        from policy_deploy.manager.state_history import LookupMiss, StateHistory

        s, ms = self.frames_settings, self.manager_settings
        rows = sorted({int(r) for r in rows})
        clock = SimClock(self.packets[0].header.main.t_pi_ns)

        class _OnePacket(PacketSource):
            def __init__(self) -> None:
                self.next: Packet | None = None

            def next_packet(self, timeout_s: float):
                p, self.next = self.next, None
                return p

            def close(self) -> None:
                pass

        health = {cam: CameraHealth(cam, s.health.stale_s, s.health.down_s, s.health.fps_window_s, clock)
                  for cam in CAMERAS}
        buffer = PacketBuffer(s.buffer.keep_s)
        selector = PacketSelector(fps={cam: self.rig.cameras[cam].fps for cam in CAMERAS},
                                  spacing_ns={"main": int(s.select.main_spacing_ms * 1e6),
                                              "ultrawide": int(s.select.ultrawide_spacing_ms * 1e6)},
                                  count=OBS_STEPS, tolerance_frames=s.select.tolerance_frames)
        source = _OnePacket()
        receiver = Receiver(source, {cam: make_decoder("jpeg", cam) for cam in CAMERAS}, self.rig, IdentityClockMap(),
                            buffer, health, s.default_exposure_ns, s.buffer.keep_s, clock,
                            PairRule(s.select.pair_max_age_ms, s.select.pair_max_lead_ms, s.select.pair_max_reuse))
        server = FramesServer.__new__(FramesServer)          # get_frames without the RpcServer (no port)
        server._buffer, server._health, server._selector = buffer, health, selector
        server._prep, server._clock = ImagePrep(self.rig), clock
        server._last_miss_log, server.served, server.misses = None, 0, 0

        st_t, st_T, st_w = self._state_times_and_poses()
        history = StateHistory(ms.history_s)
        n_added = 0
        seq0 = 1000

        def add_states_until(now_ns: int) -> None:
            nonlocal n_added
            while n_added < len(st_t) and st_t[n_added] <= now_ns:
                T = np.array(st_T[n_added], dtype=np.float64)
                history.add(ArmState(t_ns=int(st_t[n_added]), seq=seq0 + n_added, O_T_TCP=T,
                                     O_T_TCP_target=self._target_pose(int(st_t[n_added]), T),
                                     q=np.zeros(7), dq=np.zeros(7), gripper_width_m=self._c2_width(st_w[n_added]),
                                     mode="running", error="", gate=GateCounters(accepted=0, shortened=0,
                                                                                 pulled_back=0, rejected=0),
                                     loop_late=0))
                n_added += 1

        captured: dict = {}

        class _Frames:                                       # the C1 client, in-process
            def __init__(self) -> None:
                self.reply = None

            def get_frames(self):
                return self.reply

        class _Policy:                                       # the C4 client, in-process: codec round trip, hold
            def infer(self, obs):
                data = encode(obs)
                back = decode(data, Observation)
                captured["obs"], captured["c4_bytes"] = back, len(data)
                return ActionChunk(eef_pos=np.zeros((16, 3), np.float32),
                                   eef_rot6d=np.tile(np.array([1, 0, 0, 0, 1, 0], np.float32), (16, 1)),
                                   gripper_width_m=np.full((16, 1), 0.035, np.float32), width_clipped=0)

        frames_client = _Frames()
        params = inspect.signature(PoseConverter.__init__).parameters
        converter = PoseConverter(ms.exec_h, ms.lookahead_ms) if "workspace_box" not in params else \
            PoseConverter(ms.exec_h, ms.lookahead_ms, None)
        step = PolicyStep(frames_client, _Policy(), history, converter, ms, clock, IdentityPcClock())

        main_idx = self.rowmap.main_idx
        targets = {}                       # packet main frame_seq -> rows deciding at it
        for r in rows:
            if not 0 <= r < len(self.t_rows):
                raise IndexError(f"row {r} outside 0..{len(self.t_rows) - 1}")
            targets.setdefault(int(main_idx[r]), []).append(r)
        out: list[dict] = []
        chunk_id = 0
        start_pose = np.array(self.poses[0], dtype=np.float64)
        for p in self.packets:
            clock.now = int(p.header.main.t_pi_ns) + SEND_DELAY_NS
            add_states_until(clock.now)
            source.next = p
            receiver.receive_one(0.0)
            seq = int(p.header.main.frame_seq)
            if seq not in targets:
                continue
            for r in targets.pop(seq):
                clock.now = int(p.header.main.t_pi_ns) + SEND_DELAY_NS + ARRIVAL_EPS_NS
                add_states_until(clock.now)
                reply = server.get_frames(GetFramesRequest())
                reply_rt = decode(encode(reply), FramesReply)         # C1 codec round trip
                picks = selector.pick(buffer.snapshot())              # pure: the same pick get_frames made
                frames_client.reply = reply_rt
                chunk_id += 1
                captured.clear()
                result = step.run(StepJob(episode_id=1, chunk_id=chunk_id, start_pose=start_pose,
                                          last_anchor_t_ns=None))
                rec = self._record(r, p, reply_rt, picks, result, captured, history, start_pose, clock.now)
                rec["receiver"] = {"packets": dataclasses.asdict(receiver.packets) | {"latency_ms": None,
                                                                                    "pair_gap_ms": None},
                                   "counters": {c: dataclasses.asdict(receiver.counters[c]) for c in CAMERAS}}
                out.append(rec)
            if not targets:
                break
        if targets:
            raise RuntimeError(f"rows {sorted(sum(targets.values(), []))}: their main frame never came as a packet")
        return out

    def _record(self, row, packet, reply, picks, result, captured, history, start_pose, now_ns) -> dict:
        from policy_deploy.manager.policy_step import StepDone
        e = self.rowmap
        rec: dict = {"row": int(row), "now_ns": int(now_ns), "t_row_ns": int(self.t_rows[row]),
                     "newest_packet": {"packet_seq": int(packet.header.packet_seq),
                                       "main_seq": int(packet.header.main.frame_seq),
                                       "pair_seq": int(packet.header.pair_seq)},
                     "health": {cam: dataclasses.asdict(getattr(reply.health, cam)) for cam in ("main", "ultrawide")}}
        # training's frames at this row (umiproc map) for "which raw frame" comparisons
        r3 = max(row - 3, 0)
        u = int(e.uw_up[row])
        u_prev = max(u - 1, 0)
        rec["train_frames"] = {
            "main_rows": [r3, int(row)], "main_frame_idx": [int(e.main_idx[r3]), int(e.main_idx[row])],
            "main_frame_t_ns": [int(e.t_cam["main"][e.main_idx[r3]]), int(e.t_cam["main"][e.main_idx[row]])],
            "uw_local": [u_prev, u], "uw_frame_idx": [int(e.uw_sel[u_prev]), int(e.uw_sel[u])],
            "uw_frame_t_ns": [int(e.t_cam["ultrawide"][e.uw_sel[u_prev]]), int(e.t_cam["ultrawide"][e.uw_sel[u]])],
            "row_t_ns": [int(self.t_rows[r3]), int(self.t_rows[row])]}
        if picks["main"].ok and picks["ultrawide"].ok:
            rec["deploy_frames"] = {
                "main_packet_seq": [int(q.packet_seq) for q in picks["main"].packets],
                "main_frame_idx": [int(q.main_seq) for q in picks["main"].packets],
                "main_t_ns": [int(q.t_ns) for q in picks["main"].packets],
                "uw_packet_seq": [int(q.packet_seq) for q in picks["ultrawide"].packets],
                "uw_frame_idx": [int(q.pair_seq) for q in picks["ultrawide"].packets],
                "uw_frame_t_ns": [int(q.pair_t_pi_mid_ns) for q in picks["ultrawide"].packets],
                "uw_packet_t_ns": [int(q.t_ns) for q in picks["ultrawide"].packets]}
        else:
            rec["deploy_frames"] = {"miss": {c: picks[c].reason for c in picks}}
        if not isinstance(result, StepDone):
            rec["step"] = {"type": type(result).__name__,
                           "problem": getattr(getattr(result, "problem", None), "detail", None),
                           "code": str(getattr(getattr(result, "problem", None), "code", None))}
            return rec
        obs = captured["obs"]
        a = result.anchor
        poses = []
        for t in result.main_t_pc_ns:
            found = history.pose_at(int(t), a.source)
            poses.append(found.T)
        rec["step"] = {"type": "StepDone", "image_age_ms": result.timing.image_age_ms,
                       "anchor_dev_m": result.anchor_dev_m, "c4_bytes": captured["c4_bytes"],
                       "waypoints_t_ns_hold": result.waypoints.t_ns}
        rec["anchor"] = {"t_ns": a.t_ns, "t_pc_ns": a.t_pc_ns, "T": a.T, "source": a.source}
        # the alignment check's references: the rows T+3k (training's action k) and the recorded trajectory at the
        # waypoint times anchor + k*STEP_MS + lookahead (interpolated between rows as StateHistory does)
        from policy_deploy.contracts import STEP_MS
        from policy_deploy.manager import pose_math
        n = len(self.t_rows)
        fut = [min(row + 3 * k, n - 1) for k in range(16)]
        rec["future_rows"] = fut
        rec["future_row_t_ns"] = [int(self.t_rows[x]) for x in fut]
        wp_t = [int(a.t_pc_ns) + k * STEP_MS * 1_000_000 + int(self.manager_settings.lookahead_ms) * 1_000_000
                for k in range(16)]
        traj, inside = [], []
        for t in wp_t:
            j = int(np.clip(np.searchsorted(self.t_rows, t, side="right") - 1, 0, n - 2))
            al = (t - self.t_rows[j]) / (self.t_rows[j + 1] - self.t_rows[j])
            inside.append(bool(0.0 <= al <= 1.0))
            traj.append(pose_math.interpolate(self.poses[j], self.poses[j + 1], float(np.clip(al, 0.0, 1.0))))
        rec["waypoint_t_ns"] = wp_t
        rec["traj_at_waypoint_times_inside_episode"] = inside
        rec["traj_at_waypoint_times"] = np.stack(traj)
        rec["main_t_ns"] = np.asarray(result.main_t_ns)
        rec["ultrawide_t_ns"] = np.asarray(result.ultrawide_t_ns)
        rec["lookup_poses"] = np.stack(poses)
        rec["start_pose"] = start_pose
        rec["obs"] = {k: np.asarray(getattr(obs, k)) for k in ("eef_pos", "eef_rot6d", "eef_rot6d_wrt_start",
                                                               "gripper_width_m", "main", "ultrawide")}
        rec["obs"]["gripper_width_m_B"] = (self.gripper_map.to_robot(rec["obs"]["gripper_width_m"])
                                           if self.gripper_map is not None else None)
        rec["width_conventions"] = {"A": "C2 width = recorded training width (pass-through, sim D8)",
                                    "B": "C2 width = GripperMap.to_robot(recorded) (f632576 policy.gripper_map)"}
        return rec

    def describe(self) -> dict:
        return {"episode": self.ref.to_dict(), "station_copy": str(self.station), "jpeg_quality": self.jpeg_quality,
                "states": self.states, "rig_file": str(self.frames_settings.rig_file),
                "gripper_map_B": self.gripper_map.as_dict() if self.gripper_map is not None else None,
                "gripper_map_B_source": self.gripper_map_source, "target_lead_ms": self.target_lead_ms,
                "manager": {"exec_h": self.manager_settings.exec_h, "lookahead_ms": self.manager_settings.lookahead_ms,
                            "pose_source": self.manager_settings.pose_source,
                            "history_s": self.manager_settings.history_s,
                            "max_image_age_ms": self.manager_settings.max_image_age_ms,
                            "workspace_box_in_harness": None},
                "frames": {"keep_s": self.frames_settings.buffer.keep_s,
                           "default_exposure_ns": self.frames_settings.default_exposure_ns,
                           "select": dataclasses.asdict(self.frames_settings.select)},
                "clocks": "identity (Pi = laptop = PC); packet arrival = t_sensor + SEND_DELAY_NS (45.8 ms)",
                "staging": self.staging_info, "decode_and_encode_s": self.load_s,
                "n_packets": len(self.packets), "uw_upsample_matches_episode_zarr": self.rowmap.up_matches_stored,
                "main_dt_ms_all_rows": {  # umiproc: nearest main frame mid-exposure minus row time, every row
                    "mean": float(np.mean(self.rowmap.main_dt_ms)), "min": float(np.min(self.rowmap.main_dt_ms)),
                    "max": float(np.max(self.rowmap.main_dt_ms)), "n": int(len(self.rowmap.main_dt_ms))},
                "row_step_ms": {"mean": float(np.mean(np.diff(self.t_rows)) / 1e6),
                                "min": float(np.min(np.diff(self.t_rows)) / 1e6),
                                "max": float(np.max(np.diff(self.t_rows)) / 1e6)}}


F632576_GRIPPER_MAP = {"training_closed_m": 0.051, "training_open_m": 0.090, "robot_closed_m": 0.030,
                       "robot_open_m": 0.040}          # f632576 config/policy.yaml defaults


def gripper_map_for_b(policy_settings=None):
    """(GripperMap, source) for convention B: the copy's class and its configured numbers when it has them; else
    the f632576 file extracted to scratch (``git show``) with f632576's numbers."""
    try:
        from policy_deploy.policy.gripper_map import GripperMap
        cls, src = GripperMap, "copy policy/gripper_map.py"
    except ImportError:
        import importlib.util
        import subprocess
        dst = SCRATCH / "f632576_gripper_map.py"
        dst.parent.mkdir(parents=True, exist_ok=True)
        from sim_deploy import upstream
        text = subprocess.run(["git", "-C", str(upstream.repo()), "show",
                               "f632576:deployment/policy_deploy/policy_deploy/policy/gripper_map.py"],
                              capture_output=True, text=True, check=True).stdout
        dst.write_text(f"# EXTRACTED by sim_deploy/offline (W1 S1) from {upstream.repo()} f632576, "
                       "read-only reference\n" + text)
        spec = importlib.util.spec_from_file_location("f632576_gripper_map", dst)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        cls, src = mod.GripperMap, f"f632576 extracted to {dst}"
    gm = getattr(policy_settings, "gripper_map", None) if policy_settings is not None else None
    if gm is not None:
        return gm, src + " + copy policy.gripper_map config"
    return cls.from_config(F632576_GRIPPER_MAP), src + " + f632576 default numbers"


def sample_stem(index: int, row: int) -> str:
    return f"ep{index:02d}_r{row:03d}"


def save_samples(dep: DeployEpisode, recs: list[dict], out_dir: Path) -> list[Path]:
    """One ``<stem>.deploy.npz`` (arrays) + ``<stem>.deploy.json`` (everything else) per record."""
    out_dir.mkdir(parents=True, exist_ok=True)
    version = deploy_version()
    paths = []
    for rec in recs:
        stem = sample_stem(dep.ref.index, rec["row"])
        arrays = {}
        if "obs" in rec:
            arrays = {f"deploy_{k}": v for k, v in rec.pop("obs").items()}
            for k in ("main_t_ns", "ultrawide_t_ns", "lookup_poses", "start_pose", "traj_at_waypoint_times"):
                arrays[f"deploy_{k}"] = np.asarray(rec.pop(k))
            arrays["deploy_anchor_T"] = np.asarray(rec["anchor"]["T"])
        np.savez(out_dir / f"{stem}.deploy.npz", **arrays)
        write_json(out_dir / f"{stem}.deploy.json", {"deploy": rec, "deploy_setup": dep.describe(),
                                                     "deploy_version": version})
        paths.append(out_dir / f"{stem}.deploy.npz")
    return paths


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--index", type=int, nargs="+", required=True)
    ap.add_argument("--rows", type=int, nargs="+", required=True)
    ap.add_argument("--station", type=Path, default=None)
    ap.add_argument("--jpeg-quality", type=int, default=90)
    ap.add_argument("--states", default="rows", choices=["rows", "100"])
    ap.add_argument("--target-lead-ms", type=float, default=0.0)
    ap.add_argument("--out", type=Path, default=None, help="default obs/<dataset>/ (states 100: obs_100hz/<dataset>)")
    a = ap.parse_args(argv)
    variant = ("" if a.states == "rows" else "_100hz") + (f"_lead{a.target_lead_ms:g}ms" if a.target_lead_ms else "")
    out = a.out or (OBS_CACHE.parent / f"obs{variant}" / a.dataset)
    for i in a.index:
        ref = episode(a.dataset, i)
        t0 = time.monotonic()
        dep = DeployEpisode(ref, a.station, a.jpeg_quality, a.states, a.target_lead_ms)
        rows = [r for r in a.rows if r < ref.n_steps]
        recs = save_samples(dep, dep.samples(rows), out)
        print(f"{a.dataset}#{i} {ref.pi_name}: {len(recs)} samples in {time.monotonic() - t0:.1f} s "
              f"(decode+encode {dep.load_s} s) -> {out}", flush=True)
        del dep
    return 0


if __name__ == "__main__":
    sys.exit(main())
