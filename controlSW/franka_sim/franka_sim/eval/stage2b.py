"""Stage 2b: the full live stack with a demo video streamed over the emulated Wi-Fi link, teacher-forced.

Per demo: the launcher (``python -m franka_deploy run --profile P --no-tty --script ...``, a derived deploy.yaml in the
run dir) runs with ``run.sim`` = the profile's replay source (iPhone video replay sender or rig raw replay sender) on
that demo and the task's ``history_source: replay_gt`` (the demo's pose/width history, aligned to the sender's clock
through the sender ready file). Teacher forcing means the policy sees the demo's own proprio, so its chunks can be
compared with predictions made offline from the same frames (``tools.predict_demos``, same demo, same frames, the live
diffusion-noise seed): any difference comes from the live wireless pipeline (JPEG, link, clock, frame selection).

Replay clock: the history npz (built here) holds the demo's TCP poses/widths on the episode clock (t_ns from the iPhone
slot times; origin 0); the launcher adds the sender's loop span and start (sender ready file). A tap call at monotonic
time t (``info_main_ts[-1]``) shows demo frame ``interp((t - start) mod span)``.

Start alignment: a server policy loads a 2.6 GB model while the sender already streams, so the episode starts a while after
the stream does. The sender loops the demo; this stage writes ``c`` at a loop boundary (the demo's own start, from the sender
ready file) so the first policy call sees the beginning of the demo, as in training, not a random slice.

Stamp error = camera_truth t_est - sender_truth t_true joined on (seq, stream), ms. For ``arrival_minus_latency`` clocks
(the iPhone app) the SENDER models the app's capture latency (launcher passes ``--capture-latency-s`` from the sensor's
``clock.latency_s``; the link's ``latency_s`` stays the Wi-Fi hop), so the receiver's stamps are right by construction
and the stamp error measures link jitter only (it cannot validate the 80 ms constant: that needs the hardware day).

Render profiles (``image_source == render``): Stage 2b is teacher-forced on a recorded video, so the profile's replay source
(video_replay for protocol v1, rig_raw_replay for v2) is used for this stage, with a note.

INCOMPLETE (with the reason) when the launcher fails, the sender dies (``sender_died``) or no tap calls exist.
"""

from __future__ import annotations

import copy
import json
import os
import signal
import subprocess
import time
from pathlib import Path

import numpy as np
import yaml
from scipy.spatial.transform import Rotation

from franka_sim.eval import runner as rn
from franka_sim.eval import stage2a as s2a

KS = (4, 8, 15)
SKIP_S = 3.0          # stamp error: skip the first seconds (clock sync warm-up)
EDGE_FRAMES = 6       # live-vs-offline: skip calls whose history crosses the loop wrap
RAW_SUFFIX = "_left"


def replay_source_of(R: rn.Run) -> str:
    """The sender that streams the demo in Stage 2b: the profile's own replay source, or, for a render profile, the replay
    source of its protocol (v1 -> video_replay, v2 -> rig_raw_replay)."""
    if R.image_source in ("video_replay", "rig_raw_replay"):
        return R.image_source
    return "video_replay" if R.protocol == "v1" else "rig_raw_replay"


def replay_run(R: rn.Run) -> rn.Run:
    """``R`` itself for a replay profile; for a render profile a Run resolved with ``run.sim`` = the replay source (the render
    banner stays on the outer report). Shares ``R.notes``."""
    if R.image_source in ("video_replay", "rig_raw_replay"):
        return R
    src = replay_source_of(R)
    args = copy.copy(R.args)
    args.set = [x for x in (getattr(R.args, "set", None) or []) if not str(x).startswith("sim.image_source=")]
    R2 = rn.resolve(args, run_overrides={"sim": src})
    R2.notes = R.notes
    R.notes.append(f"image_source {R.image_source}: Stage 2b is teacher-forced on a recorded video, so it uses the replay source "
                   f"{src} (protocol {R.protocol}); the closed loop with rendered images is Stage 3")
    return R2


def raw_dir_of(R: rn.Run, demo_name: str) -> str:
    return demo_name + RAW_SUFFIX


def build_replay_npz(R: rn.Run, did: int, path: Path) -> dict:
    """The demo's TCP pose / width history on the episode clock for ``task --history-source replay_gt``."""
    from franka_sim import stage1_replay as s1

    dm = s1.load_demos(R.zarr, [did])[0]
    T = dm["T"]
    xyz_wxyz = np.concatenate([T[:, :3, 3], Rotation.from_matrix(T[:, :3, :3]).as_quat()[:, [3, 0, 1, 2]]], axis=1)
    t_ns = np.round(np.arange(len(T)) * 1e9 / 60.0).astype(np.int64)
    src, err = "uniform_60hz", None
    try:
        from sensor_rig.sim.iphone_reader import IphoneEpisode
        from franka_deploy.launch.launcher import resolve_replay_episode

        ep = IphoneEpisode(resolve_replay_episode({**R.station.get("sim"), "episode_dir": raw_dir_of(R, dm["name"])}))
        t = ep.main_times_unix_ns()
        if len(t) == len(T):
            t_ns, src = (t - t[0]).astype(np.int64), "iphone_slot_times"
        else:
            err = f"slot times ({len(t)}) and zarr frames ({len(T)}) differ in length"
    except Exception as e:      # noqa: BLE001 - the 60 Hz grid is within ~1 ms of the slot times, but never silently
        err = f"{type(e).__name__}: {e}"
    # the zarr rows of this demo (analyze_tap --zarr compares tapped 224 px images with them)
    lens = [len(d["T"]) for d in s1.load_demos(R.zarr)]
    np.savez(path, t_ns=t_ns, xyz_wxyz=xyz_wxyz, width=dm["w"], origin_ns=np.array(0, np.int64),
             zarr_start=np.array(int(sum(lens[:did])), np.int64), zarr_local=np.arange(len(T), dtype=np.int64))
    return {"name": dm["name"], "n": len(T), "t_ns": t_ns, "time_source": src, "time_source_error": err}


def apply_port_offset(R: rn.Run, prof: dict, o: int) -> None:
    """Shift every launcher port of a derived profile by ``o`` (shared by Stage 2b and Stage 3): wireless receiver ports,
    camera / logger, bridge, controller, inference, spacemouse, the sim state PUB (``sim_state_endpoint``, 5570+o) and the
    bridge-served hand endpoint (5565+o) when the profile sets one. ``o == 0`` leaves the profile as it is."""
    if not o:
        return
    prof["station_set"] = list(prof.get("station_set") or []) + [f"sensor.transport.ports.main={5555 + o}"] + (
        [f"sensor.transport.ports.ultrawide={5556 + o}"] if R.protocol == "v1" else [])
    prof["station_set"] = [x.replace("tcp://127.0.0.1:5565", f"tcp://127.0.0.1:{5565 + o}") if x.startswith("tooling.hand.endpoint=") else x
                           for x in prof["station_set"]]
    prof.setdefault("camera", {}).update(endpoint=f"tcp://localhost:{16702 + o}", logger_endpoint=f"tcp://localhost:{25556 + o}")
    prof.setdefault("bridge", {})["endpoint"] = f"tcp://127.0.0.1:{5565 + o}"
    prof.setdefault("controller", {})["endpoint"] = f"tcp://localhost:{26010 + o}"
    prof.setdefault("inference", {})["endpoint"] = f"tcp://0.0.0.0:{18765 + o}"
    prof.setdefault("spacemouse", {})["endpoint"] = f"tcp://localhost:{15557 + o}"
    prof["sim_state_endpoint"] = f"tcp://127.0.0.1:{5570 + o}"


def derive_deploy(R: rn.Run, did: int, info: dict, work: Path, npz: Path, port_offset: int) -> tuple[Path, dict]:
    """A deploy.yaml next to the run: the profile + this stage's overrides. Returns (path, the overrides)."""
    raw = yaml.safe_load(Path(R.cfg["config_path"]).read_text())
    prof = copy.deepcopy(raw["profiles"][R.profile])
    sets = list(R.sets)
    src = R.image_source            # a replay source: run() swaps a render profile for replay_run(R) first
    if src == "video_replay":
        sets.append(f"sim.episode_dir={raw_dir_of(R, info['name'])}")
    elif src == "rig_raw_replay":
        sets.append(f"sim.generate.episode_dir={raw_dir_of(R, info['name'])}")
    else:
        raise rn.StageIncomplete(f"Stage 2b needs a replay image source (video_replay / rig_raw_replay), not {src!r}")
    if R.policy.get("kind") == "replay":
        sets += [f"policy.episode_index={did}", "policy.start_frame_index=0"]
    prof["station_set"] = sets
    if R.policy.get("kind") in ("bp_server", "command"):
        # the laptop's BP env gets torch from ~/.local (README.md): the launcher's PYTHONNOUSERSITE=1 would hide it
        prof["user_site_children"] = sorted(set(prof.get("user_site_children") or []) | {"inference"})
    prof["work_root"] = str(work)
    prof["run"] = {**prof.get("run", {}), **R.run_ids}
    prof.setdefault("task", {}).update({"history_source": "replay_gt", "replay_npz": str(npz),
                                        "tap_images_every": int(os.environ.get("EVAL_2B_TAP_IMAGES_EVERY", 1))})
    apply_port_offset(R, prof, port_offset)
    raw["profiles"][R.profile] = prof
    p = work.parent / "deploy_stage2b.yaml"
    p.write_text(yaml.safe_dump(raw))
    return p, prof


def launch(R: rn.Run, dep: Path, script: str, cwd: Path, timeout_s: float, driver=None) -> int:
    """Run the launcher to completion. ``driver(proc, work_dir)`` (optional) runs in a thread next to it (it types the keys)."""
    import threading

    cmd = [R.python_re, "-u", "-m", "franka_deploy", "run", "--config", str(dep), "--profile", R.profile, "--no-tty",
           "--script", script]
    env = rn.child_env()
    with open(cwd / "launcher.log", "w") as lg:
        p = subprocess.Popen(cmd, stdout=lg, stderr=subprocess.STDOUT, env=env, cwd=str(cwd), start_new_session=True)
        th = threading.Thread(target=driver, args=(p, cwd / "work"), daemon=True) if driver else None
        if th:
            th.start()
        try:
            return p.wait(timeout_s)
        except subprocess.TimeoutExpired:
            os.killpg(p.pid, signal.SIGINT)
            try:
                p.wait(20)
            except subprocess.TimeoutExpired:
                os.killpg(p.pid, signal.SIGKILL)
                p.wait()
            raise rn.StageIncomplete(f"launcher timed out after {timeout_s:.0f} s; see {cwd / 'launcher.log'}")


def plan_start(start_ns: int, span_ns: int, now_ns: int, lead_s: float) -> int:
    """The first loop boundary (the demo's frame 0 on the sender's clock) at least ``lead_s`` after ``now_ns``."""
    earliest = now_ns + int(lead_s * 1e9)
    m = max(0, -(-(earliest - start_ns) // span_ns))
    return int(start_ns + m * span_ns)


def start_offset_s(R: rn.Run) -> float:
    """Seconds after the loop boundary at which ``c`` is typed: the receiver sees a frame only after the app's capture latency
    (modeled in the sender), so a ``c`` exactly at the boundary would make the first call see the END of the previous lap.
    0.5 s of margin (a few history frames) on top; ``EVAL_2B_START_OFFSET_S`` overrides the margin."""
    clock = R.station.get("sensor.clock")
    lat = float(clock["latency_s"]["main"]) if clock.get("mode") == "arrival_minus_latency" and R.protocol == "v1" else 0.0
    return lat + float(os.environ.get("EVAL_2B_START_OFFSET_S", 0.5))


def start_driver(R: rn.Run, dur_s: float, lead_s: float, out: dict):
    """Thread body: wait for the task and the sender, then type ``c`` at the next loop boundary, ``s`` after the demo and
    ``q``. Records the chosen times in ``out`` (monotonic seconds)."""
    key = "first_tick_ns" if R.image_source == "rig_raw_replay" else "start_now_ns"

    def drive(proc, work: Path):
        t_wait = time.monotonic()
        sess = None
        while proc.poll() is None and time.monotonic() - t_wait < 900:
            ss = sorted(work.glob("session_*"))
            if ss and (ss[-1] / "task_ready.json").exists() and (ss[-1] / "sender_ready.json").exists():
                sess = ss[-1]
                break
            time.sleep(0.2)
        if sess is None:
            return
        rd = json.loads((sess / "sender_ready.json").read_text())
        span = int(rd.get("loop_span_ns") or 0)
        if span <= 0:
            out["error"] = "sender ready file has no loop_span_ns"
            return
        t_c = plan_start(int(rd[key]), span, time.monotonic_ns(), lead_s) / 1e9 + start_offset_s(R)
        out.update(t_c=t_c, session=str(sess), span_s=span / 1e9)
        cmd = sess / "commands.txt"
        for t, k in ((t_c, "c"), (t_c + dur_s + 4.0, "s"), (t_c + dur_s + 7.0, "q")):
            while time.monotonic() < t and proc.poll() is None:
                time.sleep(0.01)
            if proc.poll() is not None:
                return
            with open(cmd, "a") as f:
                f.write(k + "\n")

    return drive


def read_jsonl(p: Path) -> list:
    return [json.loads(x) for x in p.read_text().splitlines() if x.strip()] if p.exists() else []


def stamp_errors(session: Path, window: tuple | None = None) -> dict:
    """{stream: signed error ms array} from camera_truth (t_est) joined with sender_truth (t_true) on (seq, stream).
    ``window`` = (t_lo, t_hi) in monotonic seconds restricts to frames captured inside it (the policy run); without it
    the whole stream after the first SKIP_S seconds."""
    tx = {(r["seq"], r["stream"]): r["t_true"] for r in read_jsonl(session / "logs" / "sender_truth.jsonl")}
    rx = [r for r in read_jsonl(session / "logs" / "camera_truth.jsonl") if (r["seq"], r["stream"]) in tx]
    if not rx:
        return {}
    t0 = min(tx[(r["seq"], r["stream"])] for r in rx)
    lo, hi = window if window else (t0 + SKIP_S, float("inf"))
    out = {}
    for s in sorted({r["stream"] for r in rx}):
        out[s] = np.array([(r["t_est"] - tx[(r["seq"], r["stream"])]) * 1e3 for r in rx
                           if r["stream"] == s and lo <= tx[(r["seq"], r["stream"])] <= hi])
    return out


def load_calls(session: Path) -> list[dict]:
    files = sorted(session.glob("task/episodes/*/*/*/episode_*/tap/call_*.npz"))
    return [dict(np.load(f, allow_pickle=False)) for f in files]


def live_chunk(c: dict) -> np.ndarray:
    return np.concatenate([c["reply_action0_eef_pos"], c["reply_action0_eef_rot_axis_angle"],
                           np.asarray(c["reply_action0_gripper_width"]).reshape(-1, 1)], axis=1).astype(float)


def frame_mapper(t_rel_ns: np.ndarray, start_ns: int, span_ns: int):
    """monotonic seconds -> (fractional demo frame, loop position in ns)."""
    idx = np.arange(len(t_rel_ns), dtype=float)

    def f(t_s: float) -> float:
        tau = (int(t_s * 1e9) - start_ns) % span_ns
        return float(np.interp(tau, t_rel_ns, idx))

    return f


def run_demo(R: rn.Run, d: Path, did: int, port_offset: int, results: dict) -> dict:
    dd = d / f"demo_{did:02d}"
    dd.mkdir(parents=True, exist_ok=True)
    work = dd / "work"
    npz = dd / "replay_gt_base.npz"
    info = build_replay_npz(R, did, npz)
    dur = float((info["t_ns"][-1] - info["t_ns"][0]) / 1e9)
    if R.image_source == "rig_raw_replay":
        n_clip = int(R.station.get("sim.generate.n_frames", 300))
        dur = min(dur, n_clip / 60.0)
    # a server policy loads a 2.6 GB model while the sender already streams: the receiver can lag seconds behind (frames
    # queue in the socket and arrive late, but arrival-stamped, so the task cannot see it); let that backlog drain first.
    lead = float(os.environ.get("EVAL_2B_START_S", 12.0 if R.policy.get("kind") in ("bp_server", "command") else 3.0))
    drv: dict = {}
    # the launcher script is only a backstop: the driver types c / s / q (c at a loop boundary = the demo's start)
    script = f"{lead + dur + 120.0:.0f}:q"
    dep, prof = derive_deploy(R, did, info, work, npz, port_offset)
    t0 = time.time()
    rc = launch(R, dep, script, dd, timeout_s=float(os.environ.get("EVAL_2B_TIMEOUT_S", 900)) + dur + 60,
                driver=start_driver(R, dur, lead, drv))
    sessions = sorted(work.glob("session_*"))
    if not sessions:
        raise rn.StageIncomplete(f"demo {did}: no session dir (launcher rc {rc}); see {dd / 'launcher.log'}")
    session = sessions[-1]
    summ_p = session / "launcher_summary.json"
    summ = json.loads(summ_p.read_text()) if summ_p.exists() else {}
    if summ.get("sender_died"):
        raise rn.StageIncomplete(f"demo {did}: sender_died {summ['sender_died']} (episode aborted)")
    if drv.get("error"):
        raise rn.StageIncomplete(f"demo {did}: {drv['error']}")
    if rc != 0 or summ.get("exit_code", 1) != 0:
        raise rn.StageIncomplete(f"demo {did}: launcher failed rc {rc}: {summ.get('exit_reason')}; see {dd / 'launcher.log'}")
    calls = load_calls(session)
    if not calls:
        raise rn.StageIncomplete(f"demo {did}: no tap calls recorded (policy never ran); see {session}")
    rg = summ["replay_gt"]
    mapf = frame_mapper(info["t_ns"] - info["t_ns"][0], rg["start_now_ns"], rg["span_ns"])
    f_max = float(np.interp(rg["span_ns"] - 1, info["t_ns"], np.arange(len(info["t_ns"]))))
    kind = R.policy.get("kind")
    frames, live, seeds_live = [], [], []
    for c in calls:
        fi = float(c["reply_replay_frame"]) if (kind == "replay" and "reply_replay_frame" in c) else mapf(float(c["info_main_ts"][-1]))
        frames.append(fi)
        live.append(live_chunk(c))
        seeds_live.append(int(c["reply_noise_seed"]) if "reply_noise_seed" in c else -1)
    frames, live, seeds_live = np.array(frames), np.array(live), np.array(seeds_live)
    start_f = mapf(float(calls[0]["info_main_ts"][0]))
    # offline predictions on the same frames, in call order, with the live noise seed
    q = [{"demo": did, "frame": float(f), "start_frame": float(start_f)} for f in frames]
    qp = dd / "queries.json"
    qp.write_text(json.dumps(q))
    seed0 = int(seeds_live[0]) if seeds_live[0] >= 0 else R.seed
    pred_p = s2a.run_predict(R, dd, None, queries=qp, seed=seed0, out_name="pred_offline.npz")
    P = dict(np.load(pred_p, allow_pickle=False))
    ok = (frames >= EDGE_FRAMES) & (frames <= f_max - EDGE_FRAMES)
    if kind in ("bp_server", "command") and (seeds_live >= 0).all():
        match = P["noise_seed"] == seeds_live
        n_seed_mismatch = int((~match & ok).sum())
        ok &= match
    else:
        n_seed_mismatch = 0
    err = s2a.chunk_errors(live[ok], P["rel_chunks"][ok]) if ok.any() else {}
    # task summary, analyze_tap, stamps, receiver health
    ts = json.loads((session / "task" / "task_summary.json").read_text()) if (session / "task" / "task_summary.json").exists() else {}
    cnt = ts.get("counters", {})
    tap_json = dd / "analyze_tap.json"
    tap_dir = calls and sorted(session.glob("task/episodes/*/*/*/episode_*/tap"))[0]
    zarr_arg = ["--zarr", R.zarr] if Path(R.zarr).exists() else []
    # with the tapped 224 px images and the dataset: image MAE vs the zarr frames and the UW frame offset vs training
    # (needs zarr + imagecodecs: the BP interpreter, like predict_demos)
    r = subprocess.run([R.python_bp if zarr_arg else R.python_re, "-m", "franka_deploy.tools.analyze_tap", "--tap-dir", str(tap_dir),
                        "--replay-npz", rg["npz"], "--start-now-ns", str(rg["start_now_ns"]), *zarr_arg, "--out", str(tap_json)],
                       capture_output=True, text=True, env=rn.child_env())
    tap = json.loads(tap_json.read_text()) if tap_json.exists() else {"analyze_tap_error": r.stderr[-500:]}
    win = (float(calls[0]["info_main_ts"][0]) - 1.0, float(calls[-1]["info_main_ts"][-1]) + 1.0)
    st = stamp_errors(session, win)
    st_all = stamp_errors(session)
    cs = (summ.get("camera_stats") or {})
    main_stats = (cs.get("streams") or {}).get("main", {})
    part = {
        "demo": did, "name": info["name"], "time_source": info["time_source"], "time_source_error": info["time_source_error"],
        "t_c_s": drv.get("t_c"), "first_call_frame": float(frames[0]), "n_calls": len(calls), "n_pairs": int(ok.sum()), "n_seed_mismatch": n_seed_mismatch,
        "err": {k: v for k, v in err.items()}, "stamp_ms": st, "stamp_ms_stream": st_all, "tap": tap, "counters": cnt, "frames": frames,
        "session": str(session), "wall_s": round(time.time() - t0, 1),
        "health": {"fps_in": main_stats.get("fps_in"), "age_p95_s": (main_stats.get("age_s") or {}).get("p95"),
                   "reconnects": main_stats.get("reconnects"), "stale_ticks": cs.get("stale_ticks"),
                   "clock_mode": (cs.get("clock") or {}).get("mode")},
        "holds": cnt.get("holds"), "stale_image": cnt.get("stale_image", 0), "obs_retries": cnt.get("obs_retries", 0),
        "policy_errors": cnt.get("policy_errors", 0), "infer_s": [float(c["info_inference_s"]) for c in calls],
    }
    return part


def aggregate(parts: list[dict], R: rn.Run) -> tuple[dict, list]:
    m: dict = {"n_demos": len(parts), "demos": [p["demo"] for p in parts], "n_calls": sum(p["n_calls"] for p in parts),
               "n_pairs": sum(p["n_pairs"] for p in parts), "n_seed_mismatch": sum(p["n_seed_mismatch"] for p in parts),
               "protocol": R.protocol}
    worst = []
    for k in KS:
        e = np.concatenate([p["err"][f"pos_{k}"] for p in parts if p["err"]] or [np.array([])])
        m[f"live_vs_offline_pos_err_mm_p50_k{k}"] = s2a.pct(e, 50)
        m[f"live_vs_offline_pos_err_mm_p95_k{k}"] = s2a.pct(e, 95)
        r_ = np.concatenate([p["err"][f"rot_{k}"] for p in parts if p["err"]] or [np.array([])])
        m[f"live_vs_offline_rot_err_deg_p95_k{k}"] = s2a.pct(r_, 95)
        if m[f"live_vs_offline_pos_err_mm_p95_k{k}"] is not None:
            worst.append(m[f"live_vs_offline_pos_err_mm_p95_k{k}"])
    m["live_vs_offline_pos_err_mm_p95"] = max(worst) if worst else None
    m["frame_selection_errors"] = int(sum(p["obs_retries"] for p in parts))
    m["stale_image_errors"] = int(sum(p["stale_image"] for p in parts))
    m["holds"] = int(sum(p["holds"] or 0 for p in parts))
    m["policy_errors"] = int(sum(p["policy_errors"] for p in parts))
    for s in ("main", "ultrawide"):
        e = np.concatenate([p["stamp_ms"].get(s, np.array([])) for p in parts] or [np.array([])])
        m[f"stamp_err_ms_p50_{s}"] = s2a.pct(np.abs(e), 50)
        m[f"stamp_err_ms_p95_{s}"] = s2a.pct(np.abs(e), 95)
        m[f"stamp_err_ms_signed_median_{s}"] = float(np.median(e)) if e.size else None
    m["cam_stamp_err_ms_p95"] = m["stamp_err_ms_p95_main"]
    allm = np.concatenate([p["stamp_ms_stream"].get("main", np.array([])) for p in parts] or [np.array([])])
    m["stamp_err_ms_p95_main_whole_stream"] = s2a.pct(np.abs(allm), 95)
    m["stamp_err_ms_max_main_whole_stream"] = float(np.abs(allm).max()) if allm.size else None
    inf = np.concatenate([p["infer_s"] for p in parts])
    m["inference_s_p50"], m["inference_s_p95"] = s2a.pct(inf, 50), s2a.pct(inf, 95)
    h = [p["health"] for p in parts]
    m["receiver_fps_in"] = float(np.mean([x["fps_in"] for x in h if x["fps_in"] is not None])) if any(x["fps_in"] for x in h) else None
    m["receiver_age_s_p95"] = max([x["age_p95_s"] for x in h if x["age_p95_s"] is not None] or [None]) if any(
        x["age_p95_s"] is not None for x in h) else None
    m["receiver_reconnects"] = int(sum(x["reconnects"] or 0 for x in h))
    m["clock_mode"] = h[0]["clock_mode"]
    taps = [p["tap"] for p in parts]
    m["contract_ok"] = bool(all(t.get("contract_ok") for t in taps))
    m["main_spacing_ok"] = bool(all(t.get("main_spacing_ok") for t in taps))
    m["uw_spacing_ok"] = bool(all(t.get("uw_spacing_ok") for t in taps))
    m["tap_teacher_forcing"] = [t.get("teacher_forcing") for t in taps]
    # image divergence next to the action gap (analyze_tap --zarr on the tapped 224 px images)
    mae = [t["image_mae_224_main_vs_zarr"] for t in taps if t.get("image_mae_224_main_vs_zarr")]
    m["image_mae_224_main_p50"] = float(np.mean([x["p50"] for x in mae])) if mae else None
    m["image_mae_224_main_p95"] = float(max(x["p95"] for x in mae)) if mae else None
    hist: dict = {}
    for t in taps:
        for k, v in (t.get("uw_frame_offset_vs_training") or {}).items():
            hist[str(k)] = hist.get(str(k), 0) + int(v)
    m["uw_frame_offset_hist"] = hist or None
    m["uw_frame_offset_nonzero_fraction"] = (1.0 - hist.get("0", 0) / sum(hist.values())) if hist else None
    m["time_source"] = sorted({p["time_source"] for p in parts})
    m["first_call_demo_frame_max"] = max(p["first_call_frame"] for p in parts)
    # everything in a live run depends on wall-clock scheduling: all of it is marked timing (reproducibility checks skip it)
    m["timing"] = sorted(k for k in m if k not in ("n_demos", "demos", "protocol", "clock_mode", "timing"))
    rows = [{"demo": p["name"][:30], "calls": p["n_calls"], "pairs": p["n_pairs"],
             "first_frame": round(p["first_call_frame"]), "live_vs_offline_p95_mm_k8": round(s2a.pct(p["err"]["pos_8"], 95), 2) if p["err"] else None,
             "stamp_p95_ms": round(s2a.pct(np.abs(p["stamp_ms"].get("main", np.array([]))), 95) or 0.0, 2),
             "holds": p["holds"], "stale": p["stale_image"], "wall_s": p["wall_s"]} for p in parts]
    return m, rows


def run(R: rn.Run, d: Path, results: dict) -> dict:
    demos = R.demo_list()
    if demos is None:
        demos = list(range(int(getattr(R.args, "episodes", None) or 1)))
    R = replay_run(R)
    if R.station.get("sensor.clock").get("mode") == "arrival_minus_latency" and R.protocol == "v1":
        lat = float(R.station.get("sensor.clock.latency_s.main"))
        R.notes.append(f"the sender models the app's capture latency (clock.latency_s {lat} s, a placeholder: the hardware day measures it); "
                       f"link.latency_s stays the Wi-Fi hop ({R.station.get('link.latency_s')} s)")
    po = getattr(R.args, "port_offset", None)
    po = int(os.environ.get("FD_TEST_PORT_OFFSET", 0)) if po is None else int(po)
    parts = [run_demo(R, d, did, po, results) for did in demos]
    m, rows = aggregate(parts, R)
    for p in parts:
        R.notes.append(f"demo {p['demo']} session {p['session']}; replay time source {p['time_source']}"
                       + (f" (fell back: {p['time_source_error']})" if p["time_source_error"] else "")
                       + f"; c typed at loop start, first call at demo frame {p['first_call_frame']:.0f}")
    return {"metrics": m, "rows": rows, "notes": [f"live run per demo under {d}/demo_NN (launcher.log, work/session_*)"]}
