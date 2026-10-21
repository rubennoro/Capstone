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

Stamp error = camera_truth t_est - sender_truth t_true joined on (seq, stream), ms. For ``arrival_minus_latency`` clocks
(the iPhone app) the sim emulates the sensor's capture latency by raising the link's ``latency_s`` to the sensor's
``clock.latency_s`` (a note says so): otherwise the 80 ms placeholder would be compared with an 8 ms Wi-Fi hop, and the
receiver's frame stamps would be 4 frames early relative to the images.

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


def raw_dir_of(R: rn.Run, demo_name: str) -> str:
    return demo_name + RAW_SUFFIX


def build_replay_npz(R: rn.Run, did: int, path: Path) -> dict:
    """The demo's TCP pose / width history on the episode clock for ``task --history-source replay_gt``."""
    from franka_sim import stage1_replay as s1

    dm = s1.load_demos(R.zarr, [did])[0]
    T = dm["T"]
    xyz_wxyz = np.concatenate([T[:, :3, 3], Rotation.from_matrix(T[:, :3, :3]).as_quat()[:, [3, 0, 1, 2]]], axis=1)
    t_ns = np.round(np.arange(len(T)) * 1e9 / 60.0).astype(np.int64)
    src = "uniform_60hz"
    try:
        from sensor_rig.sim.iphone_reader import IphoneEpisode
        from franka_deploy.launch.launcher import resolve_replay_episode

        ep = IphoneEpisode(resolve_replay_episode({**R.station.get("sim"), "episode_dir": raw_dir_of(R, dm["name"])}))
        t = ep.main_times_unix_ns()
        if len(t) == len(T):
            t_ns, src = (t - t[0]).astype(np.int64), "iphone_slot_times"
    except Exception:       # noqa: BLE001 - the 60 Hz grid is within ~1 ms of the slot times
        pass
    np.savez(path, t_ns=t_ns, xyz_wxyz=xyz_wxyz, width=dm["w"], origin_ns=np.array(0, np.int64))
    return {"name": dm["name"], "n": len(T), "t_ns": t_ns, "time_source": src}


def derive_deploy(R: rn.Run, did: int, info: dict, work: Path, npz: Path, port_offset: int) -> tuple[Path, dict]:
    """A deploy.yaml next to the run: the profile + this stage's overrides. Returns (path, the overrides)."""
    raw = yaml.safe_load(Path(R.cfg["config_path"]).read_text())
    prof = copy.deepcopy(raw["profiles"][R.profile])
    sets = list(R.sets)
    src = R.image_source
    if src == "video_replay":
        sets.append(f"sim.episode_dir={raw_dir_of(R, info['name'])}")
    elif src == "rig_raw_replay":
        sets.append(f"sim.generate.episode_dir={raw_dir_of(R, info['name'])}")
    else:
        raise rn.StageIncomplete(f"Stage 2b needs a replay image source (video_replay / rig_raw_replay), not {src!r}")
    if R.policy.get("kind") == "replay":
        sets += [f"policy.episode_index={did}", "policy.start_frame_index=0"]
    clock = R.station.get("sensor.clock")
    if clock.get("mode") == "arrival_minus_latency":
        lat = float(clock["latency_s"]["main"])
        if float(R.station.get("link.latency_s")) < lat:
            sets.append(f"link.latency_s={lat}")
            R.notes.append(f"link latency raised to the sensor's clock.latency_s ({lat} s): the sim does not model the phone's "
                           "capture pipeline, the receiver subtracts it")
    prof["station_set"] = sets
    if R.policy.get("kind") in ("bp_server", "command"):
        # the laptop's BP env gets torch from ~/.local (README.md): the launcher's PYTHONNOUSERSITE=1 would hide it
        prof["user_site_children"] = sorted(set(prof.get("user_site_children") or []) | {"inference"})
    prof["work_root"] = str(work)
    prof["run"] = {**prof.get("run", {}), **R.run_ids}
    prof.setdefault("task", {}).update({"history_source": "replay_gt", "replay_npz": str(npz), "tap_images_every": 0})
    if port_offset:
        o = port_offset
        prof["station_set"] += [f"sensor.transport.ports.main={5555 + o}"] + (
            [f"sensor.transport.ports.ultrawide={5556 + o}"] if R.protocol == "v1" else [])
        prof.setdefault("camera", {}).update(endpoint=f"tcp://localhost:{16702 + o}", logger_endpoint=f"tcp://localhost:{25556 + o}")
        prof.setdefault("bridge", {})["endpoint"] = f"tcp://127.0.0.1:{5565 + o}"
        prof.setdefault("controller", {})["endpoint"] = f"tcp://localhost:{26010 + o}"
        prof.setdefault("inference", {})["endpoint"] = f"tcp://0.0.0.0:{18765 + o}"
        prof.setdefault("spacemouse", {})["endpoint"] = f"tcp://localhost:{15557 + o}"
    raw["profiles"][R.profile] = prof
    p = work.parent / "deploy_stage2b.yaml"
    p.write_text(yaml.safe_dump(raw))
    return p, prof


def launch(R: rn.Run, dep: Path, script: str, cwd: Path, timeout_s: float) -> int:
    cmd = [R.python_re, "-u", "-m", "franka_deploy", "run", "--config", str(dep), "--profile", R.profile, "--no-tty",
           "--script", script]
    env = rn.child_env()
    with open(cwd / "launcher.log", "w") as lg:
        p = subprocess.Popen(cmd, stdout=lg, stderr=subprocess.STDOUT, env=env, cwd=str(cwd), start_new_session=True)
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
    t_c = float(os.environ.get("EVAL_2B_START_S", 12.0 if R.policy.get("kind") in ("bp_server", "command") else 3.0))
    script = f"{t_c}:c,{t_c + dur + 4.0:.1f}:s,{t_c + dur + 7.0:.1f}:q"
    dep, prof = derive_deploy(R, did, info, work, npz, port_offset)
    t0 = time.time()
    rc = launch(R, dep, script, dd, timeout_s=float(os.environ.get("EVAL_2B_TIMEOUT_S", 900)) + dur)
    sessions = sorted(work.glob("session_*"))
    if not sessions:
        raise rn.StageIncomplete(f"demo {did}: no session dir (launcher rc {rc}); see {dd / 'launcher.log'}")
    session = sessions[-1]
    summ_p = session / "launcher_summary.json"
    summ = json.loads(summ_p.read_text()) if summ_p.exists() else {}
    if summ.get("sender_died"):
        raise rn.StageIncomplete(f"demo {did}: sender_died {summ['sender_died']} (episode aborted)")
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
    r = subprocess.run([R.python_re, "-m", "franka_deploy.tools.analyze_tap", "--tap-dir", str(tap_dir), "--replay-npz",
                        rg["npz"], "--start-now-ns", str(rg["start_now_ns"]), "--out", str(tap_json)],
                       capture_output=True, text=True, env=rn.child_env())
    tap = json.loads(tap_json.read_text()) if tap_json.exists() else {"analyze_tap_error": r.stderr[-500:]}
    win = (float(calls[0]["info_main_ts"][0]) - 1.0, float(calls[-1]["info_main_ts"][-1]) + 1.0)
    st = stamp_errors(session, win)
    st_all = stamp_errors(session)
    cs = (summ.get("camera_stats") or {})
    main_stats = (cs.get("streams") or {}).get("main", {})
    part = {
        "demo": did, "name": info["name"], "n_calls": len(calls), "n_pairs": int(ok.sum()), "n_seed_mismatch": n_seed_mismatch,
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
    # everything in a live run depends on wall-clock scheduling: all of it is marked timing (reproducibility checks skip it)
    m["timing"] = sorted(k for k in m if k not in ("n_demos", "demos", "protocol", "clock_mode", "timing"))
    rows = [{"demo": p["name"][:30], "calls": p["n_calls"], "pairs": p["n_pairs"],
             "live_vs_offline_p95_mm_k8": round(s2a.pct(p["err"]["pos_8"], 95), 2) if p["err"] else None,
             "stamp_p95_ms": round(s2a.pct(np.abs(p["stamp_ms"].get("main", np.array([]))), 95) or 0.0, 2),
             "holds": p["holds"], "stale": p["stale_image"], "wall_s": p["wall_s"]} for p in parts]
    return m, rows


def run(R: rn.Run, d: Path, results: dict) -> dict:
    demos = R.demo_list()
    if demos is None:
        demos = list(range(int(getattr(R.args, "episodes", None) or 1)))
    po = getattr(R.args, "port_offset", None)
    po = int(os.environ.get("FD_TEST_PORT_OFFSET", 0)) if po is None else int(po)
    parts = [run_demo(R, d, did, po, results) for did in demos]
    m, rows = aggregate(parts, R)
    for p in parts:
        R.notes.append(f"demo {p['demo']} session {p['session']}")
    return {"metrics": m, "rows": rows, "notes": [f"live run per demo under {d}/demo_NN (launcher.log, work/session_*)"]}
