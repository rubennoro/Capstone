#!/usr/bin/env python3
"""T3 dry run: the Jetson session command, end to end, on simulated sensors -> the real iPhUMI pipeline.

    $PY sensor_rig/tests/e2e/run_collect_dryrun.py --workdir <dir> [--demos 49242,18798] [--keep]

1. ``W/orig_raw``: the real iPhone calibration 15647 + demos (per-file symlinks, read-only), as in
   run_pipeline_compare.py.
2. ``sensor_rig sim --main-fps 30 --no-import`` -> raw rig sources (30 fps main = the Jetson's fill mode) with their
   simulated VIO trajectories.
3. ``sensor_rig collect --mock-replay <sources>`` as an operator would run it (subprocess, keys on stdin): the
   gripper calibration (checked, incl. the ArUco tag check), then one demo per demo source; each episode stops when
   its source ends. The source -> live clock offset is stored in each episode's meta.json (``sim.replay_offset_ns``).
4. ``pose-import`` of each source trajectory onto the re-recorded demo (the offset is applied automatically),
   ``export``, ``validate``.
5. The commands of process_dataset.sbatch on the originals (A, session firstreal) and on the collected exports
   (B, session rigdry) in isolated work dirs; intermediate + zarr checks of run_pipeline_compare.py with the fill-mode
   tolerances, pairing B's new episode names with A's by ``meta.sim.replay_source`` and aligning B's grid to A's in
   time. ``W/report.json``; exit 0 iff every check passes. data/raw and iPhUMI's tmp_* folders are never written.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import threading
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_pipeline_compare as rpc  # noqa: E402

SESSION_B = "rigdry"


def drive_collect(py: str, sources: list[Path], out: Path, log: Path, n_demos: int) -> None:
    env = dict(os.environ, PYTHONUNBUFFERED="1")
    env.pop("PYTHONPATH", None)
    cmd = [py, "-m", "sensor_rig", "collect", "--mock-replay", *map(str, sources), "--out", str(out),
           "--session", SESSION_B, "--task-label", rpc.TASK]
    lines: list[str] = []
    with open(log, "w") as f:
        f.write("$ " + " ".join(cmd) + "\n")
        p = subprocess.Popen(cmd, cwd=str(out.parent), stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                             stderr=subprocess.STDOUT, text=True, env=env)

        def reader():
            for ln in p.stdout:
                lines.append(ln)
                f.write(ln)
                f.flush()
        threading.Thread(target=reader, daemon=True).start()

        def wait(s, n=1, t=180):
            t0 = time.time()
            while time.time() - t0 < t:
                if sum(s in ln for ln in lines) >= n:
                    return
                if p.poll() is not None:
                    break
                time.sleep(0.1)
            raise RuntimeError(f"collect: {s!r} not seen; see {log}")

        def key(k=""):
            p.stdin.write(k + "\n")
            p.stdin.flush()

        wait("-> Enter: record the GRIPPER CALIBRATION")
        key()
        wait("CAL KEPT")
        for i in range(n_demos):
            wait("-> Enter: start demo", i + 1)
            key()
            wait("DEMO KEPT", i + 1)
        key("q")
        if p.wait(60) != 0:
            raise RuntimeError(f"collect exited {p.returncode}; see {log}")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--workdir", required=True)
    ap.add_argument("--src", default=str(rpc.DEFAULT_SRC))
    ap.add_argument("--demos", default="49242,18798")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--python", default=sys.executable)
    ap.add_argument("--keep", action="store_true")
    args = ap.parse_args(argv)
    py = args.python
    W = Path(args.workdir).resolve()
    W.mkdir(parents=True, exist_ok=True)
    for forbidden in (rpc.DP.resolve(), (rpc.REPO / "data").resolve()):
        if W == forbidden or forbidden in W.parents:
            raise SystemExit(f"refusing: work dir {W} is inside {forbidden}")
    rpc.log(f"work dir {W}")
    snap0 = rpc.repo_snapshot()
    rep = rpc.Report()
    t_start = time.time()
    L = W / "logs"
    L.mkdir(exist_ok=True)
    demo_ids = [x.strip() for x in args.demos.split(",") if x.strip()]
    eps = rpc.select_sources(Path(args.src).resolve(), demo_ids)
    rpc.link_originals(eps, W / "orig_raw")
    rpc.make_calibration_yaml(W / "gripper_calibration.yaml")
    sr = [py, "-m", "sensor_rig"]

    sim_root, rig_raw, rig_exp = W / "sim", W / "rig_raw", W / "rig_export"
    if not sim_root.exists():
        rpc.log("sim (30 fps main)")
        rpc.run(sr + ["sim", "--iphone", *[str(W / "orig_raw" / e.name[:10] / e.name) for e in eps], "--out",
                      str(sim_root), "--main-fps", "30", "--no-import", "--workers", str(args.workers)], W,
                L / "sim.log")
    sources = sorted(p for p in sim_root.glob("*/*") if p.is_dir() and not p.name.startswith("."))
    if not rig_raw.exists():
        rpc.log("collect --mock-replay (calibration + demos, keys on stdin)")
        drive_collect(py, sources, rig_raw, L / "collect.log", len(demo_ids))
    summ = sorted((rig_raw / ".sensor_rig" / "sessions").glob(f"{SESSION_B}_*.json"))[-1]
    s = json.loads(summ.read_text())
    rep.add("collect: calibration kept, demos kept", s["counts"]["calibrations_kept"] == 1
            and s["counts"]["demos_kept"] == len(demo_ids), note=json.dumps(s["counts"]))
    cal = [e for e in s["episodes"] if e["kind"] == "gripper_calibration" and e["status"] == "kept"][0]
    rep.add("collect: calibration tag check", "tags: both visible" in (cal.get("reason") or ""),
            note=cal.get("reason") or "")

    from sensor_rig.raw_episode import find_raw_episodes

    id_map, extra = {}, {}
    for ep in find_raw_episodes(rig_raw):
        sim = ep.meta["sim"]
        src = Path(sim["replay_source_dir"])
        a_id, b_id = src.name.split("_")[1], ep.name.split("_")[1]
        id_map[a_id] = b_id
        from sensor_rig.raw_episode import RawEpisode

        # B string time -> A (iPhone) timeline: minus B's wall offset, minus the replay offset, plus the source's
        extra[a_id] = RawEpisode(src).wall_offset_ns() - ep.wall_offset_ns() - int(sim["replay_offset_ns"])
        if ep.kind == "demonstration" and not ep.trajectory_path():
            rpc.run(sr + ["pose-import", "--raw", str(ep.path), "--traj", str(src / "sim" / "vio_trajectory.tum"),
                          "--format", "tum", "--frame", "imu"], W, L / f"pose_import_{a_id}.log")
    for a_id, b_id in sorted(id_map.items()):
        p = next(rig_raw.glob(f"*/*_{b_id}_*/trajectory/trajectory.json"), None)
        if p is None:
            continue
        info = json.loads(p.read_text())
        rep.add(f"pose-import {a_id}: replay offset applied", info["time_offset_source"] == "meta.sim.replay_offset_ns",
                note=str(info["time_offset_applied_ns"]))
        rep.add(f"pose-import {a_id}: overlap", info["main_overlap_fraction"], None, ok=info["main_overlap_fraction"] > 0.95)
        rep.add(f"pose-import {a_id}: |traj delay| [ms]", abs(info["time_offset_check"]["traj_delay_ms"]), 5.0)
        rep.add(f"pose-import {a_id}: motion check ok", info["motion_check"]["ok"],
                note=f"gyro resid rms {info['motion_check']['gyro_resid_rms']:.4f} rad/s")
    if not rig_exp.exists():
        rpc.log("export")
        rpc.run(sr + ["export", "--raw", str(rig_raw), "--out", str(rig_exp), "--workers", str(args.workers)], W,
                L / "export.log")
    rpc.run(sr + ["validate", str(rig_exp), "--gripper-calibration", str(W / "gripper_calibration.yaml"), "--decode"],
            W, L / "validate.log")
    rep.add("sensor_rig validate (collected exports, --decode)", True,
            note=(L / "validate.log").read_text().strip().splitlines()[-1])
    modes, lags = rpc.export_modes(rig_exp)
    rep.add("every export logged grid_mode fill", set(modes.values()) == {"fill"}, note=str(modes))
    for b_id, lg in lags.items():
        rpc.lag_rows(rep, b_id, lg)

    # A timeline offsets of B's grids (per A id)
    b_first = {}
    from sensor_rig.timefmt import parse_iso

    for j in rig_exp.glob("*/*/*.json"):
        d = json.loads(j.read_text())
        if d.get("poseTimes"):
            b_first[j.stem.split("_")[1]] = parse_iso(d["poseTimes"][0])
    shift = {}
    for e in eps:
        a_id = e.name.split("_")[1]
        d = json.loads(next(e.glob("*.json")).read_text()) if e.is_dir() else None
        if d and d.get("poseTimes") and id_map.get(a_id) in b_first:
            shift[a_id] = b_first[id_map[a_id]] + extra[a_id] - parse_iso(d["poseTimes"][0])

    tol = {**rpc.TOL, **rpc.TOL_FILL}
    za = rpc.pipeline("A", W / "orig_raw", W, py, args.workers)
    zb = rpc.pipeline("B", rig_exp, W, py, args.workers, session=SESSION_B)
    rpc.intermediate_checks(W, rep, tol, id_map=id_map)
    rpc.zarr_checks(W, za, zb, rep, 1, tol, id_map=id_map, shift_ns=shift)
    diff = rpc.snapshot_diff(snap0, rpc.repo_snapshot())
    rep.add("repo untouched (iPhUMI/, scripts/, data/raw)", not diff, note="; ".join(diff[:5]))
    report = {"workdir": str(W), "ok": rep.ok, "tolerances": tol, "id_map": id_map, "grid_shift_ns": shift,
              "rows": rep.rows, "runtime_s": round(time.time() - t_start, 1), "session_summary": str(summ)}
    (W / "report.json").write_text(json.dumps(report, indent=1, default=str))
    print(rep.table())
    print(f"\n{'ALL PASS' if rep.ok else 'FAILURES'}; report: {W / 'report.json'}")
    if rep.ok and not args.keep:
        shutil.rmtree(W, ignore_errors=True)
    return 0 if rep.ok else 1


if __name__ == "__main__":
    sys.exit(main())
