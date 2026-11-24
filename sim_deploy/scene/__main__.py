"""python -m sim_deploy.scene [--home start|deploy] check     region corners: on the table, in the box, in view
python -m sim_deploy.scene sample --seed S --rollouts N   the eraser poses an eval with --seed S uses
python -m sim_deploy.scene replay replay:<zarr>#<index>   where G6 puts the eraser for that episode
python -m sim_deploy.scene replay-all <zarr>               the replay placement of every episode (distribution)
python -m sim_deploy.scene start-pose --show | --q "q1,...,q7" | --from-probe FILE | --from-git [REF]  [--force]
                                                           the start pose (stations/start_pose.yaml, P2)
python -m sim_deploy.scene start-data [--dataset ZIP] [--write]
                                                           the eraser relative to the demos' start (P2)

--home defaults to start (the eval driver's default); trainstart is an alias of start."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

from sim_deploy import upstream
from sim_deploy.scene import start_pose as sp
from sim_deploy.scene.placement import DEFAULT_MODEL, HOME_ALIASES, HOMES, Scene, canonical_home
from sim_deploy.scene.zarr_lowdim import n_episodes


def _start_pose(a) -> int:
    sources = [x for x in (a.q, a.from_probe, a.from_git) if x is not None]
    if len(sources) > 1:
        print("start-pose: give one of --q, --from-probe, --from-git", file=sys.stderr)
        return 2
    if not sources:                                       # --show (or nothing): the current one
        info = sp.show(a.model)
        if a.json:
            print(json.dumps({k: v for k, v in info.items() if k != "O_T_cam_cv"}, indent=1))
        elif not info["set"]:
            print(info["text"])
        else:
            for ln in sp.format_info(info, info["source"]):
                print(ln)
        return 0 if info.get("set") else 1
    try:
        if a.q is not None:
            q, source = sp.parse_q_text(a.q), "--q (typed)"
        elif a.from_probe is not None:
            text = sys.stdin.read() if a.from_probe == "-" else Path(a.from_probe).expanduser().read_text()
            q, source = sp.parse_probe(text), f"pc_probe --start-pose printout {a.from_probe}"
        else:
            q = sp.from_git(a.from_git)
            source = (f"git -C {upstream.repo()} show {a.from_git}:{sp.GIT_STATION} "
                      f"(franka_control.homing.q_start_rad)")
    except (sp.StartPoseError, OSError) as e:
        print(f"start-pose: {e}", file=sys.stderr)
        return 1
    if a.source:
        source = f"{a.source} [{source}]"
    info = sp.describe(q, a.model)
    for ln in sp.format_info(info, source):
        print(ln)
    if not info["ok"] and not a.force:
        print(f"start-pose: not written: the pose fails a check (--force writes it anyway); {sp.START_POSE_YAML} "
              f"unchanged", file=sys.stderr)
        return 1
    if a.dry:
        print("start-pose: --dry: not written")
        return 0
    sp.write_start_pose(q, source + ("" if info["ok"] else " (--force: fails a check)"), info)
    print(f"wrote {sp.START_POSE_YAML}  (restart the sim: run.sh merges it at each sim PC start)")
    if a.show:
        sh = sp.show(a.model)
        print(f"  {'ok  ' if sh['region']['ok'] else 'WARN'} eraser region      {sh['region']['detail']}")
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="python -m sim_deploy.scene")
    ap.add_argument("--model", default=str(DEFAULT_MODEL))
    ap.add_argument("--random", default=None, help="region file (default: the home's)")
    ap.add_argument("--home", choices=HOMES + tuple(HOME_ALIASES), default="start")
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("check")
    s = sub.add_parser("sample")
    s.add_argument("--seed", type=int, default=0)
    s.add_argument("--rollouts", type=int, default=5)
    r = sub.add_parser("replay")
    r.add_argument("target")
    r.add_argument("--behind-tip", type=float, default=None, help="m (default: half the eraser height)")
    ra = sub.add_parser("replay-all")
    ra.add_argument("dataset")
    st = sub.add_parser("start-pose", help="show or set stations/start_pose.yaml (the PC key p's q_start_rad)")
    st.add_argument("--show", action="store_true", help="print the current one (also after a write)")
    st.add_argument("--q", help='7 joint angles, rad: "q1,q2,...,q7"')
    st.add_argument("--from-probe", metavar="FILE", help="a saved `pc_probe --start-pose` printout ('-' = stdin)")
    st.add_argument("--from-git", nargs="?", const=sp.DEFAULT_GIT_REF, metavar="REF",
                    help=f"REF:{sp.GIT_STATION} (default {sp.DEFAULT_GIT_REF}, env SIM_DEPLOY_UPSTREAM_REF; "
                         f"read in env SIM_DEPLOY_UPSTREAM_REPO, default {upstream.DEFAULT_REPO})")
    st.add_argument("--source", help="text for the file's source line (default: where the q came from)")
    st.add_argument("--force", action="store_true", help="write even if a check fails")
    st.add_argument("--dry", action="store_true", help="check, do not write")
    st.add_argument("--json", action="store_true", help="--show as JSON")
    sd = sub.add_parser("start-data", help="the eraser relative to the demos' start (randomization_start.yaml)")
    sd.add_argument("--dataset", default=str(sp.DEFAULT_DATASET))
    sd.add_argument("--write", action="store_true", help=f"rewrite {sp.RANDOM_START.name} from these numbers")
    argv = list(sys.argv[1:] if argv is None else argv)
    for i in range(len(argv) - 1):                        # --q "-0.62,..." starts with '-': argparse needs --q=...
        if argv[i] == "--q":
            argv[i:i + 2] = [f"--q={argv[i + 1]}"]
            break
    a = ap.parse_args(argv)
    if a.cmd == "start-pose":
        return _start_pose(a)
    if a.cmd == "start-data":
        rel = sp.relative_to_start(Path(a.dataset).expanduser())
        for e in rel["episodes"]:
            print(f"{e['episode']:3d} ahead {e['ahead_m']:+.3f} left {e['left_m']:+.3f} below {e['below_m']:.3f} m  "
                  f"tip {e['start_tip_height_m']:.3f} m  tilt {e['start_tilt_deg']:5.1f}  yaw_rel {e['yaw_rel_deg']:+5.1f}")
        for k, v in rel["stats"].items():
            print(f"{k:<20} median {v['median']:+8.3f}  p10 {v['p10']:+8.3f}  p90 {v['p90']:+8.3f}  "
                  f"min {v['min']:+8.3f}  max {v['max']:+8.3f}")
        if a.write:
            print(f"wrote {sp.write_region(rel)}")
        return 0
    home = canonical_home(a.home)
    sc = Scene.load(a.model, a.random, home=home)
    if a.cmd == "check":
        if home == "start":
            info = sp.describe(sc.start["q"], a.model)
            print(f"home start: {sp.summary_line(info)}  ({sc.start['file']})")
            print(f"region {sc.random['_path']}: ahead {sc.random['ahead_m']} m, left {sc.random['left_m']} m, "
                  f"yaw_rel {sc.random['yaw_rel_deg']} deg (median {sc.random.get('median')})")
            if not info["ok"]:
                print("start pose checks: FAIL: " + ", ".join(k for k, c in info["checks"].items() if not c["ok"]))
        rep = sc.check_region()
        for c in rep["corners"]:
            print(c)
        if home == "start":
            n_clip = sum(sc.random_placement(0, n)["clipped"] for n in range(100))
            print(f"seed 0, rollouts 0-99: {n_clip} placements clipped to the table / box")
        print("OK" if rep["ok"] else "FAIL")
        return 0 if rep["ok"] else 1
    if a.cmd == "sample":
        for n in range(a.rollouts):
            rp = sc.random_placement(a.seed, n)
            extra = (f"  ahead {rp['ahead_m']:.3f} left {rp['left_m']:+.3f} yaw_rel {rp['yaw_rel_deg']:+.1f}"
                     f"{'  CLIPPED' if rp['clipped'] else ''}") if home == "start" else ""
            print(n, np.round(rp["pose"], 4).tolist(), extra)
        return 0
    if a.cmd == "replay":
        rp = sc.replay_placement(a.target, a.behind_tip)
        print(json.dumps({k: v for k, v in rp.items() if k not in ("G_rel_start", "O_T_grasp")}, indent=1))
        return 0
    rows = []
    for i in range(n_episodes(a.dataset)):
        rp = sc.replay_placement(f"replay:{a.dataset}#{i}")
        p = rp["pose"]
        rows.append([p[0], p[1], rp["yaw_deg"], rp["height_error_m"], rp["approach_tilt_deg"]])
        print(f"{i:3d} xy ({p[0]:.3f}, {p[1]:+.3f}) yaw {rp['yaw_deg']:+6.1f}  height_error {rp['height_error_m']:+.3f} m"
              f"  tilt {rp['approach_tilt_deg']:5.1f} deg  t_grasp {rp['grasp_t_s']:.1f} s  on_table {rp['on_table']}")
    a_ = np.array(rows)
    print("median xy", np.round(np.median(a_[:, :2], 0), 3).tolist(), "range x", np.round([a_[:, 0].min(), a_[:, 0].max()], 3).tolist(),
          "y", np.round([a_[:, 1].min(), a_[:, 1].max()], 3).tolist(), "yaw", np.round([a_[:, 2].min(), a_[:, 2].max()], 1).tolist(),
          "height_error median", round(float(np.median(a_[:, 3])), 3))
    return 0


if __name__ == "__main__":
    sys.exit(main())
