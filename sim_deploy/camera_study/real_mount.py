"""W2 step 4 (addendum "real arm"): the REAL MOUNT reference: observations made from the real Franka trials' probe
images (2026-11-20, ``~/Capstone/data/<task>/policy/<run>/deploy/*_probe_live/``: the policy's four 224x224 inputs
after the real Pi's JPEG + the deployment's ImagePrep, ~10 sets/s, written by the copy's ``frames/probe.py --live
--save``: ``NNNNNN_{main,ultrawide}_{0,1}.png``, index = the reply's order = oldest first, contracts/c1_frames.py).

Per trial: the static start sets (arm at the start pose, before the motion) are found by image differences (the first
set k > 8 with mean |main_1[k] - main_1[k-1]| > 0.6 is the motion start; the sets [3, start - 4] are static), and
``--per-trial`` (default 6) sets evenly spread over that span become observations: main / ultrawide = the probe PNGs
(_0 then _1), low-dim = the episode start (eef_pos zeros, eef_rot6d and eef_rot6d_wrt_start [1,0,0,0,1,0]), width
0.090 (= gripper_map^-1(0.040), the robot's open fingers as the policy sees them) and, as a check, 0.080
(``obs_w080.npz``). The trial's start pose (TCP 4x4 in O) comes from the session's ``manager/episodes.jsonl``.

Run (``~/envs/sim_deploy``, from ``~/Capstone-main-sim``, ``unset PYTHONPATH``)::

    python -m sim_deploy.camera_study.real_mount export [--per-trial 6]
    python -m sim_deploy.camera_study.real_mount jobs JOBS.json [--widths 090 080]

Output ``sim_evals/_pipeline/W2/attribution/real_mount/<trial>/set<NNNNNN>/{obs.npz,obs_w080.npz}``, ``<trial>/meta.json``
(run, session, episode, start pose, motion start, chosen sets, static-span image noise) and ``index.json``. Job ids
``rm__<trial>__s<NNNNNN>`` (width 0.090) and ``rm080__...``; the demo direction for ``approach`` = the mean of the
five ``first`` references' own displacements (there is no demo for a robot trial).
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
DATA = Path("~/Capstone/data").expanduser()
SESSIONS = Path("~/policy_deploy_sessions").expanduser()
OUT = REPO / "sim_evals/_pipeline/W2/attribution/real_mount"
I6 = np.array([1, 0, 0, 0, 1, 0], np.float32)
MOTION_THRESH = 0.6


def _png(path: Path) -> np.ndarray:
    import cv2
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(path)
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def trials() -> list[dict]:
    out = []
    for f in sorted(DATA.glob("*/policy/*/deploy/*_probe_live")):
        notes = (f / "NOTES.md").read_text()
        m = re.search(r"policy_deploy session: ~/policy_deploy_sessions/(\S+)", notes)
        run = f.parent.parent
        tm = re.search(r"_(trial\d+)_", f.name)
        tid = f"{run.parent.parent.name}__{run.name.split('_2026')[0]}__{f.name[11:16]}_{tm.group(1)}"
        out.append({"id": tid, "folder": f, "run": run, "session": m.group(1) if m else None,
                    "same_session_later": "later episode" in notes, "notes": notes})
    return out


def export(per_trial: int) -> int:
    import cv2
    cv2.setNumThreads(4)
    OUT.mkdir(parents=True, exist_ok=True)
    index = []
    by_session: dict[str, list] = {}
    for t in trials():
        by_session.setdefault(t["session"], []).append(t)
    for t in trials():
        f = t["folder"]
        n = len(list(f.glob("*_main_1.png")))
        diffs, prev, motion = [], None, None
        for i in range(n):
            m = _png(f / f"{i:06d}_main_1.png").astype(np.float32)
            if prev is not None:
                diffs.append(float(np.abs(m - prev).mean()))
                if i > 8 and diffs[-1] > MOTION_THRESH:
                    motion = i
                    break
            prev = m
        if motion is None:
            raise SystemExit(f"{f}: no motion start found")
        lo, hi = 3, motion - 4
        sets = sorted(set(int(round(x)) for x in np.linspace(lo, hi, per_trial)))
        # the trial's episode: the k-th trial folder of a session = its k-th episode
        eps = [json.loads(ln) for ln in (SESSIONS / t["session"] / "manager" / "episodes.jsonl").read_text().splitlines()]
        k = [x["id"] for x in sorted(by_session[t["session"]], key=lambda x: x["folder"].name)].index(t["id"])
        ep = eps[k]
        d = OUT / t["id"]
        for s in sets:
            sd = d / f"set{s:06d}"
            sd.mkdir(parents=True, exist_ok=True)
            imgs = {"main": np.stack([_png(f / f"{s:06d}_main_{j}.png") for j in (0, 1)]),
                    "ultrawide": np.stack([_png(f / f"{s:06d}_ultrawide_{j}.png") for j in (0, 1)])}
            for w, name in ((0.090, "obs.npz"), (0.080, "obs_w080.npz")):
                np.savez(sd / name, eef_pos=np.zeros((2, 3), np.float32), eef_rot6d=np.stack([I6, I6]),
                         eef_rot6d_wrt_start=np.stack([I6, I6]), gripper_width_m=np.full((2, 1), w, np.float32),
                         **imgs)
        meta = {"trial": t["id"], "folder": str(f), "run": str(t["run"]), "session": t["session"],
                "episode_index_in_session": k, "episode": {kk: ep[kk] for kk in ("episode_id", "outcome", "reason",
                                                                                  "duration_s", "wall_start")},
                "start_pose_O": ep["start_pose"], "n_sets": n, "motion_start_set": motion,
                "static_span": [lo, hi], "sets": sets,
                "static_main1_diff_mean": float(np.mean(diffs[lo:hi])) if hi > lo else None,
                "notes_first_lines": t["notes"].splitlines()[:4]}
        (d / "meta.json").write_text(json.dumps(meta, indent=1))
        index.append({"trial": t["id"], "run": str(t["run"]), "sets": sets, "motion_start_set": motion,
                      "folder": str(d), "start_pose_O": ep["start_pose"]})
        print(f"{t['id']:<45} {n:4d} sets, motion at {motion:3d}, static diff {meta['static_main1_diff_mean']:.2f}, "
              f"sets {sets}, episode {ep['episode_id']} ({ep['outcome']}/{ep['reason']}), start z "
              f"{ep['start_pose'][2][3]:.3f}")
    (OUT / "index.json").write_text(json.dumps({"trials": index}, indent=1))
    return 0


def mean_first_demo() -> list:
    from sim_deploy.camera_study.attribution_table import _refs
    from sim_deploy.camera_study.chunk_metrics import demo_displacement
    ds = [demo_displacement(r["dataset"], r["index"], r["row"]) for r in _refs() if r["label"] == "first"]
    return np.mean(ds, axis=0).tolist()


def jobs(out: Path, widths: list[str]) -> int:
    idx = json.loads((OUT / "index.json").read_text())["trials"]
    demo = mean_first_demo()
    js = []
    for t in idx:
        for s in t["sets"]:
            sd = Path(t["folder"]) / f"set{s:06d}"
            for w in widths:
                pre = "rm" if w == "090" else f"rm{w}"
                fn = "obs.npz" if w == "090" else f"obs_w{w}.npz"
                js.append({"id": f"{pre}__{t['trial']}__s{s:06d}", "obs": str(sd / fn), "demo_disp": demo})
    out.write_text(json.dumps(js, indent=1))
    print(f"{len(js)} jobs -> {out}")
    return 0


def main(argv=None) -> int:
    import sys
    sys.path.insert(0, str(REPO))
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    e = sub.add_parser("export")
    e.add_argument("--per-trial", type=int, default=6)
    j = sub.add_parser("jobs")
    j.add_argument("out", type=Path)
    j.add_argument("--widths", nargs="+", default=["090", "080"])
    a = ap.parse_args(argv)
    return export(a.per_trial) if a.cmd == "export" else jobs(a.out, a.widths)


if __name__ == "__main__":
    raise SystemExit(main())
