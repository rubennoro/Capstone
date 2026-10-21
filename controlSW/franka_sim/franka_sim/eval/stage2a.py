"""Stage 2a: the policy's predictions on held-out demos, mapped to the robot and checked with IK.

1. ``tools.predict_demos`` (BP subprocess): K = 16 relative chunks every ``exec_h`` agent steps, from the policy served
   over the robotmq protocol (any policy kind), plus the demo's own future over the same horizon.
2. Placement: a Stage 1 run dir (``--stage1``; or the Stage 1 run of this ``--stage all``; else Stage 1 runs first for the
   demos of the predictions) gives ``base_T_world`` per demo.
3. Per queried frame i: anchor = base_T_TCP(i) (the demo TCP at frame i in the base frame), absolute chunk poses =
   anchor . rel_k, ``kinematics.check_traj`` on the chunk and, for reference, on the demo's own future. The IK seed is
   the demo's tracked joint configuration at frame i (``stage1_replay.track_best`` at 20 Hz).
4. Metrics: chunk reachability, joint margin, collisions, position (p50/p95 mm), rotation (deg) and width (mm) error
   against the demo at k = 4, 8, 15, inference time p50/p95. Gates: ``criteria.stage2`` (2a keys).
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

import numpy as np

from franka_sim.eval import runner as rn
from franka_sim.eval import stage1 as s1w

KS = (4, 8, 15)


def run_predict(R: rn.Run, d: Path, demos: list | None, every: int | None = None, queries: Path | None = None,
                seed: int | None = None, out_name: str = "pred.npz") -> Path:
    """Run tools.predict_demos in the BP interpreter. ``demos``: zarr indices (None = the held-out split, limited by
    --episodes)."""
    pol = rn.write_policy_yaml(R, d / "policy_resolved.yaml")
    out = d / out_name
    cmd = [R.python_bp, "-u", "-m", "franka_deploy.tools.predict_demos", "--policy", str(pol), "--zarr", R.zarr,
           "--seed", str(R.seed if seed is None else seed), "--out", str(out), "--work-dir", str(d / "predict_work"),
           "--device", str(R.station.get("host.inference_device", "cuda"))]
    if queries is not None:
        cmd += ["--queries", str(queries)]
    elif demos is not None:
        cmd += ["--demos", ",".join(str(x) for x in demos)]
    else:
        cmd += ["--split", "val"] + (["--limit", str(R.args.episodes)] if getattr(R.args, "episodes", None) else [])
    if every:
        cmd += ["--every", str(every)]
    with open(d / f"{out_name}.log", "w") as lg:
        r = subprocess.run(cmd, stdout=lg, stderr=subprocess.STDOUT, env=rn.child_env(), cwd=str(d))
    if r.returncode != 0 or not out.is_file():
        raise rn.StageIncomplete(f"predict_demos failed (rc {r.returncode}); see {d / (out_name + '.log')}")
    return out


def ang_err_deg(Ra: np.ndarray, Rb: np.ndarray) -> np.ndarray:
    cos = (np.einsum("nij,nij->n", Ra, Rb) - 1) / 2
    return np.degrees(np.arccos(np.clip(cos, -1, 1)))


def chunk_errors(rel: np.ndarray, gt: np.ndarray, ks=KS) -> dict:
    """Per-chunk errors at step k between two (n,K,10) relative chunk arrays: pos mm, rot deg, width mm."""
    from franka_sim import frames as fr

    out = {}
    for k in ks:
        out[f"pos_{k}"] = np.linalg.norm(rel[:, k, :3] - gt[:, k, :3], axis=-1) * 1000
        Ra = np.array([fr.rot6d_to_R(x) for x in rel[:, k, 3:9]])
        Rb = np.array([fr.rot6d_to_R(x) for x in gt[:, k, 3:9]])
        out[f"rot_{k}"] = ang_err_deg(Ra, Rb)
        out[f"width_{k}"] = np.abs(rel[:, k, 9] - gt[:, k, 9]) * 1000
    return out


def pct(x, q) -> float | None:
    x = np.asarray(x, float)
    return float(np.percentile(x, q)) if x.size else None


def check_chunks(ctx, bTt_at: dict, q_seed_at: dict, demo: np.ndarray, frame: np.ndarray, chunks: np.ndarray, ok: np.ndarray):
    """IK-check each valid chunk (n,K,10) anchored at the demo's base-frame TCP pose at its frame.
    Returns arrays over chunks: reachable_fraction (n,), margin_deg (n,), collisions (n,)."""
    from franka_sim import frames as fr
    from franka_sim.kinematics import check_traj

    n, K = chunks.shape[:2]
    reach = np.full(n, np.nan)
    marg = np.full(n, np.nan)
    coll = np.zeros(n, int)
    for i in range(n):
        if not ok[i]:
            continue
        A = bTt_at[(int(demo[i]), float(frame[i]))]
        T_seq = np.empty((K, 4, 4))
        for k in range(K):
            rel = fr.make_T(fr.rot6d_to_R(chunks[i, k, 3:9]), chunks[i, k, :3])
            T_seq[k] = A @ rel @ ctx.TCP_T_EE
        r = check_traj(ctx.sm, T_seq, q_seed_at[(int(demo[i]), float(frame[i]))], frame="ee")
        q, rch = r["q"].copy(), r["reachable"].copy()
        m_rad = r["joint_margin_rad"].min(axis=1)
        # check_traj's IK only keeps 0.02 rad from the limits. Stage 1 re-solves a frame below the criterion margin with the
        # IK limit margin raised to it (Ctx.solve): do the same, so a chunk is judged by the solver that judged the demo.
        for j in range(K):
            if rch[j] and m_rad[j] >= ctx.margin_req - 1e-6:
                continue
            q2, ok2, _, _ = ctx.solve(T_seq[j], q[j - 1] if j > 0 else q_seed_at[(int(demo[i]), float(frame[i]))])
            if ok2:
                m2 = float(ctx.kin.joint_margin(q2).min())
                if not rch[j] or m2 > m_rad[j]:
                    q[j], rch[j], m_rad[j] = q2, True, m2
        reach[i] = float(rch.mean())
        mm = float(m_rad[rch].min()) if rch.any() else 0.0
        if mm >= ctx.margin_req - 1e-6:     # Stage 1's tolerance: the solver stops AT the criterion margin
            mm = max(mm, float(ctx.margin_req))
        marg[i] = float(np.rad2deg(mm))
        # Stage 1's collision test (tooling/table/self distance < 0, finger pads excluded: they dip into the table at the
        # placeholder grasp height); kin.n_contacts of check_traj also counts those pad contacts.
        coll[i] = int(sum(min(ctx.distances(qq)) < 0.0 for qq in q[rch]))
    return reach, marg, coll


def demo_tracks(ctx, demos_meta: dict, zarr: str, demo_ids: list, rec: dict):
    """Per demo: base_T_TCP (N,4,4) (stage1 placement) and the tracked joint path at 20 Hz (nearest frame lookup)."""
    from franka_sim import frames as fr
    from franka_sim import stage1_replay as s1

    demos = [s1.prepare_demo(x) for x in s1.load_demos(zarr, demo_ids)]
    out = {}
    for did, dm in zip(demo_ids, demos):
        name = dm["name"]
        if name not in rec["base_T_world"]:
            raise rn.StageIncomplete(f"stage1.json has no placement for demo {name}")
        bTw = np.array(rec["base_T_world"][name])
        bTt = bTw @ dm["T_zup"]
        T_ee = bTt @ ctx.TCP_T_EE
        idx = np.arange(0, len(T_ee), 3)
        r = s1.track_best(ctx, T_ee, idx, 20.0, early_exit=False, i_grasp=dm["i_grasp"], full=True)
        out[did] = {"name": name, "bTt": bTt, "idx": idx, "q": r["_q"], "reach": r["_reach"], "feasible": bool(r["feasible"])}
    return out


def run(R: rn.Run, d: Path, results: dict) -> dict:
    from franka_sim import stage1_replay as s1
    from franka_sim.eval import report as rpt  # noqa: F401

    exec_h = int((R.policy.get("contract") or {}).get("exec_h", 8))
    demos = R.demo_list()
    pred_p = run_predict(R, d, demos, every=exec_h)
    P = dict(np.load(pred_p, allow_pickle=False))
    meta = json.loads(str(P["meta"]))
    demo_ids = [int(x) for x in meta["demos"]]
    # Stage 1 placements
    s1_json = None
    if getattr(R.args, "stage1", None):
        s1_json = s1w.stage1_json_in(Path(R.args.stage1))
        if s1_json is None:
            raise rn.StageIncomplete(f"--stage1 {R.args.stage1}: no stage1.json")
    elif results.get("stage1_json"):
        s1_json = Path(results["stage1_json"])
    if s1_json is not None:
        res1 = json.loads(Path(s1_json).read_text())
        have = set(res1["recommendation"]["base_T_world"])
        if not {meta["demo_names"][str(i)] for i in demo_ids} <= have:       # stage 1 ran on other demos: redo for these
            s1_json = None
    if s1_json is None:
        s1_json = s1w.run_stage1_cli(R, demo_ids, d / "stage1", d / "stage1.log")
        res1 = json.loads(s1_json.read_text())
    rec = res1["recommendation"]
    spec = s1.station_spec(R.ids,
                           {"criteria": R.run_ids.get("criteria", "block_raise"), "sim": "stage1"}, R.ids["robot"],
                           None, None, R.station_sets())
    ctx = s1.Ctx(spec)
    tracks = demo_tracks(ctx, meta, R.zarr, demo_ids, rec)
    # per-chunk anchors and IK seeds
    bTt_at, q_at = {}, {}
    for i in range(len(P["frame"])):
        key = (int(P["demo"][i]), float(P["frame"][i]))
        t = tracks[key[0]]
        bTt_at[key] = t["bTt"][int(round(key[1]))]
        q_at[key] = t["q"][int(np.argmin(np.abs(t["idx"] - key[1])))]
    ok = P["gt_valid"]
    reach, marg, coll = check_chunks(ctx, bTt_at, q_at, P["demo"], P["frame"], P["rel_chunks"], ok)
    if meta["policy_kind"] == "replay":
        reach_gt, marg_gt, coll_gt = reach, marg, coll       # identical by construction
    else:
        reach_gt, marg_gt, coll_gt = check_chunks(ctx, bTt_at, q_at, P["demo"], P["frame"], P["gt_chunks"], ok)
    err = chunk_errors(P["rel_chunks"][ok], P["gt_chunks"][ok])
    m = {
        "n_chunks": int(ok.sum()), "n_demos": len(demo_ids), "demos": demo_ids,
        "chunk_reachable_fraction": float(np.nanmean(reach[ok])), "chunk_joint_margin_deg": float(np.nanmin(marg[ok])),
        "chunk_collisions": int(coll[ok].sum()), "demo_chunk_reachable_fraction": float(np.nanmean(reach_gt[ok])),
        "demo_chunk_joint_margin_deg": float(np.nanmin(marg_gt[ok])), "demo_chunk_collisions": int(coll_gt[ok].sum()),
        "inference_s_p50": pct(P["infer_s"][ok], 50) if meta["policy_kind"] in ("bp_server", "command") else None,
        "inference_s_p95": pct(P["infer_s"][ok], 95) if meta["policy_kind"] in ("bp_server", "command") else None,
        "stage1_json": str(s1_json), "policy_kind": meta["policy_kind"],
        "timing": ["inference_s_p50", "inference_s_p95", "stage1_json"],       # excluded from reproducibility checks
    }
    for k in KS:
        m[f"pred_vs_demo_pos_err_mm_p50_k{k}"] = pct(err[f"pos_{k}"], 50)
        m[f"pred_vs_demo_pos_err_mm_p95_k{k}"] = pct(err[f"pos_{k}"], 95)
        m[f"pred_vs_demo_rot_err_deg_p95_k{k}"] = pct(err[f"rot_{k}"], 95)
        m[f"pred_vs_demo_width_err_mm_p95_k{k}"] = pct(err[f"width_{k}"], 95)
    rows = []
    for did in demo_ids:
        sel = ok & (P["demo"] == did)
        if not sel.any():
            continue
        e = chunk_errors(P["rel_chunks"][sel], P["gt_chunks"][sel], (8, 15))
        rows.append({"demo": tracks[did]["name"][:30], "chunks": int(sel.sum()),
                     "reach": round(float(np.nanmean(reach[sel])), 3), "margin_deg": round(float(np.nanmin(marg[sel])), 1),
                     "pos_mm_p95_k8": round(pct(e["pos_8"], 95), 2), "pos_mm_p95_k15": round(pct(e["pos_15"], 95), 2)})
    notes = [f"predictions: {pred_p}", f"placement from {s1_json}",
             f"{m['n_chunks']} chunks on demos {demo_ids} every {exec_h} agent steps ({meta['policy_kind']}: {meta['policy_id']})"]
    return {"metrics": m, "rows": rows, "notes": notes}
