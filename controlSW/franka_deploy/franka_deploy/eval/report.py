"""Evaluation report over rollout sessions: `python -m franka_deploy eval <sessions...> --out DIR`.
Run it with the INFERENCE env's python (env_bp): reading the demo zarr needs its codecs and the OOD score unpickles
the training normalizer, which needs behavior_prompting.

Outputs DIR/report.json and DIR/report.html:
- outcomes: annotation.json if present, else the live label (y = success, n = failure, s = unlabeled);
  aborted/exception episodes are system faults (excluded from policy success, reported). Success rate with a Wilson
  95 % interval per condition: prompt index, start-pose bucket (FK distance from home), v_max, exec_h, anchor.
- similarity to the demos: TCP trajectory relative to the episode-start TCP frame (10 Hz) vs every demo of the prompt
  zarr: DTW to the nearest demo / leave-one-out demo-to-demo baseline (median nearest-neighbour DTW, cached).
- prompt fidelity: DTW to the prompted demo and its rank among all demos (1 = closest).
- phases from the measured width + TCP height (robot up = +z): approach / grasp / lift / hold durations, grasp width,
  lift height; speed profile (measured and corrected for the chunk time-scale).
- policy health: overlapping-chunk consistency (previous chunk's steps 8..15 vs next chunk's 0..7, absolute), OOD
  score = share of proprio/action values outside the training normalizer's [min, max].
- execution: tracking error |safe target (or commanded) - FK|, clamps by cause, chunk scale, settle waits, inference
  and capture->reply latency p50/p95, holds and faults per hour.
- sim ground truth (A6, additive): episodes with bridge.jsonl sim lines (MuJoCo) get ``sim_gt`` from
  franka_sim.ground_truth.evaluate_episode and the report a separate ``sim_success_rate``; human labels still decide
  ``outcome`` and the label-based rates.
"""

from __future__ import annotations

import argparse
import glob
import html
import json
import os
import pickle
import sys
from collections import defaultdict

import numpy as np
from scipy.spatial.transform import Rotation

from franka_deploy.eval import metrics as M


def rot6d_to_R(d6):
    a1, a2 = d6[..., :3], d6[..., 3:]
    b1 = a1 / np.linalg.norm(a1, axis=-1, keepdims=True)
    b2 = a2 - np.sum(b1 * a2, axis=-1, keepdims=True) * b1
    b2 /= np.linalg.norm(b2, axis=-1, keepdims=True)
    return np.stack([b1, b2, np.cross(b1, b2)], axis=-2)


def T7(p):
    T = np.eye(4)
    T[:3, :3] = Rotation.from_quat([p[4], p[5], p[6], p[3]]).as_matrix()
    T[:3, 3] = p[:3]
    return T


def find_episodes(sessions):
    out = []
    for s in sessions:
        out += sorted(glob.glob(os.path.join(s, "task", "episodes", "*", "*", "*", "episode_*")))
    return out


def outcome(ep):
    lab = str(ep["label"])
    ann = ep.get("annotation")
    if lab.startswith("aborted:") or lab.startswith("exception:") or (ann and ann.get("failure_category") == "system_fault"):
        return "system_fault"
    if ann and ann.get("success") is not None and ann.get("source") == "manual":
        return "success" if ann["success"] else "failure"
    return {"y": "success", "n": "failure"}.get(lab, "unlabeled")


def sim_gt(d):
    """franka_sim.ground_truth.evaluate_episode(d), or None (no sim lines, franka_sim not importable, or it failed)."""
    try:
        from franka_sim.ground_truth import evaluate_episode
    except ImportError:
        root = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..", "franka_sim")
        if not os.path.isfile(os.path.join(root, "franka_sim", "ground_truth.py")):
            return None
        sys.path.append(os.path.normpath(root))
        try:
            from franka_sim.ground_truth import evaluate_episode
        except Exception:
            return None
    try:
        return evaluate_episode(d)
    except Exception as e:
        print(f"[eval] sim_gt of {d} failed: {e!r}", file=sys.stderr)
        return None


def load_normalizer_ranges(path):
    try:
        norm = pickle.load(open(path, "rb"))
    except ModuleNotFoundError as e:
        raise SystemExit(f"eval must run in the inference env (env_bp): unpickling {path} needs behavior_prompting ({e})")
    sd = norm.state_dict()
    rng = {}
    for k in ("action", "gripper_right_eef_pos", "gripper_right_gripper_width"):
        lo, hi = sd.get(f"params_dict.{k}.input_stats.min"), sd.get(f"params_dict.{k}.input_stats.max")
        if lo is not None:
            rng[k] = (lo.numpy(), hi.numpy())
    return rng


def episode_metrics(d, demos10, demo_names, baseline_m, train_names, norm_rng, home_xyz):
    ep = {"dir": d}
    ep["summary"] = json.load(open(os.path.join(d, "summary.json")))
    ep["label"] = json.load(open(os.path.join(d, "label.json")))["label"]
    ap = os.path.join(d, "annotation.json")
    ep["annotation"] = json.load(open(ap)) if os.path.exists(ap) else None
    cfgp = os.path.join(d, "config", "deploy_resolved.yaml")
    import yaml

    cfg = yaml.safe_load(open(cfgp)) if os.path.exists(cfgp) else {}
    t = cfg.get("task", {})
    ep["conditions"] = {"prompt_index": t.get("prompt_index"), "v_max": t.get("v_max"), "exec_h": t.get("exec_h"),
                        "anchor": t.get("anchor")}
    evs = [json.loads(ln) for ln in open(os.path.join(d, "events.jsonl"))] if os.path.exists(os.path.join(d, "events.jsonl")) else []
    dur = max(float(ep["summary"].get("duration_s", 0.0)), 1e-6)
    ep["events"] = {"holds_per_h": 3600 * sum(e["kind"] == "hold" for e in evs) / dur,
                    "faults_per_h": 3600 * sum(e["kind"] in ("policy_exception",) for e in evs) / dur,
                    "settle_waits": int(ep["summary"].get("task_counters", {}).get("settle_waits", 0)),
                    "workspace_contacts": int(ep["summary"].get("workspace_contacts", 0))}
    cc = ep["summary"].get("controller_counters", {})
    ep["clamps"] = {k: int(cc.get(k, 0)) for k in ("clamp_pos", "clamp_rot", "clamp_joint", "clamp_within", "clamp_micro",
                                                    "clamp_box", "clamp_cone")}
    ep["chunk_scale"] = ep["summary"].get("chunk_scale")
    # taps
    taps = [dict(np.load(f)) for f in sorted(glob.glob(os.path.join(d, "tap", "call_*.npz")))]
    if taps:
        inf = np.array([float(c["info_inference_s"]) for c in taps])
        c2r = np.array([float(c["info_t_reply"] - c["info_main_ts"][-1]) for c in taps])
        ep["latency"] = {"inference_p50": float(np.median(inf)), "inference_p95": float(np.percentile(inf, 95)),
                         "capture_to_reply_p50": float(np.median(c2r)), "capture_to_reply_p95": float(np.percentile(c2r, 95))}
        cons = []
        prev = None
        for c in taps:
            if "info_anchor_pose" not in c:
                continue
            A = T7(c["info_anchor_pose"])
            R = rot6d_to_R(c["reply_action0_eef_rot_axis_angle"])
            P = c["reply_action0_eef_pos"]
            absP = np.array([(A @ np.block([[R[k], P[k][:, None]], [np.zeros((1, 3)), np.ones((1, 1))]]))[:3, 3] for k in range(len(P))])
            if prev is not None:
                cons.append(float(np.linalg.norm(prev[8:16] - absP[:8], axis=1).mean()))
            prev = absP
        ep["chunk_consistency_m"] = float(np.median(cons)) if cons else None
        if norm_rng:
            vals = []
            if "action" in norm_rng:
                a = np.concatenate([np.concatenate([c["reply_action0_eef_pos"], c["reply_action0_eef_rot_axis_angle"],
                                                    c["reply_action0_gripper_width"]], 1) for c in taps])
                vals.append(M.ood_fraction(a, *norm_rng["action"]))
            if "gripper_right_eef_pos" in norm_rng:
                vals.append(M.ood_fraction(np.concatenate([c["obs_robot0_eef_pos"] for c in taps]), *norm_rng["gripper_right_eef_pos"]))
            if "gripper_right_gripper_width" in norm_rng:
                vals.append(M.ood_fraction(np.concatenate([c["obs_robot0_gripper_width"] for c in taps]), *norm_rng["gripper_right_gripper_width"]))
            ep["ood"] = {"frac_outside": float(np.mean([v[0] for v in vals])), "max_excess": float(max(v[1] for v in vals))}
    # trace
    trp = os.path.join(d, "controller_trace.npz")
    if os.path.exists(trp):
        tr = np.load(trp)
        tt, fk = tr["t"], tr["fk"]
        ref = tr["safe_target"] if "safe_target" in tr.files else tr["commanded"]
        te = np.linalg.norm(ref[:, :3] - fk[:, :3], axis=1)
        ep["tracking_error_m"] = {"p50": float(np.median(te)), "p95": float(np.percentile(te, 95))}
        start = fk[0, :3]
        ep["start_dist_from_home_m"] = float(np.linalg.norm(start - np.asarray(home_xyz))) if home_xyz is not None else None
        rel = M.relative_positions(fk[:, :3], Rotation.from_quat(fk[0, [4, 5, 6, 3]]))
        tr10 = M.resample(tt, rel, 10.0)
        if demos10 is not None and len(tr10) > 3:
            dd = np.array([M.dtw(tr10, x) for x in demos10])
            ep["similarity"] = {"nearest_demo": demo_names[int(dd.argmin())], "nearest_dtw_m": float(dd.min()),
                                "ratio_to_demo_baseline": float(dd.min() / baseline_m)}
            pi = ep["conditions"]["prompt_index"]
            if pi is not None and train_names and pi < len(train_names) and train_names[pi] in demo_names:
                j = demo_names.index(train_names[pi])
                ep["prompt_fidelity"] = {"prompted_demo": train_names[pi], "dtw_m": float(dd[j]),
                                         "rank": int(1 + np.sum(dd < dd[j])), "n_demos": len(dd)}
        h = (fk[:, :3] - fk[0, :3]) @ np.array([0.0, 0.0, 1.0])
        ph = M.phases(tr["width_measured"], h, tt)
        ep["phases"] = ph
        sc = tr["ann_chunk_scale"] if "ann_chunk_scale" in tr.files else None
        ep["speed"] = M.speed_profile(tt, fk[:, :3], sc)
    gt = sim_gt(d)
    if gt is not None:
        ep["sim_gt"] = gt
    ep["outcome"] = outcome(ep)
    return ep


def summarize(eps):
    rep = {"n_episodes": len(eps), "outcomes": {}, "by_condition": {}}
    for o in ("success", "failure", "unlabeled", "system_fault"):
        rep["outcomes"][o] = sum(e["outcome"] == o for e in eps)
    scored = [e for e in eps if e["outcome"] in ("success", "failure")]
    k = sum(e["outcome"] == "success" for e in scored)
    p, lo, hi = M.wilson(k, len(scored))
    rep["success_rate"] = {"k": k, "n": len(scored), "p": p, "ci95": [lo, hi]}
    sim = [e for e in eps if e.get("sim_gt") and e["sim_gt"].get("success") is not None]
    if sim:                                     # physics truth, separate from the labels (A6)
        ks = sum(bool(e["sim_gt"]["success"]) for e in sim)
        p, lo, hi = M.wilson(ks, len(sim))
        rep["sim_success_rate"] = {"k": ks, "n": len(sim), "p": p, "ci95": [lo, hi], "source": "bridge.jsonl sim lines"}
    conds = defaultdict(lambda: defaultdict(list))
    for e in scored:
        c = dict(e["conditions"])
        sd = e.get("start_dist_from_home_m")
        c["start_bucket"] = None if sd is None else ("<=2cm" if sd <= 0.02 else ("<=5cm" if sd <= 0.05 else ">5cm"))
        for key, val in c.items():
            conds[key][str(val)].append(e["outcome"] == "success")
    for key, groups in conds.items():
        rep["by_condition"][key] = {}
        for val, xs in groups.items():
            p, lo, hi = M.wilson(sum(xs), len(xs))
            rep["by_condition"][key][val] = {"k": int(sum(xs)), "n": len(xs), "p": p, "ci95": [lo, hi]}
    agg = lambda key, sub: [e[key][sub] for e in eps if e.get(key) and e[key].get(sub) is not None]  # noqa: E731
    rep["execution"] = {
        "latency_inference_p50": float(np.median(agg("latency", "inference_p50"))) if agg("latency", "inference_p50") else None,
        "latency_c2r_p95": float(np.median(agg("latency", "capture_to_reply_p95"))) if agg("latency", "capture_to_reply_p95") else None,
        "tracking_p95_m": float(np.median(agg("tracking_error_m", "p95"))) if agg("tracking_error_m", "p95") else None,
        "clamps_total": {k: int(sum(e["clamps"][k] for e in eps)) for k in eps[0]["clamps"]} if eps else {},
        "holds_per_h": float(np.mean([e["events"]["holds_per_h"] for e in eps])) if eps else None,
        "faults_per_h": float(np.mean([e["events"]["faults_per_h"] for e in eps])) if eps else None}
    sims = [e["similarity"]["ratio_to_demo_baseline"] for e in eps if e.get("similarity")]
    rep["similarity_ratio_median"] = float(np.median(sims)) if sims else None
    return rep


def write_html(path, rep, eps, baseline):
    def row(cells, tag="td"):
        return "<tr>" + "".join(f"<{tag}>{html.escape(str(c))}</{tag}>" for c in cells) + "</tr>"

    sr = rep["success_rate"]
    h = ["<!doctype html><meta charset=utf-8><title>franka_deploy eval</title>",
         "<style>body{font-family:sans-serif;margin:16px}table{border-collapse:collapse;margin:8px 0}"
         "td,th{border:1px solid #999;padding:3px 6px;font-size:13px}</style>",
         "<h1>Rollout evaluation</h1>",
         f"<p>{rep['n_episodes']} episodes; outcomes {html.escape(json.dumps(rep['outcomes']))}. "
         f"Policy success {sr['k']}/{sr['n']} = {sr['p']:.2f} (Wilson 95 % [{sr['ci95'][0]:.2f}, {sr['ci95'][1]:.2f}]); "
         f"system faults excluded.</p>",
         (f"<p>Sim ground truth (bridge.jsonl, not the labels): {rep['sim_success_rate']['k']}/{rep['sim_success_rate']['n']} "
          f"= {rep['sim_success_rate']['p']:.2f}.</p>" if rep.get("sim_success_rate") else ""),
         f"<p>Demo baseline (leave-one-out nearest DTW, median): {baseline * 1000:.1f} mm. "
         f"Rollout nearest-demo DTW / baseline (median): {rep['similarity_ratio_median']}.</p>",
         "<h2>Success by condition</h2><table>" + row(["condition", "value", "k/n", "p", "95 % CI"], "th")]
    for key, groups in rep["by_condition"].items():
        for val, g in groups.items():
            h.append(row([key, val, f"{g['k']}/{g['n']}", f"{g['p']:.2f}", f"[{g['ci95'][0]:.2f}, {g['ci95'][1]:.2f}]"]))
    h.append("</table><h2>Episodes</h2><table>" + row(["episode", "label", "outcome", "dur s", "calls", "nearest DTW mm",
                                                       "ratio", "prompt rank", "lift cm", "grasp w mm", "track p95 mm",
                                                       "clamps (joint/within/micro)", "scale max", "c2r p95 s", "OOD frac",
                                                       "consistency mm", "sim_gt"], "th"))
    for e in eps:
        s, sim, pf, ph = e["summary"], e.get("similarity") or {}, e.get("prompt_fidelity") or {}, e.get("phases") or {}
        cl = e["clamps"]
        h.append(row([os.path.relpath(e["dir"], os.path.dirname(os.path.dirname(e["dir"]))), e["label"], e["outcome"],
                      f"{s.get('duration_s', 0):.1f}", s.get("policy_calls"),
                      f"{sim.get('nearest_dtw_m', float('nan')) * 1000:.1f}", f"{sim.get('ratio_to_demo_baseline', float('nan')):.2f}",
                      f"{pf.get('rank', '')}/{pf.get('n_demos', '')}", f"{ph.get('lift_height_m', float('nan')) * 100:.1f}",
                      f"{ph.get('grasp_width_m', float('nan')) * 1000:.1f}",
                      f"{(e.get('tracking_error_m') or {}).get('p95', float('nan')) * 1000:.2f}",
                      f"{cl['clamp_joint']}/{cl['clamp_within']}/{cl['clamp_micro']}",
                      f"{(e.get('chunk_scale') or {}).get('max', float('nan')):.2f}",
                      f"{(e.get('latency') or {}).get('capture_to_reply_p95', float('nan')):.3f}",
                      f"{(e.get('ood') or {}).get('frac_outside', float('nan')):.3f}",
                      f"{(e.get('chunk_consistency_m') or float('nan')) * 1000:.1f}",
                      {True: "y", False: "n"}.get((e.get("sim_gt") or {}).get("success"), "-")]))
    h.append("</table><h2>Execution</h2><pre>" + html.escape(json.dumps(rep["execution"], indent=1)) + "</pre>")
    open(path, "w").write("\n".join(h))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("sessions", nargs="+")
    ap.add_argument("--out", required=True)
    ap.add_argument("--zarr", default=None, help="demo/prompt zarr (default: from the session's checksums.json)")
    ap.add_argument("--normalizer", default=None, help="training normalizer.pkl (default: <ckpt dir>/../run_config/)")
    ap.add_argument("--no-dtw", action="store_true")
    a = ap.parse_args(argv)
    a.sessions = [os.path.abspath(x) for x in a.sessions]
    from franka_deploy.paths import check_writable_dir

    out = check_writable_dir(a.out, "eval output dir")
    out.mkdir(parents=True, exist_ok=True)
    eps_dirs = find_episodes(a.sessions)
    if not eps_dirs:
        print("no episodes found")
        return 1
    ck = json.load(open(os.path.join(eps_dirs[0], "config", "checksums.json")))
    zarr_path = a.zarr or ck["prompt_zarr"]
    norm_path = a.normalizer or os.path.join(os.path.dirname(os.path.dirname(ck["checkpoint"])), "run_config", "normalizer.pkl")
    norm_rng = load_normalizer_ranges(norm_path) if os.path.exists(norm_path) else None
    demos10 = names = None
    baseline = float("nan")
    train_names = None
    if not a.no_dtw:
        from franka_deploy.eval.demos import load_demos, loo_baseline, rel_traj_10hz

        demos = load_demos(zarr_path)
        names = [d["name"] for d in demos]
        demos10 = [rel_traj_10hz(d["t"], d["rel"]) for d in demos]
        cache = out / f"demo_baseline_{os.path.basename(zarr_path)}.json"
        if cache.exists():
            baseline = json.load(open(cache))["median_m"]
        else:
            b = loo_baseline(demos10)
            baseline = b["median_m"]
            json.dump({"median_m": baseline, "nearest_dtw_m": b["nearest_dtw_m"], "names": names}, open(cache, "w"))
        split_p = os.path.join(a.sessions[0], "server", "split_info.json")
        if os.path.exists(split_p):
            split = json.load(open(split_p))
            train_eps = {k.split(":")[0] for k, v in split.items() if v}
            train_names = [n for n in names if n in train_eps]
    import yaml

    home = None
    rp = os.path.join(eps_dirs[0], "config", "deploy_resolved.yaml")
    if os.path.exists(rp):
        home = yaml.safe_load(open(rp)).get("home", {}).get("xyz")
    eps = [episode_metrics(d, demos10, names, baseline, train_names, norm_rng, home) for d in eps_dirs]
    rep = summarize(eps)
    rep["demo_baseline_dtw_m"] = baseline
    rep["sessions"] = a.sessions
    rep["episodes"] = [{k: v for k, v in e.items() if k != "summary"} | {"summary": e["summary"]} for e in eps]
    json.dump(rep, open(out / "report.json", "w"), indent=1, default=str)
    write_html(out / "report.html", rep, eps, baseline)
    print(json.dumps({k: rep[k] for k in ("n_episodes", "outcomes", "success_rate", "similarity_ratio_median")}, default=str))
    return 0


if __name__ == "__main__":
    sys.exit(main())
