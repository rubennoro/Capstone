"""W1 step 3: per-checkpoint open-loop error on real training episodes, deploy path vs training's inputs vs the
recorded actions, for all 14 runnable checkpoints. Torch env (``~/envs/sim_policy_deploy_torch/bin/python``).

    cd ~/Capstone-main-sim && unset PYTHONPATH
    PY=~/envs/sim_policy_deploy_torch/bin/python
    $PY -m sim_deploy.offline.openloop plan                 # CPU: splits from the checkpoints -> openloop/plan.json
    $PY -m sim_deploy.offline.openloop obs                  # CPU: build_obs for every planned sample (both packages)
    GPU_LOCK_HELD=1 flock ~/Capstone-main-sim/sim_evals/.gpu.lock \
        $PY -m sim_deploy.offline.openloop run --id PICK_TEST1/run_006_2026-11-18      # GPU, one checkpoint/process
    $PY -m sim_deploy.offline.openloop summary              # CPU: summary.json + summary.md

Tolerance (fixed before running):
- T1 "model reproduces training": the direct-path (training-exact inputs) action MSE (torch mse over the full (16, 10)
  unnormalised action, plus pos / rot / width components, as train_policy_workspace.compute_action_mse) on the
  checkpoint's val episodes and on its train episodes, divided by the training log's value at the checkpoint's epoch
  (run folder epochs.csv, val/action_mse_error/all, train/action_mse_error/all and components; if the ckpt epoch row is
  empty the nearest filled row, recorded), lies in [1/3, 3].
- T2 "deploy = training": deploy-path MSE / direct-path MSE <= 1.5 per split, AND median per-sample |deploy - direct|
  k=15 position difference <= 1.5 x the median seed-to-seed spread of the direct path (seed 0 vs seed 1).
Pass = T1 and T2.

Samples. Obs cache = the superset packages (pick1-v1 for PICK_TEST1 runs, pick1-double-v2 for PICK_DOUBLE), episodes
mapped by Pi episode name (checked: every subset package's low-dim arrays and sampled main / ultrawide frames of every
episode equal the superset's). Per checkpoint: all its val episodes (checkpoint ``_extra_training_split_info``) + 3
fixed train episodes per package (train in every run on that package, evenly spaced over the smallest subset), rows
6, 21, ... up to the last row with a full action horizon (T + 45 < n_steps), stride 15.

Runs per checkpoint (deploy = convention B, the f632576 gripper_map path the copy runs): fp32 at the checkpoint's own
diffusion steps: deploy seed 0, direct seed 0, direct seed 1; deploy bf16 / 8 steps seed 0. Model-space outputs
(ModelRunner.run on the exact obs policy.infer builds; checked once per episode against policy.infer). Prompting runs:
reset + prompt(task_name, prompt_index) per episode as episode_loop does (the deploy default prompt; training's metric
used each sample's pair prompt instead).

Output: sim_evals/_pipeline/W1/openloop/{plan.json, <TASK>_<run>.json/.npz, summary.json, summary.md}.
"""
from __future__ import annotations

import argparse
import csv
import sys
import time
from pathlib import Path

import numpy as np

from sim_deploy.offline.common import OUT, deploy_version, read_json, set_cpu_threads, write_json

OL = OUT / "openloop"
SUPERSET = {"PICK_TEST1": "pick1-v1", "PICK_DOUBLE": "pick1-double-v2"}
SMALLEST = {"PICK_TEST1": "pick1-ep10-v1", "PICK_DOUBLE": "pick1-double-ep15-v1"}
RUNS = ["PICK_TEST1/run_006_2026-11-18", "PICK_TEST1/run_012_2026-11-18", "PICK_TEST1/run_014_2026-11-18",
        "PICK_TEST1/run_016_2026-11-18", "PICK_TEST1/run_020_2026-11-19", "PICK_TEST1/run_022_2026-11-19",
        "PICK_TEST1/run_024_2026-11-19", "PICK_TEST1/run_028_2026-11-19", "PICK_TEST1/run_030_2026-11-19",
        "PICK_TEST1/run_031_2026-11-19", "PICK_DOUBLE/run_002_2026-11-19", "PICK_DOUBLE/run_003_2026-11-19",
        "PICK_DOUBLE/run_004_2026-11-19", "PICK_DOUBLE/run_005_2026-11-19"]
STRIDE, ROW0, HORIZON_ROWS = 15, 6, 45
COMPONENTS = (("pos", 0, 3), ("rot", 3, 9), ("width", 9, 10))


def run_folder(rid: str) -> Path:
    task, run = rid.split("/")
    return Path.home() / "Capstone" / "data" / task / "policy" / run


def stem(rid: str) -> str:
    return rid.replace("/", "_")


# ------------------------------------------------------------------------------------------------------ plan

def split_info_cpu(rf: Path) -> dict:
    import dill
    import torch
    ck = sorted((rf / "checkpoints").glob("*.ckpt"))[0]
    with open(ck, "rb") as f:
        pl = torch.load(f, map_location="cpu", pickle_module=dill, weights_only=False)
    return {str(k): bool(v) for k, v in pl["state_dicts"]["model"]["_extra_training_split_info"].items()}


def cmd_plan() -> int:
    from sim_deploy.offline.episodes import find_index, package_episodes, split_of
    plan = {"runs": {}, "train_fixed": {}, "stride": STRIDE, "row0": ROW0}
    splits = {}
    for rid in RUNS:
        t0 = time.monotonic()
        splits[rid] = split_info_cpu(run_folder(rid))
        print(rid, f"{len(splits[rid])} eps, {sum(not v for v in splits[rid].values())} val",
              f"{time.monotonic() - t0:.1f} s", flush=True)
    for task, sup in SUPERSET.items():
        rids = [r for r in RUNS if r.startswith(task + "/")]
        small = [e["pi_name"] for e in package_episodes(SMALLEST[task])]
        ok = [n for n in small if all(split_of(splits[r], n) == "train" for r in rids)]
        pick = [ok[int(round(x))] for x in np.linspace(0, len(ok) - 1, 3)]
        plan["train_fixed"][task] = {"candidates": len(ok), "pi_names": pick,
                                     "superset_index": [find_index(sup, n) for n in pick]}
        for rid in rids:
            si = splits[rid]
            val = sorted(n.split(":", 1)[0] for n, v in si.items() if not v)
            eps = [(n, "val") for n in val] + [(n, "train") for n in pick]
            items = []
            for n, sp in eps:
                assert split_of(si, n) == sp, (rid, n, sp)
                idx = find_index(sup, n)
                nst = int(package_episodes(sup)[idx]["n_steps"])
                rows = list(range(ROW0, nst - HORIZON_ROWS, STRIDE))
                items.append({"pi_name": n, "split": sp, "index": idx, "n_steps": nst, "rows": rows})
            plan["runs"][rid] = {"dataset": sup, "n_split_entries": len(si), "episodes": items}
    plan["deploy_version"] = deploy_version()
    write_json(OL / "plan.json", plan)
    for rid, p in plan["runs"].items():
        print(rid, [(e["index"], e["split"], len(e["rows"])) for e in p["episodes"]])
    return 0


def cmd_obs() -> int:
    from sim_deploy.offline.build_obs import build_obs
    plan = read_json(OL / "plan.json")
    for sup in SUPERSET.values():
        eps = {}
        for p in plan["runs"].values():
            if p["dataset"] == sup:
                for e in p["episodes"]:
                    eps[e["index"]] = e["rows"]
        for idx, rows in sorted(eps.items()):           # one episode per call (each decodes its videos once)
            t0 = time.monotonic()
            build_obs(sup, [idx], rows)
            print(f"[openloop obs] {sup} ep{idx}: {len(rows)} rows in {time.monotonic() - t0:.1f} s", flush=True)
    return 0


# ------------------------------------------------------------------------------------------------------- run

def mse(pred: np.ndarray, gt: np.ndarray) -> dict:
    d = (np.asarray(pred, np.float64) - np.asarray(gt, np.float64)) ** 2      # (n, 16, 10)
    out = {"all": float(d.mean())}
    for name, a, b in COMPONENTS:
        out[name] = float(d[..., a:b].mean())
    return out


def cmd_run(rid: str, limit: int | None = None) -> int:
    import dataclasses
    from sim_deploy.offline.approach import approach_metrics
    from sim_deploy.offline.build_obs import load_sample
    from sim_deploy.offline.compare import chunk_errors
    from sim_deploy.offline.policy_runner import PolicyRunner, _obs
    set_cpu_threads()
    plan = read_json(OL / "plan.json")["runs"][rid]
    sup = plan["dataset"]
    t_start = time.monotonic()
    recs, arrays, desc = [], {}, {}
    samples = [(e, r) for e in plan["episodes"] for r in e["rows"]]
    if limit:
        samples = samples[:limit]
    cache = {}
    for e, r in samples:
        z, meta = load_sample(sup, e["index"], r)
        cache[(e["index"], r)] = (z, meta)
    import yaml
    cfg = yaml.safe_load((run_folder(rid) / "run_config" / "config.yaml").read_text())
    ckpt_steps = int((cfg.get("model") or cfg.get("policy"))["num_inference_steps"])
    # explicit: diffusion_steps=None would keep the station template's 8 (policy_sim.yaml), not the checkpoint's own
    for setting, (prec, steps, seeds_direct) in {"fp32_ckpt": ("fp32", ckpt_steps, (0, 1)),
                                                 "bf16_8": ("bf16", 8, ())}.items():
        runner = PolicyRunner(run_folder(rid), prec, steps)
        desc[setting] = runner.describe()
        assert desc[setting]["diffusion_steps_in_use"] == steps and desc[setting]["precision"] == prec, desc[setting]
        t0 = time.monotonic()
        last = None
        for e, r in samples:
            z, meta = cache[(e["index"], r)]
            key = f"ep{e['index']:02d}_r{r:03d}"
            if meta["deploy"]["step"]["type"] != "StepDone":
                if setting == "fp32_ckpt":
                    recs.append({"key": key, "split": e["split"], "skipped": meta["deploy"]["step"]})
                continue
            first = e["index"] != last
            if first:
                runner.new_episode()
                last = e["index"]
            obs_b = _obs(z, "deploy", "gripper_width_m_B")
            model_obs = dataclasses.replace(obs_b, gripper_width_m=runner.gripper_map.to_model(obs_b.gripper_width_m))
            arrays[f"{key}_{setting}_deploy"] = runner.infer_model(model_obs, 0)
            if first:                                   # policy.infer (map on) == the model-space call, once/episode
                chunk, raw, _ = runner.infer_deploy(obs_b, 0, "B")
                chk = float(np.max(np.abs(raw - arrays[f"{key}_{setting}_deploy"])))
                pr = float(np.max(np.abs(np.concatenate([chunk.eef_pos, chunk.eef_rot6d], -1) - raw[:, :9])))
                desc[setting].setdefault("infer_check", []).append({"key": key, "raw_vs_model_call": chk,
                                                                    "robot_pos_rot_vs_raw": pr})
            if seeds_direct:
                obs_d = _obs(z, "direct")
                for s in seeds_direct:
                    arrays[f"{key}_{setting}_direct_s{s}"] = runner.infer_model(obs_d, s)
            if setting == "fp32_ckpt":
                arrays[f"{key}_gt"] = z["action"].astype(np.float32)
                recs.append({"key": key, "split": e["split"], "index": e["index"], "row": r,
                             "pi_name": e["pi_name"], "meta": meta, "z_row_T": z["row_T"]})
        desc[setting]["eval_s"] = round(time.monotonic() - t0, 1)
        runner.close()
        del runner
    # metrics
    out = {"id": rid, "dataset": plan["dataset"], "policy": desc, "deploy_version": deploy_version(),
           "tolerance": __doc__.split("Tolerance (fixed before running):\n")[1].split("\n\nSamples.")[0]}
    per_sample, by_split = [], {"val": [], "train": []}
    for rec in recs:
        if "skipped" in rec:
            per_sample.append({"key": rec["key"], "split": rec["split"], "skipped": rec["skipped"]})
            continue
        k = rec["key"]
        gt = arrays[f"{k}_gt"]
        dep, d0, d1 = (arrays[f"{k}_fp32_ckpt_deploy"], arrays[f"{k}_fp32_ckpt_direct_s0"],
                       arrays[f"{k}_fp32_ckpt_direct_s1"])
        b8 = arrays.get(f"{k}_bf16_8_deploy")
        s = {"key": k, "split": rec["split"], "row": rec["row"], "index": rec["index"],
             "k15_deploy_vs_direct_mm": float(np.linalg.norm(dep[15, :3] - d0[15, :3]) * 1000),
             "k15_direct_seed_spread_mm": float(np.linalg.norm(d1[15, :3] - d0[15, :3]) * 1000),
             "deploy_vs_gt": chunk_errors(dep, gt)["at_k"], "direct_vs_gt": chunk_errors(d0, gt)["at_k"],
             "bf16_8_vs_gt": chunk_errors(b8, gt)["at_k"] if b8 is not None else None}
        if rec["row"] == ROW0:
            meta, zz = rec["meta"], load_sample(plan["dataset"], rec["index"], rec["row"])[0]
            anchor_T = np.asarray(meta["deploy"]["anchor"]["T"], np.float64)
            s["approach_row6"] = {"deploy": approach_metrics(dep[:, :3], anchor_T, meta, zz),
                                  "bf16_8": approach_metrics(b8[:, :3], anchor_T, meta, zz) if b8 is not None else None,
                                  "direct": approach_metrics(d0[:, :3], zz["row_T"], meta, zz),
                                  "gt": approach_metrics(gt[:, :3], zz["row_T"], meta, zz)}
        per_sample.append(s)
        by_split[rec["split"]].append(k)
    res = {}
    for sp, keys in by_split.items():
        if not keys:
            continue
        g = np.stack([arrays[f"{k}_gt"] for k in keys])
        r = {"n": len(keys), "episodes": sorted({k.split("_")[0] for k in keys})}
        for name, suf in (("deploy", "fp32_ckpt_deploy"), ("direct", "fp32_ckpt_direct_s0"),
                          ("direct_s1", "fp32_ckpt_direct_s1"), ("bf16_8", "bf16_8_deploy")):
            if all(f"{k}_{suf}" in arrays for k in keys):
                p = np.stack([arrays[f"{k}_{suf}"] for k in keys])
                r[f"mse_{name}"] = mse(p, g)
                errs = [chunk_errors(p[i], g[i]) for i in range(len(keys))]
                r[f"at_k_{name}"] = {f"k{kk}": {m: float(np.mean([e["at_k"][f"k{kk}"][m] for e in errs]))
                                                for m in ("pos_mm", "rot_deg", "width_mm")} for kk in (0, 4, 8, 15)}
        ps = [s for s in per_sample if s.get("split") == sp and "skipped" not in s]
        r["k15_deploy_vs_direct_mm_median"] = float(np.median([s["k15_deploy_vs_direct_mm"] for s in ps]))
        r["k15_direct_seed_spread_mm_median"] = float(np.median([s["k15_direct_seed_spread_mm"] for s in ps]))
        res[sp] = r
    out["splits"] = res
    out["training_log"] = training_log(rid, desc["fp32_ckpt"].get("epoch"))
    out["verdict"] = verdict(res, out["training_log"])
    out["samples"] = per_sample
    out["n_skipped"] = sum(1 for s in per_sample if "skipped" in s)
    out["wall_s"] = round(time.monotonic() - t_start, 1)
    OL.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(OL / f"{stem(rid)}.npz", **arrays)
    write_json(OL / f"{stem(rid)}.json", out)
    print(f"[openloop] {rid}: {out['verdict']} in {out['wall_s']} s", flush=True)
    return 0


def training_log(rid: str, epoch) -> dict:
    rf = run_folder(rid)
    if epoch is None:
        ck = sorted((rf / "checkpoints").glob("*.ckpt"))[0].name
        epoch = int(ck.split("epoch=")[1].split("-")[0])
    rows = list(csv.DictReader(open(rf / "epochs.csv")))
    key = "val/action_mse_error/all"
    filled = [r for r in rows if r.get(key)]
    best = min(filled, key=lambda r: abs(int(r["epoch"]) - int(epoch)))
    out = {"ckpt_epoch": int(epoch), "row_epoch": int(best["epoch"]), "exact": int(best["epoch"]) == int(epoch)}
    for sp in ("val", "train"):
        out[sp] = {"all": float(best[f"{sp}/action_mse_error/all"])}
        for name, comp in (("pos", "right_pos"), ("rot", "right_rot"), ("width", "right_gripper_width")):
            v = best.get(f"{sp}/action_mse_error_{comp}/all")
            out[sp][name] = float(v) if v else None
    return out


def verdict(res: dict, tl: dict) -> dict:
    v = {}
    for sp, r in res.items():
        ratio_t1 = r["mse_direct"]["all"] / tl[sp]["all"]
        ratio_t2 = r["mse_deploy"]["all"] / r["mse_direct"]["all"]
        k15 = r["k15_deploy_vs_direct_mm_median"] <= 1.5 * r["k15_direct_seed_spread_mm_median"]
        v[sp] = {"t1_ratio": round(ratio_t1, 3), "T1": 1 / 3 <= ratio_t1 <= 3, "t2_ratio": round(ratio_t2, 3),
                 "t2_k15": k15, "T2": ratio_t2 <= 1.5 and k15,
                 "bf16_8_over_fp32": round(r["mse_bf16_8"]["all"] / r["mse_deploy"]["all"], 3)
                 if "mse_bf16_8" in r else None}
    v["pass"] = all(x["T1"] and x["T2"] for x in v.values() if isinstance(x, dict))
    return v


# --------------------------------------------------------------------------------------------------- summary

def cmd_summary() -> int:
    rows, full = [], {}
    for rid in RUNS:
        p = OL / f"{stem(rid)}.json"
        if not p.is_file():
            rows.append(f"| {rid} | not run | | | | | | | | | |")
            continue
        d = read_json(p)
        full[rid] = {k: d[k] for k in ("dataset", "splits", "training_log", "verdict", "n_skipped", "wall_s")}
        full[rid]["kind"] = "prompting" if d["policy"]["fp32_ckpt"]["uses_prompt"] else "plain"
        full[rid]["load_s"] = d["policy"]["fp32_ckpt"]["load_s"]
        ap = [s["approach_row6"] for s in d["samples"] if "approach_row6" in s]
        full[rid]["approach_row6"] = ap
        s, tl, v = d["splits"], d["training_log"], d["verdict"]

        def f(x):
            return f"{x * 1e4:.2f}" if x is not None else "-"
        cells = []
        for sp in ("val", "train"):
            r = s.get(sp, {})
            cells.append(f"{r.get('n', 0)} | {f(tl[sp]['all'])} | {f(r['mse_direct']['all'])} | "
                         f"{f(r['mse_deploy']['all'])} | {f(r['mse_bf16_8']['all'])} | "
                         f"{v[sp]['t1_ratio']:.2f} / {v[sp]['t2_ratio']:.2f} | "
                         f"{r['at_k_deploy']['k15']['pos_mm']:.1f}")
        rows.append(f"| {rid.split('/')[0]} {rid.split('/')[1][:7]} | {full[rid]['kind']} | {d['dataset']} | "
                    f"{cells[0]} | {cells[1]} | {'PASS' if v['pass'] else 'FAIL'} |")
    head = ("| run | kind | obs package | val n | train-log val MSE | direct | deploy | bf16/8 | T1 / T2 ratio | "
            "deploy k15 pos mm | train n | train-log train MSE | direct | deploy | bf16/8 | T1 / T2 ratio | "
            "deploy k15 pos mm | verdict |\n" + "|---" * 18 + "|")
    md = ("# W1 step 3: open-loop error (MSE x 1e-4, unnormalised action; deploy = convention B, fp32 at the "
          "checkpoint's 16 steps)\n\n" + head + "\n" + "\n".join(rows) + "\n")
    write_json(OL / "summary.json", {"runs": full, "deploy_version": deploy_version(),
                                     "tolerance": __doc__.split("Tolerance (fixed before running):\n")[1]
                                     .split("\n\nSamples.")[0]})
    (OL / "summary.md").write_text(md)
    print(md)
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("cmd", choices=["plan", "obs", "run", "summary"])
    ap.add_argument("--id", default=None)
    ap.add_argument("--limit", type=int, default=None, help="first N samples only (a quick check)")
    a = ap.parse_args(argv)
    if a.cmd == "plan":
        return cmd_plan()
    if a.cmd == "obs":
        return cmd_obs()
    if a.cmd == "run":
        return cmd_run(a.id, a.limit)
    return cmd_summary()


if __name__ == "__main__":
    sys.exit(main())
