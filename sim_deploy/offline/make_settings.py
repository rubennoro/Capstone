"""Per-checkpoint deploy settings from the bench and the real-robot sessions (W1 step 4, parts 2-3).

Inputs
  - sim_evals/_pipeline/W1/bench/<TASK>_<run>[/<tag>]/{bench.jsonl, env.json}: written by bench_checkpoints.py
    (the copy's policy.bench). When a run has several attempts (re-runs in tag sub-folders) the one with the lowest
    1-min load average at lock time is used; all are listed in the entry.
  - ~/policy_deploy_sessions/2026-11-20_*/ (real robot: not _mock, _pi_collect, _frames_alone):
    manager/steps.jsonl (sent steps: image_age_ms = L, frames_ms, policy_ms, c3_rtt_ms, cycle_ms, margin_ms),
    manager/episodes.jsonl (late, loop_late), manager/session.yaml (exec_h, lookahead_ms), policy/startup.json
    (checkpoint, precision, diffusion_steps).

Budget (copy's STRUCTURE.md 7.6): the arm never holds if  L + 2*C' + d_up <= (exec_h - 1) * 50 ms,
  C' = frames + overhead + inference + RTT, d_up = RTT / 2. All terms are p95:
  L, frames, RTT = p95 over every sent step of every real session; inference = the bench p95 of the candidate.
  Overhead (look-up, conversions, C4 encode, ZMQ, decode = policy_ms minus pure inference), three ways:
    (a) per checkpoint that ran on the robot, policy_ms p50/p95 minus the bench p50/p95 at the same settings (the
        brief's method; reported, but the bench and the sessions ran at different GPU clocks, see `gpu_state`);
    (b) per episode, the manager's mean policy_ms minus the policy server's own mean inference time for the same
        episode (policy.log "episode before it: N inferences, mean X ms", same clock state, so it isolates the
        overhead);
    (c) c4_overhead.json (c4_overhead.py: to_observation + C4 encode/decode in-process, p95).
  Used: max over (b) and (c) (a's values are dominated by the clock difference).
  No extra reserve (STRUCTURE.md 11.3 adds 20 ms; the margin at the chosen exec_h shows how much is left).
  GPU state during each bench: samples.csv (bench folder; nvidia-smi + CPU package temperature every 2 s, written by
  a shell loop started next to bench_checkpoints.py) summarised per checkpoint between lock and release.

Settings per checkpoint: candidates in fidelity order (fp32, S), (bf16, S), (fp32, 8), (bf16, 8), S = the
checkpoint's own num_inference_steps; for each the smallest exec_h in 1..16 meeting the condition and its margin.
Provisional choice: the first candidate with minimal exec_h <= 12. Field names as the station uses them:
policy.precision, policy.diffusion_steps, manager.exec_h.

Outputs: sim_deploy/offline/checkpoint_settings.yaml and sim_evals/_pipeline/W1/bench/{sessions.json, summary.json}.

Run (cwd ~/Capstone-main-sim; any python 3 with numpy + PyYAML, e.g. the light env):
    unset PYTHONPATH; ~/envs/sim_deploy/bin/python sim_deploy/offline/make_settings.py
"""
from __future__ import annotations

import json
import re
import subprocess
import time
from pathlib import Path

import numpy as np
import yaml

REPO = Path.home() / "Capstone-main-sim"
BENCH = REPO / "sim_evals" / "_pipeline" / "W1" / "bench"
SESSIONS = Path.home() / "policy_deploy_sessions"
OUT_YAML = REPO / "sim_deploy" / "offline" / "checkpoint_settings.yaml"
STEP_MS = 50.0
EXEC_H_MAX = 16
EXEC_H_PICK_MAX = 12
NON_ROBOT = ("_mock", "_pi_collect", "_frames_alone")


def pct(values, q) -> float | None:
    a = np.asarray([v for v in values if v is not None], dtype=float)
    return None if a.size == 0 else round(float(np.percentile(a, q)), 2)


def stats(values) -> dict:
    a = [v for v in values if v is not None]
    return {"n": len(a), "p50": pct(a, 50), "p95": pct(a, 95), "max": round(max(a), 2) if a else None}


def run_id_of_checkpoint(ckpt: str) -> str:
    p = Path(ckpt)                                      # .../data/<TASK>/policy/<run>/checkpoints/x.ckpt
    return f"{p.parents[3].name}/{p.parents[1].name}"


# ------------------------------------------------------------------------------------------------ sessions

def read_sessions() -> list[dict]:
    out = []
    for s in sorted(SESSIONS.glob("2026-11-20_*")):
        if s.name.endswith(NON_ROBOT) or not (s / "manager" / "steps.jsonl").is_file():
            continue
        steps = [json.loads(line) for line in (s / "manager" / "steps.jsonl").read_text().splitlines() if line]
        sent = [r for r in steps if r.get("sent")]
        if not sent:
            continue
        startup = json.loads((s / "policy" / "startup.json").read_text())["settings"]
        mgr = yaml.safe_load((s / "manager" / "session.yaml").read_text())
        mgr = mgr.get("settings", mgr) if isinstance(mgr, dict) else {}
        eps = [json.loads(line) for line in (s / "manager" / "episodes.jsonl").read_text().splitlines() if line] \
            if (s / "manager" / "episodes.jsonl").is_file() else []
        margins = [r["margin_ms"] for r in sent if r.get("margin_ms") is not None]
        out.append({
            "session": s.name, "checkpoint_run": run_id_of_checkpoint(startup["checkpoint"]),
            "prompting": startup.get("prompt_dataset") is not None, "precision": startup["precision"],
            "diffusion_steps": startup["diffusion_steps"], "exec_h": mgr.get("exec_h"),
            "lookahead_ms": mgr.get("lookahead_ms"), "steps_logged": len(steps), "steps_sent": len(sent),
            "skip_reasons": sorted({r["skip_reason"].split(":")[0] for r in steps if not r.get("sent")}),
            "episodes": [{k: e.get(k) for k in ("outcome", "duration_s", "steps_sent", "late", "loop_late")}
                         for e in eps],
            "L_image_age_ms": stats([r["image_age_ms"] for r in sent]),
            "frames_ms": stats([r["frames_ms"] for r in sent]),
            "policy_ms": stats([r["policy_ms"] for r in sent]),
            "c3_rtt_ms": stats([r["c3_rtt_ms"] for r in sent]),
            "cycle_ms": stats([r["cycle_ms"] for r in sent]),
            "margin_ms": {**stats(margins), "p5": pct(margins, 5), "min": round(min(margins), 2) if margins else None,
                          "negative": sum(1 for m in margins if m < 0)},
            "_raw": {k: [r[k] for r in sent] for k in ("image_age_ms", "frames_ms", "policy_ms", "c3_rtt_ms",
                                                         "cycle_ms")},
            "server_vs_manager": server_vs_manager(s, steps),
        })
    return out


SUMMARY_RE = re.compile(r"reset in [\d.]+ ms; episode before it: (\d+) inferences, mean ([\d.]+) ms, max ([\d.]+) ms")


def server_vs_manager(s: Path, steps: list[dict]) -> list[dict]:
    """Overhead method (b): reset j (1-based) begins episode j, its log line summarises episode j - 1's inferences
    on the policy server (policy.infer only); the manager's policy_ms of that episode adds look-up, conversions and
    both C4 hops."""
    log = s / "logs" / "policy.log"
    if not log.is_file():
        return []
    resets = [line for line in log.read_text().splitlines() if "policy.server INFO reset in" in line]
    out = []
    for j, line in enumerate(resets, start=1):
        m = SUMMARY_RE.search(line)
        if j == 1 or m is None:
            continue
        ep = j - 1
        pol = [r["policy_ms"] for r in steps if r["episode_id"] == ep and r.get("policy_ms")]
        if not pol:
            continue
        n, mean, mx = int(m.group(1)), float(m.group(2)), float(m.group(3))
        out.append({"episode_id": ep, "server_inferences": n, "server_mean_ms": mean, "server_max_ms": mx,
                    "manager_steps_with_policy": len(pol), "manager_mean_policy_ms": round(float(np.mean(pol)), 2),
                    "manager_max_policy_ms": round(max(pol), 2),
                    "overhead_mean_ms": round(float(np.mean(pol)) - mean, 2)})
    return out


def gpu_state(t0: str, t1: str) -> dict | None:
    """Summary of samples.csv rows with t0 <= time <= t1 (ISO strings compare as times)."""
    f = BENCH / "samples.csv"
    if not f.is_file():
        return None
    rows = [r.split(",") for r in f.read_text().splitlines()[1:] if r.count(",") >= 13]
    rows = [r for r in rows if t0 <= r[0] <= t1]
    busy = [r for r in rows if r[12].isdigit() and int(r[12]) >= 50]       # GPU utilisation >= 50 %
    if not busy:
        return {"samples": len(rows), "busy_samples": 0}
    return {"samples": len(rows), "busy_samples": len(busy),
            "sm_mhz_p50": pct([float(r[5]) for r in busy], 50),
            "sm_mhz_min_max": [min(float(r[5]) for r in busy), max(float(r[5]) for r in busy)],
            "gpu_temp_c_p50": pct([float(r[7]) for r in busy], 50),
            "cpu_pkg_c_p50": pct([float(r[1]) / 1000 for r in busy], 50),
            "sw_thermal_slowdown_fraction": round(sum(r[10] == "Active" for r in busy) / len(busy), 2),
            "power_w_p50": pct([float(r[6]) for r in busy], 50)}


# ------------------------------------------------------------------------------------------------ bench

def read_bench() -> dict[str, dict]:
    """{run id: {"attempts": [...], "use": attempt}}; attempt = {"folder", "env", "load", "prompt", "infer"}."""
    runs: dict[str, dict] = {}
    for env_file in sorted(BENCH.glob("*/env.json")) + sorted(BENCH.glob("*/*/env.json")):
        env = json.loads(env_file.read_text())
        recs = [json.loads(line) for line in (env_file.parent / "bench.jsonl").read_text().splitlines() if line] \
            if (env_file.parent / "bench.jsonl").is_file() else []
        infer = {(r["precision"], r["steps"]): r for r in recs if r.get("what") == "infer"}
        if env.get("exit_code") != 0 or not infer:
            continue
        att = {"folder": str(env_file.parent), "env": env, "infer": infer,
               "load": next((r for r in recs if r.get("what") == "load"), {}),
               "prompt": next((r for r in recs if r.get("what") == "prompt"), None)}
        runs.setdefault(env["run"]["id"], {"attempts": []})["attempts"].append(att)
    for r in runs.values():
        r["use"] = min(r["attempts"], key=lambda a: a["env"]["at_lock"]["loadavg_1_5_15"][0])
    return runs


# ------------------------------------------------------------------------------------------------ budget

def min_exec_h(L: float, c_prime: float, d_up: float) -> tuple[int | None, float, float | None]:
    need = L + 2 * c_prime + d_up
    for h in range(1, EXEC_H_MAX + 1):
        if need <= (h - 1) * STEP_MS:
            return h, round(need, 1), round((h - 1) * STEP_MS - need, 1)
    return None, round(need, 1), None


def sh(cmd: str) -> str:
    return subprocess.run(["bash", "-c", cmd], capture_output=True, text=True).stdout.strip()


def main() -> int:
    sessions = read_sessions()
    bench = read_bench()
    pooled = {k: [v for s in sessions for v in s["_raw"][k]] for k in sessions[0]["_raw"]} if sessions else {}
    L95, F95, R95 = pct(pooled["image_age_ms"], 95), pct(pooled["frames_ms"], 95), pct(pooled["c3_rtt_ms"], 95)

    # overhead = policy_ms - pure inference, per checkpoint/settings that ran on the robot
    overhead = []
    by_cfg: dict[tuple, list[float]] = {}
    for s in sessions:
        by_cfg.setdefault((s["checkpoint_run"], s["precision"], s["diffusion_steps"]), []).extend(s["_raw"]["policy_ms"])
    for (run, prec, steps), pol in sorted(by_cfg.items()):
        b = bench.get(run)
        rec = b["use"]["infer"].get((prec, steps)) if b else None
        if rec is None:
            continue
        overhead.append({"checkpoint_run": run, "precision": prec, "diffusion_steps": steps, "steps_sent": len(pol),
                         "policy_ms_p50": pct(pol, 50), "policy_ms_p95": pct(pol, 95),
                         "bench_p50": rec["p50_ms"], "bench_p95": rec["p95_ms"],
                         "overhead_p50": round(pct(pol, 50) - rec["p50_ms"], 2),
                         "overhead_p95": round(pct(pol, 95) - rec["p95_ms"], 2)})
    for s in sessions:                                  # per session too (report)
        b = bench.get(s["checkpoint_run"])
        rec = b["use"]["infer"].get((s["precision"], s["diffusion_steps"])) if b else None
        if rec is not None:
            s["overhead_vs_bench"] = {"bench_p50": rec["p50_ms"], "bench_p95": rec["p95_ms"],
                                      "overhead_p50": round(s["policy_ms"]["p50"] - rec["p50_ms"], 2),
                                      "overhead_p95": round(s["policy_ms"]["p95"] - rec["p95_ms"], 2)}
    server = [dict(e, session=s["session"]) for s in sessions for e in s["server_vs_manager"]]
    micro_file = BENCH / "c4_overhead.json"
    micro = json.loads(micro_file.read_text())["total"] if micro_file.is_file() else None
    cand_o = [e["overhead_mean_ms"] for e in server] + ([micro["p95_ms"]] if micro else [])
    O95 = round(max(cand_o), 2)
    d_up = round(R95 / 2, 2)
    terms = {"L_p95_ms": L95, "frames_p95_ms": F95, "overhead_p95_ms": O95, "rtt_p95_ms": R95, "d_up_ms": d_up,
             "overhead_method": "max of (b) manager mean policy_ms - policy-server mean inference per episode "
                                "and (c) c4_overhead.py total p95",
             "overhead_b_server_vs_manager": server, "overhead_c_microbench": micro,
             "steps_pooled": len(pooled["image_age_ms"]), "sessions": [s["session"] for s in sessions],
             "L_p50_ms": pct(pooled["image_age_ms"], 50), "frames_p50_ms": pct(pooled["frames_ms"], 50),
             "rtt_p50_ms": pct(pooled["c3_rtt_ms"], 50), "cycle_p50_ms": pct(pooled["cycle_ms"], 50),
             "cycle_p95_ms": pct(pooled["cycle_ms"], 95), "overhead_a_policy_ms_minus_bench": overhead}

    copy_diff = sh(f"git -C {REPO} diff --stat -- deployment | tail -1") or "(empty: no diff vs HEAD)"
    copy_head = sh(f"git -C {REPO} rev-parse --short HEAD")
    gm = (REPO / "deployment/policy_deploy/policy_deploy/policy/gripper_map.py").is_file()

    entries, summary = [], []
    for run_id, b in sorted(bench.items(), key=lambda kv: (kv[1]["use"]["env"]["run"]["kind"], kv[0])):
        att = b["use"]
        info = att["env"]["run"]
        S = info["ckpt_steps"]
        cands = []
        for prec, steps in (("fp32", S), ("bf16", S), ("fp32", 8), ("bf16", 8)):
            rec = att["infer"][(prec, steps)]
            c_prime = round(F95 + O95 + rec["p95_ms"] + R95, 1)
            h, need, margin = min_exec_h(L95, c_prime, d_up)
            cands.append({"policy.precision": prec, "policy.diffusion_steps": steps, "manager.exec_h": h,
                          "margin_ms": margin, "need_ms": need, "c_prime_ms": c_prime,
                          "have_ms": None if h is None else (h - 1) * STEP_MS})
        pick = next((c for c in cands if c["manager.exec_h"] is not None and c["manager.exec_h"] <= EXEC_H_PICK_MAX),
                    None)
        vram = {"load_allocated_mb": att["load"].get("max_memory_allocated_mb"),
                "load_reserved_mb": att["load"].get("max_memory_reserved_mb"),
                "infer_reserved_mb_max": max(r.get("max_memory_reserved_mb") or 0 for r in att["infer"].values()),
                "infer_allocated_mb_max": max(r.get("max_memory_allocated_mb") or 0 for r in att["infer"].values()),
                "nvidia_smi_used_mib_max": max(r.get("smi_used_mb") or 0 for r in att["infer"].values())}
        measured = {f"{p}_{s}": {"p50": r["p50_ms"], "p95": r["p95_ms"], "max": r["max_ms"], "mean": r["mean_ms"]}
                    for (p, s), r in sorted(att["infer"].items(), key=lambda kv: (-kv[0][1], kv[0][0]))}
        smi_end = {f"{p}_{s}": {"power_w": r.get("power_draw_w"), "sm_mhz": r.get("sm_clock_mhz"),
                                "temp_c": r.get("temperature_c")} for (p, s), r in att["infer"].items()}
        env = att["env"]
        entry = {
            "run_folder": info["run_folder"], "id": run_id, "kind": info["kind"], "train_config": info["train_config"],
            "checkpoint": Path(info["checkpoint"]).name, "dataset": info["dataset"],
            ("prompt_dataset" if info["kind"] == "prompting" else "dataset_path"): info["dataset_path"],
            "checkpoint_diffusion_steps": S,
            "policy": {"precision": pick["policy.precision"] if pick else None,
                       "diffusion_steps": pick["policy.diffusion_steps"] if pick else None},
            "manager": {"exec_h": pick["manager.exec_h"] if pick else None},
            "margin_ms": pick["margin_ms"] if pick else None,
            "measured_ms": measured,
            "budget_ms": {"L": L95, "C_prime": pick["c_prime_ms"] if pick else None, "d_up": d_up,
                          "need": pick["need_ms"] if pick else None, "have": pick["have_ms"] if pick else None},
            "vram_mb": vram,
            "alternatives": [c for c in cands if c is not pick],
            "bench": {"folder": att["folder"], "infers": next(iter(att["infer"].values()))["infers"],
                      "load_1_5_15_at_lock": env["at_lock"]["loadavg_1_5_15"],
                      "load_1_5_15_after": env["after"]["loadavg_1_5_15"],
                      "other_gpu_process_at_lock": env["at_lock"]["gpu_compute_apps"],
                      "ac_online": env["at_lock"]["power_supply_online"].get("AC"),
                      "power_profile": env["at_lock"]["power_profile"], "gpu_at_end": smi_end,
                      "gpu_state": gpu_state(env["at_lock"]["time"], env["after"]["time"]),
                      "attempts": [{"folder": a["folder"], "load_1m_at_lock": a["env"]["at_lock"]["loadavg_1_5_15"][0],
                                    **{f"{p}_{s}_p50_p95": [r["p50_ms"], r["p95_ms"]]
                                       for (p, s), r in a["infer"].items()}} for a in b["attempts"]]},
        }
        entries.append(entry)
        summary.append({"id": run_id, "kind": info["kind"], **{k: [v["p50"], v["p95"]] for k, v in measured.items()},
                        "vram_reserved_mb": vram["infer_reserved_mb_max"], "pick": pick,
                        "load_1m": env["at_lock"]["loadavg_1_5_15"][0]})

    header = f"""# checkpoint_settings.yaml: deploy settings per runnable checkpoint (W1 step 4, sub-agent S4).
# Made {time.strftime('%Y-%m-%d %H:%M')} by sim_deploy/offline/make_settings.py from
#   - the copy's policy.bench per checkpoint (sim_deploy/offline/bench_checkpoints.py; 150 timed calls after 5 warm-up
#     per (precision, steps), CUDA, RTX 3050 Ti 4 GB, AC power): sim_evals/_pipeline/W1/bench/<TASK>_<run>/
#   - the real-robot sessions of 2026-11-20 ({len(sessions)} sessions, {terms['steps_pooled']} sent steps, all bf16 / 8 steps,
#     run with ~/Capstone-final-deployment): ~/policy_deploy_sessions/2026-11-20_*/manager/steps.jsonl
# Condition (copy's STRUCTURE.md 7.6): the arm never holds if  L + 2*C' + d_up <= (exec_h - 1) * 50 ms,
#   C' = frames + overhead + inference + RTT; every term is a p95; d_up = RTT/2; no extra reserve (STRUCTURE.md 11.3
#   would add 20 ms: the margin field shows what is left).
#   L (image_age_ms) p95 = {L95} ms, frames_ms p95 = {F95} ms, RTT (c3_rtt_ms) p95 = {R95} ms, d_up = {d_up} ms,
#   overhead = {O95} ms (largest of: manager mean policy_ms minus the policy server's own mean inference per episode,
#   and the in-process to_observation + C4 encode/decode p95; policy_ms minus bench is in terms_ms but not used:
#   the bench ran thermally throttled, see each entry's bench.gpu_state), inference = the bench p95 of the candidate.
# Candidates in fidelity order (fp32, S), (bf16, S), (fp32, 8), (bf16, 8), S = the checkpoint's own steps; each with
#   the smallest exec_h in 1..16 meeting the condition. Provisional choice: the first with exec_h <= 12 (to be
#   re-picked once the accuracy cost of bf16 / 8 steps is measured, W1 step 3). The other three: `alternatives`.
# Keys as the station uses them: policy.precision, policy.diffusion_steps, manager.exec_h (nested below).
# Copy version: `git diff --stat -- deployment | tail -1` = {copy_diff}; HEAD {copy_head};
#   policy/gripper_map.py exists: {gm}.
"""
    OUT_YAML.write_text(header + yaml.safe_dump({"terms_ms": {k: v for k, v in terms.items()
                                                              if k != "sessions"},
                                                 "checkpoints": entries}, sort_keys=False, width=120))
    for s in sessions:
        s.pop("_raw")
    (BENCH / "sessions.json").write_text(json.dumps({"terms": terms, "sessions": sessions}, indent=1))
    (BENCH / "summary.json").write_text(json.dumps(summary, indent=1))
    print(json.dumps(terms, indent=1))
    for s in summary:
        print(s)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
