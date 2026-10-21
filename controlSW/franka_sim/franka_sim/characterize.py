"""``python -m franka_sim characterize``: the sim arm's impedance tracking table (A4), deterministic (SimCore, no threads).

Why: hardware day measures the real arm with ``gonogo --motion steps`` (lag tau, steady-state error e). This command
runs the same profile on the sim (200 Hz targets through the bridge's real SafetyFilter into SimCore, sampled like
gonogo's client) for each translational stiffness x payload error, plus an ideal 2 cm target step per axis
(overshoot, settle time), so the two can be compared and the sim tuned after hardware day.

Usage: python -m franka_sim characterize [--robots panda,fr3] [--stiffness 600,800,1000] [--payload-error 0,0.3]
           [--out controlSW/franka_sim/franka_sim/configs/characterization_baseline.json]
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

DEFAULT_OUT = Path(__file__).resolve().parent / "configs" / "characterization_baseline.json"


def _station(robot: str, profile: str = "sim-iphone"):
    from franka_deploy.station import PROFILES, load_station

    spec = {k: dict(v) for k, v in PROFILES[profile].items()}
    spec["station"]["robot"] = robot
    return load_station(spec)


def step_metrics(core, ax: int, amp: float = 0.02, n_ms: int = 2500) -> dict:
    """Ideal ``amp`` target step on one axis, measured from the settled pose at the unchanged start target (so a
    payload-error offset, constant for any target, is not counted as overshoot)."""
    core.recover()
    core.start()
    T = core.state()["O_T_EE"].copy()
    core.step(1500)
    p0 = core.state()["O_T_EE"][ax, 3]
    T[ax, 3] += amp
    core.set_target(T)
    xs = np.empty(n_ms)
    for k in range(n_ms):
        core.step(1)
        xs[k] = core.state()["O_T_EE"][ax, 3] - p0
    out_of = np.flatnonzero(np.abs(xs - amp) > 0.05 * amp)
    return {"overshoot_pct": round(float((xs.max() - amp) / amp * 100), 2),
            "settle_5pct_ms": int(out_of[-1] + 1) if out_of.size else 0,
            "steady_err_mm": round(float(abs(xs[-1] - amp) * 1e3), 4)}


def gonogo_steps(core, safety_cfg: dict, hold_s: float = 2.0, amp: float = 0.02, rate_hz: float = 200.0) -> dict:
    """gonogo --motion steps (x, y, z in turn: ramp +amp at 0.05 m/s, hold, back, hold) and its tracking_stats."""
    from franka_deploy.bridge.safety import SafetyConfig, SafetyFilter
    from franka_deploy.tools.gonogo import STEP_HOLD_S, STEP_V, motion_offset, tracking_stats

    core.recover()
    core.start()
    T0 = core.state()["O_T_EE"].copy()
    sf = SafetyFilter(SafetyConfig.from_dict(safety_cfg))
    sf.reset(T0, 0.0)
    seg = 2 * amp / STEP_V + 2 * STEP_HOLD_S
    dur = 3 * seg
    n_tick = int(round(1000 / rate_hz))
    t_rec, cmd, fk = [], [], []
    clamps = 0
    t = 0.0
    while t < hold_s + dur:
        T = T0.copy()
        if t >= hold_s:
            T[:3, 3] += motion_offset("steps", t - hold_s, amp, 1.0, 0)
        Ts, flags = sf.apply(T, t)
        clamps += bool(flags)
        core.set_target(Ts)
        if t >= hold_s:
            t_rec.append(t)
            cmd.append(T[:3, 3].copy())
            fk.append(core.state()["O_T_EE"][:3, 3].copy())
        core.step(n_tick)
        t += n_tick / 1000.0
    st = tracking_stats(np.array(t_rec), np.array(cmd), np.array(fk))
    st = {k: (round(v, 4) if isinstance(v, float) else v) for k, v in st.items()}
    st["safety_clamps"] = clamps
    st["reflex"] = core.error
    return st


def characterize(robots=("panda", "fr3"), stiffness=(600.0, 800.0, 1000.0), payload_errors=(0.0, 0.3)) -> dict:
    from franka_sim.model import build_model, sim_config
    from franka_sim.simcore import SimCore, make_core_cfg

    out = {"what": "SimCore impedance characterization (gonogo --motion steps profile + ideal 2 cm steps)",
           "law": "panda-py CartesianImpedance as documented (simcore.py; verify-on-hardware)", "rows": []}
    for robot in robots:
        st = _station(robot)
        safety = st.bridge_fragment().get("safety", {})
        physics = sim_config(st, None)
        sm = build_model(st, physics)
        for k_t in stiffness:
            for dm in payload_errors:
                cfg = make_core_cfg(st, physics={**physics, "payload_error_kg": float(dm)})
                cfg["impedance"] = [k_t] * 3 + [float(x) for x in cfg["impedance"][3:6]]   # rotational as station
                core = SimCore(sm, cfg)
                t0 = time.perf_counter()
                row = {"robot": robot, "translational_n_m": k_t, "rotational_nm_rad": float(cfg["impedance"][3]),
                       "damping_ratio": cfg["damping_ratio"], "payload_error_kg": float(dm),
                       "gonogo_steps": gonogo_steps(core, safety),
                       "step_2cm": {a: step_metrics(core, i) for i, a in enumerate("xyz")},
                       "expected_payload_offset_mm": round(float(dm) * 9.81 / k_t * 1e3, 3)}
                row["wall_s"] = round(time.perf_counter() - t0, 1)
                out["rows"].append(row)
                g = row["gonogo_steps"]
                print(f"{robot:5s} K={k_t:6.0f} dm={dm:.1f}: lag {g.get('lag_ms')} ms, steady median "
                      f"{g.get('steady_err_mm_median')} mm, max {g.get('track_err_mm_max')} mm; step overshoot "
                      f"x/y/z {[row['step_2cm'][a]['overshoot_pct'] for a in 'xyz']} %", flush=True)
    return out


def add_arguments(p: argparse.ArgumentParser) -> None:
    p.add_argument("--robots", default="panda,fr3")
    p.add_argument("--stiffness", default="600,800,1000", help="translational stiffness values (N/m)")
    p.add_argument("--payload-error", default="0,0.3", help="physics.payload_error_kg values (configured - true)")
    p.add_argument("--out", default=str(DEFAULT_OUT))


def run(args: argparse.Namespace) -> int:
    res = characterize([r for r in args.robots.split(",") if r], [float(x) for x in args.stiffness.split(",")],
                       [float(x) for x in args.payload_error.split(",")])
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(res, indent=1) + "\n")
    print(f"wrote {out}")
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="franka_sim characterize", description=__doc__.split("\n\n")[0])
    add_arguments(ap)
    return run(ap.parse_args(argv))


if __name__ == "__main__":
    sys.exit(main())
