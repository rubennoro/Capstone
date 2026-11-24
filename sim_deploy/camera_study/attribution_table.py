"""W2 step 4: plan the first-chunk jobs and build the attribution tables (real vs sim image sets, per checkpoint).

Run from ``~/Capstone-main-sim`` (``unset PYTHONPATH``), any env with numpy + scipy (e.g. ``~/envs/sim_deploy``)::

    python -m sim_deploy.camera_study.attribution_table jobs real  JOBS.json   # 15 refs: train + deploy path, 60 neighbours
    python -m sim_deploy.camera_study.attribution_table jobs sets  JOBS.json [--looks current real ...]
    python -m sim_deploy.camera_study.attribution_table table [--runs ID ...]

``jobs`` writes a jobs file for ``first_chunk.py``. Ids: ``real__<ref>`` (training path, B1 ``obs.npz``),
``deploy__<ref>`` (B1 ``deploy_obs.npz``), ``nb__<ref>_d<+-k>`` (``real_neighbours.py``), ``<look>__<pose>__<ref>``
(step 3's sets, ``sim_evals/_pipeline/W2/sets/index.json``), and the per-camera swaps ``..__simmain`` (sim IMX519 +
real IMX219) and ``..__simuw`` (real IMX519 + sim IMX219), with the reference's low-dim in every case. Each job's demo
= the reference's own recorded displacement from its row (the neighbour's row for ``nb``).

``table`` reads ``sim_evals/_pipeline/W2/attribution/chunks/<task>__<run>/<id>.{npz,json}`` and writes
``attribution.json`` + ``attribution.md`` in ``sim_evals/_pipeline/W2/attribution/``:

* real reference per checkpoint: mean over the 15 references of ``approach_cm`` (end point along the demo's own
  displacement direction), ``z_cm`` (last step along TCP z), the demo's own displacement; the deploy path vs training
  path difference (sanity);
* tolerance per checkpoint and metric: ``tol = 2 * RMS_refs(std over {T-6, T-3, T, T+3, T+6} of the sample mean) +
  SE`` with ``SE`` = RMS over the references of the sample-mean standard error (std / sqrt(N)); for ``d_real_cm``:
  ``2 * RMS`` over references and neighbours of the distance between the neighbour's mean chunk and the reference's;
* per look and pose: means over references, the per-reference differences to the real response (mean and RMS),
  ``d_real_cm`` (RMS over steps of the distance between mean chunks, chunk_metrics.chunk_distance_cm); a cell is
  within tolerance when the RMS over references of the per-reference difference is <= tol (``d_real_cm``: the mean
  over references <= tol_d);
* ROBOT TRIALS (sim at the real arm's trial starts, ``robot_trials.py`` sets, ids ``<look>__robot_trials__<trial>``
  + swaps ``__simmain`` / ``__simuw`` against the trial's first real static set): per trial the sim's response vs the
  REAL ARM's (the mean over the trial's ``rm__`` static sets, ``real_mount.py``), trial 4 (black object) left out;
  within = RMS over trials of the difference <= the checkpoint's tolerance (z, approach; d: mean <= tol_d), the
  same tolerance as the handheld references (the static sets differ only by sensor noise, so they give no spread
  of their own);
* GEOMETRY = mount - handheld (same look, per reference); APPEARANCE = handheld(look) - real, and handheld(look) -
  handheld(current); per-camera swaps: sim IMX519 only (``simmain``) - real, sim IMX219 only (``simuw``) - real.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
REF_DIR = REPO / "sim_evals/_camera_study/B/reference"
LAYOUT = REPO / "sim_evals/_camera_study/B/layout.json"
RESULTS = REPO / "sim_evals/_pipeline/W2/attribution"
NB_DIR = RESULTS / "real_neighbours"
CHUNKS = RESULTS / "chunks"
SETS_INDEX = REPO / "sim_evals/_pipeline/W2/sets/index.json"
LOOK_ORDER = ("current", "real_nocam", "real", "real_kit", "real_bg", "robot_nocam", "robot")
RT_POSE = "robot_trials"
#: robot trials left out of the robot-cell comparison: run_022 trial 4 had a black object, not the eraser
#: (robot_trials.py does not fit / render it)
RT_EXCLUDE = ("PICK_TEST1_run_022_trial4",)
NOT_LOOKS = ("real", "deploy", "nb", "rm", "rm080")
POSE_ORDER = ("handheld", "mount", "mount_start")
SWAP_POSES = ("handheld", "mount")
METRICS = ("approach_cm", "z_cm")


def _refs() -> list[dict]:
    lay = json.loads(LAYOUT.read_text())
    out = []
    for e in lay["selection"]:
        for row, label in zip(e["rows_in_episode"], e["labels"]):
            out.append({"key": f"ep{int(e['index']):02d}_{label}", "index": int(e["index"]), "row": int(row),
                        "label": label, "dataset": str(Path(lay["dataset"]).expanduser())})
    return out


def _demo(ref: dict, row: int | None = None) -> dict:
    return {"dataset": ref["dataset"], "index": ref["index"], "row": ref["row"] if row is None else int(row)}


def jobs_real() -> list[dict]:
    jobs = []
    for r in _refs():
        d = REF_DIR / r["key"]
        jobs.append({"id": f"real__{r['key']}", "obs": str(d / "obs.npz"), "demo": _demo(r)})
        jobs.append({"id": f"deploy__{r['key']}", "obs": str(d / "deploy_obs.npz"), "demo": _demo(r)})
    nb = json.loads((NB_DIR / "index.json").read_text())["rows"]
    refs = {r["key"]: r for r in _refs()}
    for n in nb:
        r = refs[n["reference"]]
        jobs.append({"id": f"nb__{n['key']}", "obs": str(Path(n["folder"]) / "obs.npz"),
                     "demo": _demo(r, n["row_in_episode"])})
    return jobs


def _set_rows() -> list[dict]:
    """step 3's index.json -> [{look, pose, ref, folder}] (accepts a list or {'rows': [...]})."""
    if not SETS_INDEX.exists():
        return []
    doc = json.loads(SETS_INDEX.read_text())
    rows = doc.get("rows", doc.get("sets", [])) if isinstance(doc, dict) else doc
    out = []
    for r in rows:
        look = r.get("look") or r.get("look_id")
        pose = r.get("pose")
        ref = r.get("reference") or r.get("ref") or r.get("key")
        folder = Path(r["folder"])
        if not folder.is_absolute():
            folder = REPO / folder if (REPO / folder).exists() else SETS_INDEX.parent / folder
        if look and pose and ref and (folder / "obs.npz").exists():
            out.append({"look": look, "pose": pose, "ref": ref, "folder": folder, "row": r})
    return out


def jobs_sets(looks: list[str] | None, swaps: bool = True, poses: list[str] | None = None,
              swap_poses: list[str] | None = None) -> list[dict]:
    refs = {r["key"]: r for r in _refs()}
    jobs = []
    for s in _set_rows():
        if looks and s["look"] not in looks:
            continue
        if poses and s["pose"] not in poses:
            continue
        if s["pose"] == RT_POSE:
            jobs += _rt_jobs(s)
            continue
        r = refs[s["ref"]]
        base = f"{s['look']}__{s['pose']}__{s['ref']}"
        obs = str(s["folder"] / "obs.npz")
        real = str(REF_DIR / s["ref"] / "obs.npz")
        jobs.append({"id": base, "obs": obs, "demo": _demo(r)})
        if swaps and s["pose"] in (swap_poses or SWAP_POSES):
            jobs.append({"id": base + "__simmain", "obs": real, "main_from": obs, "demo": _demo(r)})
            jobs.append({"id": base + "__simuw", "obs": real, "ultrawide_from": obs, "demo": _demo(r)})
    return jobs


def _rt_norm(trial: str) -> str:
    """real_mount's trial id 'PICK_DOUBLE__run_002__17-41_trial1' -> robot_trials' set name 'PICK_DOUBLE_run_002_trial1'."""
    task, run, rest = trial.split("__")
    return f"{task}_{run}_{rest.split('_')[-1]}"


def _rt_trials() -> dict:
    """{robot_trials set name: real_mount index row}."""
    f = RESULTS / "real_mount" / "index.json"
    return {_rt_norm(t["trial"]): t for t in json.loads(f.read_text())["trials"]} if f.exists() else {}


_DEMO: list = []


def _rt_demo() -> list:
    """The demo direction of the rm__ jobs (mean of the five ``first`` references' displacements), from any finished
    rm__ job (same numbers as real_mount.mean_first_demo, which needs the zarr env)."""
    if not _DEMO:
        for f in sorted(CHUNKS.glob("*/rm__*.json")):
            d = json.loads(f.read_text()).get("job", {}).get("demo_disp")
            if d:
                _DEMO.append(d)
                break
        else:
            from sim_deploy.camera_study.real_mount import mean_first_demo
            _DEMO.append(mean_first_demo())
    return _DEMO[0]


def _rt_jobs(s: dict) -> list[dict]:
    t = _rt_trials().get(s["ref"])
    if t is None:
        return []
    base = f"{s['look']}__{RT_POSE}__{s['ref']}"
    obs = str(s["folder"] / "obs.npz")
    real = str(Path(t["folder"]) / f"set{t['sets'][0]:06d}" / "obs.npz")
    demo = _rt_demo()
    return [{"id": base, "obs": obs, "demo_disp": demo},
            {"id": base + "__simmain", "obs": real, "main_from": obs, "demo_disp": demo},
            {"id": base + "__simuw", "obs": real, "ultrawide_from": obs, "demo_disp": demo}]


# ------------------------------------------------------------------------------------------------------ the table

class Run:
    def __init__(self, folder: Path) -> None:
        self.folder = folder
        self.id = folder.name
        self.info = json.loads((folder / "run_info.json").read_text()) if (folder / "run_info.json").exists() else {}
        self.key = self.info.get("settings_key")
        self._s, self._c = {}, {}

    def has(self, jid: str) -> bool:
        f = self.folder / f"{jid}.json"
        if not f.exists():
            return False
        return self.key is None or json.loads(f.read_text()).get("settings_key") == self.key

    def s(self, jid: str) -> dict:
        if jid not in self._s:
            self._s[jid] = json.loads((self.folder / f"{jid}.json").read_text())
        return self._s[jid]

    def chunk(self, jid: str) -> np.ndarray:
        if jid not in self._c:
            self._c[jid] = np.load(self.folder / f"{jid}.npz")["eef_pos"]
        return self._c[jid]


RM_DIR = RESULTS / "real_mount"


def _ref_R(key: str) -> np.ndarray:
    """The reference's TCP rotation in the tag (table) frame at row T (z up = the table normal)."""
    m = json.loads((REF_DIR / key / "meta.json").read_text())
    return np.asarray(m["tcp_tag"]["T"]["T"], np.float64)[:3, :3]


def vertical(chunk: np.ndarray, R: np.ndarray) -> dict:
    """chunk (N, 16, 3) m in the TCP frame; R TCP -> world (z up). End point's vertical (cm, mean over samples) and
    the lowest vertical of the mean chunk over the 16 steps."""
    w = np.asarray(chunk, np.float64) @ np.asarray(R).T * 100.0
    return {"vert_cm": float(w[:, -1, 2].mean()), "vert_cm_std": float(w[:, -1, 2].std(ddof=1)),
            "vert_min_cm": float(w.mean(0)[:, 2].min()), "end_world_cm": w[:, -1].mean(0).round(2).tolist()}


def _rms(v) -> float:
    v = np.asarray(v, np.float64)
    return float(np.sqrt(np.mean(v ** 2))) if v.size else float("nan")


def _r(x, k=2):
    return None if x is None or (isinstance(x, float) and np.isnan(x)) else round(float(x), k)


def analyse(run: Run) -> dict | None:
    from sim_deploy.camera_study.chunk_metrics import chunk_distance_cm, paired_distance_cm
    refs = [r["key"] for r in _refs()]
    have = [k for k in refs if run.has(f"real__{k}")]
    if not have:
        return None
    out: dict = {"run_id": run.id, "settings": run.info.get("settings"), "n": run.info.get("n"),
                 "seed0": run.info.get("seed0"), "model_info": run.info.get("model_info"),
                 "timing": run.info.get("invocations")}
    real = {k: run.s(f"real__{k}") for k in have}
    out["real"] = {
        "refs": {k: {m: _r(real[k][m]) for m in ("approach_cm", "z_cm", "z_exec_cm", "approach_exec_cm",
                                                  "demo_along_cm", "demo_z_cm", "approach_cm_std", "z_cm_std")}
                 for k in have},
        "mean": {m: _r(np.mean([real[k][m] for k in have])) for m in
                 ("approach_cm", "z_cm", "z_exec_cm", "demo_along_cm", "demo_z_cm", "approach_exec_cm")},
        "n_refs": len(have)}
    for k in have:
        out["real"]["refs"][k].update({kk: _r(v) if not isinstance(v, list) else v
                                       for kk, v in vertical(run.chunk(f"real__{k}"), _ref_R(k)).items()})
        de = np.asarray(real[k]["demo_end_cm"], np.float64)
        out["real"]["refs"][k]["demo_vert_cm"] = _r((_ref_R(k) @ de)[2])
    out["real"]["mean"]["vert_cm"] = _r(np.mean([out["real"]["refs"][k]["vert_cm"] for k in have]))
    out["real"]["mean"]["demo_vert_cm"] = _r(np.mean([out["real"]["refs"][k]["demo_vert_cm"] for k in have]))
    out["real"]["by_label"] = {lab: {m: _r(np.mean([real[k][m] for k in have if k.endswith(lab)]))
                                     for m in ("approach_cm", "z_cm", "demo_along_cm")}
                               for lab in ("first", "approach_1of3", "approach_2of3")}
    for lab in out["real"]["by_label"]:
        for m in ("vert_cm", "demo_vert_cm"):
            out["real"]["by_label"][lab][m] = _r(np.mean([out["real"]["refs"][k][m] for k in have
                                                          if k.endswith(lab)]))
    # deploy path vs training path
    dep = [k for k in have if run.has(f"deploy__{k}")]
    if dep:
        dd = {m: [run.s(f"deploy__{k}")[m] - real[k][m] for k in dep] for m in METRICS}
        out["deploy_vs_train"] = {
            **{f"{m}_diff_mean": _r(np.mean(v)) for m, v in dd.items()},
            **{f"{m}_diff_rms": _r(_rms(v)) for m, v in dd.items()},
            "d_cm_mean": _r(np.mean([chunk_distance_cm(run.chunk(f"deploy__{k}"), run.chunk(f"real__{k}"))
                                     for k in dep])),
            "paired_d_cm_mean": _r(np.mean([paired_distance_cm(run.chunk(f"deploy__{k}"), run.chunk(f"real__{k}"))
                                            for k in dep])),
            "n_refs": len(dep)}
    # tolerance
    tol: dict = {}
    nb_keys = {k: [p.stem for p in sorted(run.folder.glob(f"nb__{k}_d*.json")) if run.has(p.stem)] for k in have}
    for m in METRICS:
        stds, ses = [], []
        for k in have:
            vals = [real[k][m]] + [run.s(j)[m] for j in nb_keys[k]]
            if len(vals) >= 3:
                stds.append(np.std(vals, ddof=1))
            ses.append(real[k][m + "_se"])
        rms_nb, se = _rms(stds), _rms(ses)
        tol[m] = {"tol": _r(2 * rms_nb + se), "rms_nb_std": _r(rms_nb), "se_mean": _r(se),
                  "n_refs_with_neighbours": len(stds), "per_ref_nb_std": {k: _r(s) for k, s in zip(have, stds)}}
    dn = [chunk_distance_cm(run.chunk(j), run.chunk(f"real__{k}")) for k in have for j in nb_keys[k]]
    tol["d_real_cm"] = {"tol": _r(2 * _rms(dn)), "rms_nb_d": _r(_rms(dn)), "n": len(dn)}
    out["tolerance"] = tol

    # the real mount reference (robot trials, addendum)
    out["real_mount"] = real_mount_block(run, out)
    # the sim sets
    looks = sorted({p.stem.split("__")[0] for p in run.folder.glob("*__*__*.json")
                    if p.stem.split("__")[1] in POSE_ORDER + ("mount_recgap", "mount_start_recgap", RT_POSE)},
                   key=lambda x: (LOOK_ORDER.index(x) if x in LOOK_ORDER else 99, x))
    cells: dict = {}

    def cell(jid_of, base_of=lambda k: f"real__{k}"):
        ks = [k for k in have if run.has(jid_of(k)) and run.has(base_of(k))]
        if not ks:
            return None
        c = {"n_refs": len(ks)}
        for m in METRICS:
            v = [run.s(jid_of(k))[m] for k in ks]
            d = [run.s(jid_of(k))[m] - run.s(base_of(k))[m] for k in ks]
            c[m] = _r(np.mean(v))
            c[f"{m}_diff_mean"] = _r(np.mean(d))
            c[f"{m}_diff_rms"] = _r(_rms(d))
            c[f"{m}_within"] = bool(_rms(d) <= tol[m]["tol"])
        dr = [chunk_distance_cm(run.chunk(jid_of(k)), run.chunk(base_of(k))) for k in ks]
        c["d_cm"] = _r(np.mean(dr))
        c["d_within"] = bool(np.mean(dr) <= tol["d_real_cm"]["tol"])
        c["per_ref"] = {k: {m: _r(run.s(jid_of(k))[m]) for m in METRICS} for k in ks}
        # by label: the static episode starts ("first") vs the references in motion ("approach_*"; the real frames
        # there carry motion blur, the sim's have none)
        for lab, sel in (("first", lambda k: k.endswith("_first")), ("moving", lambda k: "_approach_" in k)):
            kk = [k for k in ks if sel(k)]
            if kk:
                d = [run.s(jid_of(k))["approach_cm"] - run.s(base_of(k))["approach_cm"] for k in kk]
                c[f"approach_cm_diff_{lab}"] = {"mean": _r(np.mean(d)), "rms": _r(_rms(d)), "n": len(kk),
                                                "within": bool(_rms(d) <= tol["approach_cm"]["tol"])}
        return c

    for look in looks:
        for pose in POSE_ORDER:
            base = f"{look}__{pose}"
            c = cell(lambda k: f"{base}__{k}")
            if c is None:
                continue
            cells[base] = {"vs_real": c,
                           "simmain_vs_real": cell(lambda k: f"{base}__{k}__simmain"),
                           "simuw_vs_real": cell(lambda k: f"{base}__{k}__simuw")}
        # geometry: mount - handheld (same look)
        g = cell(lambda k: f"{look}__mount__{k}", lambda k: f"{look}__handheld__{k}")
        if g is not None:
            cells.setdefault(f"{look}__geometry", {})["mount_minus_handheld"] = g
        g = cell(lambda k: f"{look}__mount_start__{k}", lambda k: f"{look}__handheld__{k}")
        if g is not None:
            cells.setdefault(f"{look}__geometry", {})["mount_start_minus_handheld"] = g
        if look != "current":
            a = cell(lambda k: f"{look}__handheld__{k}", lambda k: f"current__handheld__{k}")
            if a is not None:
                cells.setdefault(f"{look}__appearance", {})["handheld_minus_current_handheld"] = a
            a = cell(lambda k: f"{look}__mount__{k}", lambda k: f"current__mount__{k}")
            if a is not None:
                cells.setdefault(f"{look}__appearance", {})["mount_minus_current_mount"] = a
    out["looks"] = looks
    out["cells"] = cells
    out["robot_trials"] = robot_trials_block(run, looks, tol)
    return out


def robot_trials_block(run: Run, looks: list[str], tol: dict) -> dict | None:
    """Sim at the real arm's trial starts vs the real arm's own images (per trial; trial 4 = black object, left out)."""
    from sim_deploy.camera_study.chunk_metrics import chunk_distance_cm
    trials = {k: t for k, t in _rt_trials().items() if k not in RT_EXCLUDE}
    res: dict = {}
    real_cache: dict = {}

    def real_of(k, t):
        if k not in real_cache:
            ids = [f"rm__{t['trial']}__s{s:06d}" for s in t["sets"]]
            ids = [i for i in ids if run.has(i)]
            real_cache[k] = None if not ids else {
                "chunks": np.concatenate([run.chunk(i) for i in ids]),
                **{m: float(np.mean([run.s(i)[m] for i in ids])) for m in METRICS}}
        return real_cache[k]

    for look in looks:
        for var, suf in (("vs_real_arm", ""), ("simmain_vs_real_arm", "__simmain"), ("simuw_vs_real_arm", "__simuw")):
            rows = {}
            for k, t in trials.items():
                jid = f"{look}__{RT_POSE}__{k}{suf}"
                rr = real_of(k, t)
                if rr is None or not run.has(jid):
                    continue
                s = run.s(jid)
                R = np.asarray(t["start_pose_O"], np.float64)[:3, :3]
                rows[k] = {**{m: _r(s[m]) for m in METRICS}, **{f"real_{m}": _r(rr[m]) for m in METRICS},
                           **{f"{m}_diff": _r(s[m] - rr[m]) for m in METRICS},
                           "vert_cm": _r(vertical(run.chunk(jid), R)["vert_cm"]),
                           "d_cm": _r(chunk_distance_cm(run.chunk(jid), rr["chunks"]))}
            if not rows:
                continue
            c = {"n_trials": len(rows), "per_trial": rows}
            for m in METRICS:
                d = [r[f"{m}_diff"] for r in rows.values()]
                c[m] = _r(np.mean([r[m] for r in rows.values()]))
                c[f"real_{m}"] = _r(np.mean([r[f"real_{m}"] for r in rows.values()]))
                c[f"{m}_diff_mean"] = _r(np.mean(d))
                c[f"{m}_diff_rms"] = _r(_rms(d))
                c[f"{m}_within"] = bool(_rms(d) <= tol[m]["tol"])
            c["d_cm"] = _r(np.mean([r["d_cm"] for r in rows.values()]))
            c["d_within"] = bool(c["d_cm"] <= tol["d_real_cm"]["tol"])
            res.setdefault(look, {})[var] = c
    return res or None


def real_mount_block(run: Run, out: dict) -> dict | None:
    if not (RM_DIR / "index.json").exists():
        return None
    idx = json.loads((RM_DIR / "index.json").read_text())["trials"]
    res: dict = {"trials": {}}
    for t in idx:
        R = np.asarray(t["start_pose_O"], np.float64)[:3, :3]
        z0 = float(np.asarray(t["start_pose_O"])[2, 3]) * 100
        row: dict = {}
        for pre in ("rm", "rm080"):
            ids = [f"{pre}__{t['trial']}__s{s:06d}" for s in t["sets"]]
            ids = [i for i in ids if run.has(i)]
            if not ids:
                continue
            ss = [run.s(i) for i in ids]
            vs = [vertical(run.chunk(i), R) for i in ids]
            v = {m: [x[m] for x in ss] for m in ("z_cm", "approach_cm", "z_exec_cm")}
            v["vert_cm"] = [x["vert_cm"] for x in vs]
            v["vert_min_cm"] = [x["vert_min_cm"] for x in vs]
            row[pre] = {**{m: _r(np.mean(x)) for m, x in v.items()},
                        **{f"{m}_set_std": _r(np.std(x, ddof=1)) if len(x) > 1 else None for m, x in v.items()},
                        "z_cm_sample_std": _r(np.mean([x["z_cm_std"] for x in ss])),
                        "end_O_z_cm": _r(z0 + np.mean(v["vert_cm"])), "min_O_z_cm": _r(z0 + np.mean(v["vert_min_cm"])),
                        "end_world_cm": np.mean([x["end_world_cm"] for x in vs], axis=0).round(2).tolist(),
                        "n_sets": len(ids)}
        if row:
            row["own_checkpoint"] = Path(t["run"]).name == out.get("model_info", {}).get("checkpoint", "").split(
                "/policy/")[-1].split("/")[0] if out.get("model_info") else None
            row["start_z_O_cm"] = _r(z0)
            res["trials"][t["trial"]] = row
    if not res["trials"]:
        return None
    main = [k for k in res["trials"] if not k.endswith("trial4") and "rm" in res["trials"][k]]
    if main:
        agg = {m: _r(np.mean([res["trials"][k]["rm"][m] for k in main])) for m in
               ("z_cm", "approach_cm", "vert_cm", "vert_min_cm", "end_O_z_cm", "min_O_z_cm")}
        agg["trial_std_z_cm"] = _r(np.std([res["trials"][k]["rm"]["z_cm"] for k in main], ddof=1)) if len(main) > 1 else None
        agg["set_std_z_cm_rms"] = _r(_rms([res["trials"][k]["rm"]["z_cm_set_std"] or 0 for k in main]))
        agg["n_trials"] = len(main)
        res["mean_excl_trial4"] = agg
        rf = out["real"]["by_label"]["first"]
        res["real_mount_minus_real_handheld_first"] = {m: _r(agg[m] - rf[m]) for m in ("z_cm", "approach_cm", "vert_cm")}
        w = [k for k in main if "rm080" in res["trials"][k]]
        if w:
            res["w080_minus_w090_z_cm"] = _r(np.mean([res["trials"][k]["rm080"]["z_cm"] - res["trials"][k]["rm"]["z_cm"]
                                                      for k in w]))
    return res


def _fmt_cell(c: dict | None, tol: dict) -> str:
    if not c:
        return "| - | - | - | - | - |"
    ok = lambda b: "yes" if b else "**no**"
    return (f"| {c['approach_cm']:+.2f} | {c['approach_cm_diff_mean']:+.2f} / {c['approach_cm_diff_rms']:.2f} "
            f"{ok(c['approach_cm_within'])} | {c['z_cm']:+.2f} | {c['z_cm_diff_mean']:+.2f} / "
            f"{c['z_cm_diff_rms']:.2f} {ok(c['z_cm_within'])} | {c['d_cm']:.2f} {ok(c['d_within'])} | {c['n_refs']} |")


GATE_LOOK = "train"          # the training-scene look the gate is read on (W2 README)
GATE_ROBOT_LOOK = "robot"    # the robot-cell look for the real-arm comparison


def _gate_summary(results: list[dict]) -> list[str]:
    """One row per checkpoint: the W2 gate cells (approach diff mean / RMS vs tol; within yes/no)."""
    def f(c, key="approach_cm"):
        if not c:
            return "-"
        ok = "yes" if c[f"{key}_within"] else "**no**"
        return f"{c[f'{key}_diff_mean']:+.1f} / {c[f'{key}_diff_rms']:.1f} {ok}"

    def lab(c, which):
        x = (c or {}).get(f"approach_cm_diff_{which}")
        return "-" if not x else f"{x['mean']:+.1f} / {x['rms']:.1f} {'yes' if x['within'] else '**no**'}"

    L = ["## Gate summary", "",
         f"approach diff (sim - real) mean / RMS over references in cm, `yes` = RMS <= the checkpoint's tolerance. "
         f"Training scene: look `{GATE_LOOK}` vs the 15 real training references (handheld = appearance only, mount = "
         f"the sim's own cameras; `first` = the 5 static episode starts, `moving` = the 10 references in motion). "
         f"Robot cell: look `{GATE_ROBOT_LOOK}` at the real arm's trial starts vs the real arm's images (trials as listed per "
         f"checkpoint below; {', '.join(RT_EXCLUDE)} left out).", "",
         "| checkpoint | N | tol | real | handheld | handheld first | handheld moving | mount | mount first | mount, sim IMX519 only | mount, sim IMX219 only | mount + IMX519 crop (opt-in) | robot cell vs real arm |",
         "|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in results:
        cells, t = r.get("cells") or {}, r.get("tolerance") or {}
        if t.get("approach_cm", {}).get("tol") is None:
            continue
        hh = (cells.get(f"{GATE_LOOK}__handheld") or {}).get("vs_real")
        mt = cells.get(f"{GATE_LOOK}__mount") or {}
        cr = (cells.get(f"{GATE_LOOK}_c519__mount") or {}).get("vs_real")
        rt = ((r.get("robot_trials") or {}).get(GATE_ROBOT_LOOK) or {}).get("vs_real_arm")
        rt_txt = "-" if not rt else f"{f(rt)} (real arm {rt['real_approach_cm']:+.1f}, sim {rt['approach_cm']:+.1f})"
        L.append(f"| {r['run_id']} | {r.get('n')} | {t['approach_cm']['tol']:.1f} | {r['real']['mean']['approach_cm']:+.1f} | "
                 f"{f(hh)} | {lab(hh, 'first')} | {lab(hh, 'moving')} | {f(mt.get('vs_real'))} | "
                 f"{lab(mt.get('vs_real'), 'first')} | {f(mt.get('simmain_vs_real'))} | {f(mt.get('simuw_vs_real'))} | "
                 f"{f(cr)} | {rt_txt} |")
    return L + [""]


def write_md(results: list[dict], path: Path) -> None:
    L = ["# W2 attribution: first-chunk response, real vs sim image sets", "",
         "Generated by `sim_deploy/camera_study/attribution_table.py table` (definitions in its docstring). cm, "
         "current TCP frame at the observation's row. `approach` = end point (step 15, 0.75 s) along the demo's own "
         "displacement direction; `z` = end point along TCP z (toward the fingertips). `diff` = per-reference "
         "difference to the base (mean / RMS over references), `within` = RMS <= tol; `d` = RMS-over-steps distance "
         "between mean chunks (vs the base), within = mean <= tol_d.", ""]
    L += _gate_summary(results)
    for r in results:
        st = r.get("settings") or {}
        t = r["tolerance"]
        rm = r["real"]["mean"]
        L += [f"## {r['run_id']}", "",
              f"settings: {st.get('precision')}, {st.get('diffusion_steps_in_use')} DDIM steps, exec_h "
              f"{st.get('exec_h')}, device {st.get('device')}, prompt {st.get('prompt')}, N {r.get('n')} paired "
              f"samples (seeds {r.get('seed0')}..), gripper_map {'on' if st.get('gripper_map') else 'off'}", "",
              f"**Real reference** (15 refs, training path): approach {rm['approach_cm']:+.2f} cm (demo "
              f"{rm['demo_along_cm']:.2f}), z {rm['z_cm']:+.2f} cm (demo {rm['demo_z_cm']:+.2f}); by label: " +
              ", ".join(f"{lab} {v['approach_cm']:+.2f} (demo {v['demo_along_cm']:.2f})"
                        for lab, v in r["real"]["by_label"].items()), ""]
        if "deploy_vs_train" in r:
            d = r["deploy_vs_train"]
            L += [f"deploy path - training path: approach {d['approach_cm_diff_mean']:+.2f} (RMS "
                  f"{d['approach_cm_diff_rms']:.2f}), z {d['z_cm_diff_mean']:+.2f} (RMS {d['z_cm_diff_rms']:.2f}), "
                  f"d {d['d_cm_mean']:.2f} cm", ""]
        if t["approach_cm"]["tol"] is None or t["d_real_cm"]["tol"] is None:
            L += ["(incomplete: too few neighbour chunks for the tolerance yet)", ""]
            continue
        L += [f"**Tolerance**: approach {t['approach_cm']['tol']:.2f} cm (2 x nb std RMS {t['approach_cm']['rms_nb_std']:.2f}"
              f" + SE {t['approach_cm']['se_mean']:.2f}), z {t['z_cm']['tol']:.2f} cm (2 x {t['z_cm']['rms_nb_std']:.2f}"
              f" + {t['z_cm']['se_mean']:.2f}), d_real {t['d_real_cm']['tol']:.2f} cm (2 x RMS nb d "
              f"{t['d_real_cm']['rms_nb_d']:.2f}, {t['d_real_cm']['n']} neighbours)", ""]
        if not r["cells"]:
            L += ["(no sim sets run yet)", ""]
            continue
        L += ["| set | vs | approach | diff mean / RMS, within | z | diff mean / RMS, within | d, within | refs |",
              "|---|---|---|---|---|---|---|---|"]
        for name, c in r["cells"].items():
            for what, cc in c.items():
                if cc is None:
                    continue
                vs = {"vs_real": "real", "simmain_vs_real": "real (sim IMX519 only)",
                      "simuw_vs_real": "real (sim IMX219 only)", "mount_minus_handheld": "handheld (GEOMETRY)",
                      "mount_start_minus_handheld": "handheld (GEOMETRY, start pose)",
                      "handheld_minus_current_handheld": "current handheld (APPEARANCE)",
                      "mount_minus_current_mount": "current mount (APPEARANCE)"}[what]
                L.append(f"| {name} | {vs} " + _fmt_cell(cc, t))
        L.append("")
        rt = r.get("robot_trials")
        if rt:
            L += ["Robot trials (sim at the real arm's trial starts vs the REAL ARM's images; `approach` / `z` real = the "
                  "real arm's mean; refs = trials):", "",
                  "| look | vs | approach (real) | diff mean / RMS, within | z (real) | diff mean / RMS, within | d, within | trials |",
                  "|---|---|---|---|---|---|---|---|"]
            ok = lambda b: "yes" if b else "**no**"
            for look, vv in rt.items():
                for what, c in vv.items():
                    vs = {"vs_real_arm": "real arm", "simmain_vs_real_arm": "real arm (sim IMX519 only)",
                          "simuw_vs_real_arm": "real arm (sim IMX219 only)"}[what]
                    L.append(f"| {look}__robot_trials | {vs} | {c['approach_cm']:+.2f} ({c['real_approach_cm']:+.2f}) | "
                             f"{c['approach_cm_diff_mean']:+.2f} / {c['approach_cm_diff_rms']:.2f} {ok(c['approach_cm_within'])} | "
                             f"{c['z_cm']:+.2f} ({c['real_z_cm']:+.2f}) | {c['z_cm_diff_mean']:+.2f} / {c['z_cm_diff_rms']:.2f} "
                             f"{ok(c['z_cm_within'])} | {c['d_cm']:.2f} {ok(c['d_within'])} | {c['n_trials']} |")
            L.append("")
    path.write_text("\n".join(L) + "\n")


def table(run_ids: list[str] | None) -> int:
    results = []
    for f in sorted(CHUNKS.glob("*__*")):
        if run_ids and f.name not in run_ids:
            continue
        r = analyse(Run(f))
        if r:
            results.append(r)
    order = ["PICK_TEST1__run_006", "PICK_TEST1__run_022", "PICK_DOUBLE__run_002"]
    results.sort(key=lambda r: next((i for i, o in enumerate(order) if r["run_id"].startswith(o)), 99))
    (RESULTS / "attribution.json").write_text(json.dumps(results, indent=1))
    write_md(results, RESULTS / "attribution.md")
    print(f"{len(results)} checkpoints -> {RESULTS / 'attribution.md'}")
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    j = sub.add_parser("jobs")
    j.add_argument("what", choices=("real", "sets"))
    j.add_argument("out", type=Path)
    j.add_argument("--looks", nargs="+")
    j.add_argument("--no-swaps", action="store_true")
    j.add_argument("--poses", nargs="+", help="only these poses (default: all in index.json)")
    j.add_argument("--swap-poses", nargs="+", help=f"poses that get the per-camera swaps (default {SWAP_POSES})")
    t = sub.add_parser("table")
    t.add_argument("--runs", nargs="+")
    a = ap.parse_args(argv)
    sys.path.insert(0, str(REPO))
    if a.cmd == "jobs":
        jobs = jobs_real() if a.what == "real" else jobs_sets(a.looks, not a.no_swaps, a.poses, a.swap_poses)
        a.out.write_text(json.dumps(jobs, indent=1))
        print(f"{len(jobs)} jobs -> {a.out}")
        return 0
    return table(a.runs)


if __name__ == "__main__":
    raise SystemExit(main())
