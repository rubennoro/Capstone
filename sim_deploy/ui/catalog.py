"""U5 catalog: the policies, replay episodes and past sim results the page lists (DESIGN.md section 2).

    from sim_deploy.ui.catalog import Catalog
    cat = Catalog(settings)          # settings: U1's object (attributes below) or None for the defaults
    cat.policies()                   # {"policies": [Policy], "replays": [Replay], "builtins": ["hold_still"]}
    cat.evals()                      # {"evals": [EvalRow]}

    python -m sim_deploy.ui.catalog [--json] [--no-replays]

Never imports torch. Everything is read only except the replay cache (``sim_deploy/ui/.cache/replays.json``).

Sources:
- policies: ``<data_root>/<TASK>/policy/run_*/``. The checkpoint comes from the eval driver's own resolver
  (``sim_deploy.eval.target.resolve``, imported lazily; a copy of its checkpoint rule is used only if that import
  fails), so a row with a checkpoint is one the driver accepts. ``train_config`` from, in order:
  ``run_config/job_info.json`` ``config_name``, the last ``TRAIN_CONFIG=`` line of ``run_config/submit_config.env``
  (file present, no line = behavior prompting, as RUNS.md says), the "Policy config" row of the run's README.md, a
  ``config_name``/``train_config`` key in job.json / train_meta.json. Not fetched runs hold only job.json, so their
  kind is "unknown" (their RUNS.md label still says which type they are).
- labels: ``<data_root>/<TASK>/RUNS.md`` (tables: run names in columns whose header says run/folder; prose: run
  names with the parenthesis after them), else "<dataset> · <job.json note>", else the dataset name.
- replays: every ``*.zarr.zip`` under ``<datasets_root>`` (default ``~/data/umi/datasets``), one row per episode,
  ``grasp_ok`` = ``|height_error_m| < 0.02`` from ``sim_deploy.scene.placement.Scene.replay_placement``, carried
  from the start pose (stations/start_pose.yaml, P2) when one is set, else from the deploy home.
- evals: ``<sim_evals>/index.csv`` (sim_deploy/eval/report.py INDEX_FIELDS), newest first.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import re
import sys
import threading
import time
import urllib.parse
from pathlib import Path

from sim_deploy import REPO, ROOT

UI_DIR = ROOT / "ui"
GRASP_OK_M = 0.02                                       # |height_error_m| below this: the replay can close on it
REPLAY_CACHE_SCHEMA = "sim_deploy.ui.replays/v1"

DEFAULTS = {
    "data_root": Path.home() / "Capstone" / "data",      # <TASK>/policy/run_*/, <TASK>/RUNS.md
    "datasets_root": Path.home() / "data" / "umi" / "datasets",
    "sim_evals": REPO / "sim_evals",
    "cache_dir": UI_DIR / ".cache",
    "model_json": ROOT / "build" / "sim_rig" / "model.json",
}
# accepted attribute / key names on the settings object, first match wins
_ALIASES = {
    "data_root": ("policy_data_root", "catalog_data_root", "data_root"),
    "datasets_root": ("datasets_root", "catalog_datasets_root"),
    "sim_evals": ("sim_evals",),
    "cache_dir": ("catalog_cache_dir",),
    "model_json": ("model_json",),
}

_RUN_FULL = re.compile(r"\brun_\d{3}_[0-9]{4}-[0-9]{2}-[0-9]{2}(?:_[A-Za-z0-9-]+)?")
_RUN_SHORT = re.compile(r"\brun_(\d{3})\b(?!_\d)")
_RUN_RANGE = re.compile(r"\brun_(\d{3})(?:_[\w-]+)?\s*(?:to|-|–|\.\.)\s*`?run_(\d{3})")
_MSE = re.compile(r"val_action_mse[A-Za-z_]*=([0-9]*\.?[0-9]+(?:[eE][-+]?\d+)?)")
_EPS = re.compile(r"(\d+)\s*eps\b")
_INT_FIELDS = ("rollouts", "seed", "successes", "faults", "stops", "timeouts")
_FLOAT_FIELDS = ("success_rate", "ci95_lo", "ci95_hi", "grasp_rate", "lift_rate")


def _setting(settings, key: str) -> Path:
    """A path from ui/settings.py's Settings (fields, then Settings.extra), a plain dict, or the default."""
    extra = getattr(settings, "extra", None) or {}
    for name in _ALIASES[key]:
        v = settings.get(name) if isinstance(settings, dict) else getattr(settings, name, None) or extra.get(name)
        if v:
            return Path(v).expanduser()
    return Path(DEFAULTS[key])


def _mtime(p: Path) -> float:
    try:
        return p.stat().st_mtime
    except OSError:
        return 0.0


def _read_json(p: Path) -> dict:
    try:
        d = json.loads(p.read_text())
        return d if isinstance(d, dict) else {}
    except (OSError, ValueError):
        return {}


def _clean(s: str) -> str:
    return re.sub(r"\s+", " ", s.replace("`", "").replace("**", "")).strip()


def kind_of(train_config: str | None) -> str:
    return {"umi_policy_dunetp": "prompting", "umi_policy_dunet": "plain"}.get(train_config or "", "unknown")


# ----------------------------------------------------------------------------------------- run folder -> checkpoint

def _driver_resolve():
    """The eval driver's resolver, or None if it cannot be imported (then _resolve_checkpoint uses a copy)."""
    try:
        from sim_deploy.eval.target import resolve
        return resolve
    except Exception:                                    # noqa: BLE001 - another agent may be editing eval/
        return None


def _resolve_checkpoint(run: Path, resolve) -> tuple[str | None, str | None]:
    """(checkpoint, error). Same rule as sim_deploy/eval/target.py: the last of sorted(checkpoints/*.ckpt)."""
    if resolve is not None:
        try:
            return resolve(str(run)).checkpoint, None
        except FileNotFoundError as e:
            return None, str(e)
        except Exception as e:                           # noqa: BLE001
            return None, f"{type(e).__name__}: {e}"
    ckpts = sorted((run / "checkpoints").glob("*.ckpt"))          # copy of target.resolve's rule
    return (str(ckpts[-1]), None) if ckpts else (None, f"{run}: no checkpoints/*.ckpt (fetch it with train/fetch.sh)")


def _train_config(run: Path, job: dict, meta: dict) -> tuple[str | None, str | None]:
    """(TRAIN_CONFIG, where it was read)."""
    cfg = _read_json(run / "run_config" / "job_info.json").get("config_name")
    if cfg:
        return cfg, "run_config/job_info.json"
    env = run / "run_config" / "submit_config.env"
    if env.is_file():
        try:
            lines = re.findall(r'^\s*TRAIN_CONFIG\s*=\s*"?([^"\s#]*)', env.read_text(), re.M)
        except OSError:
            lines = []
        return (lines[-1], "run_config/submit_config.env") if lines and lines[-1] else \
            ("umi_policy_dunetp", "run_config/submit_config.env (no TRAIN_CONFIG line = behavior prompting)")
    readme = run / "README.md"
    if readme.is_file():
        try:
            m = re.search(r"^\|\s*Policy config\s*\|\s*`?([\w.-]+)`?", readme.read_text(), re.M)
        except OSError:
            m = None
        if m:
            return m.group(1), "README.md"
    for name, d in (("job.json", job), ("train_meta.json", meta)):
        for k in ("config_name", "train_config", "TRAIN_CONFIG"):
            if d.get(k):
                return str(d[k]), name
    return None, None


def _val_mse(ckpt: str | None, run: Path) -> float | None:
    if not ckpt:
        return None
    m = _MSE.search(Path(ckpt).name)
    if m:
        return float(m.group(1))
    try:
        rows = json.loads((run / "checkpoints" / "checkpoints.json").read_text())
        for r in rows:
            if r.get("file") == Path(ckpt).name and r.get("val_action_mse") is not None:
                return float(r["val_action_mse"])
    except (OSError, ValueError, TypeError, AttributeError):
        pass
    return None


# ---------------------------------------------------------------------------------------------------- RUNS.md

def parse_runs_md(text: str) -> dict:
    """{run name or 'run_NNN' (short): label}. Tables first (they are the authoritative listing), then prose."""
    table_labels: dict = {}
    prose_labels: dict = {}
    section = para = ""
    lines = text.splitlines()
    i = 0
    while i < len(lines):
        line = lines[i]
        h = re.match(r"^#+\s*(.+)", line)
        if h:
            section, para = _clean(h.group(1)), ""
            i += 1
            continue
        if line.lstrip().startswith("|"):
            block = []
            while i < len(lines) and lines[i].lstrip().startswith("|"):
                block.append([_cell.strip() for _cell in lines[i].strip().strip("|").split("|")])
                i += 1
            if len(block) >= 2:
                _table_labels(section, block[0], [r for r in block[1:] if not _is_rule(r)], table_labels)
            continue
        if not line.strip():
            para = ""
        else:
            body = re.sub(r"^\s*[-*]\s*", "", line)
            if line.lstrip()[:1] in "-*" or not para:               # a new bullet / paragraph: its lead-in
                c, r = body.find(":"), body.find("run_")
                para = _clean(body[:c]) if 0 <= c and (r < 0 or c < r) else ""
            _prose_labels(section, para, line, prose_labels)
        i += 1
    out = dict(prose_labels)
    out.update(table_labels)
    return out


def _is_rule(row: list) -> bool:
    return all(re.fullmatch(r":?-{2,}:?", c.replace(" ", "")) or not c for c in row)


def _section_short(section: str) -> str:
    return _clean(re.split(r"[:(]", section)[0]) if section else ""


def _runs_in(cell: str) -> list:
    """Run names in a cell: full names, short run_NNN, and 'run_A to run_B' ranges (short names)."""
    cell = _clean(cell)
    found = list(_RUN_FULL.findall(cell))
    for a, b in _RUN_RANGE.findall(cell):
        found += [f"run_{n:03d}" for n in range(int(a), int(b) + 1)]
    stripped = _RUN_FULL.sub("", cell)
    found += [f"run_{n}" for n in _RUN_SHORT.findall(stripped)]
    return list(dict.fromkeys(found))


def _table_labels(section: str, header: list, rows: list, out: dict) -> None:
    run_cols = [j for j, h in enumerate(header) if re.search(r"run|folder", h, re.I)]
    if not run_cols:
        return
    for row in rows:
        info = []
        for j, c in enumerate(row):
            hj = header[j] if j < len(header) else ""
            if j in run_cols or ".ckpt" in c or re.search(r"checkpoint|on laptop", hj, re.I):
                continue
            if c.lower() in ("", "yes", "no", "not yet"):
                continue
            info.append(f"{_clean(c)} eps" if re.fullmatch(r"\d+", _clean(c)) and re.search(r"episode", hj, re.I)
                        else _clean(c))
        for j in run_cols:
            if j >= len(row):
                continue
            col = _clean(header[j])
            extra = ["smoke"] if re.search(r"smoke", col, re.I) else []
            label = " · ".join(p for p in [_section_short(section), *info, *extra] if p)
            for name in _runs_in(row[j]):
                out.setdefault(name, label)


def _prose_labels(section: str, prefix: str, line: str, out: dict) -> None:
    """Run names in a prose line, labelled with the section, the paragraph's lead-in and the parenthesis after."""
    for m in _RUN_FULL.finditer(line):
        paren = re.match(r"\(([^)]*)\)", line[m.end():].lstrip("` "))
        p = _clean(paren.group(1)) if paren else ""
        if re.fullmatch(r"\d+", p):
            p += " eps"
        out.setdefault(m.group(0), " · ".join(x for x in (_section_short(section), prefix, p) if x))


def _label_for(run_name: str, labels: dict) -> str | None:
    if run_name in labels:
        return labels[run_name]
    m = re.match(r"(run_\d{3})", run_name)
    return labels.get(m.group(1)) if m else None


# ---------------------------------------------------------------------------------------------------- the catalog

class Catalog:
    def __init__(self, settings=None) -> None:
        self.data_root = _setting(settings, "data_root")
        self.datasets_root = _setting(settings, "datasets_root")
        self.sim_evals = _setting(settings, "sim_evals")
        self.cache_dir = _setting(settings, "cache_dir")
        self.model_json = _setting(settings, "model_json")
        self._lock = threading.RLock()
        self._pol_sig = None
        self._pol = None
        self._rep_sig = None
        self._rep = None
        self._ev_sig = None
        self._ev = None
        self._ds_sig = None
        self._ds = {}
        self._disk = None                                 # .cache/replays.json

    # ---- public

    def policies(self, replays: bool = True) -> dict:
        """replays=False skips the replay scan (CLI --no-replays); the server uses the default."""
        with self._lock:
            evals = self.evals()["evals"]
            sig = (self._policy_signature(), self._ev_sig)
            if sig != self._pol_sig:
                self._pol = self._build_policies(evals)
                self._pol_sig = sig
            latest: dict = {}
            for row in evals:                             # newest first
                latest.setdefault(f"{row.get('task')}/{row.get('run_folder')}", row)
            # eval-driver names (target.py): _replay/<zarr's grandparent>/<dataset>_epNNN, _baseline/hold_still
            reps = [dict(r, last_eval=latest.get(f"_replay/{r['group']}/{r['dataset']}_ep{r['index']:03d}"))
                    for r in (self.replays() if replays else [])]
            return {"policies": self._pol, "replays": reps, "builtins": ["hold_still"],
                    "builtin_last_eval": {"hold_still": latest.get("_baseline/hold_still")}}

    def evals(self) -> dict:
        with self._lock:
            index = self.sim_evals / "index.csv"
            sig = (str(index), _mtime(index), index.stat().st_size if index.exists() else 0)
            if sig != self._ev_sig:
                self._ev = self._read_index(index)
                self._ev_sig = sig
            return {"evals": self._ev}

    def replays(self) -> list:
        with self._lock:
            zarrs = self._zarrs()
            sig = tuple((str(p), st) for p, st in zarrs) + (self._scene_key(),)
            if sig != self._rep_sig:
                self._rep = self._build_replays(zarrs)
                self._rep_sig = sig
            return self._rep

    def warm(self) -> None:
        """Fill every cache (the first replay pass reads each episode once; call it in a thread at startup)."""
        self.policies()

    # ---- policies

    def _run_dirs(self) -> list:
        out = []
        if not self.data_root.is_dir():
            return out
        for task in sorted(self.data_root.iterdir()):
            pol = task / "policy"
            if not pol.is_dir():
                continue
            out += [r for r in sorted(pol.iterdir()) if r.is_dir() and r.name.startswith("run_")]
        return out

    def _policy_signature(self) -> tuple:
        parts = [str(self.data_root), _mtime(self.data_root)]
        if self.data_root.is_dir():
            for task in sorted(self.data_root.iterdir()):
                parts += [task.name, _mtime(task / "policy"), _mtime(task / "RUNS.md")]
        for r in self._run_dirs():
            parts += [r.name, _mtime(r), _mtime(r / "checkpoints"), _mtime(r / "job.json")]
        parts.append(self._datasets_signature())
        return tuple(parts)

    def _datasets_signature(self) -> tuple:
        return tuple((str(p), st) for p, st in self._zarrs())

    def _dataset_index(self) -> dict:
        """{dataset name: (zarr path, n_episodes)} for every package under datasets_root (from the dataset cache)."""
        sig = self._datasets_signature()
        if sig != self._ds_sig:
            ds = {}
            for p, ent in self._dataset_entries(self._zarrs()):
                ds[ent["dataset"]] = (str(p), ent.get("n_episodes"))
            self._ds, self._ds_sig = ds, sig
        return self._ds

    def _build_policies(self, evals: list) -> list:
        resolve = _driver_resolve()
        ds_index = self._dataset_index()
        labels_by_task: dict = {}
        latest: dict = {}
        for row in evals:                                 # newest first
            latest.setdefault(f"{row.get('task')}/{row.get('run_folder')}", row)
        rows = []
        for run in self._run_dirs():
            task = run.parent.parent.name                 # target.py: <task>/policy/<run>
            if task not in labels_by_task:
                try:
                    labels_by_task[task] = parse_runs_md((run.parent.parent / "RUNS.md").read_text())
                except OSError:
                    labels_by_task[task] = {}
            job, meta = _read_json(run / "job.json"), _read_json(run / "train_meta.json")
            ckpt, err = _resolve_checkpoint(run, resolve)
            cfg, cfg_src = _train_config(run, job, meta)
            dataset = meta.get("dataset") or job.get("dataset") \
                or _read_json(run / "run_config" / "job_info.json").get("dataset") \
                or _read_json(run / "summary.json").get("dataset")
            zarr, n_eps = ds_index.get(dataset, (None, None)) if dataset else (None, None)
            note = meta.get("note") or job.get("note")
            if n_eps is None and note and _EPS.search(note):
                n_eps = int(_EPS.search(note).group(1))
            label = _label_for(run.name, labels_by_task[task]) \
                or (f"{dataset} · {note}" if dataset and note else dataset or note or run.name)
            pid = f"{task}/{run.name}"
            rows.append({
                "id": pid, "task": task, "run_folder": str(run), "kind": kind_of(cfg), "train_config": cfg,
                "dataset": dataset, "episodes": n_eps, "checkpoint": ckpt, "val_mse": _val_mse(ckpt, run),
                "label": label, "last_eval": latest.get(pid),
                # additions to the section 2 shape (the page may ignore them):
                "runnable": ckpt is not None, "not_runnable_reason": None if ckpt else err,
                "smoke": run.name.endswith("_smoke") or meta.get("mode") == "smoke" or job.get("mode") == "smoke",
                "note": note, "train_config_source": cfg_src, "dataset_path": zarr,
                "status": (run / "STATUS").read_text().strip() if (run / "STATUS").is_file() else None,
            })
        return rows

    # ---- replays

    def _zarrs(self) -> list:
        out = []
        root = self.datasets_root
        if not root.is_dir():
            return out
        for dirpath, dirnames, filenames in os.walk(root, followlinks=True):
            dirnames.sort()
            for f in sorted(filenames):
                if f.endswith(".zarr.zip"):
                    p = Path(dirpath) / f
                    try:
                        st = p.stat()
                    except OSError:
                        continue
                    out.append((p, (st.st_size, st.st_mtime)))
        return out

    def _scene_key(self) -> str:
        """What the replay placement depends on besides the dataset: model.json + station file + placement code."""
        model = self.model_json
        h = hashlib.sha256()
        try:
            m = json.loads(model.read_text())
            h.update(str(m.get("model_sha256")).encode())
            h.update(str(m.get("home", {}).get("O_T_tcp")).encode())
            h.update((REPO / m["station_file"]).read_bytes())
        except (OSError, ValueError, KeyError):
            h.update(b"no model")
        for f in (ROOT / "scene" / "placement.py", ROOT / "scene" / "zarr_lowdim.py", ROOT / "scene" / "start_pose.py",
                  ROOT / "stations" / "start_pose.yaml", ROOT / "stations" / "control_sim.yaml"):
            try:
                h.update(f.read_bytes())
            except OSError:
                pass
        return h.hexdigest()

    def _build_replays(self, zarrs: list) -> list:
        rows = []
        for p, ent in self._dataset_entries(zarrs):
            for ep in ent.get("episodes") or []:
                rows.append({"id": f"replay:{p}#{ep['index']}", "dataset": ent["dataset"], "index": ep["index"],
                             "grasp_ok": ep["grasp_ok"],
                             # additions to the section 2 shape:
                             "group": ent["group"], "dataset_path": str(p), "name": ep.get("name"),
                             "height_error_m": ep.get("height_error_m"), "error": ep.get("error")})
        return rows

    # ---- the dataset cache (.cache/replays.json): per zarr its episode count and every episode's replay placement

    def _dataset_entries(self, zarrs: list) -> list:
        """[(zarr path, entry)], computing (and saving) the entries whose file or scene changed."""
        cache_file = self.cache_dir / "replays.json"
        if self._disk is None:
            d = _read_json(cache_file)
            self._disk = d if d.get("schema") == REPLAY_CACHE_SCHEMA else {"schema": REPLAY_CACHE_SCHEMA}
        old = self._disk.get("datasets") or {}
        scene_key = self._scene_key()
        scene = None
        new: dict = {}
        out = []
        for p, (size, mt) in zarrs:
            ent = old.get(str(p))
            if not (ent and ent.get("size") == size and ent.get("mtime") == mt and ent.get("scene_key") == scene_key):
                if scene is None:
                    scene = self._load_scene()
                ent = self._place_dataset(p, size, mt, scene, scene_key)
            new[str(p)] = ent
            out.append((p, ent))
        if new != old:
            self._disk["datasets"] = new
            try:
                self.cache_dir.mkdir(parents=True, exist_ok=True)
                tmp = cache_file.with_suffix(".json.tmp")
                tmp.write_text(json.dumps(self._disk, indent=1))
                tmp.replace(cache_file)
            except OSError as e:
                print(f"catalog: could not write {cache_file}: {e}", file=sys.stderr)
        return out

    def _load_scene(self):
        try:
            from sim_deploy.scene import start_pose as sp
            from sim_deploy.scene.placement import Scene
            # P2: rollouts start at the start pose (--home start) when one is set, so the replay grasp is carried
            # from there; else from the deploy home
            home = "start" if sp.read_q() is not None else "deploy"
            return Scene.load(self.model_json, home=home)
        except Exception as e:                           # noqa: BLE001 - no model built yet, etc.
            return e

    def _place_dataset(self, p: Path, size: int, mt: float, scene, scene_key: str) -> dict:
        """Episode count and every episode's replay placement. The low-dim arrays are copied once to a temporary
        .zarr folder (opening the zip costs ~0.5 s and replay_placement opens it per episode); the placement itself
        is sim_deploy.scene's own replay_placement on that folder, the same reader (zarr_lowdim) either way."""
        import tempfile
        import zipfile

        from sim_deploy.scene import zarr_lowdim as zl

        ent = {"dataset": p.name.split(".zarr")[0], "group": p.parent.parent.name, "size": size, "mtime": mt,
               "scene_key": scene_key, "n_episodes": None, "episodes": []}
        keep = tuple(k + "/" for k in ("meta/episode_ends", "meta/episode_names", zl.POS_KEY, zl.ROT_KEY,
                                        zl.WIDTH_KEY))
        with tempfile.TemporaryDirectory(prefix="catalog_") as tmp:
            low = Path(tmp) / p.name.replace(".zarr.zip", ".zarr")
            try:
                with zipfile.ZipFile(p) as z:
                    for m in z.namelist():
                        if m.startswith(keep) or m.endswith(".zgroup"):
                            z.extract(m, low)
                n = zl.n_episodes(low)
            except Exception as e:                       # noqa: BLE001
                ent["error"] = f"{type(e).__name__}: {e}"
                return ent
            ent["n_episodes"] = n
            for i in range(n):
                ep = {"index": i, "grasp_ok": False, "height_error_m": None}
                if isinstance(scene, Exception):
                    ep["error"] = f"scene: {type(scene).__name__}: {scene}"
                else:
                    try:
                        rp = scene.replay_placement(f"replay:{low}#{i}")
                        ep["height_error_m"] = rp["height_error_m"]
                        ep["name"] = rp.get("episode_name") or None
                        ep["grasp_ok"] = bool(abs(rp["height_error_m"]) < GRASP_OK_M)
                    except Exception as e:               # noqa: BLE001 - e.g. "no grasp in this episode"
                        ep["error"] = f"{type(e).__name__}: {e}"
                ent["episodes"].append(ep)
        return ent

    # ---- evals

    def _read_index(self, index: Path) -> list:
        if not index.is_file():
            return []
        try:
            with open(index, newline="") as f:
                raw = list(csv.DictReader(f))
        except (OSError, csv.Error):
            return []
        rows = []
        for n, r in enumerate(raw):
            row: dict = {k: (v if v != "" else None) for k, v in r.items() if k is not None}
            for k in _INT_FIELDS:
                row[k] = _num(row.get(k), int)
            for k in _FLOAT_FIELDS:
                row[k] = _num(row.get(k), float)
            d = (row.get("eval_dir") or "").strip("/")
            q = urllib.parse.quote(d)
            row["summary_url"] = f"/evals/{q}/summary.html" if d else None
            row["folder_url"] = f"/evals/{q}/" if d else None
            row["_order"] = n
            rows.append(row)
        rows.sort(key=lambda r: (r.get("date") or "", r["_order"]), reverse=True)
        for r in rows:
            del r["_order"]
        return rows


def _num(v, typ):
    if v is None:
        return None
    try:
        return typ(float(v)) if typ is int else typ(v)
    except (TypeError, ValueError):
        return v


# ------------------------------------------------------------------------------------------------------------ CLI

def _table(cat: Catalog, with_replays: bool) -> str:
    out = []
    pol = cat.policies() if with_replays else cat.policies(replays=False)
    by_task: dict = {}
    for p in pol["policies"]:
        by_task.setdefault(p["task"], []).append(p)
    for task, rows in by_task.items():
        out.append(f"\n{task}  ({sum(r['runnable'] for r in rows)} runnable of {len(rows)})")
        out.append(f"  {'run':<26} {'state':<11} {'kind':<9} {'dataset':<22} {'eps':>4} {'val_mse':>9}  last sim  label")
        for r in rows:
            le = r["last_eval"]
            last = f"{le['successes']}/{le['rollouts']}" if le else "-"
            mse = f"{r['val_mse']:.7f}" if r["val_mse"] is not None else "-"
            out.append(f"  {Path(r['run_folder']).name:<26} {'runnable' if r['runnable'] else 'not fetched':<11} "
                       f"{r['kind']:<9} {str(r['dataset'] or '-'):<22} {str(r['episodes'] or '-'):>4} {mse:>9}  "
                       f"{last:<8}  {r['label']}")
    out.append(f"\nbuiltins: {', '.join(pol['builtins'])}")
    if with_replays:
        by_ds: dict = {}
        for r in pol["replays"]:
            by_ds.setdefault((r["group"], r["dataset"]), []).append(r)
        out.append(f"\nreplays: {len(pol['replays'])} episodes in {len(by_ds)} datasets, "
                   f"{sum(r['grasp_ok'] for r in pol['replays'])} with grasp_ok (|height_error| < {GRASP_OK_M} m)")
        for (g, d), rows in by_ds.items():
            ok = [r["index"] for r in rows if r["grasp_ok"]]
            errs = sum(1 for r in rows if r.get("error"))
            out.append(f"  {g}/{d:<22} {len(rows):3d} eps  grasp_ok {len(ok):2d}: {ok}"
                       + (f"  ({errs} errors, e.g. {next(r['error'] for r in rows if r.get('error'))})" if errs else ""))
    ev = cat.evals()["evals"]
    out.append(f"\nevals: {len(ev)} rows in {cat.sim_evals / 'index.csv'}")
    for e in ev[:15]:
        out.append(f"  {e.get('date')}  {e.get('task')}/{e.get('run_folder')}  {e.get('kind')}  "
                   f"{e.get('successes')}/{e.get('rollouts')}  tag {e.get('tag')}  {e.get('summary_url')}")
    return "\n".join(out)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="python -m sim_deploy.ui.catalog", description=__doc__.splitlines()[0])
    ap.add_argument("--json", action="store_true", help="print {policies, replays, builtins, evals} as JSON")
    ap.add_argument("--no-replays", action="store_true", help="skip the replay scan (and its cache)")
    ap.add_argument("--data-root", help=f"default {DEFAULTS['data_root']}")
    ap.add_argument("--datasets-root", help=f"default {DEFAULTS['datasets_root']}")
    ap.add_argument("--sim-evals", help=f"default {DEFAULTS['sim_evals']}")
    a = ap.parse_args(argv)
    cat = Catalog({"data_root": a.data_root, "datasets_root": a.datasets_root, "sim_evals": a.sim_evals})
    t0 = time.monotonic()
    if a.json:
        pol = cat.policies(replays=not a.no_replays)
        print(json.dumps({**pol, **cat.evals()}, indent=1))
    else:
        print(_table(cat, not a.no_replays))
        print(f"\n({time.monotonic() - t0:.2f} s; torch imported: {'torch' in sys.modules})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
