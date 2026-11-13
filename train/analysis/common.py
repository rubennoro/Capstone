"""Shared loaders and plot helpers for train/analysis/*.py.

Everything here tolerates missing files: a fetched run may be half-packaged (failed / timeout job), an old
packaged run in results/<task>/ has no checkpoints.json, train_meta.json or eval/, and a task folder may not have
a package yet. Loaders return None / empty containers instead of raising.

Layouts (train contract):
  <data root>/<folder>/                      mirror of the Jetson task folder (task.yaml, episodes.csv, episode_N/...)
  <data root>/<folder>/package/<DATASET>/    <DATASET>.zarr.zip + package.json
  <data root>/<folder>/policy/<run>/         fetched run dir: summary.json epochs.csv history.csv plots/ videos/
                                             logs/ run_config/ checkpoints/{top1_*.ckpt,checkpoints.json} wandb/
                                             train_meta.json STATUS job.json eval/<tag>/{eval.json,videos/,eval_job.json}
  <data root>/<folder>/analysis/<script>/    default output of the folder-level scripts
  <run>/analysis/                            default output of the run-level scripts

Plot style: the palette and axes helpers come from scripts/wandb_offline_to_csv.py (the user's plots), imported
through sys.path. That module imports wandb at the top; when wandb is not installed (the laptop's base python) it is
imported with stub wandb modules so its plotting functions still work (read_history then is unavailable).
"""
from __future__ import annotations

import argparse
import csv
import importlib
import json
import math
import os
import re
import sys
import types
from dataclasses import dataclass, field
from pathlib import Path

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

ANALYSIS_DIR = Path(__file__).resolve().parent
TRAIN_DIR = ANALYSIS_DIR.parent
REPO = TRAIN_DIR.parent
DEFAULT_DATA_ROOT = REPO / 'data'
SCRIPTS_DIR = REPO / 'scripts'

# Metric keys as logged by train_policy_workspace.py (and written to epochs.csv by package_run.py)
VAL_MSE = 'val/action_mse_error/all'
TRAIN_MSE = 'train/action_mse_error/all'
VAL_LOSS = 'val_loss'
TRAIN_LOSS = 'train_loss_epoch_average'
COMPONENTS = [('right_pos', 'position'), ('right_rot', 'rotation'), ('right_gripper_width', 'gripper width')]

CKPT_RE = re.compile(r'epoch=(\d+)-val_action_mse_error_all=([0-9.eE+-]+?)\.ckpt$')
CKPT_RANK_RE = re.compile(r'^top(\d+)_')
ZARR_HINT = ('zarr is not installed in this python. Install it into a venv (not the base python):\n'
             '  python3 -m venv --system-site-packages ~/envs/analysis && '
             '~/envs/analysis/bin/pip install zarr==2.18.3 numcodecs==0.13.1 imagecodecs\n'
             'then run this script with ~/envs/analysis/bin/python.')


# --------------------------------------------------------------------------------------------- style

def _load_wandb_tools():
    """scripts/wandb_offline_to_csv.py as a module, and whether wandb itself is importable."""
    if str(SCRIPTS_DIR) not in sys.path:
        sys.path.append(str(SCRIPTS_DIR))
    try:
        import wandb_offline_to_csv as mod  # noqa: F401
        return mod, True
    except ImportError as e:
        if 'wandb' not in str(e):
            return None, False
    except Exception:  # noqa: BLE001 - any other import problem: fall back to our own style constants
        return None, False
    stub_names = ['wandb', 'wandb.proto', 'wandb.proto.wandb_internal_pb2', 'wandb.sdk', 'wandb.sdk.internal',
                  'wandb.sdk.internal.datastore']
    added = []
    for name in stub_names:
        if name not in sys.modules:
            sys.modules[name] = types.ModuleType(name)
            added.append(name)
    sys.modules['wandb'].proto = sys.modules['wandb.proto']
    sys.modules['wandb.proto'].wandb_internal_pb2 = sys.modules['wandb.proto.wandb_internal_pb2']
    sys.modules['wandb'].sdk = sys.modules['wandb.sdk']
    sys.modules['wandb.sdk'].internal = sys.modules['wandb.sdk.internal']
    sys.modules['wandb.sdk.internal'].datastore = sys.modules['wandb.sdk.internal.datastore']
    try:
        sys.modules.pop('wandb_offline_to_csv', None)
        mod = importlib.import_module('wandb_offline_to_csv')
    except Exception:  # noqa: BLE001
        mod = None
    finally:
        for name in added:
            sys.modules.pop(name, None)
    return mod, False


WANDB_TOOLS, HAS_WANDB = _load_wandb_tools()

if WANDB_TOOLS is not None:
    TRAIN_COLOR, VAL_COLOR, THIRD_COLOR = WANDB_TOOLS.TRAIN_COLOR, WANDB_TOOLS.VAL_COLOR, WANDB_TOOLS.THIRD_COLOR
    SURFACE, TEXT_PRIMARY = WANDB_TOOLS.SURFACE, WANDB_TOOLS.TEXT_PRIMARY
    TEXT_SECONDARY, GRID = WANDB_TOOLS.TEXT_SECONDARY, WANDB_TOOLS.GRID
    MAX_EPOCH_TICKS = WANDB_TOOLS.MAX_EPOCH_TICKS
else:  # same values as scripts/wandb_offline_to_csv.py
    TRAIN_COLOR, VAL_COLOR, THIRD_COLOR = '#2a78d6', '#eb6834', '#1baf7a'
    SURFACE, TEXT_PRIMARY, TEXT_SECONDARY, GRID = '#fcfcfb', '#0b0b0b', '#52514e', '#e4e3dd'
    MAX_EPOCH_TICKS = 20

# Categorical slots in fixed order (the first three are the train / val / third colors above). One per run.
SERIES_COLORS = [TRAIN_COLOR, VAL_COLOR, THIRD_COLOR, '#eda100', '#e87ba4', '#008300', '#6250d6', '#e34948']
OVERFLOW_COLOR = '#a9a8a1'  # runs past the 8th slot: gray, listed in the legend
SPLIT_COLORS = {'train': TRAIN_COLOR, 'val': VAL_COLOR, 'new': THIRD_COLOR}


def style_axes(ax, title, xlabel, ylabel):
    if WANDB_TOOLS is not None:
        return WANDB_TOOLS.style_axes(ax, title, xlabel, ylabel)
    ax.set_facecolor(SURFACE)
    ax.set_title(title, loc='left', color=TEXT_PRIMARY, fontsize=11, fontweight='bold')
    ax.set_xlabel(xlabel, color=TEXT_SECONDARY)
    ax.set_ylabel(ylabel, color=TEXT_SECONDARY)
    ax.tick_params(colors=TEXT_SECONDARY, labelsize=9)
    ax.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ('top', 'right'):
        ax.spines[side].set_visible(False)
    for side in ('left', 'bottom'):
        ax.spines[side].set_color(GRID)


def new_figure(ncols=1, width=7.5, height=4.0, nrows=1):
    fig, axes = plt.subplots(nrows, ncols, figsize=(width, height), facecolor=SURFACE, squeeze=False)
    return fig, (axes[0] if nrows == 1 else axes)


def save(fig, path, quiet=False, layout=True):
    if layout:
        fig.tight_layout()
    fig.savefig(path, dpi=150, facecolor=SURFACE)
    plt.close(fig)
    if not quiet:
        print('wrote', path)
    return Path(path)


def label_point(ax, x, y, text, color_hex, dy=0, dx=8):
    """Direct label: a colored marker carries identity, the text stays in ink."""
    ax.plot([x], [y], marker='o', markersize=8, color=color_hex, markeredgecolor=SURFACE, markeredgewidth=2,
            zorder=5)
    ax.annotate(text, (x, y), xytext=(dx, dy), textcoords='offset points', va='center', fontsize=9,
                color=TEXT_PRIMARY)


def epoch_line_style(n_points):
    if n_points <= MAX_EPOCH_TICKS:
        return dict(linewidth=2, marker='o', markersize=8, markeredgecolor=SURFACE, markeredgewidth=2)
    return dict(linewidth=2)


def legend(ax, **kw):
    ax.legend(frameon=False, labelcolor=TEXT_PRIMARY, fontsize=9, **kw)


def series_color(i):
    return SERIES_COLORS[i] if i < len(SERIES_COLORS) else OVERFLOW_COLOR


# --------------------------------------------------------------------------------------------- small utils

def read_json(path, default=None):
    try:
        with open(path) as f:
            return json.load(f)
    except (OSError, ValueError):
        return default


def read_text(path, default=None):
    try:
        return Path(path).read_text().strip()
    except OSError:
        return default


def read_csv(path):
    """DataFrame or None (missing / empty / unreadable)."""
    try:
        df = pd.read_csv(path)
    except (OSError, ValueError, pd.errors.EmptyDataError):
        return None
    return df if len(df.columns) else None


def read_csv_rows(path):
    try:
        with open(path, newline='') as f:
            return list(csv.DictReader(f))
    except OSError:
        return []


def read_yaml(path, default=None):
    try:
        import yaml
        with open(path) as f:
            out = yaml.safe_load(f)
        return out if out is not None else default
    except Exception:  # noqa: BLE001 - missing file, bad yaml, no pyyaml
        return default


def isnum(x):
    return isinstance(x, (int, float, np.integer, np.floating)) and not (isinstance(x, float) and math.isnan(x)) \
        and not isinstance(x, bool)


def fmt(x, digits=4, sci_below=1e-3):
    """Numbers for tables: '-' for missing, 3 significant digits in scientific form for small values."""
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return '-'
    if isinstance(x, (bool, np.bool_)):
        return str(bool(x))
    if isinstance(x, (int, np.integer)):
        return f'{int(x):,}'
    if isinstance(x, (float, np.floating)):
        x = float(x)
        if x != 0 and abs(x) < sci_below:
            return f'{x:.3g}' if abs(x) >= 1e-4 else f'{x:.2e}'
        return f'{x:.{digits}g}' if abs(x) < 10 ** digits else f'{x:,.0f}'
    return str(x)


def fmt_minutes(minutes):
    if not isnum(minutes):
        return '-'
    h, m = divmod(int(round(minutes)), 60)
    return f'{h} h {m:02d} min' if h else f'{m} min'


def fmt_bytes(n):
    if not isnum(n):
        return '-'
    n = float(n)
    for unit in ('B', 'KB', 'MB', 'GB', 'TB'):
        if n < 1024 or unit == 'TB':
            return f'{n:.1f} {unit}' if unit != 'B' else f'{int(n)} B'
        n /= 1024


def md_table(header, rows):
    def cell(c):
        return str(c).replace('|', '\\|').replace('\n', ' ')
    lines = ['| ' + ' | '.join(cell(h) for h in header) + ' |', '|' + '|'.join('---' for _ in header) + '|']
    lines += ['| ' + ' | '.join(cell(c) for c in r) + ' |' for r in rows]
    return '\n'.join(lines)


def text_table(header, rows):
    rows = [[str(c) for c in r] for r in rows]
    w = [max([len(str(h))] + [len(r[i]) for r in rows]) for i, h in enumerate(header)]
    out = ['  '.join(str(h).ljust(w[i]) for i, h in enumerate(header)).rstrip(),
           '  '.join('-' * x for x in w)]
    out += ['  '.join(c.ljust(w[i]) for i, c in enumerate(r)).rstrip() for r in rows]
    return '\n'.join(out)


def rel(path, start):
    """Link for markdown: relative when close by (the report sits in or next to the run), else absolute."""
    r = Path(os.path.relpath(path, start)).as_posix()
    return r if not r.startswith('../../../') else Path(path).resolve().as_posix()


def write_text(path, text):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text if text.endswith('\n') else text + '\n')
    print('wrote', path)
    return path


# --------------------------------------------------------------------------------------------- locating things

def default_data_root() -> Path:
    """$LOCAL_DATA_ROOT, else LOCAL_DATA_ROOT of train/config.local.env / config.env, else <repo>/data."""
    if os.environ.get('LOCAL_DATA_ROOT'):
        return Path(os.environ['LOCAL_DATA_ROOT']).expanduser()
    for name in ('config.local.env', 'config.env'):
        try:
            text = (TRAIN_DIR / name).read_text()
        except OSError:
            continue
        for line in text.splitlines():
            m = re.match(r'\s*(?:export\s+)?LOCAL_DATA_ROOT=["\']?([^"\'#]*)["\']?', line)
            if m and m.group(1).strip() and '<' not in m.group(1):
                v = re.sub(r'\$\{TRAIN_DIR(:-[^}]*)?\}|\$TRAIN_DIR\b', str(TRAIN_DIR), m.group(1).strip())
                v = os.path.expanduser(os.path.expandvars(v))
                if '$' not in v:
                    return Path(v).resolve()
    return DEFAULT_DATA_ROOT


def add_data_root_arg(parser: argparse.ArgumentParser):
    d = default_data_root()
    parser.add_argument('--data-root', default=str(d),
                        help=f'local data root holding <folder>/ dirs (default {d}: $LOCAL_DATA_ROOT, else '
                             'LOCAL_DATA_ROOT of train/config.local.env or config.env, else <repo>/data)')


def looks_like_run_dir(p) -> bool:
    p = Path(p)
    if not p.is_dir():
        return False
    markers = ['summary.json', 'epochs.csv', 'history.csv', 'train_meta.json', 'STATUS', 'job.json']
    return any((p / m).exists() for m in markers) or (p / 'checkpoints').is_dir() or (p / 'eval').is_dir()


class LocateError(Exception):
    pass


def folder_dir(data_root, folder) -> Path:
    d = Path(data_root).expanduser() / folder
    if not d.is_dir():
        avail = sorted(x.name for x in Path(data_root).expanduser().glob('*') if x.is_dir()) \
            if Path(data_root).expanduser().is_dir() else []
        raise LocateError(f'no folder {d}' + (f' (available: {", ".join(avail)})' if avail else
                                              ' (pull it first: train/pull.sh FOLDER)'))
    return d


def run_sort_key(p: Path):
    m = re.match(r'run_(\d+)', p.name)
    return (int(m.group(1)) if m else 10 ** 9, p.name)


def list_runs(parent) -> list[Path]:
    parent = Path(parent)
    if not parent.is_dir():
        return []
    return sorted((p for p in parent.iterdir() if p.is_dir() and not p.name.startswith('.')
                   and looks_like_run_dir(p)), key=run_sort_key)


def resolve_run(data_root, folder, run=None) -> Path:
    """A run dir from (FOLDER, RUN) under the data root, or a direct path given as FOLDER or RUN.

    RUN may be a run name under <folder>/policy/, a unique prefix such as 'run_003', or omitted (= newest run)."""
    for cand in (run, folder):
        if cand and Path(cand).expanduser().is_dir() and looks_like_run_dir(Path(cand).expanduser()) \
                and (os.sep in str(cand) or cand.startswith('.')):
            return Path(cand).expanduser().resolve()
    if not folder:
        raise LocateError('give FOLDER [RUN] or a path to a run dir')
    fdir = folder_dir(data_root, folder)
    policy = fdir / 'policy'
    runs = list_runs(policy)
    if not runs:
        raise LocateError(f'no runs in {policy} (fetch one: train/fetch.sh {folder} RUN)')
    if run is None:
        return runs[-1]
    exact = [r for r in runs if r.name == run]
    if exact:
        return exact[0]
    pref = [r for r in runs if r.name.startswith(run)]
    if len(pref) == 1:
        return pref[0]
    names = ', '.join(r.name for r in runs)
    raise LocateError(f'run {run!r} {"is ambiguous" if pref else "not found"} in {policy} (runs: {names})')


def folder_of_run(run_dir: Path):
    """(data_root/<folder>) if run_dir is <folder>/policy/<run>, else None."""
    run_dir = Path(run_dir)
    if run_dir.parent.name == 'policy':
        return run_dir.parent.parent
    return None


# --------------------------------------------------------------------------------------------- the run

@dataclass
class Checkpoint:
    rank: int | None
    file: str
    epoch: int | None
    val_action_mse: float | None
    size_bytes: int | None
    present: bool


@dataclass
class Run:
    path: Path
    name: str
    summary: dict = field(default_factory=dict)
    epochs: pd.DataFrame | None = None
    train_meta: dict = field(default_factory=dict)
    job: dict = field(default_factory=dict)
    status_file: str | None = None
    checkpoints: list = field(default_factory=list)
    checkpoints_source: str = 'none'
    evals: dict = field(default_factory=dict)        # tag -> eval.json dict
    eval_jobs: dict = field(default_factory=dict)    # tag -> eval_job.json dict

    @property
    def history_path(self):
        return self.path / 'history.csv'

    def history(self):
        return read_csv(self.history_path)

    @property
    def status(self):
        return self.status_file or self.summary.get('status') or 'unknown'

    @property
    def mode(self):
        return self.train_meta.get('mode') or self.summary.get('mode') or \
            ('smoke' if self.name.endswith('_smoke') else None)

    @property
    def dataset(self):
        return self.train_meta.get('dataset') or self.summary.get('dataset') or self.job.get('dataset')

    @property
    def label(self):
        return self.train_meta.get('label') or self.summary.get('task') or self.job.get('label')

    @property
    def folder(self):
        f = folder_of_run(self.path)
        return self.train_meta.get('folder') or self.job.get('folder') or (f.name if f else None)

    @property
    def epochs_configured(self):
        return self.summary.get('num_epochs_configured') or self.train_meta.get('epochs')

    @property
    def epochs_completed(self):
        if self.summary.get('epochs_completed') is not None:
            return self.summary['epochs_completed']
        if self.epochs is not None and 'epoch' in self.epochs and len(self.epochs):
            return int(self.epochs['epoch'].max()) + 1
        return None

    def series(self, key):
        """(epochs, values) of one epochs.csv column, NaN rows dropped; empty arrays if missing."""
        if self.epochs is None or key not in self.epochs or 'epoch' not in self.epochs:
            return np.array([], dtype=int), np.array([], dtype=float)
        df = self.epochs[['epoch', key]].dropna()
        return df['epoch'].to_numpy(dtype=int), df[key].to_numpy(dtype=float)

    def best(self, key=VAL_MSE):
        """(epoch, value) of the minimum of an epochs.csv column (falls back to summary.best_checkpoint)."""
        ep, v = self.series(key)
        if len(v):
            i = int(np.argmin(v))
            return int(ep[i]), float(v[i])
        if key == VAL_MSE and self.summary.get('best_checkpoint'):
            b = self.summary['best_checkpoint']
            return b.get('epoch'), b.get('value')
        return None, None

    def final(self, key=VAL_MSE):
        """(epoch, value) of the last non-NaN value (falls back to summary.final)."""
        ep, v = self.series(key)
        if len(v):
            return int(ep[-1]), float(v[-1])
        val = (self.summary.get('final') or {}).get(key)
        return (self.epochs_completed - 1 if self.epochs_completed else None), val

    def row_at(self, epoch):
        if self.epochs is None or 'epoch' not in self.epochs:
            return {}
        df = self.epochs[self.epochs['epoch'] == epoch]
        return {} if not len(df) else {k: v for k, v in df.iloc[-1].items() if not pd.isna(v)}

    def plots(self):
        return sorted((self.path / 'plots').glob('*.png')) if (self.path / 'plots').is_dir() else []

    def videos(self):
        out = sorted((self.path / 'videos').rglob('*.mp4')) if (self.path / 'videos').is_dir() else []
        if (self.path / 'eval').is_dir():
            out += sorted((self.path / 'eval').rglob('*.mp4'))
        return out

    def wandb_dirs(self):
        w = self.path / 'wandb'
        if not w.is_dir():
            return []
        dirs = sorted(w.glob('offline-run-*')) + sorted(w.glob('run-*'))
        if not dirs and list(w.glob('run-*.wandb')):
            dirs = [w]
        return [d for d in dirs if d.is_dir()]

    def gpu_csv(self):
        for c in (self.path / 'logs' / 'gpu_memory.csv', *sorted((self.path / 'logs').glob('*gpu*.csv'))):
            if c.exists():
                return c
        return None


def parse_ckpt_name(name):
    """(rank, epoch, val_mse) from 'top1_epoch=0050-val_action_mse_error_all=0.0001481.ckpt' (rank None if no topN_)."""
    m = CKPT_RE.search(name)
    r = CKPT_RANK_RE.match(name)
    return (int(r.group(1)) if r else None,
            int(m.group(1)) if m else None,
            float(m.group(2)) if m else None)


def load_checkpoints(run_dir: Path):
    """checkpoints/checkpoints.json entries; otherwise parsed from the *.ckpt names in checkpoints/."""
    cdir = Path(run_dir) / 'checkpoints'
    data = read_json(cdir / 'checkpoints.json')
    out = []
    if isinstance(data, dict):
        data = data.get('checkpoints') or []
    if isinstance(data, list) and data:
        for i, e in enumerate(data):
            if not isinstance(e, dict):
                continue
            f = e.get('file') or e.get('name') or ''
            r, ep, v = parse_ckpt_name(Path(f).name)
            p = cdir / Path(f).name
            size = e.get('size_bytes')
            if size is None and p.exists():
                size = p.stat().st_size
            out.append(Checkpoint(rank=e.get('rank', r if r is not None else i + 1), file=Path(f).name,
                                  epoch=e.get('epoch', ep), val_action_mse=e.get('val_action_mse', v),
                                  size_bytes=size, present=p.exists()))
        return sorted(out, key=lambda c: (c.rank if c.rank is not None else 99)), 'checkpoints.json'
    files = sorted(cdir.glob('*.ckpt')) if cdir.is_dir() else []
    for p in files:
        r, ep, v = parse_ckpt_name(p.name)
        out.append(Checkpoint(rank=r, file=p.name, epoch=ep, val_action_mse=v, size_bytes=p.stat().st_size,
                              present=True))
    out.sort(key=lambda c: (c.val_action_mse if c.val_action_mse is not None else float('inf')))
    for i, c in enumerate(out):
        if c.rank is None:
            c.rank = i + 1
    return out, ('file names' if out else 'none')


def eval_tag_sort_key(tag):
    m = re.match(r'top(\d+)$', tag)
    return (0, int(m.group(1)), tag) if m else (1, 0, tag)


def load_evals(run_dir: Path):
    evals, jobs = {}, {}
    edir = Path(run_dir) / 'eval'
    if not edir.is_dir():
        return evals, jobs
    for d in sorted(edir.iterdir(), key=lambda p: eval_tag_sort_key(p.name)):
        if not d.is_dir():
            continue
        e = read_json(d / 'eval.json')
        if isinstance(e, dict):
            evals[d.name] = e
        j = read_json(d / 'eval_job.json')
        if isinstance(j, dict):
            jobs[d.name] = j
    return evals, jobs


def load_run(run_dir) -> Run:
    run_dir = Path(run_dir)
    summary = read_json(run_dir / 'summary.json', {}) or {}
    epochs = read_csv(run_dir / 'epochs.csv')
    if epochs is not None and 'epoch' in epochs:
        epochs = epochs.sort_values('epoch').reset_index(drop=True)
    ckpts, source = load_checkpoints(run_dir)
    evals, jobs = load_evals(run_dir)
    status = read_text(run_dir / 'STATUS')
    return Run(path=run_dir, name=summary.get('run') or run_dir.name, summary=summary, epochs=epochs,
               train_meta=read_json(run_dir / 'train_meta.json', {}) or {},
               job=read_json(run_dir / 'job.json', {}) or {},
               status_file=status.splitlines()[0].strip() if status else None,
               checkpoints=ckpts, checkpoints_source=source, evals=evals, eval_jobs=jobs)


# --------------------------------------------------------------------------------------------- datasets

def find_package(folder_path: Path, dataset=None):
    """(package dir, package.json dict) under <folder>/package/; the named one, or the newest if not named."""
    pdir = Path(folder_path) / 'package'
    if dataset:
        d = pdir / dataset
        return (d, read_json(d / 'package.json', {}) or {}) if d.is_dir() else (None, {})
    cands = [d for d in pdir.glob('*') if d.is_dir()] if pdir.is_dir() else []
    if not cands:
        return None, {}
    cands.sort(key=lambda d: (read_json(d / 'package.json', {}) or {}).get('created') or '')
    return cands[-1], read_json(cands[-1] / 'package.json', {}) or {}


def package_for_run(run: Run, data_root=None):
    """package.json of the run's dataset if it is in the local data tree (data/<folder>/package/<dataset>/)."""
    candidates = []
    f = folder_of_run(run.path)
    if f is not None:
        candidates.append(f)
    if data_root and run.folder:
        candidates.append(Path(data_root) / run.folder)
    for c in candidates:
        if run.dataset:
            d, pj = find_package(c, run.dataset)
            if pj:
                return pj
    return {}


def val_count(n_episodes, val_ratio=0.05):
    """Number of val demos for one task, as behavior_prompting's get_val_mask computes it (Python round)."""
    if not n_episodes:
        return 0
    return min(max(1, round(n_episodes * val_ratio)), n_episodes)


def predicted_val_indices(n_episodes, val_ratio=0.05, seed=42):
    """Episode indices get_val_mask(sample_type='task') picks for ONE label whose episodes are all non-EC and one
    task each: np.random.default_rng(seed).choice(range(n), n_val, replace=False)."""
    n_val = val_count(n_episodes, val_ratio)
    if n_episodes < 2:
        return []
    return sorted(int(i) for i in np.random.default_rng(seed=seed).choice(np.arange(n_episodes), size=n_val,
                                                                          replace=False))


def train_config_values():
    """MIN_PER_EPOCH_PER_10K_STEPS, DEFAULT_EPOCHS, SLURM_TIME, TIME_MARGIN_MIN from train/config.env (+ local)."""
    vals = {'MIN_PER_EPOCH_PER_10K_STEPS': '0.55', 'DEFAULT_EPOCHS': '60', 'SLURM_TIME': '08:00:00',
            'TIME_MARGIN_MIN': '40'}
    for name in ('config.env', 'config.local.env'):
        try:
            text = (TRAIN_DIR / name).read_text()
        except OSError:
            continue
        for line in text.splitlines():
            m = re.match(r'\s*(?:export\s+)?([A-Z0-9_]+)=["\']?([^"\'#]*)["\']?', line)
            if m and m.group(1) in vals and '$' not in m.group(2) and '<' not in m.group(2):
                vals[m.group(1)] = m.group(2).strip()
    return vals


def slurm_minutes(t):
    """'08:00:00' / '1-02:00:00' / '90' -> minutes."""
    t = str(t).strip()
    days = 0
    if '-' in t:
        d, t = t.split('-', 1)
        days = int(d)
    parts = [int(x) for x in t.split(':')]
    if len(parts) == 1:
        return days * 1440 + parts[0]
    if len(parts) == 2:
        return days * 1440 + parts[0] + parts[1] / 60
    return days * 1440 + parts[0] * 60 + parts[1] + parts[2] / 60


def training_estimate(n_steps, epochs=None):
    """{min_per_epoch, epochs, total_min, max_epochs, budget_min} with the train/config.env rule of thumb."""
    cfg = train_config_values()
    rate = float(cfg['MIN_PER_EPOCH_PER_10K_STEPS'])
    epochs = int(epochs or cfg['DEFAULT_EPOCHS'])
    mpe = rate * n_steps / 10000 if n_steps else None
    budget = slurm_minutes(cfg['SLURM_TIME']) - float(cfg['TIME_MARGIN_MIN'])
    return {'rate': rate, 'min_per_epoch': mpe, 'epochs': epochs, 'total_min': mpe * epochs if mpe else None,
            'budget_min': budget, 'slurm_time': cfg['SLURM_TIME'],
            'max_epochs': int(budget // mpe) if mpe else None}


def import_zarr():
    """(zarr module, None) or (None, hint). Registers the imagecodecs numcodecs (JPEG-XL) if available."""
    try:
        import zarr
    except ImportError:
        return None, ZARR_HINT
    try:
        from imagecodecs.numcodecs import register_codecs
        register_codecs(verbose=False)
    except Exception:  # noqa: BLE001 - images then cannot be decoded; numeric arrays still work
        pass
    return zarr, None


def common_parser(description, epilog=None):
    return argparse.ArgumentParser(description=description, epilog=epilog,
                                   formatter_class=argparse.RawDescriptionHelpFormatter)


def run_cli(main):
    """Run main(); LocateError -> message + exit 2."""
    try:
        sys.exit(main())
    except LocateError as e:
        print(f'error: {e}', file=sys.stderr)
        sys.exit(2)

