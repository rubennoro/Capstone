#!/usr/bin/env python3
"""What is in a task folder and its packaged dataset, before spending GPU hours on it.

Answers: how many recordings made it into episodes / were excluded / failed and why, how long the episodes are,
how close the validate metrics are to their thresholds, what dataproc warned about, how long training will take
and how many demos end up in the val split.

Without zarr (base python): reads <folder>/{task.yaml, episodes.csv, report.json, episode_N/processed/
{status.json, validate.json}, failed/*/reason.json} and package/<DATASET>/package.json.
With zarr (`pip install zarr==2.18.3 numcodecs==0.13.1 imagecodecs` in a venv) it also opens the packaged
<DATASET>.zarr.zip: per-episode path length and idle time at start / end, workspace extent (xy / xz), gripper
width traces, and a thumbnail grid of main-camera frames (JPEG-XL needs imagecodecs). --no-zarr skips that part.

Writes dataset_report.md + PNGs to <data root>/<FOLDER>/analysis/dataset_report/<DATASET or "folder">/ (or --out).

Examples:
  python3 train/analysis/dataset_report.py block_raise                      # newest package, if any
  python3 train/analysis/dataset_report.py block_raise --dataset br_v1
  ~/envs/analysis/bin/python train/analysis/dataset_report.py block_raise --dataset br_v1 --thumbs 6
"""
from __future__ import annotations

import re
import sys
from collections import Counter
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as C  # noqa: E402

HZ = 60.0
IDLE_SPEED_MPS = 0.02        # TCP slower than 2 cm/s counts as idle
IDLE_SMOOTH_S = 0.25         # speed is averaged over this window first
STAGES = ('check', 'slam', 'grid', 'pose', 'width', 'validate', 'images', 'export')

# validate.json metric -> (label, threshold key, threshold scale, 'max' | 'min')
VALIDATE_METRICS = [
    ('max_speed_mps', 'max TCP speed (m/s)', 'max_speed_mps', 1.0, 'max'),
    ('max_ultrawide_gap_ms', 'max ultrawide gap (ms)', 'max_ultrawide_gap_ns', 1e-6, 'max'),
    ('max_imu_gap_ms', 'max IMU gap (ms)', 'max_imu_gap_ns', 1e-6, 'max'),
    ('slam_tracked_fraction', 'SLAM tracked fraction', 'min_tracked_fraction', 1.0, 'min'),
    ('tag_fraction', 'finger-tag fraction', 'min_tag_fraction', 1.0, 'min'),
    ('kept.fraction', 'kept fraction of the button span', 'min_kept_fraction', 1.0, 'min'),
    ('kept.seconds', 'kept span (s)', 'min_kept_s', 1.0, 'min'),
]


def episode_index(name):
    m = re.match(r'episode_(\d+)$', name)
    return int(m.group(1)) if m else None


def episode_state(status):
    """exported | rejected | failed | incomplete | pending (dataproc.report.episode_summary)."""
    stages = {s: (status.get(s) or {}).get('status', 'pending') for s in STAGES}
    if stages['export'] == 'done':
        return 'exported', None, []
    for s in STAGES:
        if stages[s] in ('rejected', 'failed'):
            rec = status[s]
            return stages[s], s, list(rec.get('reasons') or [rec.get('reason') or stages[s]])
    return ('pending' if all(v == 'pending' for v in stages.values()) else 'incomplete'), None, []


def first_existing(*paths):
    for p in paths:
        if Path(p).exists():
            return Path(p)
    return None


def read_folder(fdir: Path):
    task = C.read_yaml(fdir / 'task.yaml', {}) or {}
    ledger = C.read_csv_rows(fdir / 'episodes.csv')
    pi_of = {}
    for r in ledger:
        pi_of[r.get('slot')] = r.get('pi_name')
    eps = []
    for d in sorted((p for p in fdir.glob('episode_*') if p.is_dir() and episode_index(p.name) is not None),
                    key=lambda p: episode_index(p.name)):
        sj = first_existing(d / 'processed' / 'status.json', d / 'status.json')
        vj = first_existing(d / 'processed' / 'validate.json', d / 'validate.json')
        status = C.read_json(sj, {}) if sj else {}
        state, stage, reasons = episode_state(status) if status else ('unknown', None, [])
        n_steps = ((status.get('export') or {}).get('outputs') or {}).get('n_steps')
        eps.append({'slot': d.name, 'pi_name': pi_of.get(d.name, ''), 'state': state, 'stage': stage,
                    'reasons': reasons, 'n_steps': n_steps,
                    'validate': C.read_json(vj, {}) if vj else {}})
    failed = []
    for d in sorted((p for p in (fdir / 'failed').glob('*') if p.is_dir()) if (fdir / 'failed').is_dir() else []):
        rec = C.read_json(d / 'reason.json', {}) or {}
        vj = first_existing(d / 'processed' / 'validate.json', d / 'validate.json')
        failed.append({'pi_name': d.name, 'stage': rec.get('stage'), 'status': rec.get('status'),
                       'reasons': [str(r) for r in rec.get('reasons') or []] or [rec.get('message') or
                                                                                 'no reason.json'],
                       'message': rec.get('message') or '', 'validate': C.read_json(vj, {}) if vj else {}})
    staging = sorted(p.name for p in (fdir / '.staging').glob('*') if p.is_dir()) \
        if (fdir / '.staging').is_dir() else []
    report = C.read_json(fdir / 'report.json', {}) or {}
    calib = sorted(p.name for p in (fdir / 'calibration').glob('*') if p.is_dir()) \
        if (fdir / 'calibration').is_dir() else []
    return {'task': task, 'ledger': ledger, 'episodes': eps, 'failed': failed, 'staging': staging,
            'report': report, 'calibrations': calib, 'gripper_range': C.read_json(fdir / 'gripper_range.json')}


def metric_value(v, key):
    x = (v.get('kept') or {}).get(key[5:]) if key.startswith('kept.') else (v.get('metrics') or {}).get(key)
    return x if C.isnum(x) else None


def plot_lengths(lengths_s, out_dir):
    if not lengths_s:
        return None
    fig, (ax,) = C.new_figure(height=3.6)
    bins = min(max(5, len(lengths_s) // 2), 30)
    ax.hist(lengths_s, bins=bins, color=C.TRAIN_COLOR, edgecolor=C.SURFACE, linewidth=2)
    med = float(np.median(lengths_s))
    ax.axvline(med, color=C.TEXT_SECONDARY, linestyle='--', linewidth=1)
    ax.annotate(f'median {med:.1f} s', (med, 1), xycoords=('data', 'axes fraction'), xytext=(4, -12),
                textcoords='offset points', fontsize=9, color=C.TEXT_PRIMARY)
    C.style_axes(ax, f'Episode length ({len(lengths_s)} episodes)', 'seconds (60 Hz steps / 60)', 'episodes')
    return C.save(fig, out_dir / 'episode_lengths.png')


def plot_validate(groups, out_dir):
    """Small multiples: one strip per validate metric, exported vs rejected, threshold as a dashed line."""
    panels = []
    for key, label, tkey, scale, kind in VALIDATE_METRICS:
        vals = {g: [x for x in (metric_value(v, key) for v in vs) if x is not None] for g, vs in groups.items()}
        if not any(vals.values()):
            continue
        thr = None
        for vs in groups.values():
            for v in vs:
                t = (v.get('thresholds') or {}).get(tkey)
                if C.isnum(t):
                    thr = t * scale
        panels.append((label, vals, thr, kind))
    if not panels:
        return None
    ncols = min(4, len(panels))
    nrows = -(-len(panels) // ncols)
    fig, axes = C.new_figure(ncols=ncols, nrows=nrows, width=3.6 * ncols, height=2.9 * nrows)
    axes = np.atleast_2d(axes).reshape(nrows, ncols)
    rng = np.random.default_rng(0)
    colors = {'exported': C.TRAIN_COLOR, 'rejected': C.VAL_COLOR}
    for i, ax in enumerate(axes.flat):
        if i >= len(panels):
            ax.set_visible(False)
            continue
        label, vals, thr, kind = panels[i]
        for j, (g, xs) in enumerate(vals.items()):
            if not xs:
                continue
            ax.scatter(xs, j + rng.uniform(-0.15, 0.15, len(xs)), s=28, color=colors.get(g, C.THIRD_COLOR),
                       edgecolor=C.SURFACE, linewidth=1, zorder=3, label=g)
        if thr is not None:
            ax.axvline(thr, color=C.TEXT_SECONDARY, linestyle='--', linewidth=1)
            ax.annotate(f'{kind} {thr:g}', (thr, 1), xycoords=('data', 'axes fraction'), xytext=(3, -10),
                        textcoords='offset points', fontsize=8, color=C.TEXT_SECONDARY)
        ax.set_yticks(range(len(vals)))
        ax.set_yticklabels(list(vals))
        ax.set_ylim(-0.6, len(vals) - 0.4)
        C.style_axes(ax, label, '', '')
        ax.title.set_fontsize(9)
    fig.suptitle('Validate metrics per episode (dashed: threshold)', x=0.01, ha='left', color=C.TEXT_PRIMARY,
                 fontsize=11)
    return C.save(fig, out_dir / 'validate_metrics.png')


# --------------------------------------------------------------------------------------------- zarr part

def zarr_analysis(zip_path: Path, out_dir: Path, n_thumbs: int, notes: list):
    zarr, hint = C.import_zarr()
    if zarr is None:
        notes.append(hint.replace('\n', ' '))
        return None
    try:
        store = zarr.ZipStore(str(zip_path), mode='r')
        root = zarr.open_group(store, mode='r')
        ends = np.asarray(root['meta/episode_ends'][:], dtype=np.int64)
        names = [str(x) for x in root['meta/episode_names'][:]] if 'episode_names' in root['meta'] else \
            [str(i) for i in range(len(ends))]
        pos = np.asarray(root['data/gripper_right_eef_pos'][:], dtype=np.float64)
        width = np.asarray(root['data/gripper_right_gripper_width'][:], dtype=np.float64).reshape(-1) \
            if 'gripper_right_gripper_width' in root['data'] else None
    except Exception as e:  # noqa: BLE001
        notes.append(f'could not read {zip_path.name}: {type(e).__name__}: {e}')
        return None
    starts = np.concatenate([[0], ends[:-1]])
    per = []
    win = max(1, int(IDLE_SMOOTH_S * HZ))
    for i, (a, b) in enumerate(zip(starts, ends)):
        p = pos[a:b]
        d = np.linalg.norm(np.diff(p, axis=0), axis=1) if len(p) > 1 else np.zeros(0)
        speed = np.convolve(d * HZ, np.ones(win) / win, mode='same') if len(d) else d
        moving = np.flatnonzero(speed > IDLE_SPEED_MPS)
        idle_start = (moving[0] / HZ) if len(moving) else len(p) / HZ
        idle_end = ((len(d) - 1 - moving[-1]) / HZ) if len(moving) else 0.0
        ext = p.max(axis=0) - p.min(axis=0) if len(p) else np.zeros(3)
        w = width[a:b] if width is not None else None
        per.append({'episode': names[i] if i < len(names) else str(i), 'steps': int(b - a), 'seconds': (b - a) / HZ,
                    'path_m': float(d.sum()), 'max_speed': float(speed.max()) if len(speed) else 0.0,
                    'idle_start_s': float(idle_start), 'idle_end_s': float(idle_end),
                    'extent_mm': (ext * 1000).tolist(),
                    'width_min_mm': float(w.min() * 1000) if w is not None and len(w) else None,
                    'width_max_mm': float(w.max() * 1000) if w is not None and len(w) else None})
    pngs = {}
    # workspace extent
    fig, axes = C.new_figure(ncols=2, width=11, height=4.6)
    for ax, (i, j, lab) in zip(axes, ((0, 1, 'xy (top view)'), (0, 2, 'xz (side view)'))):
        for a, b in zip(starts, ends):
            ax.plot(pos[a:b, i], pos[a:b, j], color=C.TRAIN_COLOR, linewidth=1, alpha=0.5)
            ax.plot(pos[a, i], pos[a, j], marker='o', markersize=4, color=C.THIRD_COLOR, linestyle='none')
            ax.plot(pos[b - 1, i], pos[b - 1, j], marker='s', markersize=4, color=C.VAL_COLOR, linestyle='none')
        ax.set_aspect('equal', adjustable='datalim')
        C.style_axes(ax, f'TCP path, {lab}', 'xyz'[i] + ' (m)', 'xyz'[j] + ' (m)')
    from matplotlib.lines import Line2D
    axes[0].legend(handles=[Line2D([], [], color=C.TRAIN_COLOR, label='path'),
                            Line2D([], [], color=C.THIRD_COLOR, marker='o', linestyle='none', label='start'),
                            Line2D([], [], color=C.VAL_COLOR, marker='s', linestyle='none', label='end')],
                   frameon=False, labelcolor=C.TEXT_PRIMARY, fontsize=9)
    pngs['workspace'] = C.save(fig, out_dir / 'workspace_extent.png')
    # gripper width traces
    if width is not None:
        fig, (ax,) = C.new_figure(height=3.6)
        for a, b in zip(starts, ends):
            ax.plot(np.arange(b - a) / HZ, width[a:b] * 1000, color=C.TRAIN_COLOR, linewidth=1, alpha=0.5)
        C.style_axes(ax, f'Gripper width per episode ({len(ends)} overlaid)', 'seconds from episode start', 'mm')
        pngs['width'] = C.save(fig, out_dir / 'gripper_width.png')
    # idle + path length
    fig, axes = C.new_figure(ncols=2, width=11, height=3.6)
    x = np.arange(len(per))
    axes[0].bar(x - 0.2, [p['idle_start_s'] for p in per], width=0.4, color=C.TRAIN_COLOR, label='start')
    axes[0].bar(x + 0.2, [p['idle_end_s'] for p in per], width=0.4, color=C.VAL_COLOR, label='end')
    C.style_axes(axes[0], f'Idle time (TCP < {IDLE_SPEED_MPS * 100:g} cm/s)', 'episode index', 'seconds')
    C.legend(axes[0])
    axes[1].bar(x, [p['path_m'] for p in per], color=C.TRAIN_COLOR, width=0.7)
    C.style_axes(axes[1], 'TCP path length', 'episode index', 'm')
    pngs['idle'] = C.save(fig, out_dir / 'idle_and_path.png')
    # thumbnails
    if n_thumbs and 'camera_right_main_rgb' in root['data']:
        try:
            cam = root['data/camera_right_main_rgb']
            pick = np.unique(np.linspace(0, len(ends) - 1, min(n_thumbs, len(ends))).astype(int))
            import matplotlib.pyplot as plt
            fig, axs = plt.subplots(len(pick), 3, figsize=(6.6, 2.3 * len(pick)), facecolor=C.SURFACE,
                                    squeeze=False)
            for r, e in enumerate(pick):
                a, b = int(starts[e]), int(ends[e])
                for c, (idx, lab) in enumerate(((a, 'first'), ((a + b) // 2, 'middle'), (b - 1, 'last'))):
                    axs[r][c].imshow(cam[idx])
                    axs[r][c].set_xticks([])
                    axs[r][c].set_yticks([])
                    for s in axs[r][c].spines.values():
                        s.set_visible(False)
                    axs[r][c].set_title(f'{per[e]["episode"]} {lab}' if c == 0 else lab, fontsize=8,
                                        color=C.TEXT_PRIMARY, loc='left')
            pngs['thumbs'] = C.save(fig, out_dir / 'thumbnails.png')
        except Exception as e:  # noqa: BLE001 - JPEG-XL codec missing etc.
            notes.append(f'thumbnails skipped ({type(e).__name__}: {e}); JPEG-XL frames need imagecodecs '
                         '(pip install imagecodecs)')
    store.close()
    return {'episodes': per, 'pngs': pngs, 'n_steps': int(ends[-1]) if len(ends) else 0}


# --------------------------------------------------------------------------------------------- report

def build(fdir: Path, pkg_dir, pj: dict, out_dir: Path, a):
    notes = []
    f = read_folder(fdir)
    eps, failed = f['episodes'], f['failed']
    md = [f'# Dataset report: {fdir.name}' + (f' / {pj.get("dataset")}' if pj else ''), '']
    task = f['task']
    facts = [('Folder', f'`{fdir}`'), ('Label', task.get('label') or '-'), ('Rig', task.get('rig') or '-'),
             ('Calibrations', ', '.join(f['calibrations']) or '-')]
    rep = f['report']
    if rep:
        facts.append(('Last dataproc run', f'{rep.get("created", "?")}, slam {rep.get("slam_backend", "?")}, '
                                           f'exit {rep.get("exit_code", "?")}'))
        cal = rep.get('calibration') or {}
        if cal:
            facts.append(('Calibration state', f'{cal.get("state")}' + (f' ({cal.get("name")})' if cal.get('name')
                                                                         else '')))
    if pj:
        z = pj.get('zip') or {}
        facts += [('Dataset', pj.get('dataset')), ('Package created', pj.get('created') or '-'),
                  ('Zip', f'{z.get("name")} ({C.fmt_bytes(z.get("size_bytes"))})'),
                  ('Folders in package', ', '.join(x.get('folder', '?') for x in pj.get('folders') or []))]
    md += [C.md_table(['', ''], facts), '']

    # counts
    states = Counter(e['state'] for e in eps)
    md += ['## Recordings', '']
    count_rows = [('episode_N exported', states.get('exported', 0))]
    for s in ('rejected', 'failed', 'incomplete', 'pending', 'unknown'):
        if states.get(s):
            count_rows.append((f'episode_N {s}', states[s]))
    count_rows.append(('failed/ (never packaged)', len(failed)))
    if f['staging']:
        count_rows.append(('.staging/ (in progress / waiting for SLAM)', len(f['staging'])))
    if pj:
        t = pj.get('totals') or {}
        count_rows += [('in the package', t.get('episodes')), ('excluded from the package', t.get('excluded'))]
    md += [C.md_table(['', 'count'], count_rows), '']
    reasons = Counter()
    for x in failed:
        for r in x['reasons']:
            reasons[f'{x["stage"] or "?"}: {r}'] += 1
    for e in eps:
        for r in e['reasons']:
            reasons[f'{e["stage"]}: {r}'] += 1
    if reasons:
        md += ['Failure reasons (failed/ and non-exported episode_N):', '',
               C.md_table(['stage: reason', 'recordings'], reasons.most_common()), '']
        md += ['Failed recordings:', '', C.md_table(['recording', 'stage', 'status', 'reasons', 'message'],
                                                     [(x['pi_name'], x['stage'] or '-', x['status'] or '-',
                                                       '; '.join(x['reasons']), x['message'][:80]) for x in failed]),
               ''] if failed else []
    if pj and pj.get('excluded'):
        md += ['Excluded by `dataproc package`:', '',
               C.md_table(['folder', 'entry', 'state', 'reason'],
                          [(x.get('folder'), x.get('entry'), x.get('state'), x.get('reason'))
                           for x in pj['excluded']]), '']

    # lengths
    if pj and pj.get('episodes'):
        steps = [e.get('n_steps') for e in pj['episodes'] if C.isnum(e.get('n_steps'))]
        src = 'package.json'
    else:
        steps = [e['n_steps'] for e in eps if C.isnum(e['n_steps'])]
        src = 'episode_N/processed/status.json'
    lengths = [s / HZ for s in steps]
    md += ['## Episode lengths', '']
    if lengths:
        md += [f'{len(lengths)} episodes ({src}): total {sum(steps):,} steps = {sum(lengths) / 60:.1f} min at 60 Hz; '
               f'median {np.median(lengths):.1f} s, min {min(lengths):.1f} s, max {max(lengths):.1f} s.', '']
        png = plot_lengths(lengths, out_dir)
        md += [f'![episode lengths]({C.rel(png, out_dir)})', '']
    else:
        md += ['No exported episodes with a length yet.', '']

    # validate
    groups = {'exported': [e['validate'] for e in eps if e['validate'] and e['state'] == 'exported'],
              'rejected': [e['validate'] for e in eps if e['validate'] and e['state'] != 'exported']
              + [x['validate'] for x in failed if x['validate']]}
    groups = {k: v for k, v in groups.items() if v}
    if groups:
        md += ['## Validate metrics', '']
        rows = []
        for key, label, tkey, scale, kind in VALIDATE_METRICS:
            xs = [metric_value(v, key) for v in groups.get('exported', [])]
            xs = [x for x in xs if x is not None]
            if not xs:
                continue
            thr = next((v['thresholds'][tkey] * scale for vs in groups.values() for v in vs
                        if C.isnum((v.get('thresholds') or {}).get(tkey))), None)
            worst = max(xs) if kind == 'max' else min(xs)
            ratio = worst / thr if thr else None
            # max limits: < 100 % is inside; min limits: > 100 % is inside
            rows.append((label, C.fmt(float(np.median(xs)), 3), C.fmt(worst, 3), f'{kind} {C.fmt(thr, 3)}',
                         f'{ratio:.0%} of the {kind}' if ratio is not None else '-'))
        if rows:
            md += ['Exported episodes:', '', C.md_table(['metric', 'median', 'worst', 'threshold', 'worst / threshold'],
                                                       rows), '']
        png = plot_validate(groups, out_dir)
        if png:
            md += [f'![validate metrics]({C.rel(png, out_dir)})', '']

    # warnings
    warns = list(pj.get('warnings') or []) if pj else []
    if rep.get('rig_nominal_sections'):
        warns.append(f'rig {rep.get("rig_id")} still has nominal (uncalibrated) sections: '
                     f'{", ".join(rep["rig_nominal_sections"])}')
    cal = rep.get('calibration') or {}
    if cal and cal.get('state') not in (None, 'done'):
        warns.append(f'calibration state {cal.get("state")}: {cal.get("reason")}')
    if f['staging']:
        warns.append(f'{len(f["staging"])} recording(s) still in .staging/ (waiting for SLAM?): '
                     + ', '.join(f['staging']))
    n_ep = (pj.get('totals') or {}).get('episodes') if pj else states.get('exported', 0)
    if n_ep is not None and n_ep < 2:
        warns.append(f'{n_ep} episode(s): training needs at least 2 per label (one goes to val)')
    if pj and pj.get('totals', {}).get('n_steps') and steps and sum(steps) != pj['totals']['n_steps']:
        warns.append('package.json totals.n_steps differs from the sum of its episodes')
    if pj and pkg_dir is not None:
        zp = pkg_dir / ((pj.get('zip') or {}).get('name') or f'{pj.get("dataset")}.zarr.zip')
        if not zp.exists():
            warns.append(f'{zp.name} is not in {pkg_dir} (pull it: train/pull.sh {fdir.name} --dataset '
                         f'{pj.get("dataset")})')
        elif (pj.get('zip') or {}).get('size_bytes') and zp.stat().st_size != pj['zip']['size_bytes']:
            warns.append(f'{zp.name} size differs from package.json (partial download?)')
    md += ['## Warnings', ''] + ([f'- {w}' for w in warns] or ['None.']) + ['']

    # training estimate and val split
    total_steps = (pj.get('totals') or {}).get('n_steps') if pj else sum(steps)
    md += ['## Training estimate', '']
    est = C.training_estimate(total_steps, a.epochs)
    per_label = (pj.get('totals') or {}).get('per_label') if pj else {task.get('label') or fdir.name: n_ep}
    if est['min_per_epoch']:
        md += [C.md_table(['', ''], [
            ('Steps (60 Hz)', C.fmt(total_steps)),
            ('Minutes per epoch', f'{est["min_per_epoch"]:.2f} ({est["rate"]} min per epoch per 10k steps, H200)'),
            (f'{est["epochs"]} epochs', C.fmt_minutes(est['total_min'])),
            (f'Max epochs in SLURM_TIME {est["slurm_time"]} minus margin', C.fmt(est['max_epochs']))]), '']
    val_rows = []
    for label, n in (per_label or {}).items():
        n_val = C.val_count(n or 0, a.val_ratio)
        val_rows.append((label, n, n_val, (n or 0) - n_val))
    if val_rows:
        md += [f'Val split (val_ratio {a.val_ratio} per task, at least 1; task.dataset.seed {a.seed}):', '',
               C.md_table(['label', 'episodes', 'val demos', 'train demos'], val_rows), '']
        if pj and len(per_label) == 1 and pj.get('episodes'):
            idx = C.predicted_val_indices(len(pj['episodes']), a.val_ratio, a.seed)
            if idx:
                names = [pj['episodes'][i].get('pi_name') or pj['episodes'][i].get('slot') for i in idx]
                md += [f'Predicted val demos: {", ".join(f"`{n}`" for n in names)} (reproduces get_val_mask for one '
                       'label without error-correction demos, assuming the zarr keeps package.json\'s episode order; '
                       'eval.json per_demo shows the real split).', '']

    # zarr
    z = None
    if pj and pkg_dir is not None and not a.no_zarr:
        zp = pkg_dir / ((pj.get('zip') or {}).get('name') or f'{pj.get("dataset")}.zarr.zip')
        if zp.exists():
            z = zarr_analysis(zp, out_dir, a.thumbs, notes)
    if z:
        per = z['episodes']
        md += ['## Trajectories (from the zarr)', '',
               C.md_table(['episode', 'seconds', 'path m', 'max speed m/s', 'idle start s', 'idle end s',
                           'extent x/y/z mm', 'width min-max mm'],
                          [(p['episode'], f'{p["seconds"]:.1f}', f'{p["path_m"]:.2f}', f'{p["max_speed"]:.2f}',
                            f'{p["idle_start_s"]:.1f}', f'{p["idle_end_s"]:.1f}',
                            '/'.join(f'{v:.0f}' for v in p['extent_mm']),
                            f'{C.fmt(p["width_min_mm"], 3)}-{C.fmt(p["width_max_mm"], 3)}') for p in per]), '']
        idle = [p['idle_start_s'] + p['idle_end_s'] for p in per]
        md += [f'Idle (TCP < {IDLE_SPEED_MPS * 100:g} cm/s, {IDLE_SMOOTH_S} s average) at start + end: median '
               f'{np.median(idle):.1f} s per episode, {sum(idle) / max(sum(p["seconds"] for p in per), 1e-9):.0%} of '
               'all frames.', '']
        for key, alt in (('workspace', 'workspace'), ('width', 'gripper width'), ('idle', 'idle and path'),
                         ('thumbs', 'thumbnails')):
            if z['pngs'].get(key):
                md += [f'![{alt}]({C.rel(z["pngs"][key], out_dir)})', '']
    elif pj and not a.no_zarr and not any('zarr' in n for n in notes):
        if pkg_dir is None or not (pkg_dir / ((pj.get('zip') or {}).get('name') or '')).exists():
            notes.append('zarr part skipped: the .zarr.zip is not downloaded')
    if notes:
        md += ['## Notes', ''] + [f'- {n}' for n in notes]
    C.write_text(out_dir / 'dataset_report.md', '\n'.join(md))

    lines = [f'{fdir.name}' + (f' / {pj.get("dataset")}' if pj else ' (no package)') +
             f': {states.get("exported", 0)} exported episodes, {len(failed)} failed'
             + (f', {pj["totals"].get("episodes")} in package, {pj["totals"].get("excluded")} excluded'
                if pj and pj.get('totals') else '')]
    if lengths:
        lines.append(f'  lengths: median {np.median(lengths):.1f} s, {sum(lengths) / 60:.1f} min total, '
                     f'{C.fmt(total_steps)} steps')
    if reasons:
        lines.append('  failures: ' + ', '.join(f'{k} x{v}' for k, v in reasons.most_common(5)))
    if est['min_per_epoch']:
        lines.append(f'  training: {est["min_per_epoch"]:.2f} min/epoch -> {est["epochs"]} epochs '
                     f'{C.fmt_minutes(est["total_min"])} (max {est["max_epochs"]} in {est["slurm_time"]})')
    for label, n, n_val, n_tr in val_rows:
        lines.append(f'  val split {label}: {n_val} of {n} demos')
    for w in warns:
        lines.append(f'  warning: {w}')
    for n in notes:
        lines.append(f'  note: {n}')
    print('\n'.join(lines))


def main(argv=None):
    p = C.common_parser(__doc__.split('\n\n')[0], epilog=__doc__.split('\n\n', 1)[1])
    p.add_argument('folder', help='task folder under the data root, or a path to one')
    C.add_data_root_arg(p)
    p.add_argument('--dataset', help='package name under <folder>/package/ (default: the newest)')
    p.add_argument('--no-zarr', action='store_true', help='skip opening the .zarr.zip even if zarr is installed')
    p.add_argument('--thumbs', type=int, default=4, help='episodes in the thumbnail grid (0 = none; default 4)')
    p.add_argument('--epochs', type=int, help='epochs for the time estimate (default DEFAULT_EPOCHS of config.env)')
    p.add_argument('--val-ratio', type=float, default=0.05, help='task.dataset.val_ratio (default 0.05)')
    p.add_argument('--seed', type=int, default=42, help='task.dataset.seed (default 42)')
    p.add_argument('--out', help='output dir (default <folder>/analysis/dataset_report/<dataset>/)')
    a = p.parse_args(argv)
    fp = Path(a.folder).expanduser()
    fdir = fp.resolve() if (('/' in a.folder or a.folder.startswith('.')) and fp.is_dir()) \
        else C.folder_dir(a.data_root, a.folder)
    pkg_dir, pj = C.find_package(fdir, a.dataset)
    if a.dataset and pkg_dir is None:
        raise C.LocateError(f'no package {a.dataset} in {fdir / "package"} (pull it: train/pull.sh {fdir.name} '
                            f'--dataset {a.dataset})')
    if not (fdir / 'task.yaml').exists() and not pj:
        raise C.LocateError(f'{fdir} has neither task.yaml nor a package/ (is it a pulled task folder?)')
    out = Path(a.out) if a.out else fdir / 'analysis' / 'dataset_report' / (pj.get('dataset') or 'folder')
    out.mkdir(parents=True, exist_ok=True)
    build(fdir, pkg_dir, pj, out, a)
    return 0


if __name__ == '__main__':
    C.run_cli(main)
