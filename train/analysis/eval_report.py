#!/usr/bin/env python3
"""Offline-eval report of one run: the training metric reproduced, errors in mm / degrees, the worst demos.

Answers: does the checkpoint reproduce its training val action MSE, how large the error is in physical units
along the predicted action chunk (step k = 0, 4, 8, 15 of 16), how train and val demos differ, which demos are
worst, and (with several evals) whether top2 / top3 are any better than top1.

Reads <run>/eval/<tag>/eval.json written by controlSW/franka_deploy/franka_deploy/tools/offline_eval.py (via
train/analysis/eval.sh). Writes <run>/analysis/eval_report.md and eval_per_step_<tag>.png,
eval_per_demo_<tag>.png (only with --per-demo evals), eval_compare.png (two or more evals) to <run>/analysis/.

Examples:
  python3 train/analysis/eval_report.py block_raise run_003              # top1 (or the first eval present)
  python3 train/analysis/eval_report.py block_raise run_003 --ckpt top2
Make an eval first:
  train/analysis/eval.sh block_raise run_003 --ckpt top1 --per-demo 8 && train/fetch.sh block_raise run_003
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as C  # noqa: E402

UNITS = [('pos_mm', 'position error', 'mm'), ('rot_deg', 'rotation error', 'deg'),
         ('width_mm', 'gripper width error', 'mm')]
COMP_LABELS = {'all': 'all', 'right_pos': 'position', 'right_rot': 'rotation (6D)',
               'right_gripper_width': 'gripper width'}


def ks_of(per_step):
    return sorted((int(k[1:]) for k in (per_step or {}) if k.startswith('k') and k[1:].isdigit()))


def split(e, name):
    return ((e or {}).get('splits') or {}).get(name) or {}


def plot_per_step(e, tag, out_dir):
    splits = [(n, split(e, n)) for n in ('train', 'val') if split(e, n).get('per_step')]
    if not splits:
        return None
    fig, axes = C.new_figure(ncols=3, width=13, height=3.9)
    for ax, (unit, title, ylab) in zip(axes, UNITS):
        for name, sp in splits:
            ps = sp['per_step']
            ks = ks_of(ps)
            med = [ps[f'k{k}'].get(f'{unit}_median') for k in ks]
            if all(m is None for m in med):
                continue
            color = C.SPLIT_COLORS[name]
            ax.plot(ks, med, color=color, label=f'{name} median', linewidth=2, marker='o', markersize=8,
                    markeredgecolor=C.SURFACE, markeredgewidth=2)
            mean = [ps[f'k{k}'].get(f'{unit}_mean') for k in ks]
            if all(m is not None for m in mean):
                ax.plot(ks, mean, color=color, linewidth=1, linestyle='--', alpha=0.8, label=f'{name} mean')
            p90 = [ps[f'k{k}'].get(f'{unit}_p90') for k in ks]
            if all(m is not None for m in p90):
                ax.plot(ks, p90, color=color, linewidth=1, linestyle=':', alpha=0.8, label=f'{name} p90')
        if unit == 'pos_mm':
            ps = (split(e, 'val') or split(e, 'train')).get('per_step') or {}
            ks = ks_of(ps)
            gt = [ps[f'k{k}'].get('gt_displacement_mm_median') for k in ks]
            if ks and all(g is not None for g in gt):
                ax.plot(ks, gt, color=C.TEXT_SECONDARY, linewidth=1, linestyle='-.',
                        label='ground-truth motion (median)')
        if splits:
            ax.set_xticks(ks_of(splits[0][1]['per_step']))
        C.style_axes(ax, title.capitalize(), 'action step k (60 Hz steps ahead in the chunk)'
                     if unit == 'rot_deg' else 'action step k', ylab)
        ax.set_ylim(bottom=0)
    C.legend(axes[0])
    fig.suptitle(f'{tag}: per-step error (epoch {e.get("ckpt_epoch", "?")})', x=0.01, ha='left',
                 color=C.TEXT_PRIMARY, fontsize=11)
    return C.save(fig, out_dir / f'eval_per_step_{tag}.png')


def demo_rows(e):
    rows = []
    for d in e.get('per_demo') or []:
        ps = d.get('per_step') or {}
        kmax = f'k{max(ks_of(ps))}' if ks_of(ps) else None
        last = ps.get(kmax, {}) if kmax else {}
        rows.append({'episode': str(d.get('episode')), 'split': d.get('checkpoint_split') or d.get('dataset_split')
                     or '?', 'n': d.get('n'), 'mse': (d.get('mse') or {}).get('all'),
                     'mse_parts': d.get('mse') or {}, 'k': kmax, 'pos_mm': last.get('pos_mm_median'),
                     'rot_deg': last.get('rot_deg_median'), 'width_mm': last.get('width_mm_median')})
    return rows


def plot_per_demo(rows, tag, out_dir):
    rows = [r for r in rows if r['mse'] is not None]
    if not rows:
        return None
    rows = sorted(rows, key=lambda r: r['mse'])
    n = len(rows)
    fig, axes = C.new_figure(ncols=2, width=12, height=max(3.0, 0.22 * n + 1.4))
    y = np.arange(n)
    colors = [C.SPLIT_COLORS.get(r['split'], C.OVERFLOW_COLOR) for r in rows]
    axes[0].barh(y, [r['mse'] for r in rows], color=colors, height=0.7, edgecolor=C.SURFACE, linewidth=1)
    axes[0].set_xscale('log')
    axes[0].set_yticks(y)
    axes[0].set_yticklabels([r['episode'] for r in rows], fontsize=7 if n > 30 else 8)
    C.style_axes(axes[0], 'Action MSE per demo (sorted)', 'MSE (log scale)', '')
    k = rows[0]['k'] or 'k?'
    pos = [r['pos_mm'] if r['pos_mm'] is not None else 0 for r in rows]
    axes[1].barh(y, pos, color=colors, height=0.7, edgecolor=C.SURFACE, linewidth=1)
    axes[1].set_yticks(y)
    axes[1].set_yticklabels([])
    C.style_axes(axes[1], f'Position error at {k} (median, same order)', 'mm', '')
    for ax in axes:
        ax.grid(True, axis='x', color=C.GRID, linewidth=0.8)
        ax.grid(False, axis='y')
        ax.set_ylim(-0.7, n - 0.3)
    from matplotlib.patches import Patch
    present = [s for s in ('train', 'val', 'new') if any(r['split'] == s for r in rows)]
    axes[0].legend(handles=[Patch(color=C.SPLIT_COLORS[s], label=s) for s in present], frameon=False,
                   labelcolor=C.TEXT_PRIMARY, fontsize=9, loc='lower right')
    return C.save(fig, out_dir / f'eval_per_demo_{tag}.png')


def plot_compare(evals, out_dir):
    tags = list(evals)
    if len(tags) < 2:
        return None
    fig, axes = C.new_figure(ncols=3, width=13, height=3.9)
    x = np.arange(len(tags))
    xt = [f'{t}\nepoch {evals[t].get("ckpt_epoch", "?")}' for t in tags]
    panels = [('mse', 'Action MSE (mean over noise seeds)', 'MSE'), ('pos_mm', 'Position error, last step', 'mm'),
              ('rot_deg', 'Rotation error, last step', 'deg')]
    for ax, (what, title, ylab) in zip(axes, panels):
        for j, name in enumerate(('train', 'val')):
            xs = x + (j - 0.5) * 0.18
            vals = []
            for t in tags:
                sp = split(evals[t], name)
                if what == 'mse':
                    vals.append(sp.get('mse_all_mean'))
                    seeds = sp.get('mse_all_per_seed') or []
                    ax.scatter([xs[tags.index(t)]] * len(seeds), seeds, s=10, color=C.SPLIT_COLORS[name],
                               alpha=0.45, zorder=3)
                else:
                    ps = sp.get('per_step') or {}
                    ks = ks_of(ps)
                    vals.append(ps[f'k{ks[-1]}'].get(f'{what}_median') if ks else None)
            ok = [(xx, v) for xx, v in zip(xs, vals) if v is not None]
            if ok:
                ax.plot([o[0] for o in ok], [o[1] for o in ok], linestyle='none', marker='o', markersize=9,
                        color=C.SPLIT_COLORS[name], markeredgecolor=C.SURFACE, markeredgewidth=2, label=name,
                        zorder=4)
        ax.set_xticks(x)
        ax.set_xticklabels(xt)
        ax.set_xlim(-0.6, len(tags) - 0.4)
        if what == 'mse':
            ax.set_yscale('log')
        C.style_axes(ax, title, '', ylab + (' (log scale)' if what == 'mse' else ''))
    C.legend(axes[0])
    return C.save(fig, out_dir / 'eval_compare.png')


def no_eval_message(run):
    f = run.folder or 'FOLDER'
    return (f'No eval.json in {run.path / "eval"}/<tag>/ yet. Make one with\n'
            f'  train/analysis/eval.sh {f} {run.name} --ckpt top1 --per-demo 8\n'
            f'then fetch it:\n  train/fetch.sh {f} {run.name}\n'
            f'and run this script again.')


def build(run: C.Run, tag: str, out_dir: Path):
    e = run.evals[tag]
    ck = {f'top{c.rank}': c for c in run.checkpoints}
    md = [f'# Offline eval: {run.name}', '', f'Run dir: `{run.path}`', '']
    lines = [f'{run.name} / {tag}: checkpoint epoch {e.get("ckpt_epoch", "?")}']

    # overview of every eval
    md += ['## Evals in this run', '']
    rows = []
    for t, ev in run.evals.items():
        v, tr = split(ev, 'val'), split(ev, 'train')
        train_epoch = ev.get('ckpt_epoch')
        logged = run.row_at(train_epoch).get(C.VAL_MSE) if train_epoch is not None else None
        if logged is None and t in ck:
            logged = ck[t].val_action_mse
        rows.append((t + (' (this report)' if t == tag else ''), C.fmt(train_epoch), C.fmt(v.get('mse_all_mean')),
                     C.fmt(float(np.std(v['mse_all_per_seed']))) if v.get('mse_all_per_seed') else '-',
                     C.fmt(logged), C.fmt(tr.get('mse_all_mean')),
                     C.fmt(ev.get('split_matches_checkpoint')), ev.get('precision', '-'),
                     'yes' if ev.get('per_demo') else 'no'))
    md += [C.md_table(['ckpt', 'epoch', 'val MSE (eval)', 'std over seeds', 'val MSE (training log)',
                       'train MSE (eval)', 'split matches ckpt', 'precision', 'per-demo'], rows), '']
    md += ['"val MSE (training log)" is the value training logged at that epoch: the eval reproduces the same '
           'metric (same val split, sample indices and batch size), so the two should be close; differences come '
           'from the diffusion noise (std over seeds shows its size).', '']
    if e.get('split_matches_checkpoint') is False:
        md += ['**Warning: the dataset split differs from the checkpoint\'s training split** (a different or '
               'regenerated dataset zip?); val numbers are not comparable to training.', '']
        lines.append('  WARNING: dataset split does not match the checkpoint')

    # this eval
    md += [f'## {tag}', '']
    facts = [('Checkpoint epoch', e.get('ckpt_epoch')), ('Precision', e.get('precision')),
             ('Samples per seed', e.get('sample_size')), ('Noise seeds', e.get('noise_seeds')),
             ('Index seed', e.get('index_seed')), ('Batch / micro batch', f'{e.get("batch_size")} / {e.get("micro_batch")}'),
             ('Train / val samples available', f'{C.fmt(e.get("len_train"))} / {C.fmt(e.get("len_val"))}')]
    if tag in ck:
        facts.append(('Checkpoint file', f'`{ck[tag].file}`'))
    job = run.eval_jobs.get(tag) or {}
    for k in ('job_id', 'job', 'submitted', 'dataset', 'ckpt'):
        if job.get(k) is not None:
            facts.append((f'eval_job.json {k}', job[k]))
    md += [C.md_table(['', ''], [(a, C.fmt(b) if not isinstance(b, str) else b) for a, b in facts]), '']

    comps = sorted({k for n in ('val', 'train') for k in (split(e, n).get('mse_components_mean') or {})},
                   key=lambda k: list(COMP_LABELS).index(k) if k in COMP_LABELS else 99)
    if comps:
        md += ['Action MSE by component (unnormalized, mean over noise seeds):', '',
               C.md_table(['component', 'val', 'train', 'val / train'],
                          [(COMP_LABELS.get(c, c), C.fmt(split(e, 'val').get('mse_components_mean', {}).get(c)),
                            C.fmt(split(e, 'train').get('mse_components_mean', {}).get(c)),
                            C.fmt(split(e, 'val').get('mse_components_mean', {}).get(c, 0)
                                  / split(e, 'train').get('mse_components_mean', {}).get(c), 3)
                            if split(e, 'train').get('mse_components_mean', {}).get(c) else '-') for c in comps]), '']
        v_all = split(e, 'val').get('mse_all_mean')
        t_all = split(e, 'train').get('mse_all_mean')
        lines.append(f'  val MSE {C.fmt(v_all)}, train MSE {C.fmt(t_all)}')

    step_rows = []
    for n in ('val', 'train'):
        ps = split(e, n).get('per_step') or {}
        for k in ks_of(ps):
            d = ps[f'k{k}']
            step_rows.append((n, k, C.fmt(d.get('pos_mm_median'), 3), C.fmt(d.get('pos_mm_p90'), 3),
                              C.fmt(d.get('rot_deg_median'), 3), C.fmt(d.get('width_mm_median'), 3),
                              C.fmt(d.get('gt_displacement_mm_median'), 3)))
    if step_rows:
        md += ['Per-step error along the predicted chunk (k = steps ahead, 60 Hz; median over all samples and seeds):',
               '', C.md_table(['split', 'k', 'pos mm', 'pos mm p90', 'rot deg', 'width mm', 'gt motion mm'],
                              step_rows), '']
        vps = split(e, 'val').get('per_step') or {}
        if vps:
            kk = max(ks_of(vps))
            d = vps[f'k{kk}']
            lines.append(f'  val at k={kk}: pos {C.fmt(d.get("pos_mm_median"), 3)} mm, rot '
                         f'{C.fmt(d.get("rot_deg_median"), 3)} deg, width {C.fmt(d.get("width_mm_median"), 3)} mm '
                         f'(gt motion {C.fmt(d.get("gt_displacement_mm_median"), 3)} mm)')
    png = plot_per_step(e, tag, out_dir)
    if png:
        md += [f'![per-step error]({C.rel(png, out_dir)})', '']

    drows = demo_rows(e)
    if drows:
        md += ['### Per demo', '']
        by = {}
        for r in drows:
            by.setdefault(r['split'], []).append(r)
        md += [C.md_table(['split', 'demos', 'median MSE', 'worst MSE', f'median pos mm at {drows[0]["k"]}'],
                          [(s, len(rs), C.fmt(float(np.median([r['mse'] for r in rs if r['mse'] is not None]))),
                            C.fmt(max(r['mse'] for r in rs if r['mse'] is not None)),
                            C.fmt(float(np.median([r['pos_mm'] for r in rs if r['pos_mm'] is not None])), 3)
                            if any(r['pos_mm'] is not None for r in rs) else '-')
                           for s, rs in sorted(by.items()) if any(r['mse'] is not None for r in rs)]), '']
        worst = sorted((r for r in drows if r['mse'] is not None), key=lambda r: -r['mse'])[:5]
        md += ['Worst 5 demos:', '',
               C.md_table(['episode', 'split', 'MSE', 'pos MSE', 'rot MSE', 'width MSE',
                           f'pos mm {worst[0]["k"] if worst else ""}', 'rot deg', 'width mm', 'samples'],
                          [(r['episode'], r['split'], C.fmt(r['mse']), C.fmt(r['mse_parts'].get('right_pos')),
                            C.fmt(r['mse_parts'].get('right_rot')), C.fmt(r['mse_parts'].get('right_gripper_width')),
                            C.fmt(r['pos_mm'], 3), C.fmt(r['rot_deg'], 3), C.fmt(r['width_mm'], 3), C.fmt(r['n']))
                           for r in worst]), '']
        md += ['Splits: `train` / `val` = the checkpoint\'s training split; `new` = an episode the checkpoint never '
               'saw in either split (e.g. a newer recording in the same zarr).', '']
        lines.append('  worst demos: ' + ', '.join(f'{r["episode"]} ({r["split"]}, {C.fmt(r["mse"])})'
                                                  for r in worst))
        png = plot_per_demo(drows, tag, out_dir)
        if png:
            md += [f'![per-demo error]({C.rel(png, out_dir)})', '']
    else:
        md += [f'No per-demo results in this eval (run eval.sh with `--per-demo 8` to get them).', '']

    if len(run.evals) >= 2:
        png = plot_compare(run.evals, out_dir)
        md += ['## Checkpoint comparison', '']
        if png:
            md += [f'![checkpoint comparison]({C.rel(png, out_dir)})', '']
        best = min((t for t in run.evals if split(run.evals[t], 'val').get('mse_all_mean') is not None),
                   key=lambda t: split(run.evals[t], 'val')['mse_all_mean'], default=None)
        if best:
            md += [f'Lowest eval val MSE: **{best}** (epoch {run.evals[best].get("ckpt_epoch")}).', '']
            lines.append(f'  lowest eval val MSE across {", ".join(run.evals)}: {best}')
    vids = sorted((run.path / 'eval' / tag / 'videos').glob('*.mp4')) \
        if (run.path / 'eval' / tag / 'videos').is_dir() else []
    if vids:
        md += ['Attention / rollout videos:', ''] + [f'- [{v.name}]({C.rel(v, out_dir)})' for v in vids] + ['']
    C.write_text(out_dir / 'eval_report.md', '\n'.join(md))
    print('\n'.join(lines))


def main(argv=None):
    p = C.common_parser(__doc__.split('\n\n')[0], epilog=__doc__.split('\n\n', 1)[1])
    p.add_argument('folder', help='task folder under the data root (or a path to a run dir)')
    p.add_argument('run', nargs='?', help='run name / prefix (default: newest) or a path to a run dir')
    C.add_data_root_arg(p)
    p.add_argument('--ckpt', default=None,
                   help='eval tag to report in detail (top1, top2, ...; default top1, else the first present)')
    p.add_argument('--out', help='output dir (default <run>/analysis/)')
    a = p.parse_args(argv)
    run = C.load_run(C.resolve_run(a.data_root, a.folder, a.run))
    if not run.evals:
        print(no_eval_message(run))
        return 1
    tag = a.ckpt or ('top1' if 'top1' in run.evals else next(iter(run.evals)))
    if tag not in run.evals:
        print(f'no eval for {tag} in {run.path / "eval"} (have: {", ".join(run.evals)}). Make it with\n'
              f'  train/analysis/eval.sh {run.folder or "FOLDER"} {run.name} --ckpt {tag} --per-demo 8\n'
              f'then train/fetch.sh {run.folder or "FOLDER"} {run.name}')
        return 1
    out = Path(a.out) if a.out else run.path / 'analysis'
    out.mkdir(parents=True, exist_ok=True)
    build(run, tag, out)
    return 0


if __name__ == '__main__':
    C.run_cli(main)
