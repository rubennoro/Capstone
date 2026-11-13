#!/usr/bin/env python3
"""Compare training runs side by side: one table row and one curve per run.

Answers: which run reached the lowest val action MSE and when, how the curves differ (more data, more epochs,
different overrides), what each run cost (minutes/epoch, GPU memory).

Runs: every run in <data root>/<FOLDER>/policy/ (or the RUNs named), plus with --results-dir the old packaged
runs of results/<task>/ (same package_run.py layout) as a baseline. Writes compare_runs.md, compare_runs.csv,
compare_val_mse.png (val action MSE per epoch, log y) and compare_loss.png (train / val loss per epoch, log y) to
<data root>/<FOLDER>/analysis/compare_runs/ (or --out).

Examples:
  python3 train/analysis/compare_runs.py block_raise
  python3 train/analysis/compare_runs.py block_raise --results-dir results/block_raise --no-smoke
  python3 train/analysis/compare_runs.py block_raise run_003 run_005 --results-dir results/block_raise
"""
from __future__ import annotations

import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as C  # noqa: E402

COLUMNS = ['run', 'source', 'status', 'dataset', 'mode', 'episodes', 'dataset_steps', 'epochs',
           'best_val_mse', 'best_epoch', 'final_val_mse', 'final_epoch', 'min_val_loss', 'min_val_loss_epoch',
           'steps_per_epoch', 'minutes_per_epoch', 'training_minutes', 'gpu_peak_gb', 'kept_checkpoints', 'note']


def collect(a):
    """[(label, source, Run)]."""
    runs = []
    if a.folder:
        fdir = Path(a.data_root).expanduser() / a.folder
        policy = fdir / 'policy'
        if a.runs:
            for r in a.runs:
                runs.append((C.resolve_run(a.data_root, a.folder, r), 'policy'))
        else:
            runs += [(p, 'policy') for p in C.list_runs(policy)]
            if not runs and not a.results_dir:
                raise C.LocateError(f'no runs in {policy} (and no --results-dir given)')
    for rd in a.results_dir or []:
        rd = Path(rd).expanduser()
        if C.looks_like_run_dir(rd) and not C.list_runs(rd):
            runs.append((rd, 'results'))
        else:
            found = C.list_runs(rd)
            if not found:
                print(f'warning: no runs in {rd}', file=sys.stderr)
            runs += [(p, 'results') for p in found]
    out, seen = [], set()
    for path, src in runs:
        path = path.resolve()
        if path in seen:
            continue
        seen.add(path)
        run = C.load_run(path)
        if a.no_smoke and run.mode == 'smoke':
            continue
        label = run.name if src == 'policy' else f'{run.name} (results)'
        out.append((label, src, run))
    return out


def row_for(label, src, run: C.Run, data_root):
    pj = C.package_for_run(run, data_root)
    totals = pj.get('totals') or {}
    best_ep, best = run.best(C.VAL_MSE)
    fin_ep, fin = run.final(C.VAL_MSE)
    vl_ep, vl = run.best(C.VAL_LOSS)
    s = run.summary
    return {'run': label, 'source': src, 'status': run.status, 'dataset': run.dataset, 'mode': run.mode,
            'episodes': totals.get('episodes'), 'dataset_steps': totals.get('n_steps'),
            'epochs': run.epochs_completed, 'best_val_mse': best, 'best_epoch': best_ep, 'final_val_mse': fin,
            'final_epoch': fin_ep, 'min_val_loss': vl, 'min_val_loss_epoch': vl_ep,
            'steps_per_epoch': s.get('steps_per_epoch'), 'minutes_per_epoch': s.get('minutes_per_epoch'), 'training_minutes': s.get('training_minutes'),
            'gpu_peak_gb': s.get('gpu_peak_gb'), 'kept_checkpoints': len(run.checkpoints),
            'note': run.train_meta.get('note') or s.get('note') or ''}


def plot_metric(runs, keys_titles, out_path, title_suffix=''):
    """Small multiples (one per key), one line per run, same color per run in every panel; log y."""
    n = len(keys_titles)
    fig, axes = C.new_figure(ncols=n, width=6.5 * n if n > 1 else 8, height=4.4)
    drawn = 0
    for ax, (key, title, ylabel) in zip(axes, keys_titles):
        lines = []
        for i, (label, _, run) in enumerate(runs):
            ep, v = run.series(key)
            if not len(v):
                continue
            color = C.series_color(i)
            style = C.epoch_line_style(len(v)) if len(runs) <= 4 else dict(linewidth=2)
            ax.plot(ep, v, color=color, label=label, **style)
            lines.append((ep[-1], v[-1], label, color))
            drawn += 1
        ax.set_yscale('log')
        C.style_axes(ax, title + title_suffix, 'epoch', f'{ylabel} (log scale)')
        if len(runs) <= 4 and lines:   # direct labels at the line ends; the legend stays for >= 2 runs
            merged = []   # runs ending on the same point (identical smoke runs) share one label
            for x, y, label, color in lines:
                short = label.split('_20')[0]
                for m in merged:
                    if m[0] == x and abs(m[1] - y) <= 0.02 * abs(y):
                        m[2].append(short)
                        break
                else:
                    merged.append([x, y, [short]])
            for x, y, names in merged:
                ax.annotate(', '.join(names), (x, y), xytext=(6, 0), textcoords='offset points',
                            va='center', fontsize=8, color=C.TEXT_PRIMARY)
            ax.margins(x=0.18)
            lo = min(min(r.series(key)[0]) for _, _, r in runs if len(r.series(key)[0]))
            hi = max(x for x, _, _, _ in lines)
            ax.set_xlim(left=lo - 0.03 * max(hi - lo, 1))
    if drawn == 0:
        import matplotlib.pyplot as plt
        plt.close(fig)
        return None
    handles, labels = axes[0].get_legend_handles_labels()
    layout = True
    if len(runs) > 1:
        layout = False
        fig.legend(handles, labels, loc='lower center', ncol=min(len(labels), 3 if n > 1 else 2), frameon=False, fontsize=9,
                   labelcolor=C.TEXT_PRIMARY, bbox_to_anchor=(0.5, -0.01))
        rows = -(-len(labels) // (3 if n > 1 else 2))
        fig.tight_layout(rect=(0, 0.05 + 0.055 * rows, 1, 1))
    return C.save(fig, out_path, layout=layout)


def plot_val_mse(runs, out_dir):
    path = plot_metric(runs, [(C.VAL_MSE, 'Val action MSE per epoch', 'val action MSE')], out_dir / 'compare_val_mse.png')
    return path


def plot_losses(runs, out_dir):
    return plot_metric(runs, [(C.TRAIN_LOSS, 'Train loss (epoch average)', 'train loss'),
                                     (C.VAL_LOSS, 'Val loss', 'val loss')],
                       out_dir / 'compare_loss.png')


def main(argv=None):
    p = C.common_parser(__doc__.split('\n\n')[0], epilog=__doc__.split('\n\n', 1)[1])
    p.add_argument('folder', nargs='?', help='task folder under the data root (its policy/ runs are compared)')
    p.add_argument('runs', nargs='*', help='run names / prefixes / paths (default: all runs of the folder)')
    C.add_data_root_arg(p)
    p.add_argument('--results-dir', action='append',
                   help='also include the packaged runs in this dir (e.g. results/block_raise); repeatable')
    p.add_argument('--no-smoke', action='store_true', help='leave out smoke runs')
    p.add_argument('--out', help='output dir (default <data root>/<FOLDER>/analysis/compare_runs/)')
    a = p.parse_args(argv)
    if not a.folder and not a.results_dir:
        p.error('give FOLDER and/or --results-dir')
    if a.folder and not a.runs and not a.results_dir:
        C.folder_dir(a.data_root, a.folder)
    runs = collect(a)
    if not runs:
        raise C.LocateError('no runs to compare')
    if a.out:
        out = Path(a.out)
    elif a.folder:
        out = Path(a.data_root).expanduser() / a.folder / 'analysis' / 'compare_runs'
    else:
        p.error('--out is required without FOLDER (results/ is never written to)')
    out.mkdir(parents=True, exist_ok=True)
    if len(runs) > len(C.SERIES_COLORS):
        print(f'note: {len(runs)} runs; runs past the {len(C.SERIES_COLORS)}th are drawn in gray '
              '(use --no-smoke or name the runs)', file=sys.stderr)

    rows = [row_for(label, src, run, a.data_root) for label, src, run in runs]
    with open(out / 'compare_runs.csv', 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=COLUMNS)
        w.writeheader()
        for r in rows:
            w.writerow({k: ('' if r[k] is None else r[k]) for k in COLUMNS})
    print('wrote', out / 'compare_runs.csv')

    vpng = plot_val_mse(runs, out)
    lpng = plot_losses(runs, out)

    header = ['run', 'status', 'dataset', 'mode', 'episodes', 'steps (60 Hz)', 'epochs', 'best val MSE @ epoch',
              'final val MSE', 'min val loss @ epoch', 'steps/epoch', 'min/epoch', 'train time', 'GPU peak GB']
    table = [(r['run'], r['status'], r['dataset'] or '-', r['mode'] or '-', C.fmt(r['episodes']),
              C.fmt(r['dataset_steps']), C.fmt(r['epochs']),
              f'{C.fmt(r["best_val_mse"])} @ {C.fmt(r["best_epoch"])}', C.fmt(r['final_val_mse']),
              f'{C.fmt(r["min_val_loss"])} @ {C.fmt(r["min_val_loss_epoch"])}', C.fmt(r['steps_per_epoch']),
              C.fmt(r['minutes_per_epoch']),
              C.fmt_minutes(r['training_minutes']), C.fmt(r['gpu_peak_gb'])) for r in rows]
    ranked = sorted((r for r in rows if r['best_val_mse'] is not None), key=lambda r: r['best_val_mse'])
    md = ['# Run comparison', '']
    if a.folder:
        md += [f'Folder: `{a.folder}`' + (f'; baselines from {", ".join(f"`{d}`" for d in a.results_dir)}'
                                          if a.results_dir else ''), '']
    md += [C.md_table(header, table), '']
    if ranked:
        b = ranked[0]
        md += [f'Lowest val action MSE: **{b["run"]}**, {C.fmt(b["best_val_mse"])} at epoch {b["best_epoch"]}.', '']
    md += ['Episodes / steps come from the local package.json of each run\'s dataset (blank when the dataset is '
           'not under data/<folder>/package/, e.g. the iPhone-era results/ runs).', '']
    notes = [(r['run'], r['note']) for r in rows if r['note']]
    if notes:
        md += ['Notes:', ''] + [f'- {n}: {t}' for n, t in notes] + ['']
    for png, alt in ((vpng, 'val action MSE'), (lpng, 'losses')):
        if png:
            md += [f'![{alt}]({C.rel(png, out)})', '']
    md.append(f'Table as CSV: [compare_runs.csv](compare_runs.csv)')
    C.write_text(out / 'compare_runs.md', '\n'.join(md))

    print(C.text_table(['run', 'status', 'mode', 'epochs', 'best val MSE @ ep', 'final val MSE', 'min/epoch'],
                       [(r['run'], r['status'], r['mode'] or '-', C.fmt(r['epochs']),
                         f'{C.fmt(r["best_val_mse"])} @ {C.fmt(r["best_epoch"])}', C.fmt(r['final_val_mse']),
                         C.fmt(r['minutes_per_epoch'])) for r in rows]))
    if ranked:
        print(f'lowest val action MSE: {ranked[0]["run"]} ({C.fmt(ranked[0]["best_val_mse"])} @ epoch '
              f'{ranked[0]["best_epoch"]})')
    return 0


if __name__ == '__main__':
    C.run_cli(main)
