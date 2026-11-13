#!/usr/bin/env python3
"""One training run at a glance: status, timing, best checkpoint, top-k, overfitting, per-component error.

Answers: did the run finish, which epoch is best and by how much, is it overfitting (and so how many epochs are
enough next time), which checkpoints were kept, where are its plots / videos / evals.

Writes <run>/analysis/report.md and <run>/analysis/val_mse_topk.png (val action MSE per epoch with the kept
checkpoints marked). If the run has no plots/ but has wandb/ (or history.csv + epochs.csv), the standard plots of
scripts/wandb_offline_to_csv.py are regenerated into <run>/analysis/plots/ (reading wandb/ needs `pip install
wandb`; the CSVs do not).

Examples:
  python3 train/analysis/run_report.py block_raise                 # newest run in data/block_raise/policy/
  python3 train/analysis/run_report.py block_raise run_003         # by name (or unique prefix)
  python3 train/analysis/run_report.py results/block_raise/run_001_2026-10-05 --out /tmp/r001
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as C  # noqa: E402

OVERFIT_VAL_LOSS_RISE = 0.05   # val loss rose > 5 % from the best-MSE epoch to the end
LATE_BEST_FRACTION = 0.7       # best epoch before 70 % of training -> the rest was not needed


def overfitting(run: C.Run):
    """Indicators and a one-line verdict. Every number is None when its data is missing."""
    out = {'verdict': None}
    best_ep, best = run.best(C.VAL_MSE)
    last_ep = (run.epochs_completed - 1) if run.epochs_completed else None
    out.update(best_epoch=best_ep, best=best, last_epoch=last_ep)
    vl_ep, vl = run.series(C.VAL_LOSS)
    tl_ep, tl = run.series(C.TRAIN_LOSS)
    if best_ep is None or last_ep is None:
        return out
    out['epochs_after_best'] = last_ep - best_ep
    out['fraction_after_best'] = (last_ep - best_ep) / last_ep if last_ep else 0.0
    vmin_ep, vmin = run.best(C.VAL_LOSS)
    out.update(min_val_loss_epoch=vmin_ep, min_val_loss=vmin)
    after = vl_ep >= best_ep
    if after.sum() >= 2:
        x, y = vl_ep[after], vl[after]
        slope = float(np.polyfit(x, y, 1)[0])
        out['val_loss_at_best'] = float(y[0])
        out['val_loss_final'] = float(y[-1])
        out['val_loss_change'] = float(y[-1] / y[0] - 1) if y[0] else None
        out['val_loss_slope_per_10'] = slope * 10
    t_after = tl_ep >= best_ep
    if t_after.sum() >= 2 and tl[t_after][0]:
        out['train_loss_change'] = float(tl[t_after][-1] / tl[t_after][0] - 1)
    for tag, ep in (('best', best_ep), ('final', run.final(C.VAL_MSE)[0])):
        row = run.row_at(ep) if ep is not None else {}
        v, t = row.get(C.VAL_MSE), row.get(C.TRAIN_MSE)
        out[f'gap_{tag}'] = (v / t) if (v is not None and t) else None
    # verdict
    rising = (out.get('val_loss_change') or 0) > OVERFIT_VAL_LOSS_RISE
    early = last_ep > 0 and best_ep < LATE_BEST_FRACTION * last_ep
    _, final = run.final(C.VAL_MSE)
    worse = (final / best - 1) if (final and best) else None
    out['final_vs_best'] = worse
    if last_ep < 5:
        out['verdict'] = f'too few epochs ({last_ep + 1}) to judge overfitting'
    elif rising and early:
        out['verdict'] = (f'overfitting: val action MSE bottoms out at epoch {best_ep} of {last_ep}, and val loss '
                          f'rises {out["val_loss_change"]:+.0%} after it while train loss keeps falling; '
                          f'~{int(round((best_ep + 10) / 10.0)) * 10} epochs would have been enough')
    elif early:
        out['verdict'] = (f'plateau: best val action MSE at epoch {best_ep} of {last_ep}'
                          + (f' (final is {worse:+.0%} vs best)' if worse is not None else '')
                          + '; the later epochs added nothing')
    elif best_ep >= 0.9 * last_ep:
        out['verdict'] = (f'still improving: best val action MSE at epoch {best_ep} of {last_ep}; '
                          'more epochs may help')
    else:
        out['verdict'] = f'no clear overfitting (best val action MSE at epoch {best_ep} of {last_ep})'
    return out


def components_at(run: C.Run, epoch):
    """[(component label, val MSE, train MSE)] at an epoch, for the components present in epochs.csv."""
    row = run.row_at(epoch) if epoch is not None else {}
    rows = []
    for key, label in C.COMPONENTS:
        v = row.get(f'val/action_mse_error_{key}/all')
        t = row.get(f'train/action_mse_error_{key}/all')
        if v is not None or t is not None:
            rows.append((label, v, t))
    return rows


def plot_val_mse_topk(run: C.Run, out_dir: Path):
    ep, v = run.series(C.VAL_MSE)
    if not len(v):
        return None
    fig, (ax,) = C.new_figure()
    ax.plot(ep, v, color=C.VAL_COLOR, label='val action MSE', **C.epoch_line_style(len(v)))
    tep, tv = run.series(C.TRAIN_MSE)
    if len(tv):
        ax.plot(tep, tv, color=C.TRAIN_COLOR, label='train action MSE', **C.epoch_line_style(len(tv)))
    ax.set_yscale('log')
    kept = [c for c in run.checkpoints if c.epoch is not None and c.val_action_mse]
    for c in kept:
        ax.axvline(c.epoch, color=C.TEXT_SECONDARY, linewidth=0.8, linestyle=':')
        ax.annotate(f'top{c.rank}', (c.epoch, 1), xycoords=('data', 'axes fraction'), xytext=(3, -10),
                    textcoords='offset points', fontsize=8, color=C.TEXT_SECONDARY)
    best_ep, best = run.best(C.VAL_MSE)
    C.label_point(ax, best_ep, best, f'best {C.fmt(best)} @ {best_ep}', C.VAL_COLOR, dy=-12)
    if len(ep) <= C.MAX_EPOCH_TICKS:
        ax.set_xticks(ep)
    C.style_axes(ax, 'Action MSE per epoch (dotted: kept checkpoints)', 'epoch', 'MSE (log scale)')
    C.legend(ax)
    ax.margins(x=0.08)
    return C.save(fig, out_dir / 'val_mse_topk.png')


def regenerate_plots(run: C.Run, out_dir: Path, notes: list):
    """Standard plots into <out>/plots/ when the run has none."""
    if run.plots():
        return []
    tools = C.WANDB_TOOLS
    if tools is None:
        notes.append('plots/ is missing and scripts/wandb_offline_to_csv.py could not be imported')
        return []
    plot_dir = out_dir / 'plots'
    history, epochs = None, None
    wdirs = run.wandb_dirs()
    if wdirs and C.HAS_WANDB:
        try:
            history = tools.read_history(tools.find_wandb_file(str(wdirs[-1])))
            epochs = tools.epoch_table(history)
            plot_dir.mkdir(parents=True, exist_ok=True)
            history.to_csv(plot_dir / 'history.csv', index=False)
            epochs.to_csv(plot_dir / 'epochs.csv', index=False)
        except Exception as e:  # noqa: BLE001
            notes.append(f'reading {wdirs[-1]} failed: {type(e).__name__}: {e}')
            history = None
    if history is None:
        history, epochs = run.history(), run.epochs
        if history is None or epochs is None:
            if wdirs and not C.HAS_WANDB:
                notes.append('plots/ is missing; regenerating them from wandb/ needs wandb '
                             '(pip install wandb into a venv, same version as the cluster bp env)')
            else:
                notes.append('plots/ is missing and there is no wandb/ or history.csv + epochs.csv to rebuild them')
            return []
    try:
        tools.make_plots(history, epochs, str(plot_dir), str(run.gpu_csv()) if run.gpu_csv() else None)
    except Exception as e:  # noqa: BLE001 - e.g. a column missing in a half-written history
        notes.append(f'regenerating plots failed: {type(e).__name__}: {e}')
    return sorted(plot_dir.glob('*.png'))


def build(run: C.Run, out_dir: Path):
    notes = []
    ov = overfitting(run)
    best_ep, best = ov['best_epoch'], ov['best']
    final_ep, final = run.final(C.VAL_MSE)
    s, tm = run.summary, run.train_meta
    regen = regenerate_plots(run, out_dir, notes)
    topk_png = plot_val_mse_topk(run, out_dir)

    md = [f'# Run report: {run.name}', '', f'Run dir: `{run.path}`', '']
    facts = [
        ('Status', run.status + ('' if not run.status_file or not s.get('status') or s['status'] == run.status_file
                                 else f' (summary.json says {s["status"]})')),
        ('Mode', run.mode or '-'),
        ('Folder / label', f'{run.folder or "-"} / {run.label or "-"}'),
        ('Dataset', run.dataset or '-'),
        ('Epochs', f'{C.fmt(run.epochs_completed)} completed of {C.fmt(run.epochs_configured)} configured'),
        ('Job', s.get('job_id') or run.job.get('job') or run.job.get('job_id') or '-'),
        ('Submitted', tm.get('submitted') or run.job.get('submitted') or '-'),
        ('Packaged', s.get('packaged') or '-'),
        ('Training time', C.fmt_minutes(s.get('training_minutes'))),
        ('Speed', f'{C.fmt(s.get("iterations_per_second"))} steps/s, {C.fmt(s.get("steps_per_epoch"))} steps/epoch, '
                  f'{C.fmt(s.get("minutes_per_epoch"))} min/epoch'),
        ('Peak GPU memory', f'{C.fmt(s.get("gpu_peak_gb"))} GB'),
        ('Top-k kept', tm.get('topk') or '-'),
        ('Note', tm.get('note') or s.get('note') or '-'),
    ]
    if tm.get('overrides'):
        ov_list = tm['overrides'] if isinstance(tm['overrides'], list) else [tm['overrides']]
        facts.append(('Extra overrides', ' '.join(f'`{o}`' for o in ov_list)))
    if tm.get('clone_git_commit'):
        facts.append(('Cluster clone commit', f'`{tm["clone_git_commit"]}`'))
    md += [C.md_table(['', ''], facts), '']

    md += ['## Result', '']
    res = [('Best val action MSE', f'**{C.fmt(best)}** at epoch {C.fmt(best_ep)}'),
           ('Final val action MSE', f'{C.fmt(final)} at epoch {C.fmt(final_ep)}'),
           ('Min val loss', f'{C.fmt(ov.get("min_val_loss"))} at epoch {C.fmt(ov.get("min_val_loss_epoch"))}')]
    sb = s.get('best_checkpoint') or {}
    if sb and sb.get('epoch') is not None and sb.get('epoch') != best_ep:
        res.append(('summary.json best checkpoint', f'epoch {sb.get("epoch")} ({C.fmt(sb.get("value"))})'))
    md += [C.md_table(['', ''], res), '']

    comps = components_at(run, best_ep)
    if comps:
        md += [f'Per-component action MSE at the best epoch ({best_ep}):', '',
               C.md_table(['component', 'val MSE', 'train MSE', 'val / train'],
                          [(lab, C.fmt(v), C.fmt(t), C.fmt(v / t, 3) if (v is not None and t) else '-')
                           for lab, v, t in comps]), '']

    md += ['## Kept checkpoints', '']
    if run.checkpoints:
        md += [C.md_table(['rank', 'epoch', 'val action MSE', 'size', 'on disk', 'file'],
                          [(c.rank, C.fmt(c.epoch), C.fmt(c.val_action_mse), C.fmt_bytes(c.size_bytes),
                            'yes' if c.present else 'no', f'`{c.file}`') for c in run.checkpoints]),
               '', f'(from {run.checkpoints_source})', '']
    else:
        md += ['No checkpoints in this run dir' + (' (fetch.sh may have been run without them, or the job failed '
                                                   'before saving)' if run.status != 'ok' else '') + '.', '']
    scratch = C.read_text(run.path / 'logs' / 'checkpoints_on_scratch.txt')
    if scratch:
        md += ['Checkpoints left on cluster scratch (logs/checkpoints_on_scratch.txt):', '', '```',
               *scratch.splitlines()[:8], '```', '']

    md += ['## Overfitting', '']
    if ov.get('verdict'):
        md += [f'**{ov["verdict"]}.**', '']
    rows = [('Epoch of min val action MSE vs last epoch', f'{C.fmt(best_ep)} vs {C.fmt(ov.get("last_epoch"))}'
             + (f' ({ov["epochs_after_best"]} epochs after the best, {ov["fraction_after_best"]:.0%} of training)'
                if ov.get('epochs_after_best') is not None else '')),
            ('Val loss from the best epoch to the end',
             f'{C.fmt(ov.get("val_loss_at_best"))} -> {C.fmt(ov.get("val_loss_final"))}'
             + (f' ({ov["val_loss_change"]:+.0%}; slope {C.fmt(ov["val_loss_slope_per_10"])} per 10 epochs)'
                if ov.get('val_loss_change') is not None else '')),
            ('Train loss over the same epochs', f'{ov["train_loss_change"]:+.0%}' if ov.get('train_loss_change')
             is not None else '-'),
            ('Val / train action MSE at best', C.fmt(ov.get('gap_best'), 3)),
            ('Val / train action MSE at the end', C.fmt(ov.get('gap_final'), 3)),
            ('Final vs best val action MSE', f'{ov["final_vs_best"]:+.1%}' if ov.get('final_vs_best') is not None
             else '-')]
    md += [C.md_table(['indicator', 'value'], rows), '']

    if run.evals:
        md += ['## Offline evals', '', C.md_table(
            ['ckpt', 'epoch', 'val MSE (mean of seeds)', 'train MSE', 'split matches ckpt'],
            [(tag, C.fmt(e.get('ckpt_epoch')), C.fmt(((e.get('splits') or {}).get('val') or {}).get('mse_all_mean')),
              C.fmt(((e.get('splits') or {}).get('train') or {}).get('mse_all_mean')),
              C.fmt(e.get('split_matches_checkpoint'))) for tag, e in run.evals.items()]),
            '', f'Details: `python3 train/analysis/eval_report.py {run.folder or "FOLDER"} {run.name}`', '']

    md += ['## Files', '']
    if topk_png:
        md += [f'![val MSE and kept checkpoints]({C.rel(topk_png, out_dir)})', '']
    plots = run.plots() or regen
    if regen:
        md.append('Regenerated plots (the run had no plots/):')
    for p in plots:
        md.append(f'- [{p.name}]({C.rel(p, out_dir)})')
    for p in plots:
        if p.name in ('loss_per_epoch.png', 'action_mse_per_epoch.png'):
            md += ['', f'![{p.stem}]({C.rel(p, out_dir)})']
    vids = run.videos()
    if vids:
        md += ['', 'Videos:']
        md += [f'- [{v.relative_to(run.path).as_posix()}]({C.rel(v, out_dir)})' for v in vids]
    for extra in ('README.md', 'logs/train_log.txt', 'logs/slurm_job.txt'):
        if (run.path / extra).exists():
            md.append(f'- [{extra}]({C.rel(run.path / extra, out_dir)})')
    if not run.evals:
        md += ['', 'No offline eval yet. Make one with '
               f'`train/analysis/eval.sh {run.folder or "FOLDER"} {run.name} --ckpt top1 --per-demo 8`, then '
               f'`train/fetch.sh {run.folder or "FOLDER"} {run.name}` and `train/analysis/eval_report.py`.']
    if notes:
        md += ['', '## Notes', ''] + [f'- {n}' for n in notes]
    C.write_text(out_dir / 'report.md', '\n'.join(md))

    # stdout
    lines = [f'{run.name}: status {run.status}, mode {run.mode or "-"}, dataset {run.dataset or "-"}, '
             f'{C.fmt(run.epochs_completed)}/{C.fmt(run.epochs_configured)} epochs, '
             f'{C.fmt_minutes(s.get("training_minutes"))}',
             f'  best val action MSE {C.fmt(best)} @ epoch {C.fmt(best_ep)}; final {C.fmt(final)} @ {C.fmt(final_ep)}']
    if comps:
        lines.append('  at best: ' + ', '.join(f'{lab} {C.fmt(v)}' for lab, v, _ in comps))
    if run.checkpoints:
        lines.append('  kept: ' + ', '.join(f'top{c.rank} ep {c.epoch} ({C.fmt(c.val_action_mse)})'
                                            for c in run.checkpoints))
    if ov.get('verdict'):
        lines.append(f'  {ov["verdict"]}')
    for n in notes:
        lines.append(f'  note: {n}')
    print('\n'.join(lines))
    return ov


def main(argv=None):
    p = C.common_parser(__doc__.split('\n\n')[0], epilog=__doc__.split('\n\n', 1)[1])
    p.add_argument('folder', help='task folder under the data root (or a path to a run dir)')
    p.add_argument('run', nargs='?', help='run name or unique prefix under <folder>/policy/ (default: newest), '
                                          'or a path to a run dir')
    C.add_data_root_arg(p)
    p.add_argument('--out', help='output dir (default <run>/analysis/)')
    a = p.parse_args(argv)
    run_dir = C.resolve_run(a.data_root, a.folder, a.run)
    run = C.load_run(run_dir)
    out = Path(a.out) if a.out else run_dir / 'analysis'
    out.mkdir(parents=True, exist_ok=True)
    build(run, out)
    return 0


if __name__ == '__main__':
    C.run_cli(main)
