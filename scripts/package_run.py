#!/usr/bin/env python3
"""Package one training run into the standard results folder (results/<task>/<run>/).

Every run, smoke or full, successful or not, gets the same layout and analysis:
  README.md  summary.json  epochs.csv  history.csv
  plots/        loss / action MSE / lr + grad norm / GPU memory (scripts/wandb_offline_to_csv.py)
  videos/       epoch_XXXX/ per rollout + final/
  logs/         train_log.txt  slurm_job.txt  gpu_memory.csv
  run_config/   config.yaml  overrides.yaml  normalizer.pkl  job_info.json
  checkpoints/  best checkpoint only, slimmed (gitignored; stays on the cluster)

Never deletes anything from the source run dir, so it can be re-run on old runs.
Needs the `bp` env (wandb, pandas, matplotlib, pyyaml, torch for slimming).

Usage:
  python package_run.py --run-dir <hydra run dir> --out results/<task>/<run> --status ok \
      [--job-info job_info.json] [--wandb-run <offline-run dir>] [--gpu-csv <csv>] [--slurm-log <.out>] [--note "..."]
"""
import argparse
import datetime
import glob
import json
import os
import re
import shutil
import sys

import yaml

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from wandb_offline_to_csv import epoch_table, find_wandb_file, make_plots, read_history  # noqa: E402

CLUSTER_LOGIN = 'noroian.r@xfer.discovery.neu.edu'
PROGRESS_LINE = re.compile(r'(it/s|s/it)\]?\s*$')


def load_json(path):
    if path and os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return {}


def strip_progress_lines(src, dst):
    """Copy a console log without tqdm progress-bar redraws."""
    with open(src, errors='replace') as f:
        lines = f.read().replace('\r', '\n').split('\n')
    kept = [line for line in lines if line.strip() and not PROGRESS_LINE.search(line)]
    with open(dst, 'w') as f:
        f.write('\n'.join(kept) + '\n')
    return kept


def find_best_checkpoint(ckpt_dir):
    """Lowest val action MSE in the filename; falls back to train loss, then latest.ckpt."""
    for pattern, key in [('*val_action_mse_error_all=*.ckpt', 'val_action_mse_error_all'),
                         ('*train_action_mse_error_all=*.ckpt', 'train_action_mse_error_all'),
                         ('*val_loss=*.ckpt', 'val_loss'), ('*train_loss=*.ckpt', 'train_loss')]:
        candidates = []
        for path in glob.glob(os.path.join(ckpt_dir, pattern)):
            match = re.search(rf'epoch=(\d+)-{key}=([0-9.eE+-]+)\.ckpt$', os.path.basename(path))
            if match:
                candidates.append((float(match.group(2)), int(match.group(1)), path, key))
        if candidates:
            value, epoch, path, key = min(candidates)
            return {'path': path, 'epoch': epoch, 'metric': key, 'value': value}
    latest = os.path.join(ckpt_dir, 'latest.ckpt')
    if os.path.exists(latest):
        return {'path': latest, 'epoch': None, 'metric': None, 'value': None}
    return None


def copy_videos(run_dir, out_dir):
    vis_dir = os.path.join(run_dir, 'vis')
    copied = []
    if not os.path.isdir(vis_dir):
        return copied
    for epoch_dir in sorted(glob.glob(os.path.join(vis_dir, 'epoch_*'))):
        dst = os.path.join(out_dir, 'videos', os.path.basename(epoch_dir))
        shutil.copytree(epoch_dir, dst, dirs_exist_ok=True)
        copied.append(os.path.basename(epoch_dir))
    finals = glob.glob(os.path.join(vis_dir, '*.mp4'))
    if finals:
        os.makedirs(os.path.join(out_dir, 'videos', 'final'), exist_ok=True)
        for video in finals:
            shutil.copy2(video, os.path.join(out_dir, 'videos', 'final'))
        copied.append('final')
    return copied


def fmt(value, digits=4):
    if value is None or (isinstance(value, float) and value != value):
        return 'n/a'
    if isinstance(value, float):
        return f'{value:.{digits}f}'
    return str(value)


def gpu_peak_gb(gpu_csv):
    if not gpu_csv or not os.path.exists(gpu_csv):
        return None
    peak = 0.0
    with open(gpu_csv) as f:
        for line in f:
            match = re.search(r',\s*([0-9.]+) MiB', line)
            if match:
                peak = max(peak, float(match.group(1)) / 1024)
    return round(peak, 1) if peak else None


def build_summary(args, cfg, epochs, history, best, job_info, gpu_csv):
    summary = {
        'run': os.path.basename(os.path.normpath(args.out)),
        'task': job_info.get('task'),
        'dataset': job_info.get('dataset'),
        'mode': job_info.get('mode'),
        'status': args.status,
        'note': args.note,
        'job_id': job_info.get('job_id'),
        'packaged': datetime.datetime.now().isoformat(timespec='seconds'),
        'num_epochs_configured': cfg.get('training', {}).get('num_epochs'),
        'gpu_peak_gb': gpu_peak_gb(gpu_csv),
    }
    if history is not None and len(history):
        summary['epochs_completed'] = int(history['epoch'].max()) + 1
        summary['steps'] = int(history['global_step'].max()) + 1
        summary['steps_per_epoch'] = round(summary['steps'] / summary['epochs_completed'])
        if 'iterations_per_second' in history:
            summary['iterations_per_second'] = round(float(history['iterations_per_second'].dropna().median()), 2)
            summary['minutes_per_epoch'] = round(summary['steps_per_epoch'] / summary['iterations_per_second'] / 60, 2)
        if '_runtime' in history:
            summary['training_minutes'] = round(float(history['_runtime'].max()) / 60, 1)
    if epochs is not None and len(epochs):
        last = epochs.iloc[-1]
        summary['final'] = {k: (None if last[k] != last[k] else float(last[k])) for k in
                            ['train_loss_epoch_average', 'val_loss', 'val/action_mse_error/all', 'train/action_mse_error/all']
                            if k in epochs}
    if best:
        summary['best_checkpoint'] = {'epoch': best['epoch'], 'metric': best['metric'], 'value': best['value'],
                                      'file': best.get('packaged_name')}
    return summary


def write_readme(path, args, cfg, overrides, epochs, summary, job_info, videos, best, log_tail):
    task, run = summary['task'], summary['run']
    training, dataloader = cfg.get('training', {}), cfg.get('dataloader', {})
    lines = [f'# {task}: {run}', '']
    status_line = {'ok': 'completed', 'failed': '**FAILED**', 'timeout': '**stopped at the time limit**'}.get(args.status, args.status)
    lines += [f"**Status:** {status_line}  ·  **Mode:** {summary.get('mode') or 'n/a'}  ·  **Job:** {summary.get('job_id') or 'n/a'}"
              f"  ·  **Packaged:** {summary['packaged']}", '']
    if args.note:
        lines += [f'> {args.note}', '']

    lines += ['## Data', '']
    dataset_cfg = cfg.get('task', {}).get('dataset', {})
    rows = [
        ('Dataset', f"`{job_info.get('dataset', 'n/a')}`"),
        ('Path', f"`{dataset_cfg.get('dataset_path', 'n/a')}`"),
        ('Train / val split', f"val_ratio {dataset_cfg.get('val_ratio')} per task, seed {dataset_cfg.get('seed')}"),
    ]
    lines += ['| | |', '|---|---|'] + [f'| {k} | {v} |' for k, v in rows] + ['']

    lines += ['## Training', '']
    rows = [
        ('Policy config', f"`{job_info.get('config_name', 'umi_policy_dunetp')}`"),
        ('Epochs', f"{summary.get('epochs_completed', 'n/a')} completed of {training.get('num_epochs')} configured"),
        ('Batch size', dataloader.get('batch_size')),
        ('Learning rate', f"{cfg.get('optimizer', {}).get('lr')} ({training.get('lr_scheduler')}, warmup {training.get('lr_warmup_proportion')})"),
        ('Precision', training.get('mixed_precision')),
        ('Evaluate / rollout every', f"{training.get('val_every')} / {training.get('rollout_every')} epochs"),
        ('Hardware', f"{job_info.get('gpu_name', 'n/a')} on `{job_info.get('node', 'n/a')}` ({job_info.get('partition', 'n/a')})"),
        ('Command-line overrides', '<br>'.join(f'`{o}`' for o in overrides) or 'none'),
    ]
    lines += ['| | |', '|---|---|'] + [f'| {k} | {v} |' for k, v in rows] + ['']

    lines += ['## Results', '']
    if epochs is not None and len(epochs):
        cols = [('train_loss_epoch_average', 'Train loss'), ('val_loss', 'Val loss'), ('val/action_mse_error/all', 'Val action MSE')]
        cols = [(k, label) for k, label in cols if k in epochs]
        evaluated = epochs.dropna(subset=['val_loss']) if 'val_loss' in epochs else epochs
        table = evaluated if len(evaluated) else epochs
        lines += ['| Epoch | ' + ' | '.join(label for _, label in cols) + ' |', '|---' * (len(cols) + 1) + '|']
        for _, row in table.iterrows():
            lines.append(f"| {int(row['epoch'])} | " + ' | '.join(fmt(row[k]) for k, _ in cols) + ' |')
        lines.append('')
    else:
        lines += ['_No metrics were logged._', '']
    stats = [
        ('Speed', f"{fmt(summary.get('iterations_per_second'), 2)} steps/s, {summary.get('steps_per_epoch', 'n/a')} steps/epoch ≈ {fmt(summary.get('minutes_per_epoch'), 1)} min/epoch"),
        ('Training time', f"{fmt(summary.get('training_minutes'), 1)} min"),
        ('Peak GPU memory', f"{fmt(summary.get('gpu_peak_gb'), 1)} GB"),
    ]
    if best:
        where = f"`checkpoints/{best['packaged_name']}`" if best.get('packaged_name') else f"`{best['path']}`"
        stats.append(('Best checkpoint', f"epoch {best['epoch']}, {best['metric']} = {fmt(best['value'], 5)}: {where} (slimmed, not in git)"))
    lines += [f'- **{k}:** {v}' for k, v in stats] + ['']
    if best and best.get('packaged_name'):
        lines += ['Fetch the checkpoint to the laptop (from the repo root):', '```bash',
                  f"rsync -a {CLUSTER_LOGIN}:CapstoneProject/Capstone/results/{task}/{run}/checkpoints/ results/{task}/{run}/checkpoints/",
                  '```', '']
    if args.status != 'ok' and log_tail:
        lines += ['### Last log lines', '```', *log_tail, '```', '']

    lines += ['## Files', '', '| Path | Contents |', '|---|---|',
              '| `plots/loss_per_step.png` | Train loss at every gradient step (log scale) |',
              '| `plots/loss_per_epoch.png` | Epoch-average train loss vs val loss |',
              '| `plots/action_mse_per_epoch.png` | Action error split into position / rotation / gripper width, train vs val |',
              '| `plots/lr_and_grad_norm.png` | Learning-rate schedule and gradient norm |',
              '| `plots/gpu_memory.png` | GPU memory over the job |',
              '| `epochs.csv`, `history.csv` | Per-epoch metrics; every logged step |',
              '| `summary.json` | Machine-readable summary of this page |']
    if videos:
        lines.append(f"| `videos/` | Rollout + attention videos per evaluation: {', '.join(f'`{v}`' for v in videos)} |")
    lines += ['| `run_config/` | Hydra config + overrides, normalizer, job_info.json |',
              '| `logs/` | Training log, Slurm console log (progress bars stripped), GPU memory samples |',
              '| `checkpoints/` | Best checkpoint only, slimmed (no optimizer state); gitignored, lives on the cluster |', '',
              '**Videos:** `rollout_*.mp4` replays a training demo at the policy\'s input resolution; '
              '`attention_and_rollout_*.mp4` shows the current frame (top) and the prompt demo\'s frames (bottom), each tinted by '
              'how much the policy attends to it (pale green = low, darker blue = high). In a well-trained policy, the '
              'high-attention band moves through the prompt as the task progresses.', '']
    with open(path, 'w') as f:
        f.write('\n'.join(lines))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run-dir', required=True, help='Hydra run dir of the training job')
    parser.add_argument('--out', required=True, help='results/<task>/<run> folder to (re)create')
    parser.add_argument('--status', default='ok', choices=['ok', 'failed', 'timeout'])
    parser.add_argument('--job-info', help='default: <run-dir>/job_info.json')
    parser.add_argument('--wandb-run', help='default: newest <run-dir>/wandb/offline-run-*')
    parser.add_argument('--gpu-csv')
    parser.add_argument('--slurm-log')
    parser.add_argument('--note', default='')
    parser.add_argument('--no-checkpoint', action='store_true', help='skip finding/slimming the best checkpoint')
    args = parser.parse_args()

    run_dir, out = args.run_dir, args.out
    os.makedirs(out, exist_ok=True)
    job_info = load_json(args.job_info or os.path.join(run_dir, 'job_info.json'))

    # Config
    cfg, overrides = {}, []
    os.makedirs(os.path.join(out, 'run_config'), exist_ok=True)
    for name in ['config.yaml', 'overrides.yaml']:
        src = os.path.join(run_dir, '.hydra', name)
        if os.path.exists(src):
            shutil.copy2(src, os.path.join(out, 'run_config', name))
    if os.path.exists(os.path.join(run_dir, '.hydra', 'config.yaml')):
        cfg = yaml.safe_load(open(os.path.join(run_dir, '.hydra', 'config.yaml'))) or {}
    if os.path.exists(os.path.join(run_dir, '.hydra', 'overrides.yaml')):
        overrides = yaml.safe_load(open(os.path.join(run_dir, '.hydra', 'overrides.yaml'))) or []
    for src in [os.path.join(run_dir, 'normalizer.pkl'), args.job_info or os.path.join(run_dir, 'job_info.json')]:
        if src and os.path.exists(src):
            shutil.copy2(src, os.path.join(out, 'run_config'))

    # Metrics + plots
    history = epochs = None
    wandb_run = args.wandb_run or (sorted(glob.glob(os.path.join(run_dir, 'wandb', 'offline-run-*'))) or [None])[-1]
    if wandb_run:
        try:
            history = read_history(find_wandb_file(wandb_run))
            if len(history):
                epochs = epoch_table(history)
                history.to_csv(os.path.join(out, 'history.csv'), index=False)
                epochs.to_csv(os.path.join(out, 'epochs.csv'), index=False)
                make_plots(history, epochs, os.path.join(out, 'plots'), args.gpu_csv)
        except Exception as e:  # a broken/partial wandb file must not stop the rest of the packaging
            print(f'WARNING: could not read wandb run {wandb_run}: {e}')

    # Logs
    os.makedirs(os.path.join(out, 'logs'), exist_ok=True)
    if os.path.exists(os.path.join(run_dir, 'train.log')):
        shutil.copy2(os.path.join(run_dir, 'train.log'), os.path.join(out, 'logs', 'train_log.txt'))
    log_tail = []
    if args.slurm_log and os.path.exists(args.slurm_log):
        log_tail = strip_progress_lines(args.slurm_log, os.path.join(out, 'logs', 'slurm_job.txt'))[-40:]
    if args.gpu_csv and os.path.exists(args.gpu_csv):
        shutil.copy2(args.gpu_csv, os.path.join(out, 'logs', 'gpu_memory.csv'))

    videos = copy_videos(run_dir, out)

    # Best checkpoint, slimmed
    best = None if args.no_checkpoint else find_best_checkpoint(os.path.join(run_dir, 'checkpoints'))
    if best:
        from slim_checkpoint import slim_checkpoint  # needs torch; imported only when there's a checkpoint
        os.makedirs(os.path.join(out, 'checkpoints'), exist_ok=True)
        best['packaged_name'] = 'best_' + os.path.basename(best['path'])
        slim_checkpoint(best['path'], os.path.join(out, 'checkpoints', best['packaged_name']))

    summary = build_summary(args, cfg, epochs, history, best, job_info, args.gpu_csv)
    with open(os.path.join(out, 'summary.json'), 'w') as f:
        json.dump(summary, f, indent=2)
    write_readme(os.path.join(out, 'README.md'), args, cfg, overrides, epochs, summary, job_info, videos, best, log_tail)
    print(f"Packaged {summary['run']} ({args.status}) into {out}")


if __name__ == '__main__':
    main()
