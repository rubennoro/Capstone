#!/usr/bin/env python3
"""Turn an offline wandb run into CSVs and loss-curve PNGs, no wandb account needed.

Reads the binary run-<id>.wandb file with wandb's own reader and writes:
  history.csv            every logged step (train_loss, lr, grad_norm, ...)
  epochs.csv             one row per epoch (epoch-average train loss, val loss, action MSE)
  loss_per_step.png      train loss over every gradient step
  loss_per_epoch.png     epoch-average train loss vs val loss
  action_mse_per_epoch.png   action error split into position / rotation / gripper width
  lr_and_grad_norm.png   learning-rate schedule and gradient norm
  gpu_memory.png         (only with --gpu-csv) GPU memory from an nvidia-smi CSV log

Needs wandb (same version that wrote the run), pandas and matplotlib; the cluster's `bp` env has all three.
Usage:
  python wandb_offline_to_csv.py <offline-run-dir or run-*.wandb> -o <out_dir> [--gpu-csv smoke_gpu.csv]
"""
import argparse
import glob
import json
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd
from wandb.proto import wandb_internal_pb2
from wandb.sdk.internal import datastore

# Validated categorical slots (light surface), assigned in fixed order: train, val, third series
TRAIN_COLOR = '#2a78d6'
VAL_COLOR = '#eb6834'
THIRD_COLOR = '#1baf7a'
SURFACE = '#fcfcfb'
TEXT_PRIMARY = '#0b0b0b'
TEXT_SECONDARY = '#52514e'
GRID = '#e4e3dd'
MAX_EPOCH_TICKS = 20  # above this, let matplotlib choose ticks and skip per-epoch boundary lines


def read_history(wandb_file):
    store = datastore.DataStore()
    store.open_for_scan(wandb_file)
    rows = []
    while True:
        data = store.scan_data()
        if data is None:
            break
        record = wandb_internal_pb2.Record()
        record.ParseFromString(data)
        if record.WhichOneof('record_type') != 'history':
            continue
        row = {}
        for item in record.history.item:
            key = item.key or '/'.join(item.nested_key)
            row[key] = json.loads(item.value_json)
        # Drop media entries (videos/images): either a dict value or flattened fields under a `<prefix>/_type`
        media_prefixes = [k[:-len('/_type')] for k in row if k.endswith('/_type')]
        row = {k: v for k, v in row.items()
               if not isinstance(v, dict) and not any(k.startswith(p + '/') for p in media_prefixes)}
        rows.append(row)
    return pd.DataFrame(rows)


def style_axes(ax, title, xlabel, ylabel):
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


def new_figure(ncols=1, width=7.5, height=4.0):
    fig, axes = plt.subplots(1, ncols, figsize=(width, height), facecolor=SURFACE, squeeze=False)
    return fig, axes[0]


def save(fig, path):
    fig.tight_layout()
    fig.savefig(path, dpi=150, facecolor=SURFACE)
    plt.close(fig)
    print('wrote', path)


def label_line_end(ax, x, y, text, color_hex, fmt='{:.4f}', dy=0):
    # Direct label at the last point; text stays in ink, a small colored marker carries identity.
    # dy (points) nudges the label up/down so labels of nearby series don't collide.
    ax.plot([x], [y], marker='o', markersize=8, color=color_hex, markeredgecolor=SURFACE, markeredgewidth=2, zorder=5)
    ax.annotate(f'{text} {fmt.format(y)}', (x, y), xytext=(8, dy), textcoords='offset points',
                va='center', fontsize=9, color=TEXT_PRIMARY)


def plot_loss_per_step(history, epoch_ends, out_dir):
    df = history.dropna(subset=['train_loss'])
    fig, (ax,) = new_figure()
    ax.plot(df['global_step'], df['train_loss'], color=TRAIN_COLOR, alpha=0.25, linewidth=1)
    smooth = df['train_loss'].rolling(25, min_periods=1).mean()
    ax.plot(df['global_step'], smooth, color=TRAIN_COLOR, linewidth=2)
    if len(epoch_ends) <= MAX_EPOCH_TICKS:
        for step in epoch_ends[:-1]:
            ax.axvline(step, color=TEXT_SECONDARY, linewidth=0.8, linestyle=':')
    ax.set_yscale('log')
    style_axes(ax, 'Train loss per step (25-step average; raw in light)', 'gradient step', 'train loss (log scale)')
    label_line_end(ax, df['global_step'].iloc[-1], smooth.iloc[-1], 'final avg', TRAIN_COLOR)
    save(fig, os.path.join(out_dir, 'loss_per_step.png'))


def epoch_line_style(n_points):
    # Markers only when points are sparse enough to read individually
    if n_points <= MAX_EPOCH_TICKS:
        return dict(linewidth=2, marker='o', markersize=8, markeredgecolor=SURFACE, markeredgewidth=2)
    return dict(linewidth=2)


def set_epoch_ticks(ax, epochs):
    if len(epochs) <= MAX_EPOCH_TICKS:
        ax.set_xticks(epochs['epoch'])


def plot_loss_per_epoch(epochs, out_dir):
    fig, (ax,) = new_figure()
    series = [(key, label, color_hex) for key, label, color_hex in
              [('train_loss_epoch_average', 'train', TRAIN_COLOR), ('val_loss', 'val', VAL_COLOR)] if key in epochs]
    finals = {key: epochs[key].dropna().iloc[-1] for key, _, _ in series}
    for key, label, color_hex in series:
        df = epochs.dropna(subset=[key])
        ax.plot(df['epoch'], df[key], color=color_hex, label=label, **epoch_line_style(len(df)))
        # Higher final value gets its label nudged up, lower one down
        dy = 7 if finals[key] == max(finals.values()) else -7
        label_line_end(ax, df['epoch'].iloc[-1], df[key].iloc[-1], label, color_hex, dy=dy if len(finals) > 1 else 0)
    set_epoch_ticks(ax, epochs)
    style_axes(ax, 'Loss per epoch', 'epoch', 'diffusion loss')
    ax.legend(frameon=False, labelcolor=TEXT_PRIMARY)
    ax.margins(x=0.15)
    save(fig, os.path.join(out_dir, 'loss_per_epoch.png'))


def plot_action_mse(epochs, out_dir):
    parts = [('right_pos', 'Position'), ('right_rot', 'Rotation'), ('right_gripper_width', 'Gripper width')]
    fig, axes = new_figure(ncols=3, width=12, height=3.8)
    for ax, (part, title) in zip(axes, parts):
        for split, color_hex in (('train', TRAIN_COLOR), ('val', VAL_COLOR)):
            key = f'{split}/action_mse_error_{part}/all'
            if key not in epochs:
                continue
            df = epochs.dropna(subset=[key])
            ax.plot(df['epoch'], df[key], color=color_hex, label=split, **epoch_line_style(len(df)))
        set_epoch_ticks(ax, epochs)
        ax.ticklabel_format(axis='y', style='sci', scilimits=(-3, 3))
        style_axes(ax, f'{title} action MSE', 'epoch', 'MSE')
    axes[0].legend(frameon=False, labelcolor=TEXT_PRIMARY)
    save(fig, os.path.join(out_dir, 'action_mse_per_epoch.png'))


def plot_lr_and_grad_norm(history, out_dir):
    # Two measures on different scales -> small multiples, never a dual axis
    fig, (ax_lr, ax_grad) = new_figure(ncols=2, width=12, height=3.8)
    df = history.dropna(subset=['lr'])
    ax_lr.plot(df['global_step'], df['lr'], color=TRAIN_COLOR, linewidth=2)
    style_axes(ax_lr, 'Learning rate', 'gradient step', 'lr')
    df = history.dropna(subset=['grad_norm'])
    ax_grad.plot(df['global_step'], df['grad_norm'], color=TRAIN_COLOR, alpha=0.25, linewidth=1)
    ax_grad.plot(df['global_step'], df['grad_norm'].rolling(25, min_periods=1).mean(), color=TRAIN_COLOR, linewidth=2)
    if 'grad_norm_clipped' in df:
        clip = df['grad_norm_clipped'].dropna()
        if len(clip):
            ax_grad.axhline(clip.iloc[-1], color=TEXT_SECONDARY, linewidth=1, linestyle='--')
            ax_grad.annotate(f'clip at {clip.iloc[-1]:g}', (df['global_step'].iloc[0], clip.iloc[-1]),
                             xytext=(0, 4), textcoords='offset points', fontsize=9, color=TEXT_SECONDARY)
    style_axes(ax_grad, 'Gradient norm (before clipping; 25-step average)', 'gradient step', 'grad norm')
    save(fig, os.path.join(out_dir, 'lr_and_grad_norm.png'))


def plot_gpu_memory(gpu_csv, out_dir):
    df = pd.read_csv(gpu_csv, skipinitialspace=True)
    df.columns = [c.strip() for c in df.columns]
    time_col = [c for c in df.columns if c.startswith('timestamp')][0]
    mem_col = [c for c in df.columns if c.startswith('memory.used')][0]
    df['t'] = pd.to_datetime(df[time_col], format='%Y/%m/%d %H:%M:%S.%f')
    df['minutes'] = (df['t'] - df['t'].iloc[0]).dt.total_seconds() / 60
    df['gb'] = df[mem_col].astype(str).str.replace(' MiB', '').astype(float) / 1024
    fig, (ax,) = new_figure(height=3.5)
    ax.plot(df['minutes'], df['gb'], color=TRAIN_COLOR, linewidth=2)
    peak = df['gb'].idxmax()
    label_line_end(ax, df['minutes'][peak], df['gb'][peak], 'peak', TRAIN_COLOR, fmt='{:.1f} GB', dy=-10)
    style_axes(ax, 'GPU memory used', 'minutes since job start', 'GB')
    save(fig, os.path.join(out_dir, 'gpu_memory.png'))


def find_wandb_file(run):
    return run if run.endswith('.wandb') else glob.glob(os.path.join(run, 'run-*.wandb'))[0]


def epoch_table(history):
    # Epoch-level metrics are logged on the last step of each epoch
    epoch_keys = [c for c in history.columns if c.startswith(('val', 'train/')) or c == 'train_loss_epoch_average']
    epochs = history.dropna(subset=epoch_keys, how='all')[['epoch', 'global_step'] + epoch_keys]
    return epochs.groupby('epoch', as_index=False).last()


def make_plots(history, epochs, out_dir, gpu_csv=None):
    os.makedirs(out_dir, exist_ok=True)
    plot_loss_per_step(history, list(epochs['global_step']), out_dir)
    plot_loss_per_epoch(epochs, out_dir)
    plot_action_mse(epochs, out_dir)
    plot_lr_and_grad_norm(history, out_dir)
    if gpu_csv and os.path.exists(gpu_csv):
        plot_gpu_memory(gpu_csv, out_dir)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('run', help='offline-run-* directory or its run-*.wandb file')
    parser.add_argument('-o', '--out-dir', required=True)
    parser.add_argument('--gpu-csv', help='optional nvidia-smi CSV log to plot GPU memory')
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    history = read_history(find_wandb_file(args.run))
    history.to_csv(os.path.join(args.out_dir, 'history.csv'), index=False)
    epochs = epoch_table(history)
    epochs.to_csv(os.path.join(args.out_dir, 'epochs.csv'), index=False)
    print(f'{len(history)} steps, {len(epochs)} epochs')
    make_plots(history, epochs, args.out_dir, args.gpu_csv)


if __name__ == '__main__':
    main()
