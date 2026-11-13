"""Synthetic inputs for the analysis tests, mirroring the train contract layout and the real file schemas:
package_run.py (summary.json, epochs.csv, history.csv), train.sbatch (checkpoints/checkpoints.json, STATUS,
train_meta.json), offline_eval.py (eval.json), dataproc (task folder, package.json, report.json, reason.json)."""
from __future__ import annotations

import csv
import json
import math
import shutil
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[3]
RESULTS = REPO / 'results' / 'block_raise'
COMPS = ['right_pos', 'right_rot', 'right_gripper_width']


def wjson(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=1))


def make_run(data_root, folder, name, n_epochs=30, best_epoch=12, val_every=1, mode='full', status='ok',
             topk=3, dataset='ds1', write_ckpt_files=True, with_summary=True, with_epochs=True):
    """A fetched run dir under <data_root>/<folder>/policy/<name>/ with a U-shaped val MSE (min at best_epoch)."""
    run = Path(data_root) / folder / 'policy' / name
    run.mkdir(parents=True, exist_ok=True)
    rows, hist = [], []
    steps_per_epoch = 100
    for ep in range(n_epochs):
        tl = 0.5 * math.exp(-ep / 4) + 0.01 * (1 - ep / (3 * n_epochs))
        row = {'epoch': ep, 'global_step': (ep + 1) * steps_per_epoch - 1, 'train_loss_epoch_average': tl}
        if ep % val_every == 0 or ep == n_epochs - 1:
            v = 1e-4 * (1 + 0.02 * (ep - best_epoch) ** 2) + (0.2 if ep == 0 else 0)
            row['val/action_mse_error/all'] = v
            row['train/action_mse_error/all'] = v / (1 + ep / 5)
            row['val_loss'] = 0.02 * (1 + 0.01 * max(0, ep - best_epoch) ** 1.5) + (0.2 if ep == 0 else 0)
            for i, c in enumerate(COMPS):
                f = [0.5, 1.4, 0.03][i]
                row[f'val/action_mse_error_{c}/all'] = v * f
                row[f'train/action_mse_error_{c}/all'] = v * f / (1 + ep / 5)
        rows.append(row)
        for s in range(0, steps_per_epoch, 25):
            hist.append({'global_step': ep * steps_per_epoch + s, 'epoch': ep, 'train_loss': tl * 1.1,
                         'lr': 1e-4, 'grad_norm': 1.0, 'grad_norm_clipped': 0.5, 'iterations_per_second': 4.6,
                         '_runtime': (ep * steps_per_epoch + s) / 4.6})
        hist[-1].update(row)
    if with_epochs:
        keys = sorted({k for r in rows for k in r}, key=lambda k: (k != 'epoch', k != 'global_step', k))
        with open(run / 'epochs.csv', 'w', newline='') as f:
            w = csv.DictWriter(f, fieldnames=keys)
            w.writeheader()
            w.writerows(rows)
        hkeys = sorted({k for r in hist for k in r})
        with open(run / 'history.csv', 'w', newline='') as f:
            w = csv.DictWriter(f, fieldnames=hkeys)
            w.writeheader()
            w.writerows(hist)
    vals = {r['epoch']: r['val/action_mse_error/all'] for r in rows if 'val/action_mse_error/all' in r}
    ranked = sorted(vals.items(), key=lambda kv: kv[1])[:topk]
    if with_summary:
        wjson(run / 'summary.json', {
            'run': name, 'task': 'block_raise', 'dataset': dataset, 'mode': mode, 'status': status, 'note': 'syn',
            'job_id': '123', 'packaged': '2026-11-13T10:00:00', 'num_epochs_configured': n_epochs,
            'gpu_peak_gb': 30.5, 'epochs_completed': n_epochs, 'steps': n_epochs * steps_per_epoch,
            'steps_per_epoch': steps_per_epoch, 'iterations_per_second': 4.6, 'minutes_per_epoch': 0.36,
            'training_minutes': n_epochs * 0.36,
            'final': {'val/action_mse_error/all': rows[-1].get('val/action_mse_error/all')},
            'best_checkpoint': {'epoch': ranked[0][0], 'metric': 'val_action_mse_error_all',
                                'value': round(ranked[0][1], 7), 'file': None}})
    (run / 'STATUS').write_text(status + '\n')
    wjson(run / 'train_meta.json', {'folder': folder, 'label': 'block_raise', 'dataset': dataset, 'run': name,
                                    'mode': mode, 'epochs': n_epochs, 'topk': topk, 'note': f'note {name}',
                                    'overrides': ['training.lr=1e-4'], 'submitted': '2026-11-13T08:00:00',
                                    'clone_git_commit': 'abc1234'})
    ck = []
    (run / 'checkpoints').mkdir(exist_ok=True)
    for rank, (ep, v) in enumerate(ranked, 1):
        fname = f'top{rank}_epoch={ep:04d}-val_action_mse_error_all={v:.7f}.ckpt'
        if write_ckpt_files:
            (run / 'checkpoints' / fname).write_bytes(b'x' * (100 * rank))
        ck.append({'rank': rank, 'file': fname, 'epoch': ep, 'val_action_mse': round(v, 7),
                   'size_bytes': 2_500_000_000})
    wjson(run / 'checkpoints' / 'checkpoints.json', ck)
    (run / 'plots').mkdir(exist_ok=True)
    return run


def per_step(scale, rng):
    out = {}
    for k in (0, 4, 8, 15):
        g = 1 + k / 4
        out[f'k{k}'] = {'pos_mm_median': scale * g, 'pos_mm_mean': scale * g * 1.2, 'pos_mm_p90': scale * g * 2,
                        'rot_deg_median': 0.5 * scale * g, 'rot_deg_mean': 0.6 * scale * g,
                        'width_mm_median': 0.2 * scale * g, 'width_mm_mean': 0.25 * scale * g,
                        'gt_displacement_mm_median': 3.0 * k + float(rng.uniform(0, 0.1))}
    return out


def make_eval(run_dir, tag='top1', epoch=12, val_mse=1.6e-4, n_demos=12, per_demo=True, seed=0):
    rng = np.random.default_rng(seed)
    res = {'ckpt_epoch': epoch, 'split_matches_checkpoint': True, 'precision': 'fp32', 'sample_size': 256,
           'index_seed': 42, 'batch_size': 64, 'micro_batch': 4, 'len_train': 30000, 'len_val': 1500,
           'noise_seeds': [0, 1, 2, 3, 4], 'splits': {}}
    for name, m, sc in (('val', val_mse, 2.0), ('train', val_mse / 3, 1.0)):
        seeds = list(m * (1 + 0.05 * rng.standard_normal(5)))
        res['splits'][name] = {'mse_all_per_seed': seeds, 'mse_all_mean': float(np.mean(seeds)),
                               'mse_components_mean': {'all': float(np.mean(seeds)), 'right_pos': m * 0.5,
                                                       'right_rot': m * 1.4, 'right_gripper_width': m * 0.03},
                               'episodes_sampled': [0, 1, 2], 'per_step': per_step(sc, rng), 'seconds': 12.0}
    if per_demo:
        demos = []
        for i in range(n_demos):
            sp = 'val' if i in (3, 7) else ('new' if i == n_demos - 1 else 'train')
            m = val_mse * (2 if sp == 'val' else 0.5) * (1 + rng.uniform(0, 2)) * (10 if i == 5 else 1)
            demos.append({'episode': f'demo_{i:03d}', 'dataset_split': 'val' if sp == 'val' else 'train',
                          'checkpoint_split': sp, 'n': 8,
                          'mse': {'all': m, 'right_pos': m * 0.5, 'right_rot': m * 1.4,
                                  'right_gripper_width': m * 0.03},
                          'per_step': per_step(1000 * m, rng)})
        res['per_demo'] = demos
    wjson(Path(run_dir) / 'eval' / tag / 'eval.json', res)
    wjson(Path(run_dir) / 'eval' / tag / 'eval_job.json', {'job_id': '999', 'ckpt': tag})
    return res


def copy_results_run(src: Path, data_root, folder, name=None):
    """Copy a packaged results/ run into <data_root>/<folder>/policy/ without its (multi-GB) checkpoint files:
    each *.ckpt becomes an empty file with the same name."""
    dst = Path(data_root) / folder / 'policy' / (name or src.name)
    shutil.copytree(src, dst, ignore=shutil.ignore_patterns('*.ckpt', '*.mp4'))
    for c in (src / 'checkpoints').glob('*.ckpt') if (src / 'checkpoints').is_dir() else []:
        (dst / 'checkpoints').mkdir(exist_ok=True)
        (dst / 'checkpoints' / c.name).write_bytes(b'')
    for v in (src / 'videos').rglob('*.mp4') if (src / 'videos').is_dir() else []:
        t = dst / v.relative_to(src)
        t.parent.mkdir(parents=True, exist_ok=True)
        t.write_bytes(b'')
    return dst


def _status(n_steps=None, fail_stage=None, reasons=()):
    stages = ['check', 'slam', 'grid', 'pose', 'width', 'validate', 'images', 'export']
    st, stopped = {}, False
    for s in stages:
        if stopped:
            st[s] = {'status': 'skipped', 'reason': f'{fail_stage} rejected', 'reasons': [], 'outputs': {}}
        elif s == fail_stage:
            st[s] = {'status': 'rejected', 'reason': ', '.join(reasons), 'reasons': list(reasons), 'outputs': {}}
            stopped = True
        else:
            st[s] = {'status': 'done', 'reason': None, 'reasons': [], 'outputs': {}}
    if not stopped:
        st['export']['outputs'] = {'dir': 'episode.zarr', 'n_steps': n_steps, 'n_uw': n_steps // 6,
                                   'name': 'x', 'task_label': 'block_raise'}
    return st


def _validate(n, speed, status='done', reasons=()):
    return {'status': status, 'reasons': list(reasons),
            'kept': {'i0': 0, 'i1': n, 'n': n, 'seconds': n / 60, 'fraction': 0.97, 't_lo_ns': 0, 't_hi_ns': 1},
            'metrics': {'n_grid': n, 'n_usable': n, 'n_pose_valid': n, 'n_width_valid': n, 'n_main_valid': n,
                        'max_speed_mps': speed, 'max_ultrawide_gap_ms': 33.3, 'max_imu_gap_ms': 2.4,
                        'slam_tracked_fraction': 0.99, 'slam_scale_problems': [], 'tag_fraction': 0.93},
            'thresholds': {'main_max_dt_ns': 25000000, 'min_kept_s': 2.0, 'min_kept_fraction': 0.8,
                           'max_speed_mps': 3.0, 'extreme_speed_mps': 7.0, 'max_ultrawide_gap_ns': 100000000,
                           'max_imu_gap_ns': 20000000, 'min_tracked_fraction': 0.95, 'min_tag_fraction': 0.8}}


def make_task_folder(data_root, folder='block_raise', n_episodes=6, dataset='ds1', with_package=True):
    """A pulled dataproc task folder (no raw/) with n exported episodes, one failed (validate: speed) and one failed
    (slam), plus package/<dataset>/package.json (zip absent unless make_zarr writes it)."""
    t = Path(data_root) / folder
    t.mkdir(parents=True, exist_ok=True)
    (t / 'task.yaml').write_text("label: block_raise\nrig: rig_pi_v3\ncreated: '2026-11-13T05:21:13+00:00'\n"
                                 "notes: ''\n")
    wjson(t / 'gripper_range.json', {'min_width': 0.03, 'max_width': 0.09})
    ledger = [['slot', 'pi_name', 'session_id', 'episode_number', 'kind', 'rig_id', 'start_unix_ns',
               'intake_unix_ns', 'manifest_sha256'],
              ['calibration/pi_cal', 'pi_cal', 's', 0, 'calibration', 'rig_pi_v3', 1, 1, 'a']]
    eps = []
    for i in range(1, n_episodes + 1):
        n = 600 + 60 * i
        d = t / f'episode_{i}' / 'processed'
        wjson(d / 'status.json', _status(n))
        wjson(d / 'validate.json', _validate(n, 0.4 + 0.1 * i))
        (d / 'episode.zarr').mkdir(parents=True, exist_ok=True)
        ledger.append([f'episode_{i}', f'pi_ep{i:03d}', 's', i, 'demonstration', 'rig_pi_v3', 1, 2, 'b'])
        eps.append({'folder': folder, 'slot': f'episode_{i}', 'pi_name': f'pi_ep{i:03d}', 'label': 'block_raise',
                    'n_steps': n, 'calibration': 'pi_cal'})
    fv = t / 'failed' / 'pi_fast' / 'processed'
    wjson(fv / 'status.json', _status(None, 'validate', ['speed']))
    wjson(fv / 'validate.json', _validate(500, 3.6, 'rejected', ['speed']))
    wjson(t / 'failed' / 'pi_fast' / 'reason.json', {'pi_name': 'pi_fast', 'source_name': 'episode_000009',
                                                     'stage': 'validate', 'status': 'rejected',
                                                     'reasons': ['speed'], 'message': 'speed', 'time': 'x'})
    wjson(t / 'failed' / 'pi_noslam' / 'reason.json', {'pi_name': 'pi_noslam', 'stage': 'slam', 'status': 'failed',
                                                       'reasons': ['slam_tracking_lost'], 'message': 'lost'})
    ledger.append(['failed/pi_fast', 'pi_fast', 's', 9, 'demonstration', 'rig_pi_v3', 1, 2, 'c'])
    with open(t / 'episodes.csv', 'w', newline='') as f:
        csv.writer(f).writerows(ledger)
    wjson(t / 'report.json', {'folder': folder, 'label': 'block_raise', 'rig_id': 'rig_pi_v3',
                              'slam_backend': 'orbslam3', 'exit_code': 0, 'error': None,
                              'calibration': {'state': 'done', 'name': 'pi_cal'},
                              'recordings': [{'source_name': 'episode_000009', 'pi_name': 'pi_fast',
                                              'outcome': 'failed', 'slot': 'failed/pi_fast', 'stage': 'validate',
                                              'reasons': ['speed'], 'message': 'speed'}],
                              'totals': {'episode': n_episodes, 'failed': 2, 'waiting': 0, 'skipped': 0},
                              'rig_nominal_sections': ['imu.noise'], 'created': '2026-11-13T05:30:00+00:00'})
    if with_package:
        wjson(t / 'package' / dataset / 'package.json', {
            'dataset': dataset, 'label_dir': 'block_raise', 'labels': ['block_raise'],
            'folders': [{'folder': folder, 'label': 'block_raise', 'rig': 'rig_pi_v3'}], 'rig': 'rig_pi_v3',
            'episodes': eps, 'excluded': [{'folder': folder, 'entry': 'failed/pi_fast', 'state': 'failed',
                                           'reason': 'validate: speed'}],
            'totals': {'episodes': len(eps), 'n_steps': sum(e['n_steps'] for e in eps),
                       'per_label': {'block_raise': len(eps)}, 'excluded': 1},
            'warnings': ['synthetic warning'], 'zip': {'name': f'{dataset}.zarr.zip', 'size_bytes': 0, 'sha256': ''},
            'dataproc_version': '0.1.0', 'created': '2026-11-13T06:00:00+00:00'})
    return t


def make_zarr(zip_path, n_episodes=3, n=240, images=True):
    """A small ReplayBuffer-layout .zarr.zip (needs zarr; JPEG-XL images need imagecodecs)."""
    import zarr
    from numcodecs import Blosc  # noqa: F401
    ends = np.cumsum([n] * n_episodes).astype(np.int64)
    N = int(ends[-1])
    t = np.arange(N) / 60.0
    pos = np.zeros((N, 3), np.float32)
    width = np.zeros((N, 1), np.float32)
    for e in range(n_episodes):
        s = slice(e * n, (e + 1) * n)
        tt = np.linspace(0, 1, n)
        move = np.clip((tt - 0.2) / 0.6, 0, 1)   # idle for the first 20 % and the last 20 %
        pos[s, 0] = 0.3 + 0.1 * move + 0.01 * e
        pos[s, 1] = 0.05 * np.sin(np.pi * move)
        pos[s, 2] = 0.1 + 0.15 * move
        width[s, 0] = 0.08 - 0.04 * (move > 0.5)
    store = zarr.ZipStore(str(zip_path), mode='w')
    root = zarr.group(store)
    data, meta = root.create_group('data'), root.create_group('meta')
    data.array('gripper_right_eef_pos', pos)
    data.array('gripper_right_eef_rot_axis_angle', np.tile(np.array([[0.1, 0.2, 0.3]], np.float32), (N, 1)))
    data.array('gripper_right_gripper_width', width)
    if images:
        img = np.zeros((N, 224, 224, 3), np.uint8)
        img[..., 0] = (t[:, None, None] * 50 % 255).astype(np.uint8)
        try:
            from imagecodecs.numcodecs import JpegXl, register_codecs
            register_codecs(verbose=False)
            data.array('camera_right_main_rgb', img, chunks=(1, 224, 224, 3), compressor=JpegXl(level=99))
        except Exception:  # noqa: BLE001
            data.array('camera_right_main_rgb', img, chunks=(1, 224, 224, 3))
    meta.array('episode_ends', ends)
    meta.array('episode_names', np.array([f'pi_ep{i + 1:03d}' for i in range(n_episodes)], dtype=object),
               object_codec=__import__('numcodecs').VLenUTF8())
    store.close()
    return zip_path
