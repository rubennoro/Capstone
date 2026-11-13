"""Tests for train/analysis/. Run from the repo root:

    python3 -m unittest discover -s train/analysis/tests -v
    ZARR_PYTHON=/path/to/venv/bin/python train/analysis/tests/run.sh     # also the zarr path, in a venv

Everything is written into temp dirs; results/ is only read (its runs are copied into a temp data root).
"""
from __future__ import annotations

import contextlib
import io
import json
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parent
ANALYSIS = HERE.parent
sys.path.insert(0, str(ANALYSIS))
sys.path.insert(0, str(HERE))

import common as C  # noqa: E402
import compare_runs  # noqa: E402
import dataset_report  # noqa: E402
import eval_report  # noqa: E402
import fixtures as F  # noqa: E402
import run_report  # noqa: E402

try:
    import zarr  # noqa: F401
    HAVE_ZARR = True
except ImportError:
    HAVE_ZARR = False


def run_main(mod, *argv):
    out = io.StringIO()
    with contextlib.redirect_stdout(out):
        rc = mod.main(list(argv))
    return rc, out.getvalue()


class TempRoot(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp(prefix='train_analysis_'))
        self.root = self.tmp / 'data'
        self.root.mkdir()

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)


class TestCommon(TempRoot):
    def test_resolve_run(self):
        F.make_run(self.root, 'block_raise', 'run_001_2026-11-13', n_epochs=5, best_epoch=2)
        r2 = F.make_run(self.root, 'block_raise', 'run_002_2026-11-14_smoke', n_epochs=3, best_epoch=2, mode='smoke')
        self.assertEqual(C.resolve_run(self.root, 'block_raise').name, r2.name)          # newest
        self.assertEqual(C.resolve_run(self.root, 'block_raise', 'run_001').name, 'run_001_2026-11-13')
        self.assertEqual(C.resolve_run(self.root, str(r2)), r2.resolve())                # direct path as FOLDER
        self.assertEqual(C.resolve_run(self.root, 'x', str(r2)), r2.resolve())           # direct path as RUN
        with self.assertRaises(C.LocateError):
            C.resolve_run(self.root, 'block_raise', 'run_009')
        with self.assertRaises(C.LocateError):
            C.resolve_run(self.root, 'run_')                                             # no such folder

    def test_checkpoints_json_and_names(self):
        r = F.make_run(self.root, 'f', 'run_001_x', n_epochs=10, best_epoch=4, topk=3)
        run = C.load_run(r)
        self.assertEqual(run.checkpoints_source, 'checkpoints.json')
        self.assertEqual([c.rank for c in run.checkpoints], [1, 2, 3])
        self.assertEqual(run.checkpoints[0].epoch, 4)
        self.assertTrue(all(c.present for c in run.checkpoints))
        (r / 'checkpoints' / 'checkpoints.json').unlink()
        run = C.load_run(r)
        self.assertEqual(run.checkpoints_source, 'file names')
        self.assertEqual(run.checkpoints[0].epoch, 4)
        self.assertEqual(C.parse_ckpt_name('top2_epoch=0050-val_action_mse_error_all=0.0001481.ckpt'),
                         (2, 50, 0.0001481))
        self.assertEqual(C.parse_ckpt_name('best_slim_epoch=0050-val_action_mse_error_all=0.0001481.ckpt'),
                         (None, 50, 0.0001481))

    def test_missing_files_tolerated(self):
        r = self.root / 'f' / 'policy' / 'run_003_2026-11-13'
        r.mkdir(parents=True)
        (r / 'STATUS').write_text('failed\n')
        run = C.load_run(r)
        self.assertEqual(run.status, 'failed')
        self.assertEqual(run.best(), (None, None))
        self.assertEqual(run.checkpoints, [])
        self.assertEqual(run.evals, {})

    def test_val_split_and_estimate(self):
        self.assertEqual(C.val_count(45), 2)       # round(2.25)
        self.assertEqual(C.val_count(30), 2)       # round(1.5) == 2 (banker's rounding)
        self.assertEqual(C.val_count(10), 1)       # round(0.5) == 0 -> at least 1
        self.assertEqual(C.val_count(1), 1)
        self.assertEqual(len(C.predicted_val_indices(45)), 2)
        self.assertEqual(C.predicted_val_indices(1), [])
        e = C.training_estimate(37971, 60)
        self.assertAlmostEqual(e['min_per_epoch'], 0.55 * 3.7971, places=6)
        self.assertEqual(C.slurm_minutes('08:00:00'), 480)
        self.assertEqual(C.slurm_minutes('1-00:30:00'), 1470)

    def test_palette_matches_user_plots(self):
        self.assertEqual((C.TRAIN_COLOR, C.VAL_COLOR, C.THIRD_COLOR), ('#2a78d6', '#eb6834', '#1baf7a'))
        self.assertIsNotNone(C.WANDB_TOOLS)        # imported (with wandb stubs if wandb is missing)


class TestRunReport(TempRoot):
    def test_synthetic_topk(self):
        r = F.make_run(self.root, 'block_raise', 'run_001_2026-11-13', n_epochs=40, best_epoch=12)
        rc, out = run_main(run_report, 'block_raise', 'run_001', '--data-root', str(self.root))
        self.assertEqual(rc, 0)
        md = (r / 'analysis' / 'report.md').read_text()
        self.assertIn('at epoch 12', md)
        self.assertIn('overfitting', md)
        self.assertEqual(md.count('top1_epoch=0012'), 1)
        self.assertIn('| 3 |', md)                                    # three ranks in the top-k table
        self.assertIn('position', md)
        self.assertTrue((r / 'analysis' / 'val_mse_topk.png').exists())
        # plots/ is empty in the fixture -> regenerated from history.csv + epochs.csv
        self.assertTrue((r / 'analysis' / 'plots' / 'loss_per_epoch.png').exists())
        self.assertIn('best val action MSE', out)

    def test_single_checkpoint(self):
        r = F.make_run(self.root, 'block_raise', 'run_001_2026-11-13', n_epochs=20, best_epoch=8, topk=1)
        rc, out = run_main(run_report, 'block_raise', '--data-root', str(self.root))
        self.assertEqual(rc, 0)
        md = (r / 'analysis' / 'report.md').read_text()
        self.assertIn('top1_epoch=0008', md)
        self.assertNotIn('top2_', md)
        self.assertIn('kept: top1 ep 8', out)

    def test_half_packaged_run(self):
        r = self.root / 'block_raise' / 'policy' / 'run_004_2026-11-13'
        r.mkdir(parents=True)
        (r / 'STATUS').write_text('timeout\n')
        F.wjson(r / 'train_meta.json', {'mode': 'full', 'dataset': 'd', 'epochs': 60, 'topk': 1})
        rc, out = run_main(run_report, str(r))
        self.assertEqual(rc, 0)
        self.assertIn('status timeout', out)
        self.assertIn('No checkpoints', (r / 'analysis' / 'report.md').read_text())

    def test_real_results_runs(self):
        for name in ('run_001_2026-10-05', 'run_002_2026-10-06_smoke'):
            src = F.RESULTS / name
            if not src.is_dir():
                self.skipTest(f'{src} missing')
            F.copy_results_run(src, self.root, 'block_raise')
        rc, out = run_main(run_report, 'block_raise', 'run_001', '--data-root', str(self.root))
        self.assertEqual(rc, 0)
        md = (self.root / 'block_raise' / 'policy' / 'run_001_2026-10-05' / 'analysis' / 'report.md').read_text()
        self.assertIn('**0.000148** at epoch 50', md)
        self.assertIn('140 completed of 140', md)
        self.assertIn('overfitting', md)
        self.assertIn('best_slim_epoch=0050', md)
        self.assertIn('videos/final/rollout_block_raise.mp4', md)
        rc, out = run_main(run_report, 'block_raise', 'run_002', '--data-root', str(self.root))
        self.assertEqual(rc, 0)
        self.assertIn('too few epochs', out)
        self.assertIn('0.0192 @ epoch 2', out)


class TestCompareRuns(TempRoot):
    def test_folder_plus_results_baseline(self):
        F.make_run(self.root, 'block_raise', 'run_001_2026-11-13', n_epochs=30, best_epoch=10)
        F.make_run(self.root, 'block_raise', 'run_002_2026-11-14_smoke', n_epochs=3, best_epoch=2, mode='smoke')
        F.make_task_folder(self.root, 'block_raise', dataset='ds1')
        res = self.tmp / 'results_copy'
        if F.RESULTS.is_dir():
            for name in ('run_001_2026-10-05', 'run_002_2026-10-06_smoke'):
                F.copy_results_run(F.RESULTS / name, res.parent, 'results_copy_unused', name)
            res = self.tmp / 'results_copy_unused' / 'policy'
        rc, out = run_main(compare_runs, 'block_raise', '--data-root', str(self.root), '--results-dir', str(res))
        self.assertEqual(rc, 0)
        od = self.root / 'block_raise' / 'analysis' / 'compare_runs'
        rows = (od / 'compare_runs.csv').read_text().splitlines()
        self.assertEqual(rows[0].split(',')[:3], ['run', 'source', 'status'])
        n_expected = 2 + (2 if F.RESULTS.is_dir() else 0)
        self.assertEqual(len(rows) - 1, n_expected)
        self.assertIn('6', rows[1].split(',')[5])                       # episodes from the local package.json
        for f in ('compare_runs.md', 'compare_val_mse.png', 'compare_loss.png'):
            self.assertTrue((od / f).exists(), f)
        if F.RESULTS.is_dir():
            self.assertIn('0.000148 @ 50', (od / 'compare_runs.md').read_text())
        rc, out = run_main(compare_runs, 'block_raise', '--data-root', str(self.root), '--no-smoke')
        self.assertNotIn('smoke', out.split('lowest')[0].replace('--no-smoke', ''))

    def test_named_runs_and_real_results_dir_readonly(self):
        F.make_run(self.root, 'block_raise', 'run_001_2026-11-13', n_epochs=10, best_epoch=4)
        F.make_run(self.root, 'block_raise', 'run_002_2026-11-14', n_epochs=10, best_epoch=6)
        rc, out = run_main(compare_runs, 'block_raise', 'run_002', '--data-root', str(self.root))
        self.assertEqual(rc, 0)
        self.assertIn('run_002_2026-11-14', out)
        self.assertNotIn('run_001_2026-11-13', out)
        if F.RESULTS.is_dir():
            before = sorted(p.as_posix() for p in F.RESULTS.rglob('*'))
            rc, out = run_main(compare_runs, '--results-dir', str(F.RESULTS), '--out', str(self.tmp / 'cmp'))
            self.assertEqual(rc, 0)
            self.assertIn('run_001_2026-10-05 (results)', out)
            self.assertEqual(before, sorted(p.as_posix() for p in F.RESULTS.rglob('*')))


class TestEvalReport(TempRoot):
    def test_no_eval_yet(self):
        F.make_run(self.root, 'block_raise', 'run_001_2026-11-13', n_epochs=5, best_epoch=2, topk=1)
        rc, out = run_main(eval_report, 'block_raise', 'run_001', '--data-root', str(self.root))
        self.assertEqual(rc, 1)
        self.assertIn('train/analysis/eval.sh block_raise run_001_2026-11-13 --ckpt top1 --per-demo 8', out)
        self.assertIn('train/fetch.sh block_raise run_001_2026-11-13', out)

    def test_single_eval(self):
        r = F.make_run(self.root, 'block_raise', 'run_001_2026-11-13', n_epochs=20, best_epoch=8, topk=1)
        F.make_eval(r, 'top1', epoch=8)
        rc, out = run_main(eval_report, 'block_raise', '--data-root', str(self.root))
        self.assertEqual(rc, 0)
        a = r / 'analysis'
        md = (a / 'eval_report.md').read_text()
        self.assertIn('Worst 5 demos', md)
        self.assertIn('demo_005', md)               # the fixture's worst demo
        self.assertIn('| new |', md)
        self.assertTrue((a / 'eval_per_step_top1.png').exists())
        self.assertTrue((a / 'eval_per_demo_top1.png').exists())
        self.assertFalse((a / 'eval_compare.png').exists())
        self.assertNotIn('Checkpoint comparison', md)
        self.assertIn('val at k=15', out)

    def test_three_evals_and_mismatch(self):
        r = F.make_run(self.root, 'block_raise', 'run_001_2026-11-13', n_epochs=20, best_epoch=8, topk=3)
        F.make_eval(r, 'top1', epoch=8, val_mse=1.8e-4)
        F.make_eval(r, 'top2', epoch=7, val_mse=1.5e-4, seed=1)
        e = F.make_eval(r, 'top3', epoch=9, per_demo=False, seed=2)
        e['split_matches_checkpoint'] = False
        F.wjson(r / 'eval' / 'top3' / 'eval.json', e)
        rc, out = run_main(eval_report, 'block_raise', 'run_001', '--data-root', str(self.root), '--ckpt', 'top3')
        self.assertEqual(rc, 0)
        md = (r / 'analysis' / 'eval_report.md').read_text()
        self.assertIn('Checkpoint comparison', md)
        self.assertIn('Lowest eval val MSE: **top2**', md)
        self.assertIn('split differs', md)
        self.assertIn('No per-demo results', md)
        self.assertTrue((r / 'analysis' / 'eval_compare.png').exists())
        rc, out = run_main(eval_report, 'block_raise', 'run_001', '--data-root', str(self.root), '--ckpt', 'top5')
        self.assertEqual(rc, 1)
        self.assertIn('--ckpt top5', out)


class TestDatasetReport(TempRoot):
    def test_synthetic_folder_without_zarr(self):
        F.make_task_folder(self.root, 'block_raise', n_episodes=6, dataset='ds1')
        rc, out = run_main(dataset_report, 'block_raise', '--data-root', str(self.root), '--no-zarr')
        self.assertEqual(rc, 0)
        od = self.root / 'block_raise' / 'analysis' / 'dataset_report' / 'ds1'
        md = (od / 'dataset_report.md').read_text()
        self.assertIn('| episode_N exported | 6 |', md)
        self.assertIn('| failed/ (never packaged) | 2 |', md)
        self.assertIn('validate: speed', md)
        self.assertIn('slam: slam_tracking_lost', md)
        self.assertIn('synthetic warning', md)
        self.assertIn('nominal', md)
        self.assertIn('| block_raise | 6 | 1 | 5 |', md)        # val split
        self.assertIn('Predicted val demos', md)
        self.assertIn('ds1.zarr.zip is not in', md)
        for f in ('episode_lengths.png', 'validate_metrics.png'):
            self.assertTrue((od / f).exists(), f)
        self.assertIn('min/epoch', out)

    def test_folder_without_package(self):
        F.make_task_folder(self.root, 'block_raise_2', n_episodes=1, with_package=False)
        rc, out = run_main(dataset_report, 'block_raise_2', '--data-root', str(self.root))
        self.assertEqual(rc, 0)
        self.assertIn('(no package)', out)
        self.assertIn('training needs at least 2', out)
        self.assertTrue((self.root / 'block_raise_2' / 'analysis' / 'dataset_report' / 'folder' /
                         'dataset_report.md').exists())

    def test_missing_folder_and_dataset(self):
        with self.assertRaises(C.LocateError):
            run_main(dataset_report, 'nope', '--data-root', str(self.root))
        F.make_task_folder(self.root, 'block_raise')
        with self.assertRaises(C.LocateError):
            run_main(dataset_report, 'block_raise', '--data-root', str(self.root), '--dataset', 'other')

    def test_zarr_missing_message_or_zarr_path(self):
        t = F.make_task_folder(self.root, 'block_raise', n_episodes=3, dataset='ds1')
        zp = t / 'package' / 'ds1' / 'ds1.zarr.zip'
        if not HAVE_ZARR:
            zp.write_bytes(b'not really a zip')
            rc, out = run_main(dataset_report, 'block_raise', '--data-root', str(self.root))
            self.assertEqual(rc, 0)
            self.assertIn('zarr==2.18.3 numcodecs==0.13.1 imagecodecs', out)
            return
        F.make_zarr(zp, n_episodes=3, n=240)
        pj = json.loads((t / 'package' / 'ds1' / 'package.json').read_text())
        pj['zip']['size_bytes'] = zp.stat().st_size
        F.wjson(t / 'package' / 'ds1' / 'package.json', pj)
        rc, out = run_main(dataset_report, 'block_raise', '--data-root', str(self.root), '--thumbs', '2')
        self.assertEqual(rc, 0)
        od = t / 'analysis' / 'dataset_report' / 'ds1'
        md = (od / 'dataset_report.md').read_text()
        self.assertIn('Trajectories (from the zarr)', md)
        for f in ('workspace_extent.png', 'gripper_width.png', 'idle_and_path.png'):
            self.assertTrue((od / f).exists(), f)
        # fixture: idle for the first and last 20 % of 4 s -> ~0.8 s each (minus the smoothing window)
        row = [line for line in md.splitlines() if line.startswith('| pi_ep001 |')][0].split('|')
        self.assertAlmostEqual(float(row[5]), 0.8, delta=0.2)
        self.assertAlmostEqual(float(row[6]), 0.8, delta=0.2)
        self.assertNotIn('size differs', md)


class TestCli(unittest.TestCase):
    def test_help(self):
        for s in ('run_report.py', 'compare_runs.py', 'eval_report.py', 'dataset_report.py'):
            p = subprocess.run([sys.executable, str(ANALYSIS / s), '-h'], capture_output=True, text=True)
            self.assertEqual(p.returncode, 0, p.stderr)
            self.assertIn('usage:', p.stdout)
            self.assertIn('--data-root', p.stdout)

    def test_locate_error_exit_code(self):
        with tempfile.TemporaryDirectory() as d:
            p = subprocess.run([sys.executable, str(ANALYSIS / 'run_report.py'), 'nofolder', '--data-root', d],
                               capture_output=True, text=True)
            self.assertEqual(p.returncode, 2)
            self.assertIn('error: no folder', p.stderr)


if __name__ == '__main__':
    unittest.main()
