# train/analysis

Reports on a task folder, its dataset and the training runs fetched into `data/<folder>/policy/`. Each script
prints a short summary and writes a markdown report with PNG plots (same palette and style as
`scripts/wandb_offline_to_csv.py`). Every script has `-h`.

Inputs are named as `FOLDER [RUN]` under the data root. The data root is `$LOCAL_DATA_ROOT`, else
`LOCAL_DATA_ROOT` from `train/config.local.env` / `config.env`, else `<repo>/data`. Override it with
`--data-root`. `RUN` can be a full name, a unique prefix (`run_003`) or left out (newest run). A path to a run dir
also works in place of `FOLDER`.

| Script | Answers | Writes | Needs |
|---|---|---|---|
| `run_report.py FOLDER [RUN]` | Did it finish? Best val action MSE and its epoch, the kept checkpoints, overfitting (best epoch vs last epoch, val loss after the best, val/train gap), per-component error (pos/rot/width), links to plots, videos and evals | `<run>/analysis/report.md`, `val_mse_topk.png`; `analysis/plots/` if the run has no `plots/` | base python (rebuilding plots from `wandb/` needs `wandb`; from `history.csv` + `epochs.csv` it does not) |
| `compare_runs.py FOLDER [RUN ...] [--results-dir results/block_raise] [--no-smoke]` | Which run is best, how the curves differ, what each run cost | `data/<folder>/analysis/compare_runs/{compare_runs.md,.csv, compare_val_mse.png, compare_loss.png}` | base python |
| `eval_report.py FOLDER [RUN] [--ckpt top1]` | Does the checkpoint reproduce its training val MSE? Error in mm / degrees along the action chunk (k = 0, 4, 8, 15), train vs val vs new demos, the worst 5 demos, top1 vs top2/top3 | `<run>/analysis/eval_report.md`, `eval_per_step_<tag>.png`, `eval_per_demo_<tag>.png`, `eval_compare.png` (only with 2+ evals) | base python; an `eval/<tag>/eval.json` from `eval.sh` |
| `dataset_report.py FOLDER [--dataset NAME]` | How many recordings were exported / failed / excluded and why, episode lengths, how close the validate metrics come to their thresholds, dataproc warnings, training-time estimate, val split size (and which demos it predicts) | `data/<folder>/analysis/dataset_report/<dataset>/dataset_report.md` + PNGs | base python; **zarr** for trajectories (path length, idle time, workspace xy/xz, gripper width traces) and **imagecodecs** for the thumbnail grid |

Runs keep one checkpoint by default (`checkpoints/top1_*.ckpt`). The kept-checkpoints table then has one row,
`eval_report.py` reports top1, and the checkpoint comparison only appears once several `eval/<tag>/` exist.

## block_raise examples

```bash
# the dataset, before training (pulled with train/pull.sh block_raise --dataset br_v1)
python3 train/analysis/dataset_report.py block_raise --dataset br_v1

# a fetched run
python3 train/analysis/run_report.py block_raise run_003

# every run of the folder, with the old iPhone-era runs as a baseline (results/ is only read)
python3 train/analysis/compare_runs.py block_raise --results-dir results/block_raise --no-smoke

# offline eval: make it on the cluster, fetch it, then report
train/analysis/eval.sh block_raise run_003_2026-11-13 --ckpt top1 --per-demo 8
train/fetch.sh block_raise run_003_2026-11-13
python3 train/analysis/eval_report.py block_raise run_003
```

## zarr (dataset_report only)

The base python has no zarr, and it should stay that way. Use a venv that sees the base packages (matplotlib,
pandas):

```bash
python3 -m venv --system-site-packages ~/envs/analysis
~/envs/analysis/bin/pip install zarr==2.18.3 numcodecs==0.13.1 imagecodecs
~/envs/analysis/bin/python train/analysis/dataset_report.py block_raise --dataset br_v1 --thumbs 6
```

Without zarr, `dataset_report.py` still writes everything that comes from the JSON files, and says what it skipped.
`--no-zarr` skips the zip even when zarr is installed.

## Notes

- Estimated training time = `MIN_PER_EPOCH_PER_10K_STEPS` (0.55 on an H200) x steps / 10k per epoch, with
  `DEFAULT_EPOCHS`, `SLURM_TIME` and `TIME_MARGIN_MIN` from `train/config.env`.
- Val split: `min(max(1, round(n * 0.05)), n)` demos per label, as in `get_val_mask`. A label with only one demo
  cannot train, because no demo would be left for train. The predicted val demos assume that the zarr keeps the
  episode order of package.json. The `per_demo` section of eval.json shows the real split.
- Overfitting verdict: "overfitting" means the best val action MSE falls before 70 % of training and val loss
  rises more than 5 % after it.
- Tests: `train/analysis/tests/run.sh` (set `ZARR_PYTHON=~/envs/analysis/bin/python` to also cover the zarr path).
