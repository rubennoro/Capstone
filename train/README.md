# train/: from Jetson data to a trained policy

Laptop scripts that move a processed task folder from the Jetson to the laptop, upload its dataset to the Explorer
cluster, train there, and bring the results back. Every script takes `-h` (help) and `--dry-run` (print every
ssh/rsync command, run and write nothing).

```
Jetson  ~/data/tasks/<folder>/ + ~/data/datasets/<label>/<dataset>/
   │ pull.sh
laptop  data/<folder>/ ─ upload.sh ─▶ cluster ~/CapstoneProject/datasets/<label>/<dataset>/
                                        │ submit.sh  (Slurm job on 1 H200)
laptop  data/<folder>/policy/<run>/ ◀─ fetch.sh ─ ~/CapstoneProject/policies/<label>/<run>/
```

## One-time setup

1. **Fill `train/config.env`.** Every setting has a comment and an example. At least `JETSON_HOST` (still a `<...>`
   placeholder; a script that needs an unfilled setting stops and names it). Personal overrides can go in
   `train/config.local.env` (same syntax, not tracked by git, read after `config.env`).
2. **ssh keys** so `ssh <JETSON_HOST>` and `ssh noroian.r@login.explorer.northeastern.edu` work from the laptop
   (`ssh-copy-id <host>`). The transfer node `xfer.discovery.neu.edu` shares the cluster home. `CLUSTER_SSH_OPTS`
   turns on ssh connection sharing (ControlMaster), so Duo asks once per 10 minutes, not once per connection.
3. **Cluster:** the conda env `bp` already exists. The repo clone `~/CapstoneProject/Capstone` must be on `main`, at
   `4be9982` or newer for `eval.sh` (it needs `controlSW/franka_deploy`): `ssh <login> git -C CapstoneProject/Capstone pull`.
4. Try it without touching anything: `train/go.sh block_raise br-nov13 --package --dry-run`.

## Workflow

Examples use the task folder `block_raise` and a new dataset name `br-nov13`.

| Step | Command |
|---|---|
| 1. Pull from the Jetson (optionally packaging first) | `train/pull.sh block_raise --package br-nov13 --no-raw` |
| 2. Upload the dataset | `train/upload.sh block_raise br-nov13` |
| 3. Submit training (smoke, then full) | `train/submit.sh block_raise br-nov13 --smoke-first --note "first rig data"` |
| 4. Watch it | `train/status.sh block_raise run_002_2026-11-13 --wait` |
| 5. Fetch the results | `train/fetch.sh block_raise run_002_2026-11-13` |
| 6. Analyse | see [analysis/README.md](analysis/README.md); offline eval on the cluster: `train/analysis/eval.sh block_raise run_002_2026-11-13 --ckpt top1` |

Steps 1–3 in one command: `train/go.sh block_raise br-nov13 --package --no-raw --note "first rig data"`
(`--pull` instead of `--package` for a dataset already packaged on the Jetson, `--no-smoke` to skip the smoke run).

Details:
- **pull.sh** never deletes local files and skips `.staging/`. `--dataset NAME` pulls an existing package;
  `--package NAME` runs `dataproc package NAME FOLDER` on the Jetson first. The zip's sha256 is checked against its
  `package.json`.
- **upload.sh** checks the sha256 on the cluster. A zip already there with the same sha256 is skipped; a different
  one is refused (a dataset name is never reused: package under a new name).
- **submit.sh** prints the run names and job ids and writes `data/<folder>/policy/<run>/job.json`. Extra Hydra
  overrides go after `--`, e.g. `-- training.lr_scheduler=constant`. `--smoke-first` = a 3-epoch smoke run, then the
  full run, which only starts if the smoke run succeeded.
- **status.sh** without a run name uses the newest `job.json`. `--wait` checks every 5 min until the run is `ok`,
  `failed` or `timeout`.
- **fetch.sh** copies everything (summary, csv, plots, videos, logs, `wandb/`, `eval/`) plus the best `--top N`
  checkpoints (default `TOPK` = 1) and checks their sizes. It refuses a run that is still queued or running
  (`--force` takes what exists); the state also comes from Slurm, so a killed job counts as failed. `--all`
  fetches every run in `data/<folder>/policy/`; `--no-checkpoints` skips them
  (e.g. to pick up `eval/` results later).

## Data layout on the laptop

```
data/<folder>/                        mirror of the Jetson task folder (task.yaml, episodes.csv, calibration/,
                                      episode_N/{raw,processed}, failed/, report.json; no .staging/)
data/<folder>/package/<dataset>/      <dataset>.zarr.zip + package.json
data/<folder>/policy/<run>/           job.json (written at submit), README.md, summary.json, epochs.csv,
                                      history.csv, plots/, videos/, logs/, run_config/, train_meta.json, STATUS,
                                      checkpoints/{top1_epoch=...ckpt, checkpoints.json}, wandb/offline-run-*/,
                                      eval/<top1>/{eval.json, videos/}
data/<folder>/analysis/               default output folder of the analysis scripts
```

`data/` is ignored by git.

## Disk space

The laptop has only ~23 GB free. A task folder with raw recordings is roughly 300 MB per episode (processed + raw);
`--no-raw` leaves out the raw part. Each run keeps one slim checkpoint by default (~2.5 GB per run, on the cluster
and on the laptop); `submit.sh --topk K` keeps more and `fetch.sh --top N` fetches more (~2.5 GB each).
`pull.sh` and `fetch.sh` print the transfer size first and refuse if less than `MIN_FREE_GB` (5) would be left
(`--force` to override). Runs on the cluster live in the home quota: delete old ones there once fetched.

## Epoch sizing

Training takes about 0.55 min per epoch per 10,000 dataset steps on one H200. `submit.sh` reads `totals.n_steps` from
`package.json` and fits the epochs into `SLURM_TIME` (8 h) minus `TIME_MARGIN_MIN` (40 min):

    max epochs = (480 − 40) / (0.55 × n_steps / 10,000)

`firstreal-single` (~38,000 steps): 2.1 min per epoch, at most 210 epochs; the default 60 epochs take ~2 h. More epochs
than fit are lowered with a warning. 60 is the default because run_001 overfit after ~50 epochs.

## Troubleshooting

- **`X is not set (value: '<...>')`**: fill X in `train/config.env`.
- **ssh asks for a password or hangs**: check `ssh <host>` by hand; set up keys (setup step 2).
- **`sha256 mismatch`** after a pull: delete the zip in `data/<folder>/package/<dataset>/` and pull again.
- **`a different ... is already on the cluster`**: that name is taken by another zip; package under a new name.
- **upload interrupted**: run it again; it resumes in `<dataset>/.upload/` and only moves the zip into place once its
  sha256 matches.
- **`only N GB would be left`**: free space, or use `--no-raw` / `--top 1` / `--no-checkpoints`.
- **`checkpoint size mismatch`**: the copy was interrupted; run `fetch.sh` again (it resumes).
- **Job failed**: `status.sh` shows the log tail; the full log is `~/CapstoneProject/logs/bp_train_<label>_<job>.out`.
- **eval.sh: `controlSW/franka_deploy missing`**: the cluster clone is older than `4be9982`; `git pull` it.

## Tests

All three run offline (fake ssh/rsync/sbatch or fixtures in a temporary folder):

- `bash train/tests/run_tests.sh`: the laptop scripts (pull, upload, submit, status, fetch, go, eval.sh);
  `SHELLCHECK=/path/to/shellcheck` adds a shellcheck pass.
- `bash train/tests/cluster/run_tests.sh`: the cluster-side `submit_job.sh` and sbatch scripts.
- `bash train/analysis/tests/run.sh`: the analysis scripts.
