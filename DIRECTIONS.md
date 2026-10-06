# Directions: training on a new dataset

These are the steps from "we recorded new demos" to "results are in the repo". Day to day it's one command on the
laptop plus a `git pull` later. For how the pieces fit together, see [DESIGN.md](DESIGN.md).

---

## 0. One-time setup (do once, before the first run with this pipeline)

Wait until the first full training job (`10851655`) has finished: its last step uses scripts from the cluster clone.
Then, from the laptop:

```bash
# a) Update the cluster clone and give it a commit identity (training jobs commit and push their results)
ssh noroian.r@login.explorer.northeastern.edu "cd ~/CapstoneProject/Capstone && git pull --rebase --autostash \
  && git config user.name '$(git log -1 --format=%an)' && git config user.email '$(git log -1 --format=%ae)'"

# b) Move the existing dataset to the standard dataset location
ssh noroian.r@login.explorer.northeastern.edu 'mkdir -p ~/CapstoneProject/datasets/block_raise/firstreal-single && \
  mv ~/CapstoneProject/Capstone/iPhUMI/python_package/iphumi/demonstration_processing/tmp_sessions/firstreal-single/replay_buffer_firstreal-single.zarr.zip \
     ~/CapstoneProject/datasets/block_raise/firstreal-single/firstreal-single.zarr.zip'

# c) Delete stale script copies from before the pipeline existed
ssh noroian.r@login.explorer.northeastern.edu 'cd ~/CapstoneProject && rm -f build_env.sbatch get_example_data.sbatch build_iphumi_env.sbatch process_firstreal.sbatch train_smoke.sh train_full.sbatch'
```

d) Convert the first full run (`block_raise_bp_v1`) into the standard results folder. Its run dir is under
`/scratch/noroian.r/bp_runs/` and its wandb run is the newest one in `train_network/wandb/`. Ask Claude to repackage it with
`scripts/package_run.py` into `results/block_raise/run_001_2026-10-05/`.

---

## 1. Record the demos (iPhone app)

- **Session ID:** use one new, descriptive session name per recording session, e.g. `oct12-block-raise`. The pipeline
  selects demos by session name.
- **Gripper calibration:** record one at the start of every session (the app requires it).
- **Task label:**
  - **"No label" (X icon):** the pipeline labels each demo as one task spanning the whole recording, using the
    `--task` name you give it. This is fine for single-task data like `block_raise`.
  - **Predefined task list / narration:** the labels are kept as recorded. Use this when one demo contains several tasks.
- **Side:** recording as `L` or `R` both work for single-arm data. The dataset always stores it as `right`.
- Supported phones: iPhone 15 Pro and iPhone 16 Pro (calibration entries in
  `iPhUMI/python_package/iphumi/demonstration_processing/calibration/gripper_calibration.yaml`). Any other model needs
  a new entry there first.

## 2. Export to the laptop

Export from the app and put everything from one day in **one folder**, `data/raw/<YYYY-MM-DD>/`. `data/` is gitignored, so
raw data never goes through git.

Check the session name: it's the third `_`-separated field of each export folder name, e.g.
`2026-10-12T15-03-11.123Z_12345_oct12-block-raise_demonstration_left` → `oct12-block-raise`.

## 3. Pick names

| Name | Rule | Example |
|---|---|---|
| `--task` | snake_case, one per task; it becomes the `results/<task>/` folder and the task label | `block_raise` |
| `--dataset` | unique per task, never reused (processing refuses to overwrite) | `oct12-single` |
| `--sessions` | comma-separated session names from step 2 (several sessions → one dataset) | `oct12-block-raise` |

## 4. Run the pipeline (one command on the laptop)

Commit and push any code changes first: the cluster runs whatever is on GitHub.

**New data → process → smoke test → full training:**
```bash
scripts/run_pipeline.sh all --task block_raise --dataset oct12-single --sessions oct12-block-raise \
  --raw-dir data/raw/2026-10-12 --smoke-first --note "second session, brighter lighting"
```
This copies the raw folder to the cluster, updates the cluster's repo clone, and submits three chained jobs. Each job only
starts if the previous one succeeded:
1. **Processing** (CPU, ≤4 h): builds `~/CapstoneProject/datasets/<task>/<dataset>/<dataset>.zarr.zip`.
2. **Smoke test** (GPU, 3 epochs, ~15 min): proves the dataset trains → `results/<task>/run_NNN_<date>_smoke/`.
3. **Full training** (GPU, 140 epochs, ≤8 h) → `results/<task>/run_NNN_<date>/`.

Drop `--smoke-first` to skip the smoke test.

**Train again on an existing dataset** (different settings, or just a repeat):
```bash
scripts/run_pipeline.sh train --task block_raise --dataset firstreal-single --note "200 epochs" -- training.num_epochs=200
```
Everything after `--` is passed to the training config as Hydra overrides. Add `--smoke` for a 3-epoch test.

**Only process** (build the dataset now, train later): `scripts/run_pipeline.sh process --task ... --dataset ... --sessions ... --raw-dir ...`

You can close the laptop as soon as the job ids print. Everything runs on the cluster.

### Will it fit in the 8-hour limit?
Training time scales with the number of frames. Measured: **about 0.55 min per epoch per 10,000 frames** on one H200
(`firstreal-single`: 37,971 frames → 2.1 min/epoch → 140 epochs ≈ 5.2 h with evaluation).
- 140 epochs fit for datasets up to **about 57,000 frames** (about 65 demos the length of ours).
- For bigger datasets, lower the epochs so it fits: `-- training.num_epochs=N`, with
  `N ≈ 440 / (0.55 × frames / 10,000)`.
- If a run hits the limit anyway, it's stopped 10 min early, and whatever exists is packaged with status `timeout`.

## 5. Check on it (optional)

```bash
ssh noroian.r@login.explorer.northeastern.edu 'squeue -u noroian.r; ls -t ~/CapstoneProject/logs | head -5'
```
- `PD` = waiting in the queue, `R` = running. A job disappears from `squeue` when it's done.
- Live console output: `~/CapstoneProject/logs/bp_process_<job>.out` / `bp_train_<job>.out`.
- If the processing job fails, the training jobs after it never start. Slurm shows them as `DependencyNeverSatisfied`;
  clear them with `scancel`.

## 6. Get the results

```bash
git pull
```
Each finished training job pushes its own `results/<task>/run_NNN_<date>/` folder. Start with its `README.md`: status, dataset,
settings, per-epoch table, speed and GPU memory, plus links to the plots and videos.

The **best checkpoint** (slimmed, ~3 GB) stays on the cluster. The run README has the exact `rsync` line to fetch it.
Laptop disk space is limited, so only fetch the checkpoints you need.

## 7. If something goes wrong

| Symptom | Where to look / what to do |
|---|---|
| Run folder says **FAILED** | Its README has the last log lines. The full log is in `logs/slurm_job.txt`. Fix the cause, then submit `train` again; it gets a new run number. |
| `dataset '<name>' already exists` | Dataset names are never reused. Pick a new `--dataset`. |
| `raw dir ... not found` | Pass `--raw-dir` relative to the repo (`data/raw/<date>`); `run_pipeline.sh` copies it over. |
| `dataset not found` when training | Process the dataset first, or check `--task` / `--dataset` match the folder under `~/CapstoneProject/datasets/`. |
| Results never appeared on GitHub | The job's console log ends with `pushed: 0` → the push failed. The folder is still in the cluster clone; push it from there. |
| Status **timeout** | The dataset was too big for 140 epochs in 8 h; resubmit with fewer epochs (see step 4). |
| Processing dropped demos | The processing log line `Align validation complete: X passed, Y failed` shows how many. Usually caused by bad tracking (see the SLAM tips in `iPhUMI/docs/usage.md`). |
