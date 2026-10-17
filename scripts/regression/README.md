# iPhone path regression guard

Three checks that show the existing iPhone path (record, process, train, deploy) still works exactly as it did
before the sensor rig work started. The baseline is `0ddedef`, the last commit before `sensor_rig/`.

| Check | What it shows | Runtime | Needs |
|---|---|---|---|
| `check_protected.sh` | No file the iPhone path depends on changed. The only exceptions are additions that are explicitly allowed. | ~1 s | git, python3 with PyYAML |
| `iphone_pipeline_check.py` | The real iPhUMI processing turns 5 fixed firstreal episodes into a zarr identical to the golden one. | ~50 s (20 cores, 4 workers) | iphumi env, `data/raw/2026-10-01` |
| `test_iphone_deploy_unchanged.py` | real-env's `iphone` camera server composes and constructs exactly as at the baseline. | ~3 s | python with hydra-core + pytest |

## When to run

- **Before every commit that touches shared code**: anything outside `sensor_rig/`, or any file in the protected
  list.
- **Before every `git pull` on the cluster** (run it on the laptop for the commit you are about to pull). Also after
  rebuilding the `iphumi` env.
- `check_protected.sh` is cheap enough to run before every commit.

## How to run

```bash
# 1. protected files vs 0ddedef (working tree incl. untracked files; --head = committed state only)
scripts/regression/check_protected.sh [REF] [--head]

# 2. processing (exit 1 on mismatch). --python = the iphumi env's python (default: the interpreter running it)
python scripts/regression/iphone_pipeline_check.py [--python ~/.conda/envs/iphumi/bin/python] [--keep]
#    cluster: run it inside a CPU job (srun/sbatch), not on the login node, and pass --src if data/raw lives elsewhere

# 3. deployment (unset PYTHONPATH if ROS is sourced: its launch_testing pytest plugin breaks collection)
env -u PYTHONPATH python -m pytest scripts/regression/test_iphone_deploy_unchanged.py -v
```

All three exit 0 on success.

## Details

**Protected list** (`protected_paths.txt`, syntax documented at the top of the file):
- All of `iPhUMI/`: the app, the processing package, its configs, the env and setup files.
- The pipeline scripts: `run_pipeline.sh`, `cluster/*`, `package_run.py`, `slim_checkpoint.py`,
  `wandb_offline_to_csv.py`, `label_whole_demo_as_task.py`, `check_job.sh`.
- All of `behavior_prompting/`: training, deployment, icrt, LIBERO, and real-env.
- Special rules:
  - `gripper_calibration.yaml`: the baseline bytes must be an exact prefix of the current file, and every baseline
    YAML key must keep its value. New devices may only be appended.
  - real-env `pyproject.toml`: lines may only be added.
  - New real-env files are allowed only if they match `peripherals/rig_*.py` or `configs/peripherals/rig_*.yaml`.

**Pipeline check**:
- Runs the commands of `cluster/process_dataset.sbatch` steps 1–4 in a temp work dir. Paths are absolute Hydra
  overrides, and the inputs are symlinked. The repo's `tmp_demonstrations/`, `tmp_sessions/` and `data/raw` are never
  written; a before/after snapshot verifies this.
- One deviation from the sbatch: the label overlay video is not rendered. `--with-label-video` renders it, which makes
  the commands identical to the sbatch.
- The episodes are calibration `15647` and demos `49242 18798 87784 73132` (4 episodes, 2965 frames).
- `--code-root DIR` runs the iPhUMI code of another checkout. It prepends `DIR/iPhUMI/python_package` to
  `PYTHONPATH`. `iphumi` is a namespace package, so the script checks `iphumi.__path__[0]` and three module files.
- What is compared against the golden file:
  - **Inputs**: sha256 of every input file.
  - **Intermediates**: valid flags, gripper range, and the aligned CSVs.
  - **Zarr structure**: shapes, dtypes, chunks and compressors.
  - **Values**: the sha256 of the decoded values of every array, exact.
- Three runs on one machine gave byte-identical zips: JPEG-XL runs with `numthreads=1`, and every step is
  deterministic. The only tolerances are for a different library build, such as the cluster env:
  - **Images**: if the decoded pixels differ, the check falls back to per-frame mean RGB. It passes with a warning if
    the max |diff| ≤ 0.5 and every episode's mean ≤ 0.1. `--strict` makes this a failure.
  - **Low-dim arrays**: they fail on any difference unless you pass `--lowdim-atol`. The golden file stores the full
    arrays, so a failure prints the max |diff| and the first differing row.
- **Regenerating the golden file**: only do this deliberately, at a commit whose iPhone path you trust, and with
  inputs that are known to be good:
  ```bash
  git worktree add --detach /tmp/wt 0ddedef
  python scripts/regression/iphone_pipeline_check.py --make-golden --code-root /tmp/wt
  git worktree remove /tmp/wt
  ```
  The current golden file was made this way from `0ddedef`, with python 3.10.20, numpy 2.2.6, opencv 4.12.0 and zarr
  2.18.3.

**Deploy check**:
- Extracts the baseline `behavior_prompting/deps/real-env` with `git archive`. For both it and the working tree, a
  subprocess composes every `iphone_*.yaml` and calls the real `run_iphone()` for each one. `iPhone.run` is replaced
  by a capture.
- The test then compares the composed configs, the constructed camera's full state (endpoints, ports, UDID, RMQ
  topics, VideoLogger attrs, config_str), the constructor signatures, the method lists, and the `iphone` console
  script.
- `robotmq`, `robologger`, `robot_utils`, `loguru` and `pypeertalk` are always replaced by local fakes, so no sockets
  or USB devices are opened. The fakes are defined in the test on purpose, not imported from `sensor_rig/tests`, so the
  guard doesn't depend on the code it guards.
- Set `BASELINE_REF=<ref>` to compare against another baseline.
