# dataproc: running it on the Jetson

`dataproc` runs on the Jetson. It turns the episodes the Pi 5 records into one training file, a `.zarr.zip`.

The flow:

1. The Pi records an episode and copies it to the Jetson's `/data/rig/incoming/`.
2. `dataproc intake` checks the episode and files it under a task.
3. `dataproc process` turns the task's episodes into `output/<task>.zarr.zip`.
4. You copy that zip to the cluster and train on it.

The steps below are in order. Part A is done once per Jetson. Part B is an optional practice run on fake data.
Part C is what you do every time you record.

> **Not working yet: real recordings.** `process` needs ORB-SLAM3 to work out where the camera was, and ORB-SLAM3
> isn't set up on the Jetson yet (see [Later: ORB-SLAM3](#later-orb-slam3)). Until it is, you can intake real
> episodes, but `process` only works on fake ones (`--slam oracle`).

---

## Part A: one-time setup

Assumes JetPack 6 (Ubuntu 22.04, which ships Python 3.10). Run everything as your normal user unless it says `sudo`.

**A1. Install the system packages.**

```bash
sudo apt update
sudo apt install -y git python3.10-venv
```

**A2. Get the code.** The code is on the `rnoro/data-work` branch.

```bash
cd ~
git clone git@github.com:rubennoro/Capstone.git
cd ~/Capstone
git checkout rnoro/data-work
```

If you already have a clone, run `cd ~/Capstone && git fetch && git checkout rnoro/data-work && git pull` instead.

**A3. Make a Python environment and install `dataproc`.** This can take several minutes.

```bash
python3.10 -m venv ~/envs/env_dataproc
~/envs/env_dataproc/bin/pip install --upgrade pip
cd ~/Capstone
~/envs/env_dataproc/bin/pip install -e "data-processing[test]"
```

**A4. Turn the environment on.** Do this in every new terminal before using `dataproc`:

```bash
source ~/envs/env_dataproc/bin/activate
```

To have it on automatically, add that line to the end of `~/.bashrc`.

**A5. Check the install.** Both commands should succeed. The tests take a few minutes and should end with
`505 passed, 1 skipped` (or similar, with no failures).

```bash
dataproc --help
cd ~/Capstone
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest data-processing/tests -q
```

**A6. Make the data folder** and give it to your user. The Pi copies episodes into `/data/rig/incoming/`.

```bash
sudo mkdir -p /data/rig/incoming /data/rig/rigs
sudo chown -R $USER:$USER /data/rig
```

Keep all of `/data/rig` on one disk. Intake moves episodes instead of copying them, so `incoming/` and `tasks/`
have to be on the same filesystem.

**A7. Copy in the rig description.** It tells `process` about the cameras, IMU and gripper.

```bash
cp ~/Capstone/data-processing/configs/rigs/rig_v1.yaml /data/rig/rigs/
```

For now its numbers are nominal (datasheet and CAD guesses). After the rig is calibrated, put the real values in
`/data/rig/rigs/rig_v1.yaml`. The `rig_id` in each episode's `meta.json` has to match a file name in
`/data/rig/rigs/`.

Setup is done.

---

## Part B (optional): practice run on fake data

This runs the whole flow on fake episodes that `dataproc sim` makes, so you can see what each command prints. It
uses its own folder, `~/dataproc_practice`, so nothing gets mixed into the real `/data/rig`.

```bash
source ~/envs/env_dataproc/bin/activate
export DATAPROC_ROOT=~/dataproc_practice        # only for this terminal
mkdir -p $DATAPROC_ROOT/incoming

# 1. Fake "the Pi sent a calibration episode", then intake it as calibration
dataproc sim --out $DATAPROC_ROOT/incoming --calibration --episodes 1 --fast --seed 1
dataproc intake --task practice --calibration

# 2. Fake "the Pi sent 3 demos", then intake them
dataproc sim --out $DATAPROC_ROOT/incoming --episodes 3 --fast --seed 2
dataproc intake --task practice

# 3. Process (oracle = use the fake data's known answer instead of ORB-SLAM3)
dataproc process --task practice --slam oracle
dataproc status --task practice
```

What you should see:

- The calibration intake prints `... -> calibration/...` and `1 ingested`.
- The demo intake prints `-> episode_0`, `-> episode_1`, `-> episode_2` and `3 ingested`.
- `process` prints `calibration: done` and `exported` for each episode, then writes
  `~/dataproc_practice/tasks/practice/output/practice.zarr.zip`.
- `process` also prints a `note: rig ... still has nominal sections`. That's expected: it means the rig isn't
  calibrated yet.

When you're done, close the terminal (or `unset DATAPROC_ROOT`) so later commands use `/data/rig` again. You can
delete `~/dataproc_practice` whenever you like.

---

## Part C: every recording session

Pick a task name first, for example `block_raise`. Letters, numbers, `_` and `-` only. Use the same name for every
command below.

```bash
source ~/envs/env_dataproc/bin/activate
```

**C1. Record the gripper calibration episode first.** On the Pi, record one short episode where you open and close
the gripper all the way a few times, with both finger tags in view of the ultrawide camera. Wait for it to arrive,
then:

```bash
dataproc intake --task block_raise --calibration
```

It should print `-> calibration/<name>` and `1 ingested`. Only one episode is used as calibration: the oldest one
waiting. Any others stay in `incoming/` for the next step.

You need a calibration once per task folder. Redo it (same command) if the fingers or tags change. `process`
always uses the newest one.

**C2. Record the demos** on the Pi, as many as you like.

**C3. Intake the demos.** Run this after recording, or any time in between. It's safe to run as often as you want.

```bash
dataproc intake --task block_raise
```

Each episode gets one line:

| Line | Meaning |
| --- | --- |
| `-> episode_N` | Good. Filed as `tasks/block_raise/episode_N/`. |
| `skipped: not verified yet` or `skipped: transfer in progress` | Still arriving from the Pi. Run intake again later. |
| `quarantined: <reasons>` | Rejected and moved to `/data/rig/quarantine/<name>/`. See [Troubleshooting](#troubleshooting). |

To start a fresh folder for the same task (for example, a new day you want kept separate), add `--new`. That makes
`block_raise_1`, then `block_raise_2`, and so on. Then use that folder name (`--task block_raise_1`) for the
rest of that session, including C1.

**C4. Process.**

```bash
dataproc process --task block_raise
```

> For now this only works on fake data with `--slam oracle`. Real episodes need ORB-SLAM3 (see the note at the top).

It runs every episode through all its stages and prints `exported` or `rejected`/`failed` (with reasons) for each
one. Then it writes:

- `/data/rig/tasks/block_raise/output/block_raise.zarr.zip`: the training file, built from every exported episode;
- `/data/rig/tasks/block_raise/output/report.json`: what happened to each episode.

It's safe to run again: stages whose inputs haven't changed are skipped, so after adding a few demos only the new
ones are processed. Two useful options:

- `--episode 3`: process only `episode_3` (the zip is still rebuilt from all exported episodes);
- `--force`: redo every stage, even unchanged ones.

**C5. Check where things stand** at any time:

```bash
dataproc status --task block_raise
```

It shows each episode's last stage and status, whether calibration is done, and whether the zip is up to date or
`stale` (episodes changed since it was built, so run `process` again).

**C6. Copy the zip off the Jetson** to the cluster, into the standard dataset folder. From the Jetson:

```bash
ssh noroian.r@login.explorer.northeastern.edu "mkdir -p ~/CapstoneProject/datasets/block_raise/<dataset-name>"
scp /data/rig/tasks/block_raise/output/block_raise.zarr.zip \
    noroian.r@xfer.discovery.neu.edu:~/CapstoneProject/datasets/block_raise/<dataset-name>/<dataset-name>.zarr.zip
```

Replace `<dataset-name>` with a name for this dataset, for example `rig-first-single`. Training is covered in
[DIRECTIONS.md](../DIRECTIONS.md).

---

## Troubleshooting

**An episode was quarantined.** Look at `/data/rig/quarantine/<name>/reason.json`. Its `reasons` list says why,
for example:

- `hash_mismatch`: a file got corrupted in transfer;
- `duplicate`: that episode was already taken in;
- `rig_mismatch`: the episode's `rig_id` doesn't match the task's rig.

The full list of codes is in [docs/CONTRACT.md](docs/CONTRACT.md). Nothing is ever deleted. To retry an episode
after fixing the problem, move it back:

```bash
mv /data/rig/quarantine/<name>/raw /data/rig/incoming/<name>
```

**Nothing happens on intake** (`nothing waiting`). Check that the episode is in `/data/rig/incoming/` and has a
`.verified` file (`ls -a`). The Pi writes `.verified` after it has checked the checksums on the Jetson. Without it,
intake waits.

**`another intake is running`.** Only one intake can run at a time. Wait for the other one to finish.

**An episode was `rejected` or `failed` in process.** `dataproc status --task <task>` shows the reasons, and
`report.json` has the details. Common ones:

- `no_calibration`: do step C1 first, then run `process` again;
- `oracle_no_truth`: `--slam oracle` was used on a real recording. It only works on `dataproc sim` episodes.

**`run intake first`.** The task folder doesn't exist yet (or the name is misspelled). Intake creates it.

**`dataproc: command not found`.** Turn the environment on (step A4).

---

## Later: ORB-SLAM3

Real recordings need the ORB-SLAM3 Docker image (`dataproc/orbslam3:v1.0`) on the Jetson. It has been built on the
laptop, but never on the Jetson (arm64) and never run on data. Build instructions:
[docker/orbslam3/README.md](docker/orbslam3/README.md). Once it's built, `dataproc process --task <task>` (no
`--slam` option) uses it.

## More detail

| File | What's in it |
| --- | --- |
| [docs/CONTRACT.md](docs/CONTRACT.md) | Exactly what the Pi must send (files, fields, checksums). For whoever writes the Pi side. |
| [docs/DESIGN.md](docs/DESIGN.md) | Folder layout, the output zarr format, conventions. |
| [docs/INTAKE.md](docs/INTAKE.md) | Every rule intake follows. |
| [docs/PROCESS.md](docs/PROCESS.md) | Every processing stage, its checks and thresholds. |
| [docs/SIM.md](docs/SIM.md) | The fake-episode generator. |
| [docs/GEOMETRY.md](docs/GEOMETRY.md) | Frames, poses and how gripper width is measured. |
