# Deployment: day-0 directions

How to take a brand-new PC to a running robot. Follow the steps in order and copy the commands as they are. You don't
need to read any other file.

**What you end up with:** two commands, run from `deployment/`:

- `deploy check`: checks that everything is ready and prints one table. Every row should say `pass` (`warn` and
  `info` are fine).
- `deploy run`: starts everything. You then drive the robot from the keyboard.

**Two ways to run it:**

| | Needs | Use it for |
|---|---|---|
| `--mock` | only this PC | trying the whole system with a fake camera and a fake arm. Do this first. |
| real | the arm, the gripper's Raspberry Pi, the network | the real robot |

---

## 1. What you need

- A PC running **Ubuntu 22.04** (24.04 works too, see step 3), with internet access and `sudo`.
- For the real run:
  - an Ethernet cable from the PC straight to the Franka **controller**;
  - the gripper's **Raspberry Pi 5**, powered, with its camera sender running;
  - a Wi-Fi network that both the PC and the Pi are on.

## 2. Get the code

```bash
cd ~
git clone https://github.com/rubennoro/Capstone.git
cd Capstone
git checkout rnoro/deployment
cd deployment
```

From here on, every command starts in `~/Capstone/deployment` unless it says otherwise.

## 3. Python environment

The code needs **Python 3.10** exactly.

**Ubuntu 22.04** (Python 3.10 is built in):

```bash
sudo apt install -y python3.10-venv
python3.10 -m venv ~/envs/deploy
```

**Ubuntu 24.04, or no `python3.10`:** fetch Python 3.10 with `uv` (no `sudo` needed):

```bash
python3 -m venv ~/envs/uv-tool && ~/envs/uv-tool/bin/pip install uv
~/envs/uv-tool/bin/uv python install 3.10
~/envs/uv-tool/bin/uv venv --python 3.10 --seed ~/envs/deploy
```

**Then, either way, install the code:**

```bash
~/envs/deploy/bin/pip install -e './contracts[test]' -e './deploy[test]' -e './process_frames[test]' \
                              -e './message_comms[test]' -e './policy[test]'
```

**Check:** `~/envs/deploy/bin/deploy --help` prints the usage.

If you put the environment somewhere other than `~/envs/deploy`, set `python.light` in `station.yaml` to its
`bin/python`.

## 4. Build the arm controller (`franka_control`)

This is needed even for `--mock`: the fake arm is inside the same program.

```bash
franka_control/build.sh
```

- It installs build packages with `sudo apt`, downloads and builds libfranka 0.9.2 into `~/franka/`, then builds
  `franka_control/build/franka_control`. It takes several minutes the first time.
- Running it again is safe: finished steps are skipped.
- `franka_control/build.sh --dry-run` shows every command without running anything.
- **Check:** the script ends by printing a line ending in `libfranka 0.9.2`.

**libfranka version:** 0.9.2 is right for a Panda with robot system 4.x. To see your robot's system version, open Desk
(step 6) → Settings → System. If it differs, ask whoever maintains this setup before changing `libfranka_version` in
`franka_control/config.yaml`. A wrong version fails at connect with "incompatible library version". It never works
halfway.

## 5. First run, without any hardware (`--mock`)

**`--mock` needs a recorded episode, and a new PC doesn't have one.** The fake camera replays a real recording from
the gripper's cameras: the folder `episode_000045`. It is **not in the git repository**, so a fresh clone doesn't
include it. Without it, `deploy check --mock` fails with `mock episode ... not found` and `deploy run --mock` won't
start.

Copy it from the Jetson, where it is at `/data/incoming/episode_000045`:

```bash
scp -r <user>@<jetson-ip>:/data/incoming/episode_000045 ~/Downloads/
ls ~/Downloads/episode_000045/meta.json          # must list the file
```

`~/Downloads/episode_000045` is where `station.yaml` looks. If you put it elsewhere, set
`process_frames.mock_episode` in `station.yaml` to that folder.

If you can't get the recording, skip this step. You can't try the system without hardware then, but the real run
(steps 6–9) doesn't need it.

**Then run it.** Use a normal terminal window. `deploy run` reads your keys from it, so it can't run in the
background or over a pipe.

```bash
~/envs/deploy/bin/deploy check --mock
~/envs/deploy/bin/deploy run --mock
```

- **`deploy check --mock`:** every row is `pass`, `warn` or `info`.
- **`deploy run --mock`:**
  - it prints `deploy: all 4 processes ready (session: ...)`, then a status line with the key list;
  - press `h` (home), `c` (start an episode), wait a few seconds, `y` (end it as a success), then `q` (quit);
  - it ends with `deploy: done`.

If both work, the PC is set up. The rest is connecting the real hardware.

## 6. Connect the arm

**Network** (once). The arm is at `172.16.0.2`. Give the PC's Ethernet port a fixed address on the same subnet:

```bash
ip link                                   # find the Ethernet port's name, e.g. enp3s0
sudo nmcli con add type ethernet ifname <PORT> con-name franka ipv4.method manual ipv4.addresses 172.16.0.1/24
sudo nmcli con up franka
ping -c 3 172.16.0.2                      # must answer
```

**Every session, in Desk.** Open `https://172.16.0.2` in a browser on the PC.

1. Unlock the joints (the brakes).
2. Activate FCI.
3. Release the external stop button (the user stop). Keep it within reach.

## 7. Connect the Pi (the cameras)

1. **Same network:** the PC and the Pi must be on the same Wi-Fi.
2. **Find the Pi (`admin@RaspberryPi5`).** Try these on the PC in order and stop at the first that works. What it
   gives you is `<PI>` below. The IP can change between days, but the name and the hardware address don't.
   - **By name.** If it answers, `<PI>` is `RaspberryPi5.local`.

     ```bash
     ping -c 2 RaspberryPi5.local
     ```

   - **By the Pi's Wi-Fi hardware address.** This pings every address on this PC's networks, then prints the Pi's
     line. `<PI>` is the IP at the start of that line.

     ```bash
     ( for net in $(ip -4 -o addr show scope global | awk '{print $4}' | cut -d. -f1-3 | sort -u); do
         for i in $(seq 1 254); do ping -c1 -W1 $net.$i >/dev/null 2>&1 & done; done; wait )
     ip neigh | grep -i 88:a2:9e:a1:70:d4
     ```

   - **On the Pi itself.** `<PI>` is the first address printed.

     ```bash
     hostname -I
     ```

3. **Start the camera sender on the Pi.** It runs on the Pi and isn't part of this repository.
4. **Check that the PC can reach it:**

   ```bash
   timeout 2 bash -c '</dev/tcp/<PI>/5555' && echo open
   ```

5. **Put the address in `station.yaml`:** set `endpoints.pi_frames` to `tcp://<PI>:5555`. Also set
   `endpoints.pi_time_sync` to the same address with port `5601`. The station file requires this line, but the
   current Pi stream doesn't use it.
6. **Optional, watch the raw stream:**

   ```bash
   cd process_frames
   ~/envs/deploy/bin/python -m process_frames.pi_collect --station ../station.yaml --seconds 10
   cd ..
   ```

   Good: about `30.0 pkt/s`, `gaps 0`, `bad 0`, and both cameras `640x480`.

## 8. Set the arm's limits (before the first real run)

Open `franka_control/config.yaml` and fill in the lines marked `TODO`. The ones that matter for safety:

| Setting | What to put |
|---|---|
| `safety.workspace_min_m` / `workspace_max_m` | the box (metres, robot base frame) the gripper may move in. Measure it around your table. |
| `homing.q_home_rad` | the home pose. Check that the gripper is inside the box there. |
| `load.mass_kg`, `com_m` | the weight of everything beyond the Franka Hand: printed fingers, Pi, cameras. While it's 0, `deploy check` warns. |
| `gripper.width_offset_m` | the real fingertip opening minus the hand's reading, measured with the printed fingers |

Leave everything else as it is. `station.yaml` already has `franka_control.realtime: ignore`, which works on a normal
kernel; `deploy check` shows a `warn` for it.

## 9. The real run

```bash
~/envs/deploy/bin/deploy check
~/envs/deploy/bin/deploy run
```

- `deploy check` must show no `fail` row. `deploy run` refuses to start if any check fails.
- **The policy:** `station.yaml` has `policy.kind: hold_still`, which keeps the arm still at the pose where each
  episode started. It is the first test on the real arm: it proves the cameras, timing and safety work before a
  trained model moves anything. Using a trained model is step 11.
- **Keep a hand on the external stop button** the whole time.
- **The fingers move once at start-up:** when `franka_control` connects, the Franka Hand runs its own homing (the
  fingers close and reopen to calibrate). Keep the fingers clear of objects when you start `deploy run`. The arm
  itself doesn't move at start-up: it holds where it is.

## 10. Driving it

`deploy run` starts four programs, prints `deploy: all 4 processes ready`, and shows a status line. Then you use the
keys.

The keys only work while both cameras are streaming. What the cameras see doesn't matter: a covered lens is fine.
Frames just have to keep arriving at the right size. If a camera stops for more than 2 s, the run ends.

| Key | What it does |
|---|---|
| `h` | move the whole arm (all 7 joints) to the home pose, `homing.q_home_rad`, in a slow joint move of a few seconds. The fingers don't move. Only between episodes. Keys do nothing while it moves; Ctrl-C or the external stop button stops it. |
| `c` | start an episode from where the arm is now. Press `h` first. |
| `y` / `n` | end the episode as a success / failure. The arm stops and holds. |
| `s` | stop the arm and end the episode, recorded as stopped. Everything keeps running: `h` then `c` for the next one. |
| `q` or Ctrl-C | stop the arm and quit. Everything shuts down. |

**Where results go:** `~/deploy_sessions/<date>_<time>/` holds:

- `logs/<program>.log` for each program, and `logs/deploy.log` for the launcher;
- `message_comms/`, with the episode records (success, fail, stopped, aborted);
- a copy of `station.yaml` as it was for this run.

If something goes wrong during a run, everything stops in a safe order and one line says which program failed, why,
and which log to read. The arm is safe even then: when its controller exits, the robot stops itself.

## 11. Optional: a trained policy

Skip this until a policy has been trained on data from this gripper (none has yet). It needs a GPU.

1. Install `mamba` (Miniforge), then create the policy environment:

   ```bash
   cd ~/Capstone/behavior_prompting
   mamba env create -f environment.yml          # creates the env "bp" (torch, CUDA)
   mamba run -n bp pip install -e . -e ../deployment/contracts -e ../deployment/policy
   cd ../deployment
   ```

   The `torch` lines in `environment.yml` depend on the PC's CUDA version. Read its comments before creating the
   environment.

2. In `station.yaml`:

   ```yaml
   python:
     policy: ~/miniforge3/envs/bp/bin/python     # the env's python: `mamba env list` shows the folder
   policy:
     kind: diffusion
     checkpoint: /path/to/the.ckpt
   ```

3. Run `deploy check` again. The `policy` rows check the environment, the checkpoint and the GPU.

## 12. When something fails

`deploy check` and `deploy run` name the program and what to check. The common ones:

| Message | Fix |
|---|---|
| `franka_control version ... rebuild` | run `franka_control/build.sh` (step 4) |
| `arm not reachable at 172.16.0.2:1337` | cable; the PC's address (step 6); FCI activated in Desk |
| `franka_control: not ready` at start-up | Desk: brakes unlocked, FCI on, user stop released. Then read `logs/franka_control.log`. |
| `Pi reachable ... no TCP connection` | Pi on and on the same Wi-Fi; sender running; `endpoints.pi_frames` matches its address (step 7) |
| `frame size ..., rig file says ...` | the Pi's stream size changed. Ask for a matching rig file in `process_frames/rigs/` and set `process_frames.rig_file` to it. |
| `port tcp://...: cannot bind` | a previous run is still alive. Find it with `ss -ltnp \| grep 56`, stop it, then run again. |
| `mock episode ... not found` | step 5 (copy the recording) |
| `no terminal for the keyboard` | run `deploy run` in a normal terminal window, not through a pipe or in the background |
| `python.policy not found` | step 11, or set `policy.kind: hold_still` |
| `geometry tcp triangle` fails | `gripper.yaml`, `franka_control`'s `F_T_TCP` and the rig file disagree. Run `~/envs/deploy/bin/python -m deploy.geometry`, then `deploy check` again. |

**One program at a time:** each can be started alone, to watch it in its own terminal. Pick the line you need and run
it from `deployment/`. Add `--mock` for the stand-in.

```bash
franka_control/build/franka_control --station station.yaml
(cd process_frames && ~/envs/deploy/bin/python -m process_frames --station ../station.yaml)
(cd policy         && ~/envs/deploy/bin/python -m policy         --station ../station.yaml)
(cd message_comms  && ~/envs/deploy/bin/python -m message_comms  --station ../station.yaml)
```

**Tests** (optional, about a minute each). Run them inside each folder:

```bash
cd <folder> && PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 ~/envs/deploy/bin/python -m pytest tests -q
```

The folders are `contracts`, `deploy`, `process_frames`, `message_comms` and `policy`.
`PYTEST_DISABLE_PLUGIN_AUTOLOAD=1` only matters on a PC with ROS installed.

## The files you edit

| File | What's in it |
|---|---|
| `station.yaml` | this PC's settings: Python paths, the Pi's address, ports, which policy, which rig file |
| `franka_control/config.yaml` | the arm: safety box, home pose, payload, gains, libfranka version |
| `gripper.yaml` | where the camera sits on the gripper. After a change, run `python -m deploy.geometry`. |

Everything else is code. More detail on any part: `PLAN.md`, and `STRUCTURE.md` in each folder.
