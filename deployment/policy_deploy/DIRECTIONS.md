# The laptop: run the policy

The laptop receives the Pi's two cameras, runs the policy and sends the moves to the PC, which drives the arm
(`deployment/control_deploy/DIRECTIONS.md`). The laptop never starts or stops anything on the PC. Copy the commands in
order.

Every command below runs from `~/Capstone/deployment/policy_deploy` (step 2), in a terminal where `echo $PYTHONPATH`
prints nothing (if it shows `/opt/ros/...`, run `unset PYTHONPATH` first).

> **Daily routine (once set up)**
>
> 1. **The PC first:** `control_deploy run` is running there and shows `ready` (the PC's directions).
> 2. The Pi is on and its camera sender is running. The laptop is on ASUS_5g and plugged in.
> 3. On the laptop:
>
>    ```bash
>    cd ~/Capstone/deployment/policy_deploy && unset PYTHONPATH
>    ~/envs/policy_deploy/bin/policy_deploy check
>    ~/envs/policy_deploy/bin/policy_deploy run
>    ```
>
> 4. Drive it with the keys. Keep a hand on the arm's user stop.
>
> | Key | What it does |
> |---|---|
> | `h` | home the arm (all 7 joints to the home pose). Refused during an episode or while the PC is homing. |
> | `c` | start an episode from where the arm is |
> | `y` / `n` | end the episode as a success / failure; the arm holds |
> | `s` | stop: the arm holds, the episode is recorded as stopped |
> | `q` or Ctrl-C | quit the laptop. The PC keeps running and holding the arm. |
>
> Usual cycle: `h`, `c`, ..., `y` or `n`, then `h`, `c` again. `q` when done.

The policy that runs is the one `policy_deploy use` last chose. **`hold_still`** (the start) keeps the arm at the pose
where the episode started: it tests the cameras, the link and the timing without moving anything.

> **Plug in a trained policy** (details: "A trained policy" below)
>
> ```bash
> ~/envs/policy_deploy/bin/policy_deploy use ~/Capstone/data/PICK_TEST1/policy/run_006_2026-11-18 --precision bf16 --diffusion-steps 8 --exec-h 12
> ~/envs/policy_deploy/bin/policy_deploy check
> ~/envs/policy_deploy/bin/policy_deploy run
> ```
>
> Another run: the same `use` line with its folder. Back to holding still: `policy_deploy use hold_still`.
> On the real arm `check` refuses it until `../contracts/frames.yaml` says `settled: true` (step 4 there).

---

## Setting up, from scratch

### 1. What you need

- The laptop (Ubuntu 22.04 with Python 3.10, NVIDIA driver for the trained policy later), internet, about 10 GB free.
- Wi-Fi on the lab router **ASUS_5g** (the PC and the Pi are on it too).

### 2. Get the code

Downloads the project and switches to the deployment branch (it must already be pushed to GitHub).

```bash
cd ~
git clone https://github.com/rubennoro/Capstone.git
cd Capstone
git checkout rnoro/final-deployment
cd deployment
```

- **Good:** `git log --oneline -1` shows the same commit as on the PC.
- **If not:** `destination path 'Capstone' already exists`: use it instead:
  `cd ~/Capstone && git fetch && git checkout rnoro/final-deployment && git pull && cd deployment`.

### 3. The environment

Creates `~/envs/policy_deploy`, the Python environment that runs everything on the laptop except a trained model. It
uses the laptop's own Python 3.10 (`python3.10 -m venv`; no `uv` needed).

```bash
unset PYTHONPATH
python3.10 -m venv ~/envs/policy_deploy
~/envs/policy_deploy/bin/pip install --upgrade pip
~/envs/policy_deploy/bin/pip install -e contracts -e "policy_deploy[light,test]"
cd policy_deploy
~/envs/policy_deploy/bin/policy_deploy --help
```

- **Good:** the last command prints `usage: policy_deploy [-h] {check,run} ...`. Warnings about "Editable
  installation" during the install are normal.
- **If not:** `ensurepip is not available`: `sudo apt install -y python3.10-venv`, delete `~/envs/policy_deploy`, redo.

From here on you are in `~/Capstone/deployment/policy_deploy`.

### 4. The addresses on ASUS_5g (once)

The laptop must know the PC's and the Pi's addresses, and the PC's firewall must know the laptop's.

1. Connect the laptop to ASUS_5g and read its address:

   ```bash
   hostname -I
   ```

   Pick the four-number address on the Wi-Fi (not `172.17.0.1`, docker; `ip -4 -o addr show scope global` shows the
   card, Wi-Fi cards start with `wl`). Get the PC's address the same way on the PC.
2. **Find the Pi** (`admin@RaspberryPi5`). Try these in order and stop at the first that works; it gives `<PI>`:
   - By name: `ping -c 2 RaspberryPi5.local`. `<PI>` is the address in brackets on the first line.
   - By the Pi's Wi-Fi hardware address: this pings every address on the laptop's networks, then prints the Pi's line;
     `<PI>` is the address at its start.

     ```bash
     ( for net in $(ip -4 -o addr show scope global | awk '{print $4}' | cut -d. -f1-3 | sort -u); do
         for i in $(seq 1 254); do ping -c1 -W1 $net.$i >/dev/null 2>&1 & done; done; wait )
     ip neigh | grep -i 88:a2:9e:a1:70:d4
     ```

   - On the Pi itself: `hostname -I`; `<PI>` is the first address.
3. Recommended: reserve the three addresses on the router so they never change (the ASUS web page, usually
   `http://192.168.50.1`: LAN → DHCP Server → manually assigned IP). Otherwise redo this step when one changes.
4. Edit `../contracts/link.yaml`: `pc: address:` is the PC's address, `laptop: addresses: [...]` the laptop's. Make the
   **same edit on the PC** (or commit it once and `git pull` there). The PC then reinstalls its firewall.
5. Edit `station.yaml`: `pi: address:` is `<PI>` (numbers only, not the name).

- **Good:** `~/envs/policy_deploy/bin/policy_deploy check` shows `laptop address PASS` and `Pi reachable PASS` (and,
  with the PC running, the `manager` rows about the PC `PASS`).
- **If not:** "When something fails".

### 5. Watch the Pi stream (once)

Checks that the Pi's frames arrive, and measures how the two cameras are paired, which sets one value
(`pair_max_lead_ms`).

```bash
~/envs/policy_deploy/bin/python -m policy_deploy.frames.pi_collect --station station.yaml --seconds 30
```

- **Good:** about `30.0 pkt/s`, `gaps 0`, `bad 0`; then read the pair-gap report as the `pi_collect` section says.
- **If not:** nothing arrives: the Pi's sender and `pi: address` (step 4).

### 6. Check the link to the PC (once, and when the Wi-Fi is bad)

With `control_deploy run` running on the PC (and no laptop run), checks the link speed and the clock match.

```bash
~/envs/policy_deploy/bin/python -m policy_deploy.manager.pc_probe --seconds 30
```

- **Good:** every row at the end says `PASS` (details in the `pc_probe` section).
- **If not:** a `FAIL` row names what is wrong; no C2 at all: the PC's `run`, ASUS_5g, `link.yaml` (step 4).

### 7. The real run (hold still)

The first run with the real arm. The PC must already have done its bring-up (the PC's directions, step 10).

```bash
~/envs/policy_deploy/bin/policy_deploy check
~/envs/policy_deploy/bin/policy_deploy run
```

- **Good:** see the `run` section. `h`, `c`: the arm stays still (hold still), the status line counts steps; `y`.
- **If not:** "When something fails".

### 8. Stopping, restarting, and after a fault

- **End an episode:** `y`, `n` or `s`. The arm holds; `h` then `c` for the next one.
- **Quit:** `q` or Ctrl-C. The PC keeps holding the arm; `policy_deploy run` again whenever you want.
- **When the run stops by itself**, it prints one line: `policy_deploy: FAILED <process> ...: <what failed>: <what
  to check>`. Fix that (table at the end) and run again.
- **After a fault on the PC** (`the PC reports fault: ...`): on the PC, Ctrl-C, release the user stop if it was a
  reflex, `control_deploy run`; then `policy_deploy run` here.
- Turn off sleep/suspend on the laptop during runs: if it sleeps, the PC stops the arm.

---

## The commands

### `policy_deploy check`

**When:** before every run. It moves nothing (it talks to the PC for about 2 s: never run it during a run).

```bash
~/envs/policy_deploy/bin/policy_deploy check
```

**Good:** a table where no row says `FAIL`, ending `policy_deploy check: ready (N warning(s))`. `WARN Wi-Fi` (not
on ASUS_5g) is fine when the laptop is wired. With the PC running it also shows the PC's rows (`hello`, `C2 rate`,
`clock agreement`, `PC commit`, ...). `WARN PC commit` means the two machines are on different commits: `git pull`
on both (and rebuild on the PC).

### `policy_deploy run`

**When:** every session, after the PC is `ready`.

```bash
~/envs/policy_deploy/bin/policy_deploy run
```

**Keys:** `h c y n s q` (table at the top).

**Good:** the check table, then `policy_deploy: all 3 processes ready (session: ...)` and a status line, for example:

```
[idle] arm idle | cams ok 30/ok 30 fps | late loops 0 | rtt 3 ms c2 100 Hz clock ok | keys: c start  h home  q quit
[ep 1 running 2.0 s] step 59 | rtt 3 ms c2 100 Hz clock ok | cycle 41 ms (policy 3) | img 74 ms | margin +239 ms | keys: y n s q
```

`cams` is both cameras' state and frame rate (30 each), `rtt` the link's round trip, `c2` the PC's state rate (about
100 Hz), `margin` how far ahead of the arm the policy is (positive is good). Messages appear above it:
`home pose reached`, `episode 1 started`, `episode 1: success (operator), 181 steps, 6.0 s`. It ends with
`policy_deploy: done`. Records: `~/policy_deploy_sessions/<date>_<time>/` (`logs/`, `manager/episodes.jsonl`).

While homing the keys are inactive; Ctrl-C quits, the arm's user stop stops it. A refused key prints why, e.g.
`h: end the episode first (y / n / s)` or `c: the PC is homing (its h or g key): wait for idle`.

### `policy_deploy use`

**When:** to choose which policy `run` runs (a trained run, or hold still). It moves nothing and needs nothing
running; it only rewrites the `policy:` and `manager:` sections of `station.yaml` (every other line stays).

```bash
~/envs/policy_deploy/bin/policy_deploy use <run folder> [--prompt-index N] [--precision bf16] [--diffusion-steps N] [--exec-h N]
~/envs/policy_deploy/bin/policy_deploy use hold_still
```

**Good:** the changed lines, then `policy_deploy use: station.yaml now runs <run>. Next: policy_deploy check`.
Options not given keep their current value. Rarely needed: `--task NAME` (a dataset with several tasks),
`--prompt-dataset FILE --prompt-split all` (demos the policy never trained on), `--station FILE`.

**If not:** `policy_deploy use: FAIL <what>` and a `hint:` line; `station.yaml` is not changed.

### The link probe: `pc_probe`

**When:** with `control_deploy run` on the PC and **no** `policy_deploy run` going, to check the link (once at
set-up, again when going wired or when runs stop with "PC not answering"). It only reads; it never moves the arm.

```bash
~/envs/policy_deploy/bin/python -m policy_deploy.manager.pc_probe --seconds 30
```

**Good:** one line per second, then the checks:

```
round_trip +1006.288326 s +0.0 ppm | one_way +1006.287812 s +0.0 ppm | diff +0.51 ms | newest C2 3.3 ms old | C2 100 Hz gaps 0 | rtt p50 1.0 p95 1.3 max 1.4 ms
PASS  H2: C2 mode idle: no other C3 client
PASS  hello: answered; RTT p50 0.9 ms, p95 1.3 ms, max 1.8 ms (237 requests)
PASS  C2 rate: 100 Hz, 0 seq gaps, 0 t_ns repeats seen
PASS  compare_hello: all equal
```

`--axes` adds three lines saying where the TCP's axes point (the check before `settled: true`, "A trained
policy" step 4). `C2` about 100 Hz with few gaps; `rtt p95` a few ms on Wi-Fi (under 1 ms wired); `diff` (the two clock estimates)
within a few ms. Any `FAIL` names what differs. Without the PC: `--mock` against the fake PC
(`python -m policy_deploy.fake_pc --mock` in another terminal).

### Watch the Pi stream: `pi_collect`

**When:** once at set-up, and whenever the cameras seem wrong. Receive only: it never sends anything to the Pi. Run it
with `policy_deploy run` stopped.

```bash
~/envs/policy_deploy/bin/python -m policy_deploy.frames.pi_collect --station station.yaml --seconds 30
```

**Good:** a stats line per second, then the **pair-gap report** (an example; your numbers differ):

```
 30.0 pkt/s | packet_no gaps 0 | main 640x480 18 kB frame_no gaps 0 | ultrawide 640x480 35 kB frame_no gaps 0 reused 0 | bad 0
total: 900 packets, 0 bad, 0 packet_no gaps, 0 with uw_reused
pair gap (main.t_mid_ns - ultrawide.t_mid_ns, from t_mid_ns; negative = ultrawide newer): 900 packets, min -16.8 ms, median -16.3 ms, max -15.6 ms
longest run of consecutive packets with the same pair: 1 (ultrawide frame_no 191)
```

About `30.0 pkt/s`, `gaps 0`, `bad 0`, both cameras `640x480`.

**Reading the pair-gap report (needed once):** each main frame is sent with one ultrawide frame; the gap says how far
apart they were taken. A negative `min` means the ultrawide is newer by that much (the live Pi: about −16 ms). The
laptop accepts an ultrawide up to `pair_max_lead_ms` (25 ms) newer and up to `pair_max_age_ms` (60 ms) older:

- If `−min` plus 5 ms is above 25, change the line `frames: {}` in `station.yaml` to
  `frames: {select: {pair_max_lead_ms: <−min + 5, rounded up>}}`.
- `max` must stay below 60, and `longest run` at 1 or 2. If not, the Pi's ultrawide camera is freezing: check the Pi.

The headers and a few images are saved in `~/policy_deploy_sessions/<date>_<time>_pi_collect/`.

### The frames probe

**When:** to look at the images the policy gets (camera mount, lighting), with `policy_deploy run` stopped. Start the
frames process alone in one terminal, ask it once from another, then Ctrl-C the first:

```bash
~/envs/policy_deploy/bin/python -m policy_deploy.frames
~/envs/policy_deploy/bin/python -m policy_deploy.frames.probe --save ~/probe_images
```

**Good:** the first prints `ready`; the probe prints both cameras `ok` at `30.0` fps, the four image times and ages,
and writes `main_0.png`, `main_1.png`, `ultrawide_0.png`, `ultrawide_1.png` into `~/probe_images`: open them and look.

### Policy timing: `policy.bench`

**When:** once, before choosing `exec_h` for a trained policy (it needs the torch environment, section below). Laptop
on power, browser closed, `powerprofilesctl set performance`.

```bash
~/envs/policy_deploy_torch/bin/python -m policy_deploy.policy.bench --steps 16,8 --precision fp32,bf16 --infers 500
```

**Good:** a `load: ...` line, then one line per setting:
`steps 16 fp32: mean ... ms, p50 ..., p95 ... ms, max ... ms over 500 (...); VRAM allocated ... MB, ...`.

Take `p95` for `steps 16 fp32` and choose `exec_h` (the number of moves sent per policy step) in `station.yaml`
(change the line `manager: {}` to `manager: {exec_h: ...}`). The table assumes `pc_probe` shows `rtt p95` of 10 ms or less:

| p95 | `exec_h` |
|---|---|
| up to 89 ms | 8 (the default) |
| up to 139 ms | 10 |
| up to 189 ms | 12 |
| more | `policy: {precision: bf16}` and use the `bf16` line; still too slow: `diffusion_steps: 8` (changes the policy's behaviour) |

`--checkpoint /path/to.ckpt` instead reports a real checkpoint's load time and memory. In the light environment it
prints `FAIL: torch is not importable`: use the torch environment.

---

## A trained policy

Once a policy has been trained on data from this rig and fetched to the laptop (`train/fetch.sh`, into
`~/Capstone/data/<folder>/policy/<run>/`). You never edit `station.yaml` by hand for this: `policy_deploy use` does it.

1. **The torch environment** (once; large: several GB):

   ```bash
   cd ~/Capstone && unset PYTHONPATH
   python3.10 -m venv ~/envs/policy_deploy_torch
   ~/envs/policy_deploy_torch/bin/pip install --upgrade pip
   ~/envs/policy_deploy_torch/bin/pip install torch==2.7.1 torchvision==0.22.1 --index-url https://download.pytorch.org/whl/cu118
   ~/envs/policy_deploy_torch/bin/pip install -r controlSW/franka_deploy/requirements-inference.txt
   ~/envs/policy_deploy_torch/bin/pip install -e behavior_prompting -e deployment/contracts -e "deployment/policy_deploy[model,test]"
   cd deployment/policy_deploy
   ```

   Good: `~/envs/policy_deploy_torch/bin/python -c "import torch; print(torch.cuda.is_available())"` prints `True`.

2. **Choose the policy:** give `use` the run folder. Example, the 30-demo pick policy `run_006`:

   ```bash
   ~/envs/policy_deploy/bin/policy_deploy use ~/Capstone/data/PICK_TEST1/policy/run_006_2026-11-18 --precision bf16 --diffusion-steps 8 --exec-h 12
   ```

   Good: it prints the lines of `station.yaml` it changed, then
   `policy_deploy use: station.yaml now runs run_006_2026-11-18. Next: policy_deploy check`.

   It finds everything in the run folder by itself: the checkpoint (`checkpoints/*.ckpt`), whether it is a prompting
   checkpoint, and for one that is, the demos it trained on (`../../package/<dataset>/<dataset>.zarr.zip`) and the
   task name in them. A prompt is not text: it is one recorded demo, shown to the policy before each episode.

   The three speed options are what this laptop's RTX 3050 Ti needs to keep up with these checkpoints. Give them once;
   later `use` calls keep them. To measure a new checkpoint: `policy.bench --checkpoint <the .ckpt>` ("Policy timing").

   | To | Run |
   |---|---|
   | switch to another run (e.g. the 25-demo one) | `policy_deploy use ~/Capstone/data/PICK_TEST1/policy/run_016_2026-11-18` |
   | show it another demo (0 is the first) | `policy_deploy use <run folder> --prompt-index 3` |
   | go back to holding still | `policy_deploy use hold_still` |

   Each session folder keeps a copy of `station.yaml`, so every trial records which policy and demo it ran.
   If anything is missing it prints `policy_deploy use: FAIL ...` with a hint, and `station.yaml` is not changed. The
   usual one: `no checkpoint in .../checkpoints` (the run was not fetched: `train/fetch.sh <folder> <run>`).

3. **Try it without the arm (optional):** `policy_deploy run --mock` runs the chosen policy against a fake Pi and a
   fake PC. It needs a recorded Pi episode in `station.yaml`: `frames: {mock_episode: <a folder with meta.json>}`.
   Good: the status line's `margin` stays positive during an episode.

4. **On the real arm, once: settle the gripper axes.** `check` refuses a trained policy
   (`settled FAIL trained policy refused on the real arm`) until `../contracts/frames.yaml` says `settled: true`.
   Set it only after both checks below pass. Only `h` moves the arm. "Right" means standing behind the robot,
   facing the way it faces.

   1. PC: `git pull`, `control_deploy run`, press `h` (home). Laptop, with `policy_deploy run` not running:

      ```bash
      ~/envs/policy_deploy/bin/python -m policy_deploy.manager.pc_probe --seconds 3 --axes
      ```

      Good: `PASS  compare_hello: all equal` (the PC has the same `frames.yaml`), then
      `TCP x axis -> ...: to the robot's right`, `TCP y axis -> ...: back (toward the base)`, `TCP z axis -> ...: down`.
   2. Camera: save images with the frames probe (section "The frames probe"), holding your hand to the robot's right
      of the fingers. Good: in `ultrawide_0.png` the fingers are at the bottom and your hand is on the **right**.
   3. Both good: set `settled: true` in `../contracts/frames.yaml`, commit, push, `git pull` on the PC and restart
      `control_deploy run` there. Either one wrong (x "to the robot's left", or your hand on the left): **don't**
      settle; the camera is on the other face and `F_T_TCP` needs the other rotation (PLAN §9).
   4. First trained episode: hand on the user stop. If the first move does not head for the object, press `s`.

5. **Every run after that:** the daily routine at the top (`check`, then `run`). The chosen policy stays chosen.

6. **Once episodes work: record what the policy sees.** Off by default. Add under `manager:` in `station.yaml`
   (`use` keeps these lines):

   ```yaml
   manager:
     save_tap: true          # one .npz per step sent to the arm
     tap_images_every: 1     # the images in every one (default 10)
   ```

   Each episode goes to `<session>/manager/tap/ep_NNNN/chunk_K.npz`: the policy's exact input images (`main`,
   `ultrawide`: the last 2 frames, 224×224 RGB), the poses, the predicted chunk, the waypoints and the PC's reply.
   One file per policy step (a few per second, not the camera's 30 fps); about 600 KB each, tens of MB per episode.
   Turn it off (`false`) while debugging timing.

**What the laptop needs.** Measured with `run_001` (the same model size) on an RTX 3050 Ti laptop GPU (4 GB, 35 W
power cap, a low-end GPU):

| | Needed | Measured with `run_001` |
|---|---|---|
| GPU | NVIDIA with CUDA; driver 535 or newer (tested) | |
| GPU memory | 4 GB at least; 6 GB or more for headroom | peak with fp32 about 2.0 GB allocated / 2.2 GB reserved; with bf16 about 2.8 GB / 2.8 GB (about 3.0 GB in `nvidia-smi`): bf16 keeps extra copies of the weights |
| `precision: bf16` | an RTX 30-series GPU or newer; on older GPUs keep `fp32` (bf16 runs there but emulated, so it is not faster) | |
| Speed | a policy step that fits `exec_h` (the table in "Policy timing") | in a `--mock` run: fp32 16 steps about 300 ms, bf16 16 steps about 226 ms, bf16 8 steps about 186 ms |
| RAM | 16 GB | peak about 5 GB while the checkpoint loads (about 10 s) |
| Disk | about 10 GB | torch environment 6.1 GB, checkpoint 2.6 GB, demos 0.5-0.9 GB |

Plugged in, on AC power: on battery or power-saver the start-up limits are missed.

---

## When something fails

`check` names the row; a run that stops prints `policy_deploy: FAILED <process> ...: FATAL <process>: <what failed>:
<what to check>`.

| You see | Do this |
|---|---|
| `manager C2 FAIL no C2 state from tcp://...:5620 within 20 s` | start `control_deploy run` on the PC first; same Wi-Fi; `link.yaml pc: address` (step 4) |
| `laptop address FAIL none of link.yaml laptop.addresses ... is on this laptop` | step 4: the laptop's address changed or `link.yaml` is wrong (then also the PC's firewall) |
| `Pi reachable FAIL no TCP connection to <PI>:5555` | the Pi is on, on ASUS_5g, its sender is running; `station.yaml pi: address` (step 4) |
| `Wi-Fi WARN not on ASUS_5g` | fine when wired; otherwise join ASUS_5g |
| `terminal FAIL no terminal for the keyboard` | run it in a normal terminal window, not through a pipe |
| `... cannot bind tcp://127.0.0.1:56..: is another policy_deploy run ... still running?` | another `policy_deploy run` (or a `control_deploy run --mock` on this laptop) is still open: quit it |
| `settled FAIL trained policy refused on the real arm` | set `kind: hold_still` in `station.yaml` ("A trained policy", item 3) |
| `PC not answering: ...: check franka_control on the PC and the link` | is `control_deploy run` still running? Wi-Fi; then run again |
| `the PC stopped the arm (watchdog or the PC's 's')` | a Wi-Fi stall over 1 s, or someone pressed `s` on the PC: run again |
| `the PC reports fault: ...` | on the PC: Ctrl-C, release the user stop if it was a reflex, `control_deploy run`; then run again |
| `franka_control restarted on the PC` | the PC was restarted during the run: run again |
| `the PC is driving an episode for another C3 client` | stop the other laptop run, `pc_probe` or the PC's `console` |
| `FAILED frames during start-up: not ready after 30 s` | the Pi stream isn't arriving: `pi_collect` |
| `main camera down` / `ultrawide camera down` | the Pi or its Wi-Fi: check the Pi, then run again |
| `frame size ..., rig file says ...` | the Pi now streams another size: the rig file in `rigs/` must match it |
| `clock mismatch: ...` or a `hello` mismatch | both machines on the same commit, `link.yaml` the same on both; then `pc_probe` |
| `SKIPPING` or a negative `margin` on the status line | the policy or the link is too slow: the arm just holds; raise `exec_h` (policy timing) |

## Tests (optional)

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 ~/envs/policy_deploy/bin/python -m pytest tests -q
```

**Good:** `... passed, ... skipped` and no `failed` (about 2 minutes). Don't run it during a `policy_deploy run` (same ports).

## The files you edit

| File | What's in it |
|---|---|
| `station.yaml` | this laptop's settings: the Pi's address, the policy (set it with `policy_deploy use`), overrides of `config/*.yaml` |
| `../contracts/link.yaml` | the PC's and the laptop's addresses: the same on both machines |

How it is built and why: `deployment/PLAN.md` and `policy_deploy/STRUCTURE.md`.
