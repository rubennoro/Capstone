# The laptop: collect teleop demonstrations

The laptop drives the Franka Panda from a page in the browser with the keyboard and records each demonstration: both
Pi cameras and every arm state. The PC runs the arm (`deployment/control_deploy/DIRECTIONS.md`); the laptop never
starts, stops or changes anything on the PC or the Pi. Copy the commands in order.

Every command below runs from `~/Capstone-teleop-collect-deploy/deployment/teleop_collect` (step 3), in a terminal
where `echo $PYTHONPATH` prints nothing (if it shows `/opt/ros/...`, run `unset PYTHONPATH` first).

> **Daily routine (once set up)**
>
> 1. Hardware (§9): physical, then the PC (`control_deploy run`, `h`, `p`), then the Pi (LED blue).
> 2. Laptop (§10): `teleop_collect check`, then `teleop_collect run`; the page opens.
> 3. Location (§11): the header's **Storage** (default `~/CapstoneTeleop`).
> 4. Task (§12): pick one in **Task**, or **New task**.
> 5. Every episode (§13): `h`, `p` on the PC, **Start episode**, drive, **Stop episode**, **Save: success** /
>    **Save: fail** / **Discard**.
> 6. Quit (§15): Ctrl-C, the Pi's C button; end of the day `q` on the PC and lock the brakes.

---

## Setting up, from scratch

### 1. What you need

- The laptop with Ubuntu 22.04 and Python 3.10 (`python3.10 --version`), internet for the install, and a browser.
- Free disk: about 110 MB per 30 s episode, under `~/CapstoneTeleop`.
- Wi-Fi on the lab router **ASUS_5g** (the PC and the Pi are on it too).
- The PC set up and brought up first (`deployment/control_deploy/DIRECTIONS.md`, steps 1–10, the start pose
  included).
- For the first test without hardware: the recorded Pi episode `~/Downloads/data/incoming/episode_000152` (step 5 says
  how to get it).

### 2. Get the code

Downloads the project, on the teleop branch, into its own folder `~/Capstone-teleop-collect-deploy`, so it never
touches a `~/Capstone` clone that `policy_deploy` runs from. **The branch must already be pushed to GitHub** from the machine it was built on.

```bash
git clone -b rnoro/teleop-collect-deploy https://github.com/rubennoro/Capstone.git ~/Capstone-teleop-collect-deploy
cd ~/Capstone-teleop-collect-deploy/deployment
```

`-b` picks the branch; the last argument is the folder.

- **Good:** `ls` shows `contracts`, `control_deploy` and `teleop_collect`.
- **If not:**
  - `destination path '...' already exists`: it is already cloned; update it:
    `cd ~/Capstone-teleop-collect-deploy && git pull && cd deployment`.
  - `Remote branch rnoro/teleop-collect-deploy not found`: the branch is not pushed yet.

### 3. The environment

Creates `~/envs/teleop_collect`, the one Python environment of this laptop side, with this branch's
`deploy_contracts` and `teleop_collect` installed editable.

```bash
unset PYTHONPATH
python3.10 -m venv ~/envs/teleop_collect
~/envs/teleop_collect/bin/pip install --upgrade pip
~/envs/teleop_collect/bin/pip install -e "contracts[test]" -e "teleop_collect[test]"
cd teleop_collect
~/envs/teleop_collect/bin/teleop_collect --help
```

**Never install into `~/envs/policy_deploy*`:** those have `deploy_contracts` installed from the
`rnoro/final-deployment` checkout, and this install would point them at this folder instead.

- **Good:** the last command prints `usage: teleop_collect [-h] {check,run} ...`. Warnings about "Editable
  installation" are normal.
- **If not:** `ensurepip is not available`: `sudo apt install -y python3.10-venv`, delete `~/envs/teleop_collect`,
  redo.

From here on you are in `~/Capstone-teleop-collect-deploy/deployment/teleop_collect`.

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
4. Edit `../contracts/link.yaml`: `pc: address:` is the PC's address, `laptop: addresses: [...]` the laptop's.
5. **Make the same edit on the PC**, by hand, in its `~/Capstone/deployment/contracts/link.yaml` (the PC's directions,
   step 6). There it stays a local edit: never commit it to `rnoro/final-deployment`. The two files must be equal.
6. Edit `station.yaml`: `pi: address:` is `<PI>` (numbers only, not the name).

- **Good:** step 8 shows `laptop address PASS` and `Pi reachable PASS`.
- **If not:** "When something fails" (§19).

### 5. A first run without hardware (`--mock`)

Runs everything on the laptop alone: a fake PC and a fake Pi that plays `~/Downloads/data/incoming/episode_000152`. Nothing
connects to the PC or the Pi. Its episodes go to `~/CapstoneTeleop_mock` (`station.yaml mock_data_root`), never
among the real ones in `~/CapstoneTeleop`.

The recorded episode comes from the lab laptop (a folder recorded on the Pi: `meta.json`, `cam0.mp4`, `cam1.mp4` and
their CSVs). On a fresh machine, copy it over first:

```bash
mkdir -p ~/Downloads/data/incoming
scp -r <USER>@<LAB_LAPTOP>:~/Downloads/data/incoming/episode_000152 ~/Downloads/data/incoming/
```

`<USER>@<LAB_LAPTOP>` is the lab laptop's login and address.

```bash
~/envs/teleop_collect/bin/teleop_collect check --mock
~/envs/teleop_collect/bin/teleop_collect run --mock
```

`--mock` replaces the Pi and the PC with stand-ins on `127.0.0.1`. Add `--no-browser` to only print the page's
address.

In the page:

1. **New task**, a name such as `mock_test`, **Create**.
2. Wait for the state card's **Ready** and `At the start-up pose ✓ ready to start (the PC checks it at Start)`. The
   fake PC starts at its start pose, as if `h` and `p` had been pressed on the PC.
3. **Start episode**; click the page; wait about 1 s; hold `W` for 1 s; press `N`, then `M`.
4. **Stop episode**, then **Save: success**. About 3 s later the state card shows `At the start pose ✓ ready to
   start`: the fake PC stands in for you at the PC and does `h`, then `p` by itself, 1 s after every episode's stop.
5. Ctrl-C in the terminal. The episode stays in `~/CapstoneTeleop_mock/mock_test/`.

- **Good:**
  - `check --mock` ends with `teleop_collect check: ready (0 warning(s))`.
  - `run --mock` prints `teleop_collect: teleop: the start pose is the arm's pose at start-up: the fake PC starts at
    its start pose`, `teleop_collect: all 4 processes ready (session: ...)`, then
    `teleop_collect: ready: open http://127.0.0.1:8765`, and the page opens.
  - The page shows the **Mock PC** pill, green **Arm**, **Cameras** and **Link** dots, and both cameras moving with
    `30 fps`. The **Activity log** shows `start pose: the arm's pose now (0.450, 0.000, 0.300 m); every episode starts
    here (h, then p on the PC)`. While you hold `W`, **Diagnostics** → TCP position `x` grows by about 50 mm per
    second.
  - After the save, **Episodes** shows `episode_000 Success ...`. Ctrl-C prints `teleop_collect: stopped by
    Ctrl-C`.
- **If not:** `mock episode FAIL not found (no meta.json)`: copy the recorded episode (above), or point
  `frames: {mock_episode: <folder>}` in `station.yaml` at one. Anything else: "When something fails" (§19).

### 6. Watch the Pi stream (once)

Checks that the Pi's frames arrive, and measures how the two cameras are paired, which sets one value
(`pair_max_lead_ms`). Press the Pi's C button first (LED blue), with `teleop_collect run` stopped.

```bash
~/envs/teleop_collect/bin/python -m teleop_collect.frames.pi_collect --station station.yaml --seconds 30
```

`--station` takes the Pi's address from `station.yaml`; `--seconds` is how long it listens.

- **Good:** a stats line per second, then the report, for example:

  ```
   30.0 pkt/s | packet_no gaps 0 | main 640x480 18 kB frame_no gaps 0 | ultrawide 640x480 35 kB frame_no gaps 0 reused 0 | bad 0
  total: 900 packets, 0 bad, 0 packet_no gaps, 0 with uw_reused
  pair gap (main.t_mid_ns - ultrawide.t_mid_ns, from t_mid_ns; negative = ultrawide newer): 900 packets, min -16.8 ms, median -16.3 ms, max -15.6 ms
  longest run of consecutive packets with the same pair: 1 (ultrawide frame_no 191)
  ```

  About `30.0 pkt/s`, `gaps 0`, `bad 0`, both cameras `640x480`. Then read the pair gap:
  - If `−min` plus 5 ms is above 25, change the line `frames: {}` in `station.yaml` to
    `frames: {select: {pair_max_lead_ms: <−min + 5, rounded up>}}`.
  - `max` must stay below 60, and `longest run` at 1 or 2. If not, the Pi's ultrawide camera is freezing: check the
    Pi.
- **If not:** nothing arrives: the Pi's C button (LED blue), ASUS_5g, `pi: address` (step 4).

### 7. Check the link to the PC (once, and when the Wi-Fi is bad)

With `control_deploy run` running on the PC and no `teleop_collect run` going, measures the link and checks that both
machines agree on the contract files. It never moves the arm.

```bash
~/envs/teleop_collect/bin/python -m teleop_collect.teleop.pc_probe --seconds 30
```

`--seconds` is how long it probes.

- **Good:** one line per second, then the checks:

  ```
  round_trip +1006.288326 s +0.0 ppm | one_way +1006.287812 s +0.0 ppm | diff +0.51 ms | newest C2 3.3 ms old | C2 100 Hz gaps 0 | rtt p50 1.0 p95 1.3 max 1.4 ms
  PASS  H2: C2 mode idle: no other C3 client
  PASS  hello: answered; RTT p50 0.9 ms, p95 1.3 ms, max 1.8 ms (237 requests)
  PASS  C2 rate: 100 Hz, 0 seq gaps, 0 t_ns repeats seen
  PASS  compare_hello: all equal
  ```

  `C2` about 100 Hz with few gaps, `rtt p95` a few ms on Wi-Fi, `diff` within a few ms.
- **If not:** `FAIL  C2: no C2 state from tcp://<PC>:5620 within 20 s: ...`: start `control_deploy run` on the PC; ASUS_5g;
  `link.yaml pc: address` (step 4). `FAIL  compare_hello: ...` names what differs: make `link.yaml` equal on both
  machines.

### 8. `teleop_collect check`

Every check of a real run, without starting anything: the station, the addresses, the Pi, the PC link (it talks to
the PC for about 2 s, so never run it during a run), the ports, the disk. The PC's `control_deploy run` must be up and
the Pi streaming.

```bash
~/envs/teleop_collect/bin/teleop_collect check
```

- **Good:** a table of rows `<process>  <check>  <status>  <detail>` where no status is `FAIL`, ending
  `teleop_collect check: ready (N warning(s))`. Expected among them:

  ```
  station  laptop address   PASS  192.168.50.20 is on this laptop
  station  commit           INFO  laptop commit <40 letters and digits>+dirty
  frames   Pi reachable     PASS  TCP connection to <PI>:5555 opened
  teleop   H2               PASS  C2 mode idle: no other C3 client
  teleop   hello equality   PASS  schemas, mock, F_T_TCP, limits and watchdog equal
  teleop   PC commit        WARN  PC commit <b> differs from laptop commit <a>+dirty
  teleop   PC mock          INFO  the PC runs the real arm
  teleop   PC margins       INFO  teleop assumes the PC's safety.box_margin_m 0.05 and cone_margin_deg 10 (...)
  ```

  - `PC commit WARN`: the laptop runs `rnoro/teleop-collect-deploy` and the PC `rnoro/final-deployment`, so the
    commits always differ. It is recorded, never refused.
  - `commit INFO ... +dirty` after editing `link.yaml` or `station.yaml`.
  - `Wi-Fi WARN` when the laptop is wired.
  - `PC margins INFO`: the PC's `station.yaml` must be on its teleop block, whose margins are these (the PC's
    directions, step 8, "Teleop settings"). The PC does not report them, so nothing here can check it. With smaller PC
    margins a key held into the floor can fault the PC.
- **If not:** the `FAIL` row says what to fix; see "When something fails" (§19).

---

## Every session

### 9. Hardware ready (all of it before the laptop)

Order: the PC, then the Pi, then the laptop (§10).

**Part A — Physical**

1. Router on, arm controller on.
2. User stop plugged in and within reach.
3. Table clear except the scene.

**Part B — The PC** (its directions: `deployment/control_deploy/DIRECTIONS.md`)

1. Desk (`https://172.16.0.2`): unlock the brakes, activate FCI, release the user stop.
2. `control_deploy/station.yaml` on its **teleop block**, so no software stops get in the way of teleop (the PC's
   directions, step 8, "Teleop settings").
3. `control_deploy run`: starts the arm's control; leave it running all session.
   - **Good:** `control_deploy: ready`; the fingers home once and the arm holds. (Its `start policy_deploy on the
     laptop` means `teleop_collect run` here.)
4. `h`: all 7 joints slowly to home.
   - **Good:** `homing finished`.
5. `p`: to the start pose, hand open. Hand on the user stop.
   - **Good:** `start pose reached`.
   - **If not:** `key p refused: homing.q_start_rad is not set: ...`: no start pose measured yet; press only `h`
     (below).

**Part C — The Pi**

1. Press its C button: starts the camera stream.
   - **Good:** the LED turns blue.
   - **If not:** check the stream with `pi_collect` (§17).

**Go on only when:**

- the PC shows `start pose reached` (or `homing finished` without a start pose)
- the Pi's LED is blue
- your hand is near the user stop

**About the start pose:** it is the PC's `homing.q_start_rad`, measured once (the PC's directions, step 10;
`pc_probe --start-pose`, §17). `teleop_collect run` takes the arm's pose at its start-up as the session's start pose:
that is why `h`, `p` come first. Without `homing.q_start_rad` the start pose is home and the PC checks nothing at
Start. Collect and deploy with the same start pose: changing `homing.q_start_rad` means recording again.

### 10. Start the laptop

```bash
cd ~/Capstone-teleop-collect-deploy/deployment/teleop_collect && unset PYTHONPATH
~/envs/teleop_collect/bin/teleop_collect check
~/envs/teleop_collect/bin/teleop_collect run
```

1. `cd ... && unset PYTHONPATH`: the folder every command runs from, without ROS's Python path.
2. `check`: every check of a run, nothing started (§8).
3. `run`: starts the four processes and opens the page. **The arm does not move:** its pose now becomes the session's
   start pose.

- **Good:**
  - The terminal: `teleop_collect: all 4 processes ready (session: ...)`, then
    `teleop_collect: ready: open http://127.0.0.1:8765`.
  - The page opens (or open that address): the **Arm**, **Cameras** and **Link** dots green, both cameras
    (**Wrist camera**, **Wide view**) moving with `30 fps`.
- **If not:** `teleop_collect: not started: N check(s) failed (see above)` or `teleop_collect: FAILED <process> ...`:
  "When something fails" (§19).

### 11. Location: where the episodes are saved

Default `~/CapstoneTeleop`. The header's **Storage** shows the folder in use.

To change it (not while recording):

1. **Change**, type a full path (`/...` or `~/...`), **Save**.
   - **Good:** the new path in the header; the **Task** list reloads (no task selected).
   - **If not:** the reason next to the field, e.g. `Give a full path: /... or ~/...`.
2. Or start the laptop with `~/envs/teleop_collect/bin/teleop_collect run --data-root PATH`.

It is remembered for later runs in `storage.local.yaml` (next to `station.yaml`); delete that file to go back to
`~/CapstoneTeleop`.

### 12. New task (or pick one)

**Part A — Pick a task:** choose it in **Task**.

- **Good:** its description under the selector, its episodes under **Episodes**.

**Part B — New task**

1. **New task**, a name (`a-z`, `0-9`, `_`), the task in one line, **Create**.
   - **Good:** the new task selected, its description shown, **Episodes** says
     `No episodes yet. Press Start episode to record one.`
   - **If not:** the reason under the form.

**Part C — Delete a whole task** (its folder and every episode; not while recording)

1. **Delete task…**, type the task's name, **Delete task** (enabled only when the name matches exactly).
   - **Good:** the task is gone from **Task**.

### 13. Repeated episodes

One cycle per demonstration.

**Part A — Reset**

1. On the PC: `h`, wait for `homing finished`; `p`, wait for `start pose reached`.
2. Set up the scene.
   - **Good:** the state card (right column): **Ready** and `At the start pose ✓ ready to start` (the session's first
     episode: `At the start-up pose ✓ ready to start (the PC checks it at Start)`).
   - **If not:** **Not ready** and why, e.g. `Not at the start pose: h, then p on the PC` (§19).

**Part B — Record**

1. **Start episode**: starts recording.
   - **Good:** the state card shows **Recording episode_NNN** with a timer; `REC 0:00` on the camera.
2. Click the page once (it needs the keyboard). Wait about 1 s: the recording opens on a still scene.
3. Drive: hold keys (§14). Release every key: the arm holds.
   - **Good:** the controls show `Keys live`; held keys light up.

**Part C — End**

1. **Stop episode**: keys off, the arm holds; it records until you choose.
2. Choose:
   - **Save: success** or **Save: fail**: saved, complete on disk.
   - **Discard**, then **Yes, discard** (deleted from disk) or **Back**.
   - **Resume recording**: back to driving.
   - **Good:** the row in **Episodes**, e.g. `episode_012 Success 12.3 s 370 frames`.

**Part D — Next:** back to Part A.

**Something went wrong mid-episode:**

- **Stop arm** or `Esc` (any time): the arm holds, the episode is kept **Unsaved**. The PC's `s` and the watchdog do
  the same.
- Later, in its row: **Save ✓** (success), **Save ✗** (fail) or the trash icon. Training reads saved episodes only.
- Only **Discard** and the trash icon delete an episode.
- A camera or disk problem never stops the arm or the episode, and never blocks Save (noted in its `meta.yaml`).
- `At workspace limit` on the camera, at the table: normal, the arm stops going down.

**Delete one episode** (not while recording): the trash icon in its row, then **Yes, delete** (or **No**).

### 14. The keys

Only while the page has focus and an episode records; the keycaps on the page light up while held.

| Key | Motion |
|---|---|
| `W` / `S` | +x / −x (away from / toward the robot base) |
| `A` / `D` | +y / −y (left / right, seen from behind the robot) |
| `R` / `F` | +z / −z (up / down) |
| `U` / `O` | twist the hand about its own approach axis |
| `I` / `K` | tilt forward / back |
| `J` / `L` | tilt left / right |
| `N` | close the fingers (grasp with force) |
| `M` | open the fingers fully |
| `Shift` (held) | slow mode (a quarter of the speed) |
| `Esc` | **Stop arm** (any time; an open episode is kept unsaved) |

Speeds: 5 cm/s per axis, 10 °/s, held keys add up per axis and opposite keys cancel. The physical user stop is the
emergency stop: nothing in the page replaces it.

### 15. Quit

1. Ctrl-C in the terminal: stops the run (an open episode is kept unsaved; the PC keeps holding the arm).
   - **Good:** `teleop_collect: stopped by Ctrl-C`; the page shows `Disconnected: ...`.
2. The Pi's C button: the stream stops, LED off.
3. End of the day, on the PC: `q`, then lock the brakes in Desk.

### 16. Where the episodes are

`<storage folder>/<task>/episode_NNN/` (the storage folder: §11).

| File | What's in it |
|---|---|
| `task.yaml` | the task's name and one-line description, when it was created, the next episode number, the settings |
| `episode_NNN/meta.yaml` | outcome, times, duration, counts, the PC's full `hello` (with `F_T_TCP`), commits, the rig file (the deployment's `contracts/rigs/rig_pi_v3_640.yaml`), the teleop settings, both clock fits, `warnings` (camera down or stale while recording: code, what, seconds into the episode) |
| `episode_NNN/main.mkv`, `ultrawide.mkv` | every camera JPEG exactly as the Pi sent it, one video frame each (plays in any video player) |
| `episode_NNN/main_frames.csv`, `ultrawide_frames.csv` | one row per video frame: `video_index` (its frame in the `.mkv`), frame number, Pi and laptop times, size |
| `episode_NNN/packets.csv` | one row per Pi packet: times, frame numbers, the pairing verdict |
| `episode_NNN/robot_states.csv` | every arm state (100 per second): times, mode, TCP and target poses, joints, finger width, gate counters |
| `episode_NNN/commands.csv` | every request sent to the PC, with its waypoints, twist and reply |
| `episode_NNN/keys.csv` | every key change from the page |
| `episode_NNN/clock_pi.csv`, `clock_pc.csv` | the Pi and PC clock fits (and the PC's raw samples) |
| `episode_NNN/events.jsonl` | start, warnings and stop, with laptop times |
| `episode_NNN/COMPLETE` | written last, when saved: a folder without it is **Unsaved** (listed with **Save ✓** / **Save ✗** and the trash icon, never deleted automatically, never read by training) |

`--mock` runs write the same layout under `~/CapstoneTeleop_mock/<task>/`. Episode numbers only count up, so gaps are
normal. The logs of every run: `~/teleop_collect_sessions/<date>_<time>/` (`logs/<process>.log`,
`teleop/events.jsonl`; `_mock` at the end with `--mock`).

---

## 17. The tools

### `pi_collect`: watch the Pi stream

**When:** once at set-up (step 6), and whenever the cameras seem wrong, with `teleop_collect run` stopped. Receive
only: it never sends anything to the Pi.

```bash
~/envs/teleop_collect/bin/python -m teleop_collect.frames.pi_collect --station station.yaml --seconds 30
```

**Good:** as step 6. The headers and a few JPEGs are saved in `~/teleop_collect_sessions/<stamp>_pi_collect/`.
`--endpoint tcp://<PI>:5555` instead of `--station` takes the address directly.

### `pc_probe`: check the link to the PC

**When:** at set-up (step 7), when going wired, when runs stop with `PC not answering`, and once to measure the
start pose (below). With `control_deploy run`
on the PC and no `teleop_collect run` going (one C3 client at a time). It never moves the arm.

```bash
~/envs/teleop_collect/bin/python -m teleop_collect.teleop.pc_probe --seconds 30
```

**Good:** as step 7.

**Checking the TCP axes** (the read-only check of `contracts/frames.yaml` before `settled: true`): with the arm at
home (press `h` on the PC first), add `--axes`:

```bash
~/envs/teleop_collect/bin/python -m teleop_collect.teleop.pc_probe --seconds 3 --axes
```

**Good:** `PASS  compare_hello: all equal`, then `TCP x axis -> ...: to the robot's right`,
`TCP y axis -> ...: back (toward the base)`, `TCP z axis -> ...: down`. "Right" means standing behind the robot,
facing the way it faces.

**Measuring the start pose** (once, during the PC's directions step 10, "The start pose"): with the arm hand-guided to
the pose and `control_deploy run` started again (it holds the arm there), add `--start-pose`:

```bash
~/envs/teleop_collect/bin/python -m teleop_collect.teleop.pc_probe --seconds 3 --start-pose
```

**Good:** after the checks, five lines (your numbers differ):

```
q_start_rad: [0.0000, -0.4000, 0.0000, -2.1000, 0.0000, 2.4000, 0.7854]   # paste under franka_control: homing: in the PC's control_deploy/station.yaml
fingertip (TCP) height: z 0.160 m in the robot base frame
fingertip direction (TCP z axis) -> base (+0.74, +0.00, -0.67): tilted 48.0 deg from straight down, pointing forward (away from the base)
gripper width: 80.0 mm
demos start: fingertips tilted ~48 deg from straight down (38-64), ~16 cm above the table (6-27), hand open; object ~16 cm ahead horizontally and ~17 cm below the fingertips
```

Compare the height and the tilt with the `demos start` line, then paste the `q_start_rad: [...]` line into the PC's
`control_deploy/station.yaml`, and on the PC commit and push it on `rnoro/final-deployment` (the PC's directions, step
10, say where and how). It is the one value that ties these demos to the deployment, so it is kept in git.

### Trying the tools without hardware

The stand-ins of `--mock`, each alone, in a second terminal (stop them with Ctrl-C):

```bash
~/envs/teleop_collect/bin/python -m teleop_collect.fake_pc --mock
~/envs/teleop_collect/bin/python -m teleop_collect.teleop.pc_probe --mock --seconds 5
```

```bash
~/envs/teleop_collect/bin/python -m teleop_collect.frames.fake_pi
~/envs/teleop_collect/bin/python -m teleop_collect.frames.pi_collect --endpoint tcp://127.0.0.1:5650 --seconds 5
```

`--mock` connects to `127.0.0.1` and expects the fake PC; `5650` is `station.yaml ports.mock_pi`. With
`--endpoint`, `pi_collect` writes into `~/pi_runs/<stamp>/`.

**Good:** `pc_probe` ends with four `PASS` rows (`H2`, `hello`, `C2 rate`, `compare_hello: all equal`); `pi_collect`
shows `30.0 pkt/s` and both cameras `640x480`. Never run the fake PC while the PC runs `control_deploy run --mock
--local` on this laptop: they use the same ports.

## 18. Tests (optional)

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 ~/envs/teleop_collect/bin/python -m pytest tests -q -p no:cacheprovider
(cd ../contracts && PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 ~/envs/teleop_collect/bin/python -m pytest tests -q -p no:cacheprovider)
```

The first runs this package's tests, the second those of the shared C2/C3 code in `../contracts`.

- **Good:** both end `... passed` with no `failed` (about 3.5 minutes, then a few seconds). Tests that need `~/Downloads/data/incoming/episode_000152`, or the
  libfranka-less `franka_control` already built in `~/Capstone-final-deployment`, are skipped without them.
- **If not:** don't run it during a `teleop_collect run` or with anything on ports 5620/5630 (the integration tests
  use them, and are skipped when they are busy).

## 19. When something fails

`check` names the row. A run that stops prints `teleop_collect: FAILED <process> ...: FATAL <process>: <what
failed>: <what to check>`. Page text appears in the state card (Start refusals, first letter capitalized), the
**Activity log** (messages, as written) or the banner at the top. A message that ended an episode means it was kept
**Unsaved** (**Save ✓** / **Save ✗** or the trash icon in its row); the app stays up. A warn line that says
`the episode goes on` changes nothing: keep going or Stop.

| You see | Do this |
|---|---|
| `laptop address FAIL none of link.yaml laptop.addresses (...) is on this laptop` | step 4: the laptop's address changed or `link.yaml` is wrong (then also the PC's copy and its firewall) |
| `Pi reachable FAIL no TCP connection to <PI>:5555: is the Pi on ASUS_5g and its sender running?` | press the Pi's C button (LED blue); ASUS_5g; `station.yaml pi: address` (step 4) |
| `PC FAIL ...` or `C2 FAIL no C2 state from tcp://...:5620 within 20 s` | start `control_deploy run` on the PC first; same Wi-Fi; `link.yaml pc: address` (step 4) |
| `H2 FAIL C2 still shows holding after 2 s: the PC is driving an episode for another C3 client` | stop the other laptop run (`policy_deploy`), `pc_probe` or the PC's `console` |
| `deploy_contracts FAIL ...` | step 3 again (the install points at another clone) |
| `... port FAIL cannot bind tcp://127.0.0.1:...: is another teleop_collect run, or this process started alone, still running?` | quit the other run; `ss -ltnp \| grep -E '56(20\|30\|5[0-3])\|8765'` shows who holds the port |
| `mock episode FAIL not found (no meta.json): ...` | step 5, "If not" |
| `data_root FAIL ...` or `disk WARN ... (below 2 GB)` | free disk space under `~/CapstoneTeleop` (`~/CapstoneTeleop_mock` with `--mock`), or set `data_root` / `mock_data_root` in `station.yaml` |
| `Wi-Fi WARN not on ASUS_5g` | fine when wired; otherwise join ASUS_5g |
| `PC commit WARN ...` | expected: the two machines run different branches |
| `FAILED frames during start-up: not ready after 30 s` | the Pi stream isn't arriving: `pi_collect` (step 6) |
| `PC not answering: ...: check franka_control on the PC and the link` | is `control_deploy run` still running on the PC? Wi-Fi; then `teleop_collect run` again |
| `franka_control restarted on the PC: restart teleop_collect run` | the PC's run was restarted: `teleop_collect run` again |
| `the PC reports fault: ...` | on the PC: Ctrl-C, release the user stop if it was a reflex, `control_deploy run`; then `teleop_collect run` |
| `the PC's gate rejected waypoint ...: check the workspace box and teleop.motion` | as a fault (above); then check that the PC's `station.yaml` is on its teleop block, with `box_margin_m` 0.05 and `cone_margin_deg` 10 (the `PC margins` row; the PC's directions, step 8, "Teleop settings") |
| `the PC is driving an episode for another C3 client` | stop the other laptop run, `pc_probe` or the PC's `console` |
| `clock mismatch: ...` or a `hello` mismatch | `link.yaml` equal on both machines (step 4); then `pc_probe` (step 7) |
| `hello equality FAIL` / `compare_hello FAIL` naming a schema, a limit or `F_T_TCP` | the PC's `rnoro/final-deployment` has moved past `447910b` (its `contracts/` = `927a5ba`'s, which this branch's `../contracts` copies): on the PC, back to `447910b` and rebuild (the PC's directions, step 2), or re-copy `contracts/` from the PC's commit into this branch |
| state card: `Select or create a task first` | pick one in **Task**, or **New task** (§12) |
| state card: `Put the arm at the start pose: h, then p on the PC` | no start pose known yet: on the PC `h`, wait for `homing finished`, then `p` |
| state card: `Not at the start pose: h, then p on the PC` | on the PC `h`, then `p`. If it stays after both: this run took another pose as its start pose (it was started before `h`, `p`, with `homing.q_start_rad` unset): Ctrl-C, `h` and `p` on the PC, `teleop_collect run` again. If the arm lands a little off each time instead, raise `teleop: {start_pose: {tolerance_rad: ...}}` in `station.yaml` (default 0.01, keep it below the PC's 0.05) |
| state card: `The PC refused the start here (not the start pose): press p on the PC` | the PC refused a Start here, so this pose is not its start pose: `h` if needed, then `p` on the PC; the pose after the PC's move becomes the start pose |
| state card: `Start episode refused: the PC refused start: refused: not at the start pose: press h, then p on the PC` | the episode was not started (nothing recorded): on the PC `h`, then `p`; then Start again |
| Activity log: `start pose not learned: the PC is ...; the pose after its move is the start pose` | `run` started while the PC was moving the arm; once its `h` / `p` ends, that pose is the start pose |
| state card: `Main camera down` / `Ultrawide camera stale` (or `down`) | the Pi: C button, LED blue, ASUS_5g; Start comes back once both are `ok` |
| Activity log: `episode_NNN: main camera down ...; the episode goes on (noted in its meta.yaml)` (or `stale`) | during an episode: the arm and the recording go on. Keep going or Stop; the episode is saved with the warning (frames are missing there). Then the Pi: C button, LED blue, ASUS_5g |
| state card: `Main camera down: check the Pi (after a Pi restart: Ctrl-C and run teleop_collect again)` | the Pi's stream restarted (C button off and on, a Pi reboot): both cameras stay down for this run. Ctrl-C, then `teleop_collect run` again |
| state card: `Waiting for the PC clock map` | wait a few seconds; if it stays, `pc_probe` |
| state card: `The PC is homing` | the PC's `h`, `p` or `g` is moving: wait for it to end |
| state card: `The PC is fault` | as `the PC reports fault` |
| Activity log: ``... the PC stopped the arm (watchdog or the PC's `s`): check the link and the PC`` | the episode was kept **Unsaved** (**Save ✓** / **Save ✗** or the trash icon in its row): someone pressed `s` on the PC, or Wi-Fi stalled over 1 s. `h`, `p` on the PC, Start again |
| Activity log: `episode_NNN: recording could not keep up / disk error: ...; the episode goes on (data may be missing; noted in its meta.yaml)` | keep going or Stop; Save works (the failure is in its meta.yaml `warnings`); free disk space, check the frames log |
| Activity log: `episode_NNN saved: ... (N warning(s) in its meta.yaml)` | a failure found at Stop is listed in its meta.yaml `warnings`; check the episode before training on it |
| Activity log: `episode_NNN ended: operator stop: nothing to check. Kept unsaved: ...` | you pressed **Stop arm** or `Esc`: **Save ✓**, **Save ✗** or the trash icon in its row |
| Activity log: `unsaved episode kept from an earlier run: <task>/episode_NNN ...` | select that task in **Task**; **Save ✓** / **Save ✗** or the trash icon in its row |
| banner: `Teleop not answering` or `Frames not answering` | wait a second; if it stays, the terminal shows a `FAILED` line |
| banner: `Disconnected: the ui is not running (teleop_collect stopped?) · retrying` | the run ended: read the terminal, fix, `teleop_collect run` again |
| `teleop_collect: could not open a browser` / `no display` | open `http://127.0.0.1:8765` by hand |

## The files you edit

| File | What's in it |
|---|---|
| `station.yaml` | this laptop's settings: the Pi's address, the data and log folders, the ports, overrides of `config/*.yaml` |
| `storage.local.yaml` | written for you (the header's **Storage** → **Change**, or `--data-root`; §11): the storage folder in use, not in git |
| `../contracts/link.yaml` | the PC's and the laptop's addresses: equal to the PC's copy |

How it is built: `DESIGN.md` (in this folder), `PLAN.md` and `STRUCTURE.md`.
