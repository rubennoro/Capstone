# The laptop: collect teleop demonstrations

The laptop drives the Franka Panda from a page in the browser with the keyboard and records each demonstration: both
Pi cameras and every arm state. The PC runs the arm (`deployment/control_deploy/DIRECTIONS.md`); the laptop never
starts, stops or changes anything on the PC or the Pi. Copy the commands in order.

Every command below runs from `~/Capstone-teleop-collect-deploy/deployment/teleop_collect` (step 3), in a terminal
where `echo $PYTHONPATH` prints nothing (if it shows `/opt/ros/...`, run `unset PYTHONPATH` first).

> **Daily routine (once set up)**
>
> 1. Router on, arm controller on, user stop plugged in and within reach, table clear apart from the scene.
> 2. **The PC first:** Desk unlocked, FCI on, `station.yaml` on its teleop block (the PC's directions, step 8,
>    "Teleop settings"), then `control_deploy run` shows `control_deploy: ready`. Then on the PC: `h`, wait for
>    `homing finished`, then `p`, wait for `start pose reached`.
> 3. **Then the Pi:** press its C button; the LED turns blue.
> 4. **Then the laptop:**
>
>    ```bash
>    cd ~/Capstone-teleop-collect-deploy/deployment/teleop_collect && unset PYTHONPATH
>    ~/envs/teleop_collect/bin/teleop_collect check
>    ~/envs/teleop_collect/bin/teleop_collect run
>    ```
>
>    The arm does not move: the arm's pose now becomes this session's start pose. The page opens at
>    `http://127.0.0.1:8765`.
> 5. For every episode: on the PC `h`, then `p`; in the page wait for `at the start pose ✓ ready to start`, set up the
>    scene, **Start episode**, wait about 1 s, drive, **Stop episode**, then **Save: success**, **Save: fail** or
>    **Discard**.
>    An episode that ended by itself (Stop arm, the PC's `s`, the watchdog, a crash, Ctrl-C) is never deleted: it is
>    listed `unsaved` with **save success**, **save fail** and **delete**.
> 6. Quit: Ctrl-C in the terminal (or close it). Then the Pi's C button (LED off). End of the day, on the PC: `q`, then
>    lock the brakes in Desk.

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
- **If not:** "When something fails".

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

1. **+ new task**, a name such as `mock_test`, **Create**.
2. Wait for `at the start-up pose ✓ ready to start (the PC checks it at Start)`. The fake PC starts at its start
   pose, as if `h` and `p` had been pressed on the PC.
3. **Start episode**; wait about 1 s; hold `W` for 1 s; press `N`, then `M`.
4. **Stop episode**, then **Save: success**. About 3 s later the page shows `at the start pose ✓ ready to start`: the
   fake PC stands in for you at the PC and does `h`, then `p` by itself, 1 s after every episode's stop.
5. Ctrl-C in the terminal. The episode stays in `~/CapstoneTeleop_mock/mock_test/`.

- **Good:**
  - `check --mock` ends with `teleop_collect check: ready (0 warning(s))`.
  - `run --mock` prints `teleop_collect: teleop: the start pose is the arm's pose at start-up: the fake PC starts at
    its start pose`, `teleop_collect: all 4 processes ready (session: ...)`, then
    `teleop_collect: ready: open http://127.0.0.1:8765`, and the page opens.
  - The page shows `mock PC` and `connected` at the top, both previews moving, and `cams` `ok 30 / ok 30 fps`. Its
    messages show `start pose: the arm's pose now (0.450, 0.000, 0.300 m); every episode starts here (h, then p on the
    PC)`. While you hold `W`, `tcp` x grows by about 0.050 per second.
  - After the save, the episode list shows `episode_000 success ...`. Ctrl-C prints `teleop_collect: stopped by
    Ctrl-C`.
- **If not:** `mock episode FAIL not found (no meta.json)`: copy the recorded episode (above), or point
  `frames: {mock_episode: <folder>}` in `station.yaml` at one. Anything else: "When something fails".

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
- **If not:** the `FAIL` row says what to fix; see "When something fails".

---

## Every session

### 9. The session routine

**Always the PC first, then the Pi, then the laptop.**

1. **Physical:** router on, arm controller on, user stop plugged in and within reach, table clear apart from the
   scene.
2. **PC:** Desk (`https://172.16.0.2`): unlock the brakes, activate FCI, release the user stop. Check that
   `control_deploy/station.yaml` is on its teleop block (the PC's directions, step 8, "Teleop settings"). Then
   `control_deploy run` and wait for `control_deploy: ready`. The fingers home once and the arm holds. Leave it
   running all session. (It says `start policy_deploy on the laptop`: here that means `teleop_collect run`.)
3. **PC, the start pose:** press `h` (all 7 joints slowly to the home pose; wait for `homing finished`), then `p` (a
   slow move to the PC's start pose `homing.q_start_rad`, the hand opens; wait for `start pose reached`). Hand on the
   user stop. **Do this before `teleop_collect run`:** teleop takes the arm's pose at its start-up as this session's
   start pose.
4. **Pi:** press the C button; the LED turns blue.
5. **Laptop:**

   ```bash
   ~/envs/teleop_collect/bin/teleop_collect check
   ~/envs/teleop_collect/bin/teleop_collect run
   ```

   `run` runs the same checks, then the start-up: C2 from the PC, the `hello` checks, the clock map, both cameras,
   then it takes the arm's pose as the start pose. **The arm does not move.**

- **Good:** `teleop_collect: teleop: the start pose is the arm's pose at start-up: press h, then p on the PC first`,
  then `teleop_collect: all 4 processes ready (session: ...)` and `teleop_collect: ready: open http://127.0.0.1:8765`.
  The browser opens the page (if not, open that address). Its messages show `start pose: the arm's pose now (x, y, z
  m); every episode starts here (h, then p on the PC)` and `teleop assumes the PC's safety.box_margin_m 0.05 and
  cone_margin_deg 10 (...)`.
- **If not:** `teleop_collect: not started: N check(s) failed (see above)` or `teleop_collect: FAILED <process> ...`:
  "When something fails". The message `start pose not learned: the PC is homing; ...`: `run` started while the PC
  was still moving; the pose after that move is the start pose (nothing to do).

**The start pose** is the PC's `homing.q_start_rad`, measured once on the PC (the PC's directions, step 10, "The
start pose", with `pc_probe --start-pose` here, "The tools"). The first **Start episode** of a session confirms it
with the PC; from then on it is fixed for the session. Once `homing.q_start_rad` is set, the PC itself refuses a
start anywhere else.

**Not measured yet** (`p` answers `key p refused: homing.q_start_rad is not set: ...`): press only `h`. The start pose
is then home, and the PC checks nothing at Start.

**Collect and deploy with the same start pose.** A policy starts where its demos started. Demos recorded with
`homing.q_start_rad` unset start at home: deploy that policy with `homing.q_start_rad` unset too, or record them again
after setting it. Changing `homing.q_start_rad` later means recording again. The value is committed on
`rnoro/final-deployment` (the PC's directions, step 10), so the deployment uses the same one.

### 10. Collecting

All in the page. **Start episode** stays disabled until the arm is at the start pose and idle, the link and both
cameras are healthy; the line under the buttons says why.

1. **Task:** pick one in `task`, or **+ new task**: a name (lowercase letters, digits, `_`) and the task in one line,
   then **Create**. The list below shows that task's episodes.
2. **The start pose before every episode:** on the PC, `h`, wait for `homing finished`, then `p`. Wait for the page's
   `at the start pose ✓ ready to start`, then set up the scene. (The session's first episode shows `at the start-up
   pose ✓ ready to start (the PC checks it at Start)` instead.)
3. **Start episode:** the panel shows `REC episode_NNN 0:00`. **Wait about a second before the first key**, so the
   recording opens on the still scene.
4. **Drive:** hold keys to move, release to stop (step 11). `N` closes the fingers (a grasp with force), `M` opens
   them. Hands off the keys: the arm holds at once.
5. **Stop episode**, then **Save: success**, **Save: fail** or **Discard** (**Keep recording** goes back). Motion
   stops and the arm holds where it is. A saved episode is complete on disk at once; a discarded one is deleted.
6. Repeat from 2: `h`, `p` on the PC, scene, Start.
7. **Delete** an episode: `delete` next to it in the list, then confirm. A delete is refused while an episode
   records: stop it first.
8. **Stop arm** (button, or `Esc`) at any time: the arm holds (a move of the PC's `h` or `p` stops too); an open
   episode ends and is kept **unsaved** (below).
9. **Quit:** Ctrl-C in the terminal (closing the terminal does the same). An open episode is kept unsaved, the page
   shows `disconnected`, and the PC keeps holding the arm. Then the Pi's C button (LED off). End of the day, on the
   PC: `q`, then lock the brakes in Desk.
10. **Unsaved episodes:** nothing ever deletes an episode except your **Discard** or **delete**. One that ended
    without your choice (Stop arm, the PC's `s`, the watchdog, a PC fault, Ctrl-C, a crash) is listed `unsaved`:
    click **save success** or **save fail** next to it (or **delete**). One left by an earlier run is named in the
    messages at start-up: select its task to see it. Training reads only saved ones. A row with `0.0 s` has no
    meta.yaml (a hard crash): it can only be deleted.

- **Good:** after a save the list shows `episode_NNN success 12.3 s 370 frames`, and the messages
  `episode_NNN saved: success, ...`. At the table (the PC's box floor) the arm stops going down and the panel shows
  `at workspace limit`: normal. Nothing else in the software stops the arm while you drive: only Franka's own limits
  and reflexes, you (keys released, **Stop arm**, the PC's `s`, the user stop) and a dead link (1 s).
- **A data problem never stops the arm or the episode, and never blocks Save.** A camera down or stale shows one warn line, e.g.
  `episode_004: main camera down: check the Pi and ASUS_5g; the episode goes on (noted in its meta.yaml)`. Keep going
  or **Stop episode**; it is saved with the warning in its `meta.yaml` (training may leave it out: frames are
  missing). A failed recording shows `...; the episode goes on (data may be missing; noted in its meta.yaml)`: Save
  still works; check the episode before training on it.
- **If not:** an episode that ended by itself (the PC stopped the arm: its `s` or the watchdog) is kept unsaved and
  the messages say why in one line. Save or delete it in the list, then `h`, `p` on the PC and Start again. A line starting `FATAL` or a page
  showing `disconnected`: the run ended; read the terminal ("When something fails").

### 11. The keys

Only while the page has focus and an episode records. Clicking outside the page releases every key.

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

### 12. Where the episodes are

`~/CapstoneTeleop/<task>/` (`station.yaml data_root`):

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
| `episode_NNN/COMPLETE` | written last, when saved: a folder without it is unsaved (listed with save / delete buttons, never deleted automatically, never read by training) |

`--mock` runs write the same layout under `~/CapstoneTeleop_mock/<task>/` (`station.yaml mock_data_root`) instead.
Episode numbers only count up: a deleted or discarded number is never reused, so gaps are normal. The logs of every
run are in `~/teleop_collect_sessions/<date>_<time>/` (`logs/<process>.log`, `teleop/events.jsonl`); with `--mock`
the folder name ends in `_mock`.

---

## 13. The tools

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

## 14. Tests (optional)

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 ~/envs/teleop_collect/bin/python -m pytest tests -q -p no:cacheprovider
(cd ../contracts && PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 ~/envs/teleop_collect/bin/python -m pytest tests -q -p no:cacheprovider)
```

The first runs this package's tests, the second those of the shared C2/C3 code in `../contracts`.

- **Good:** both end `... passed` with no `failed` (about 3.5 minutes, then a few seconds). Tests that need `~/Downloads/data/incoming/episode_000152`, or the
  libfranka-less `franka_control` already built in `~/Capstone-final-deployment`, are skipped without them.
- **If not:** don't run it during a `teleop_collect run` or with anything on ports 5620/5630 (the integration tests
  use them, and are skipped when they are busy).

## 15. When something fails

`check` names the row. A run that stops prints `teleop_collect: FAILED <process> ...: FATAL <process>: <what
failed>: <what to check>`. A message in the page that ended an episode means the episode was kept unsaved (save or
delete it in the list); the app stays up. A warn line that says `the episode goes on` changes nothing: keep going or Stop.

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
| page: `select or create a task first` | pick a task, or **+ new task** |
| page: `put the arm at the start pose: h, then p on the PC` | no start pose known yet: on the PC `h`, wait for `homing finished`, then `p` |
| page: `not at the start pose: h, then p on the PC` | on the PC `h`, then `p`. If it stays after both: this run took another pose as its start pose (it was started before `h`, `p`, with `homing.q_start_rad` unset): Ctrl-C, `h` and `p` on the PC, `teleop_collect run` again. If the arm lands a little off each time instead, raise `teleop: {start_pose: {tolerance_rad: ...}}` in `station.yaml` (default 0.01, keep it below the PC's 0.05) |
| page: `the PC refused the start here (not the start pose): press p on the PC` | the PC refused a Start here, so this pose is not its start pose: `h` if needed, then `p` on the PC; the pose after the PC's move becomes the start pose |
| page: `the PC refused start: refused: not at the start pose: press h, then p on the PC` | the episode was not started (nothing recorded): on the PC `h`, then `p`; then Start again |
| page: `start pose not learned: the PC is ...; the pose after its move is the start pose` | `run` started while the PC was moving the arm; once its `h` / `p` ends, that pose is the start pose |
| page: `main camera down` / `ultrawide camera stale` (or `down`) as the Start refusal | the Pi: C button, LED blue, ASUS_5g; Start comes back once both are `ok` |
| page: `episode_NNN: main camera down ...; the episode goes on (noted in its meta.yaml)` (or `stale`) | during an episode: the arm and the recording go on. Keep going or Stop; the episode is saved with the warning (frames are missing there). Then the Pi: C button, LED blue, ASUS_5g |
| page: `main camera down ... (after a Pi restart: Ctrl-C and run teleop_collect again)` | the Pi's stream restarted (C button off and on, a Pi reboot): both cameras stay down for this run. Ctrl-C, then `teleop_collect run` again |
| page: `waiting for the PC clock map` | wait a few seconds; if it stays, `pc_probe` |
| page: `the PC is homing` | the PC's `h`, `p` or `g` is moving: wait for it to end |
| page: `the PC is fault` | as `the PC reports fault` |
| page: ``... the PC stopped the arm (watchdog or the PC's `s`): check the link and the PC`` | the episode was kept unsaved (save or delete it in the list): someone pressed `s` on the PC, or Wi-Fi stalled over 1 s. `h`, `p` on the PC, Start again |
| page: `episode_NNN: recording could not keep up / disk error: ...; the episode goes on (data may be missing; noted in its meta.yaml)` | keep going or Stop; Save works (the failure is in its meta.yaml `warnings`); free disk space, check the frames log |
| page: `episode_NNN saved: ... (N warning(s) in its meta.yaml)` | a failure found at Stop is listed in its meta.yaml `warnings`; check the episode before training on it |
| page: `episode_NNN ended: operator stop: nothing to check. Kept unsaved: ...` | you pressed **Stop arm** or `Esc`: **save success**, **save fail** or **delete** it in the list |
| page: `unsaved episode kept from an earlier run: <task>/episode_NNN ...` | select that task; save or delete it in the list |
| page: `teleop not answering` or `frames not answering` | wait a second; if it stays, the terminal shows a `FAILED` line |
| page: `disconnected: the ui is not running (teleop_collect stopped?)` | the run ended: read the terminal, fix, `teleop_collect run` again |
| `teleop_collect: could not open a browser` / `no display` | open `http://127.0.0.1:8765` by hand |

## The files you edit

| File | What's in it |
|---|---|
| `station.yaml` | this laptop's settings: the Pi's address, the data and log folders, the ports, overrides of `config/*.yaml` |
| `../contracts/link.yaml` | the PC's and the laptop's addresses: equal to the PC's copy |

How it is built: `DESIGN.md` (in this folder), `PLAN.md` and `STRUCTURE.md`.
