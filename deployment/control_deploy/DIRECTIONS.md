# The PC: run the arm for teleop collection

This PC is cabled to the Franka Panda and runs only the arm side, `control_deploy run` from the branch
`rnoro/final-deployment` at `447910b`, **unchanged**. The laptop drives the arm from its page and records the episodes
(`deployment/teleop_collect/DIRECTIONS.md` on the branch `rnoro/teleop-collect-deploy`). This file is the part of
`rnoro/final-deployment`'s own `deployment/control_deploy/DIRECTIONS.md` that teleop collection needs; that file has
the rest. Copy the commands in order.

Every command below runs from `~/Capstone/deployment` (step 2), in a terminal where `echo $PYTHONPATH` prints nothing
(if it shows `/opt/ros/...`, run `unset PYTHONPATH` first).

> **Daily routine (once set up)**
>
> 1. **Desk**: open `https://172.16.0.2` in a browser on this PC. Unlock the brakes, activate FCI, release the user
>    stop (the external stop button). Keep the stop button within reach the whole time.
> 2. On this PC, with `station.yaml` on its teleop values (step 8, "Teleop settings"):
>    `cd ~/Capstone/deployment && ~/envs/control_deploy/bin/control_deploy run`
> 3. Wait for `control_deploy: ready: ...`. The fingers close and reopen once; the arm holds where it is.
> 4. Press `h`, wait for `homing finished`, then `p`, wait for `start pose reached`. Hand on the user stop.
> 5. **Only now** the Pi (its C button) and then the laptop (`teleop_collect run`). **Always the PC first.** The
>    laptop takes the arm's pose at its start-up as the session's start pose; it never moves the arm to a pose. Leave
>    this PC running across laptop runs.
> 6. **Before every episode:** `h`, then `p` here, then **Start episode** on the laptop.
> 7. End of the day: press `q` here, then lock the brakes in Desk.
>
> | Key on the PC | What it does |
> |---|---|
> | `h` | home the arm: all 7 joints to the home pose (fingertips straight down), slow joint move |
> | `p` | start pose: from home only, a slow joint move to `homing.q_start_rad` (where every episode starts), and the hand opens fully |
> | `g` | home the hand: the fingers close and reopen to calibrate, then go back to their width |
> | `s` | stop: the arm holds where it is (an open laptop episode is deleted) |
> | `q` or Ctrl-C | stop the arm and quit |
>
> `h`, `p` and `g` are refused while the laptop is running an episode. `s` and `q` always work. Once
> `homing.q_start_rad` is set (step 10), the PC refuses the laptop's Start unless the arm is at the start pose. Until
> then `p` is refused (`key p refused: homing.q_start_rad is not set: ...`): press only `h`.

---

## Setting up, from scratch

### 1. What you need

- This PC with **Ubuntu 20.04**, `sudo`, and internet (the first build downloads code).
- Two network connections: an Ethernet cable from the PC straight to the Franka **controller**, and Wi-Fi on the lab
  router **ASUS_5g** (the laptop and the Pi are on it too).
- A browser on this PC for Desk.
- A tape measure (the workspace box), a scale (the rig's mass), calipers (the finger opening) and a ruler (the start
  pose's height).

### 2. Get the code

Downloads the project and switches to the deployment branch. The PC always runs `rnoro/final-deployment`, never the
laptop's teleop branch.

```bash
cd ~
git clone https://github.com/rubennoro/Capstone.git
cd Capstone
git checkout rnoro/final-deployment
cd deployment
```

- **Good:** `git log --oneline -1` shows the newest commit of `rnoro/final-deployment`.
- **If not:** `destination path 'Capstone' already exists`: use the existing clone instead:
  `cd ~/Capstone && git fetch && git checkout rnoro/final-deployment && git pull && cd deployment`.

After every `git pull`, rebuild (step 4).

**The PC runs this branch at commit `447910b`** (`927a5ba`'s start pose key `p` and TCP 235.65 mm out, plus the
default box's `workspace_min_m` x 0.20). The laptop's contract files are a copy of `927a5ba`'s, which `447910b` does
not change. A PC still at `ea9f7fb` or older: `git pull`, then `control_deploy/franka_control/build.sh` (the C++
changed). Your own commit of the start pose (step 10) on top of it changes only `control_deploy/station.yaml`: fine.
If the laptop's `teleop_collect check` or `pc_probe` reports a `hello` mismatch naming a schema, a limit or
`F_T_TCP`, this branch has moved past that commit. Go back to it, then rebuild (step 4):

```bash
git checkout 447910b
```

git keeps your local `link.yaml` and `station.yaml` edits, unless that commit changes those files: then it refuses
and names them. Once the start pose is committed (step 10), going back drops it from `station.yaml`: paste the
`q_start_rad` line back (it is in `git log -p control_deploy/station.yaml`). The other way is to copy the new
`contracts/` into the laptop's branch.

### 3. Python 3.10 and the environment

Ubuntu 20.04 ships Python 3.8; the code needs 3.10. `uv` downloads Python 3.10 for you, without `sudo`.

```bash
unset PYTHONPATH
wget -qO- https://astral.sh/uv/install.sh | sh
~/.local/bin/uv venv --python 3.10 ~/envs/control_deploy
~/.local/bin/uv pip install --python ~/envs/control_deploy/bin/python -e "contracts[test]" -e "control_deploy[test]"
~/envs/control_deploy/bin/control_deploy --help
```

- **Good:** the last command prints `usage: control_deploy [-h] {check,run,console,firewall} ...`.
- **If not:** if the installer put `uv` somewhere else, it prints the folder: use that path instead of
  `~/.local/bin/uv`.

### 4. Build `franka_control`

Builds the C++ program that drives the arm. The first time, use `--test`: it also builds and runs the C++ tests.

```bash
control_deploy/franka_control/build.sh --test
```

`--test` adds the C++ tests; `--dry-run` shows the commands without running them. It installs the build packages with
`sudo apt` (asks for your password), builds libfranka 0.9.2 into `~/franka/libfranka-0.9.2` (skipped with
`already installed` if it is there), then `franka_control` (the first build downloads msgpack).

- **Good:** a line `franka_control commit=<40 letters and digits> dirty=... libfranka=0.9.2`, and at the end
  `100% tests passed, 0 tests failed out of ...`.
- **If not:**
  - `No tests were found!!!`: run them by hand:
    `(cd control_deploy/franka_control/build-test && ctest --output-on-failure)`. Good: `100% tests passed`.
  - Anything else: "When something fails".

From now on, after every `git pull`: `control_deploy/franka_control/build.sh` (without `--test`). Moving or re-cloning
the folder also needs a rebuild: `rm -rf control_deploy/franka_control/build*` first.

### 5. The cable to the arm

Gives the PC's Ethernet port a fixed address next to the arm's (`172.16.0.2`). Done once; already done on the lab PC.

```bash
ping -c 3 172.16.0.2
```

- **Good:** three replies.
- **If not:** set the port up (find its name with `ip link`, e.g. `enp3s0`), then ping again:

  ```bash
  sudo nmcli con add type ethernet ifname <PORT> con-name franka ipv4.method manual ipv4.addresses 172.16.0.1/24
  sudo nmcli con up franka
  ```

### 6. The addresses on ASUS_5g (once)

The laptop must know the PC's address, and the PC's firewall must know the laptop's. Both go in
`contracts/link.yaml`, which must be **identical on both machines**.

1. Connect this PC to the ASUS_5g Wi-Fi. Then read its address:

   ```bash
   hostname -I
   ```

   Pick the four-number address on the Wi-Fi network (not `172.16.0.1`, the arm cable, and not `172.17.0.1`,
   docker). `ip -4 -o addr show scope global` shows which card each address belongs to (Wi-Fi cards start with
   `wl`). Ask the laptop's user for the laptop's address (they get it the same way).
2. Recommended: reserve both addresses on the router, so they never change (the ASUS web page, usually
   `http://192.168.50.1`: LAN → DHCP Server → manually assigned IP). Otherwise redo this step whenever one changes.
3. Edit `contracts/link.yaml`: set `pc: address:` to this PC's address and `laptop: addresses: [...]` to the laptop's.
   **The laptop makes the same edit** in its own copy (`rnoro/teleop-collect-deploy`). Here it is a **local edit:
   never commit it** to `rnoro/final-deployment`.

- **Good:** `~/envs/control_deploy/bin/control_deploy check` shows `link address  PASS`.
- **If not:** the row says which addresses this PC has; fix `link.yaml`.

### 7. OPTIONAL - The firewall (once, and again after changing the laptop's address)

Only the laptop (and the PC itself) may send commands to the arm. `control_deploy firewall` prints the rule from
`link.yaml`; these commands install it and keep it after a reboot.

```bash
sudo apt install -y nftables
sudo mkdir -p /etc/nftables.d
~/envs/control_deploy/bin/control_deploy firewall | sudo tee /etc/nftables.d/control_deploy.nft
sudo nft delete table inet control_deploy 2>/dev/null; sudo nft -f /etc/nftables.d/control_deploy.nft
grep -q control_deploy.nft /etc/nftables.conf || echo 'include "/etc/nftables.d/control_deploy.nft"' | sudo tee -a /etc/nftables.conf
sudo systemctl enable nftables
sudo nft list table inet control_deploy
```

- **Good:** the last command prints the table, with the laptop's address in the `ip saddr` line.
- **If not:** if `ufw` is on (`sudo ufw status` says active), also allow the laptop:
  `sudo ufw allow from <LAPTOP> to any port 5620,5630 proto tcp`.

`control_deploy check` then usually shows `firewall  WARN  cannot read the firewall without root`: that is expected;
the `sudo nft list` above is the real check.

### 8. The arm's values that must be measured (before the first real run)

Put the measured values in `control_deploy/station.yaml`, under `franka_control:` (it overrides
`control_deploy/franka_control/config.yaml` key by key). Edit it on this PC; it is committed once, with the start pose
(step 10), while it is on its deployment block. Switching to the teleop block ("Teleop settings") stays a local edit.

```yaml
franka_control:
  realtime: ignore                 # already there: keep it until the PC has a real-time kernel
  safety:
    workspace_min_m: [0.20, -0.30, -0.005] # the box the fingertips may move in (m, robot base frame)
    workspace_max_m: [0.70, 0.30, 0.60]
  load:
    mass_kg: 0.45                  # everything beyond the Franka Hand: printed fingers, Pi, cameras, mount
    com_m: [0.0, 0.0, 0.05]        # its centre of mass from the flange (m, from CAD)
  gripper:
    width_offset_m: 0.0            # real fingertip opening minus the width the hand reports
  # homing:                       # the start pose, once measured (step 10, "The start pose"): uncomment and paste
  #   q_start_rad: [...]           # the line pc_probe --start-pose prints (7 joint angles, rad)
```

In this order:

1. **The workspace box first.** It is the box of the bring-up (step 10) and of a policy run. For teleop only its floor
   is kept and its sides are opened ("Teleop settings" below). Measure around the table from the robot base (x forward,
   y to the left, z up), a few cm inside the real limits. Leave room toward the base: demos pull back to about x 0.20
   after the grasp (the default `workspace_min_m` x since `447910b`). **The floor (`workspace_min_m` z):** rest the
   fingertips on the table, read `tcp z` on the status line, and use that minus 0.005 (about `-0.005` with the arm
   standing on the table; the default). Grasps end at the table: a higher floor keeps the fingers off objects on the
   table (and a policy trained on these demos would fault the gate there). The gate pulls a move back up to
   `box_margin_m` below the floor (1 cm by default, 5 cm with the teleop settings) and faults beyond that; teleop never
   goes that far as long as the margins match (below).
2. **The payload** (`load.mass_kg`, `load.com_m`): weigh the rig without the Franka Hand; the centre of mass from CAD.
   While it is 0, `check` warns and the arm sags a few mm.
3. **`gripper.width_offset_m`:** in step 10, measure the fingertip gap with calipers and compare it with `grip` on the
   status line.

Keep the default home pose (`homing.q_home_rad`); in step 10, after `h`, the `tcp x y z` on the status line must be
inside the box. The start pose (`homing.q_start_rad`) comes in step 10, "The start pose". Until it is set, `p` is
refused (`key p refused: homing.q_start_rad is not set: ...`) and episodes start at home.

- **Good:** `~/envs/control_deploy/bin/control_deploy check` shows `config  PASS`, `payload  PASS`, and the `todo`
  row no longer lists these keys (it may still list the gains, collision thresholds and cone: those are fine).
- **If not:** `config  FAIL` names the key and the problem.

#### Teleop settings (station.yaml while collecting)

During teleop the operator has full authority: nothing in the software should stop the arm except Franka's own limits
and reflexes, the operator (key release, **Stop arm**, `s`, the user stop) and a dead link (the 1 s watchdog). So for
collecting, the box sides and the tilt cone are opened and the collision thresholds raised. **The floor is kept:** it
protects the printed finger adapter and the table, and grasps end at it.

Keep both blocks in `control_deploy/station.yaml`, under `franka_control:`, one of them commented out. The other keys
(`realtime`, `load`, `gripper`, `homing`) stay as they are, both ways. Station keys override `config.yaml` key by key,
so every key not listed here keeps its default.

```yaml
franka_control:
  # ... realtime, load, gripper, homing: as in step 8 and step 10, unchanged
  # --- deployment (bring-up, console, running a policy): the measured box, the defaults otherwise
  # safety:
  #   workspace_min_m: [0.20, -0.30, -0.005]
  #   workspace_max_m: [0.70, 0.30, 0.60]
  # --- teleop collection: open except the floor
  safety:
    workspace_min_m: [-1.0, -1.0, -0.005]   # z: the floor as measured (step 8); x, y: open
    workspace_max_m: [1.0, 1.0, 1.5]
    cone_max_deg: 179.0                     # the loader needs < 180
    box_margin_m: 0.05                      # = teleop.yaml motion.assumed_pc_box_margin_m
    cone_margin_deg: 10.0                   # = teleop.yaml motion.assumed_pc_cone_margin_deg
  control:
    collision_torque_thresholds_nm: [60.0, 60.0, 54.0, 54.0, 48.0, 42.0, 36.0]   # e.g. 3x the defaults; raise on the arm
    collision_force_thresholds_n: [60.0, 60.0, 60.0, 75.0, 75.0, 75.0]           # if a reflex still fires in a grasp
```

- **The risk:** with the sides open, the arm goes wherever the keys send it within its reach, and with the raised
  thresholds it pushes harder before a reflex stops it. The operator is the guard: slow mode near objects, a hand on
  the user stop.
- **The margins must match the laptop's.** `box_margin_m` and `cone_margin_deg` here must equal the laptop's
  `teleop.yaml` `motion.assumed_pc_box_margin_m` and `motion.assumed_pc_cone_margin_deg` (0.05 and 10). Teleop's
  moves are sized so that a key held into the floor stays within these margins: the gate pulls it back onto the
  floor. With the defaults (0.01 and 2) a press into the floor can make the gate reject a waypoint, and a rejection
  faults the PC (Ctrl-C, `control_deploy run` again, `h`, `p`).
- **Before running a policy, switch back** to the deployment block: the deployment needs the gate. The policy
  reproduces the space used while collecting, so choose the deployment box (and cone) to cover the motions in the
  collected episodes, and the start pose.
- **The bring-up console (step 10) needs the deployment block** too: with the teleop block its `out_of_box` step is
  pulled back and accepted instead of refused.
- **After every change** to `station.yaml`: Ctrl-C `control_deploy run` and start it again (it reads the file only at
  start). `control_deploy check` shows `config  PASS`, or names the key that is wrong.

### 9. A first run without the arm (`--mock`)

Runs everything with a fake arm on this PC only, to check the install. `--mock` is the fake arm; `--local` keeps it
on this PC (`127.0.0.1`).

```bash
~/envs/control_deploy/bin/control_deploy check --mock --local
~/envs/control_deploy/bin/control_deploy run --mock --local
```

Press `h`, wait for `homing finished`, then `g`, wait for `hand homing finished`, then `s`, then `q`. (`p` answers
`key p refused: homing.q_start_rad is not set: ...` until the start pose is measured, step 10.)

- **Good:** `binds 127.0.0.1` in the check table, `control_deploy: ready: C2 tcp://127.0.0.1:5620, ...`, the status
  line starting `idle | tcp 0.450 0.000 0.450 m | grip 0.080 m`. `h` prints `key h: homing the arm` then
  `homing finished`; `g` prints `key g: homing the hand` then `hand homing finished`; `s` prints `key s: stopped`. It
  ends with `control_deploy: franka_control stopped cleanly`.
- **If not:** "When something fails".

### 10. The first real run and the bring-up (before any teleop)

The first time on the real arm, test it alone, without the laptop. **The arm must follow waypoints in the console
before the laptop ever drives it.** Keep a hand on the user stop.

1. `station.yaml` on its **deployment** block (step 8, "Teleop settings"). Desk (daily routine, step 1). Then
   `~/envs/control_deploy/bin/control_deploy check`: no `FAIL` row.
2. `~/envs/control_deploy/bin/control_deploy run`: the fingers home once, then `control_deploy: ready`, mode `idle`.
   Watch it for a minute: the arm doesn't drift, `late` on the status line stays at or near 0.
3. Press `g`: the fingers close, reopen, go back. Measure `width_offset_m` now (step 8); restart `run` after editing.
4. Press `h`: the arm goes to home, mode `idle`. Check the `tcp x y z` is inside your box.
5. Press `h`, then `s` while it moves: expect `stopped`, holding where it stopped. If it shows `fault` instead, write
   down the text, Ctrl-C and start `run` again. Then `h` once more: it must home normally.
6. In a **second terminal**, while `run` shows `idle` or `stopped`:

   ```bash
   ~/envs/control_deploy/bin/control_deploy console
   ```

   It runs six steps in order: `hold`, `move` (1.5 cm up and back), `rotate` (2° and back), `gripper` (2 cm and
   back), `watchdog` (sends nothing for 1 s: the arm must stop and hold), `out_of_box` (one move above the box: it
   must be refused). Before each moving step it says what will move and waits for **Enter** (Ctrl-C sends stop and
   quits). The last step ends in `fault` **on purpose**: Ctrl-C `run` and start it again. (Running the console again
   once `homing.q_start_rad` is set: `h`, then `p` in `run` first, since the PC refuses its start elsewhere.)
7. **The start pose** (once; again if the table or the object's place changes). Every episode starts here, and a
   policy trained on these demos is deployed from the same pose. Aim, from the training demos (measured straight up
   from the table): fingertips tilted about 48° from straight down (38–64°), pointing away from the robot, camera on
   top; fingertips about 16 cm above the table (6–27 cm); the object about 16 cm ahead of the fingertips
   horizontally (8–23 cm), centred left–right, on the table. Within about ±10° and ±5 cm is fine.
   1. Ctrl-C `run`. In Desk, put the arm in guiding mode (FCI off) and move it by hand to that pose, with the object
      in its place. Use a ruler for the height.
   2. Back in Desk: activate FCI. Start `run` again: the arm holds where you left it.
   3. On the laptop (teleop_collect's directions, "The tools", `pc_probe`):
      `~/envs/teleop_collect/bin/python -m teleop_collect.teleop.pc_probe --seconds 3 --start-pose`.
      It prints the tilt, the height and the hand opening (compare them with the aim above), and a
      `q_start_rad: [...]` line. On this PC, paste that line into `control_deploy/station.yaml` under
      `franka_control:` → `homing:` (uncomment the two lines of step 8's example). Ctrl-C `run` and start it again.
      Then commit and push it on `rnoro/final-deployment`, while `station.yaml` is still on its deployment block (as
      in this whole step), so the deployment (`policy_deploy`'s clone) and the simulation get the same value. It is
      the one value that ties the teleop demos to the deployment:

      ```bash
      git commit -m "control_deploy/station.yaml: measured homing.q_start_rad" control_deploy/station.yaml && git push
      ```

      `link.yaml` stays out of it (the command names only `station.yaml`).
   4. Press `h`, then `p` (hand on the user stop): the arm goes to that pose and the hand opens, `start pose reached`.
      `p` again: refused (`key p refused: not at home: press h first`).
8. Only then the laptop: switch `station.yaml` to its teleop block (step 8, "Teleop settings"), then restart `run`.

- **Good:** every item behaves as described; the console prints `console: preflight ok: ...`, one `PASS` line per
  step, and `console: all steps done (.../results.yaml)`. Keep its `results.yaml`. Seeing `stopped: watchdog` in the
  `run` terminal while the console waits for Enter is normal.
- **If not:** `WARN` in the console means it worked but the numbers are not great (tune later); `FAIL` stops the
  console and sends stop. `refused: franka_control not running` means `run` is not started. Otherwise "When something
  fails".

---

## Every session

### 11. `control_deploy run`

**When:** every session, before the Pi and the laptop. Leave it running.

```bash
~/envs/control_deploy/bin/control_deploy run
```

- **Good:** the check table, then `control_deploy: ready: C2 tcp://<PC>:5620, C3 tcp://<PC>:5630; start policy_deploy
  on the laptop. Keys: h home arm, p start pose, g home hand, s stop, q quit` (for teleop the laptop runs
  `teleop_collect run` instead). Then a status line at the bottom:

  ```
  idle | tcp 0.307 0.000 0.354 m | grip 0.080 m | gate a0 s0 p0 r0 | late 0 (+0/10s) | C2 100 Hz | C3 peer 192.168.50.20 | h p g s q
  ```

  (After `h`: the home pose puts the TCP, between the fingertips, at about x 0.307, y 0, z 0.354 m.)

  Mode (`idle`, `holding`, `running`, `stopped`, `homing`, `fault: <reason>`), the fingertip point (x y z, m), the
  finger opening, the moves the safety gate accepted / shortened / pulled back / rejected, late control cycles, the
  state rate (about 100 Hz), and the laptop's address once it is connected. While the laptop drives, the mode is
  `running` during a move and `holding` once it ends; `p` (pulled back) counts up at a box wall. A `1 kHz loop: ...`
  log line every second is normal. Everything is saved in `~/control_deploy_sessions/<date>_<time>_control_deploy/`.
- **If not:** `check` rows that are normal here: `WARN realtime` (no real-time kernel yet), `WARN firewall` (can't read
  it without root), `WARN todo` (values not yet measured), `WARN version ... uncommitted changes` (your local
  `link.yaml` and `station.yaml` edits), `INFO tcp frame`. Anything with `FAIL`: "When something fails".

### 12. The keys during collection

- **Before `teleop_collect run`:** `station.yaml` on its teleop block (step 8, "Teleop settings"), then `h`, then
  `p`. The laptop takes the arm's pose at its start-up as the session's start pose.
- **Before every episode:** `h` (the whole arm slowly to the home pose; wait for `homing finished`), then `p` (to the
  start pose, the hand opens; wait for `start pose reached`). The laptop allows Start only at the start pose, and once
  `homing.q_start_rad` is set this PC refuses a start anywhere else. `h` and `p` work while the PC shows `idle` or
  `stopped`, which is the case between episodes; `p` only from home.
- `g`: homes the hand only; it does not count as a move to a pose.
- `s`: the arm holds where it is; an episode open on the laptop is deleted.
- `q` or Ctrl-C: quit. The laptop's run then stops with `PC not answering` until `run` is started again.

### 13. Stopping, restarting, and after a fault

- **Stop the arm:** `s` (or the laptop's **Stop arm** / `Esc`). The arm holds where it is.
- **Quit:** `q` or Ctrl-C. The laptop then can't run until you start `run` again.
- **After a fault** (the status line shows `fault: ...`, the laptop stops with `the PC reports fault`):
  1. Read the text on the status line.
  2. Press Ctrl-C.
  3. If it was a reflex (a collision, the arm stopped hard, Desk shows an error): press and release the user stop.
  4. `~/envs/control_deploy/bin/control_deploy run` again, then `h`, then `p`. The laptop then runs `teleop_collect
     run` again.

---

## When something fails

`check` and `run` say what failed and what to check. Lines from `franka_control` start with `FATAL franka_control:`.

| You see | Do this |
|---|---|
| `binary FAIL ... build it` | step 4 |
| `version FAIL built from <a>, checkout is <b>` | you pulled without rebuilding: `control_deploy/franka_control/build.sh` |
| `baked paths FAIL the checkout moved since the build` | `rm -rf control_deploy/franka_control/build*`, then `control_deploy/franka_control/build.sh` |
| `env FAIL deploy_contracts comes from ...` | step 3 again (the install points at another clone) |
| `link address FAIL link.yaml pc.address ... is not on this PC` | step 6: the PC's Wi-Fi address changed or `link.yaml` is wrong |
| `ports FAIL ... in use: is control_deploy run already running?` | another `run` is still open: find it with `ss -ltnp \| grep 56`, quit it |
| `firewall FAIL no table inet control_deploy` or `... differs from link.yaml` | step 7 |
| `fci network FAIL no 172.16.0.x address` | the cable; `sudo nmcli con up franka` (step 5) |
| `arm not reachable at 172.16.0.2:1337` | the cable; FCI activated in Desk |
| `franka_control not ready after 90 s` | Desk: brakes unlocked, FCI active, user stop released; then read `logs/franka_control.log` in the session folder |
| `FATAL franka_control: arm reports ...: press and release the user stop` | press and release the user stop, check Desk, `run` again |
| `FATAL franka_control: gripper: ...` | the Franka Hand in Desk (end effector set to Franka Hand) and its cable |
| `FATAL franka_control: cannot bind the C2 endpoint ...` | the PC's Wi-Fi address is not `link.yaml pc.address` (step 6), or another `franka_control` is running |
| `libfranka 0.9.2 does not match the robot's system version` | Desk → Settings → System must be 4.x for libfranka 0.9.2; ask the maintainer before changing `libfranka_version` |
| `fault: safety gate rejected waypoint ...` | a move outside the box (expected after the console's `out_of_box`); Ctrl-C, `run` again; if it happens while the laptop drives, check that `station.yaml` is on its teleop block and its margins equal the laptop's (step 8, "Teleop settings") |
| `key p refused: homing.q_start_rad is not set: ...` | the start pose is not measured yet: step 10, "The start pose" (until then, `h` only) |
| `key p refused: not at home: press h first` | press `h`, wait for `homing finished`, then `p` |
| `fault: ... homing.q_start_rad ...` (outside the box or the cone) | the start pose is outside the safety box or tilt cone: measure it again (step 10), or widen the box (step 8) if it is really safe |
| the laptop: `the PC refused start: refused: not at the start pose: press h, then p on the PC` | exactly that: `h`, then `p` here, then Start on the laptop |
| `stopped` during a laptop episode, `stopped: watchdog` in the log | the laptop went quiet for 1 s (Wi-Fi stall, laptop closed): the arm holds; the laptop deletes that episode and goes on |
| `realtime WARN` and late cycles or reflexes while holding | expected without a real-time (PREEMPT_RT) kernel; install one when possible, then set `realtime: enforce` in `station.yaml` |
| build: `Could not find a configuration file for package "Eigen3" ... version "3.4"` | Ubuntu 20.04's Eigen is 3.3: install 3.4 into `~/franka` without `sudo`, then build with it (below) |
| build: `No tests were found!!!` | step 4, "If not" |

Eigen 3.4 without `sudo`:

```bash
mkdir -p ~/franka/src && cd ~/franka/src
wget https://gitlab.com/libeigen/eigen/-/archive/3.4.0/eigen-3.4.0.tar.gz && tar xf eigen-3.4.0.tar.gz
cmake -S eigen-3.4.0 -B eigen-3.4.0/build -DCMAKE_INSTALL_PREFIX=$HOME/franka/eigen-3.4.0 && cmake --install eigen-3.4.0/build
cd ~/Capstone/deployment && CMAKE_PREFIX_PATH=$HOME/franka/eigen-3.4.0 control_deploy/franka_control/build.sh --test
```

## The files you edit

| File | What's in it |
|---|---|
| `control_deploy/station.yaml` | this PC's settings, the measured arm values (step 8), the teleop and deployment blocks (step 8, "Teleop settings") and the start pose (step 10). Committed once, with the start pose, on its deployment block (step 10); the switch to the teleop block stays a local edit |
| `contracts/link.yaml` | the PC's and the laptop's addresses: identical to the laptop's copy (step 6). A local edit, never committed |
