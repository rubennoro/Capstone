# The PC: run the arm

This PC is cabled to the Franka Panda and runs only the arm side (`franka_control`). The laptop runs the cameras and
the policy and sends the arm its moves (`deployment/policy_deploy/DIRECTIONS.md`). Copy the commands in order.

Every command below runs from `~/Capstone/deployment` (step 2), in a terminal where `echo $PYTHONPATH` prints nothing
(if it shows `/opt/ros/...`, run `unset PYTHONPATH` first).

> **Daily routine (once set up)**
>
> 1. **Desk**: open `https://172.16.0.2` in a browser on this PC. Unlock the brakes, activate FCI, release the user
>    stop (the external stop button). Keep the stop button within reach the whole time.
> 2. On this PC: `cd ~/Capstone/deployment && ~/envs/control_deploy/bin/control_deploy run`
> 3. Wait for `control_deploy: ready: ...`. The fingers close and reopen once; the arm holds where it is.
> 4. **Only now** start the laptop (`policy_deploy run`). **Always the PC first, then the laptop.** Leave this PC
>    running across laptop runs.
> 5. End of the day: press `q` here, then lock the brakes in Desk.
>
> | Key on the PC | What it does |
> |---|---|
> | `h` | home the arm: all 7 joints to the home pose, slow joint move |
> | `g` | home the hand: the fingers close and reopen to calibrate, then go back to their width |
> | `s` | stop: the arm holds where it is |
> | `q` or Ctrl-C | stop the arm and quit |
>
> `h` and `g` are refused while the laptop is running an episode. `s` and `q` always work.

---

## Setting up, from scratch

### 1. What you need

- This PC with **Ubuntu 20.04**, `sudo`, and internet (the first build downloads code).
- Two network connections: an Ethernet cable from the PC straight to the Franka **controller**, and Wi-Fi on the lab
  router **ASUS_5g** (the laptop and the Pi are on it too).
- A browser on this PC for Desk.

### 2. Get the code

Downloads the project and switches to the deployment branch (it must already be pushed to GitHub).

```bash
cd ~
git clone https://github.com/rubennoro/Capstone.git
cd Capstone
git checkout rnoro/final-deployment
cd deployment
```

- **Good:** `git log --oneline -1` shows the same commit as on the laptop.
- **If not:** `destination path 'Capstone' already exists`: use the existing clone instead:
  `cd ~/Capstone && git fetch && git checkout rnoro/final-deployment && git pull && cd deployment`.

Both machines must be on the same commit. After every `git pull`, rebuild (step 4).

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

Builds the C++ program that drives the arm. The first time, use `--test`: it also builds and runs the C++ tests, since
this is the first full build of the arm code on this PC.

```bash
control_deploy/franka_control/build.sh --test
```

What it does (`--dry-run` shows the commands without running them):

1. Installs the build packages with `sudo apt` (asks for your password): `build-essential cmake git pkg-config
   libpoco-dev libeigen3-dev libzmq3-dev libyaml-cpp-dev googletest`.
2. Builds libfranka 0.9.2 into `~/franka/libfranka-0.9.2`. If it is already there (it is on the lab PC, from the old
   build), this step says `already installed` and is skipped.
3. Builds `control_deploy/franka_control/build/franka_control`. The first build downloads msgpack (needs internet).
4. Prints its version, then builds `build-test/` and runs the tests (several minutes).

- **Good:** a line `franka_control commit=<40 letters and digits> dirty=... libfranka=0.9.2`, and at the end
  `100% tests passed, 0 tests failed out of ...`.
- **If not:**
  - It ends with `No tests were found!!!`: no test ran. Run them by hand:
    `(cd control_deploy/franka_control/build-test && ctest --output-on-failure)`. Good: `100% tests passed`.
  - Anything else: see "When something fails" at the end.

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
`contracts/link.yaml`, which must be **the same on both machines**.

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
   Make the **same edit on the laptop** (or commit it once and `git pull` on the other machine).

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

`control_deploy check` then usually shows `firewall  WARN  cannot read the firewall without root`: that is expected
(it can't read it without `sudo`); the `sudo nft list` above is the real check.

### 8. The arm's values that must be measured (before the first real run)

Some settings in `control_deploy/franka_control/config.yaml` are still marked `TODO`. Put the measured values in
`control_deploy/station.yaml`, under `franka_control:` (it overrides `config.yaml` key by key), for example:

```yaml
franka_control:
  realtime: ignore                 # already there: keep it until the PC has a real-time kernel
  safety:
    workspace_min_m: [0.30, -0.30, 0.10]   # the box the fingertips may move in (m, robot base frame)
    workspace_max_m: [0.70, 0.30, 0.60]
  load:
    mass_kg: 0.45                  # everything beyond the Franka Hand: printed fingers, Pi, cameras, mount
    com_m: [0.0, 0.0, 0.05]        # its centre of mass from the flange (m, from CAD)
  gripper:
    width_offset_m: 0.0            # real fingertip opening minus the width the hand reports
```

| Value | How to get it |
|---|---|
| `workspace_min_m`, `workspace_max_m` | Measure around the table from the robot base (x forward, y to the left, z up), a few cm inside the real limits. `min z` must keep the fingertips above the table. |
| the home pose (`homing.q_home_rad`) | Keep the default. In step 10, after `h`, the `tcp x y z` on the status line must be inside the box. |
| `load.mass_kg`, `load.com_m` | Weigh the rig without the Franka Hand. While it is 0, `check` warns and the arm sags a few mm. |
| `gripper.width_offset_m` | In step 10: measure the fingertip gap with calipers and compare it with `grip` on the status line. |

- **Good:** `~/envs/control_deploy/bin/control_deploy check` shows `config  PASS`, `payload  PASS`, and the `todo`
  row no longer lists these keys (it may still list the gains, collision thresholds and cone: those are fine).
- **If not:** `config  FAIL` names the key and the problem.

### 9. A first run without the arm (`--mock`)

Runs everything with a fake arm on this PC only, to check the install. `--local` keeps it on this PC (`127.0.0.1`).

```bash
~/envs/control_deploy/bin/control_deploy check --mock --local
~/envs/control_deploy/bin/control_deploy run --mock --local
```

Press `h`, wait for `homing finished`, then `g`, wait for `hand homing finished`, then `s`, then `q`.

- **Good:** see the `run --mock` section below. It ends with `control_deploy: franka_control stopped cleanly`.
- **If not:** "When something fails".

### 10. The first real run (bring-up)

The first time on the real arm, test it alone with the console before any laptop run. Keep a hand on the user stop.

1. Desk (daily routine, step 1). Then `~/envs/control_deploy/bin/control_deploy check`: no `FAIL` row.
2. `~/envs/control_deploy/bin/control_deploy run`: the fingers home once, then `control_deploy: ready`, mode `idle`.
   Watch it for a minute: the arm doesn't drift, `late` on the status line stays at or near 0.
3. Press `g`: the fingers close, reopen, go back. Measure `width_offset_m` now (step 8); restart `run` after editing.
4. Press `h`: the arm goes to home, mode `idle`. Check the `tcp x y z` is inside your box.
5. Press `h`, then `s` while it moves: expect `stopped`, holding where it stopped. If it shows `fault` instead, write
   down the text, Ctrl-C and start `run` again. Then `h` once more: it must home normally.
6. In a second terminal, `console` (section below), one step at a time. Its last step ends in `fault` on purpose:
   Ctrl-C `run` and start it again.
7. Only then the laptop.

- **Good:** every item behaves as described, and the console ends with `all steps done`. Keep its `results.yaml`.
- **If not:** "When something fails"; a `WARN` in the console is not a stop (tune the values later).

### 11. Stopping, restarting, and after a fault

- **Stop the arm:** `s` (or the laptop's `s`). The arm holds where it is.
- **Quit:** `q` or Ctrl-C. The laptop then can't run until you start `run` again.
- **After a fault** (the status line shows `fault: ...`, the laptop stops with "the PC reports fault"):
  1. Read the text on the status line.
  2. Press Ctrl-C.
  3. If it was a reflex (a collision, the arm stopped hard, Desk shows an error): press and release the user stop.
  4. `~/envs/control_deploy/bin/control_deploy run` again. The laptop just runs `policy_deploy run` again.

---

## The commands

### `control_deploy check`

**When:** before `run`, and whenever something seems wrong. It only reads; it never moves the arm.

```bash
~/envs/control_deploy/bin/control_deploy check
```

**Good:** a table where no row says `FAIL`. Normal on this PC: `WARN realtime` (no real-time kernel yet), `WARN
firewall` (can't read it without root), `WARN todo` (values not yet measured), `WARN version ... uncommitted
changes` (you edited `link.yaml` or `station.yaml`), `INFO tcp frame`. With `--mock` it
skips the arm rows; with `--local` it skips the address and firewall rows. `run` runs the same checks first.

### `control_deploy run`

**When:** every session, before the laptop. Leave it running.

```bash
~/envs/control_deploy/bin/control_deploy run
```

**Keys:** `h` home arm, `g` home hand, `s` stop, `q` quit (table at the top).

**Good:** the check table, then `control_deploy: ready: C2 tcp://<PC>:5620, C3 tcp://<PC>:5630; start policy_deploy
on the laptop. Keys: h home arm, g home hand, s stop, q quit`. Then a status line at the bottom:

```
idle | tcp 0.450 0.000 0.450 m | grip 0.080 m | gate a0 s0 p0 r0 | late 0 (+0/10s) | C2 100 Hz | C3 peer 192.168.50.20 | h g s q
```

Mode (`idle`, `holding`, `running`, `stopped`, `homing`, `fault: <reason>`), the fingertip point (x y z, m), the
finger opening, the moves the safety gate accepted / shortened / pulled back / rejected, late control cycles, the
state rate (about 100 Hz), and the laptop's address once it is connected. A `1 kHz loop: ...` log line every second is
normal. Everything is also saved in `~/control_deploy_sessions/<date>_<time>_control_deploy/logs/`.

When it ends: `control_deploy: franka_control stopped cleanly (session ...)`.

### `control_deploy run --mock`

**When:** to test this PC without the arm (after an install, or a pull). `--local` keeps it on `127.0.0.1`.
**Never** run it on the laptop at the same time as `policy_deploy run --mock`: both use the same ports.

```bash
~/envs/control_deploy/bin/control_deploy run --mock --local
```

**Keys:** the same as `run`.

**Good:** `binds 127.0.0.1` in the check table, `control_deploy: ready: C2 tcp://127.0.0.1:5620, ...`, the status line
starting `idle | tcp 0.450 0.000 0.450 m | grip 0.080 m`. `h` prints `key h: homing the arm` then `homing finished`;
`g` prints `key g: homing the hand` (the `grip` value goes down and back up) then `hand homing finished`; `s` prints
`key s: stopped`; `q` prints `shut down cleanly`.

### `control_deploy console`

**When:** at bring-up (step 10), to test moves on the real arm without the laptop. Run it in a **second terminal**
while `run` shows `idle` or `stopped`. **Never** while the laptop is running.

```bash
~/envs/control_deploy/bin/control_deploy console
```

It runs six steps in order: `hold`, `move` (1.5 cm up and back), `rotate` (2° and back), `gripper` (2 cm and back),
`watchdog` (sends nothing for 1 s: the arm must stop and hold), `out_of_box` (one move above the box: it must be
refused). Before each moving step it says what will move and waits for **Enter** (Ctrl-C sends stop and quits). To run
some steps only: `console --only hold move`.

**Good:** `console: preflight ok: ...`, then one `PASS` line per step, and `console: all steps done (.../results.yaml)`.
After `out_of_box` the arm is in `fault` **on purpose**: Ctrl-C `run` and start it again. Seeing `stopped: watchdog`
in the `run` terminal while the console waits for Enter is normal.

**If not:** `WARN` means it worked but the numbers are not great (tune later); `FAIL` stops the console and sends stop.
`refused: franka_control not running` means `run` is not started.

Without the arm: `console --mock --local` in a second terminal while `run --mock --local` runs (add `--yes` to skip the
Enter prompts; only allowed with `--mock`).

### `control_deploy firewall`

**When:** setting up (step 7), and after changing `laptop: addresses` in `link.yaml`.

```bash
~/envs/control_deploy/bin/control_deploy firewall
```

**Good:** it prints a `table inet control_deploy { ... }` with the laptop's address in the `ip saddr` line. It only
prints; step 7 installs it.

### `franka_control` alone (fallback)

**When:** only if `control_deploy run` itself misbehaves. No checks, no status line, nothing saved but its settings.

```bash
control_deploy/franka_control/build/franka_control --station control_deploy/station.yaml --keyboard
```

**Keys:** the same as `run`. **Good:** `ready: C2 on tcp://<PC>:5620, C3 on tcp://<PC>:5630; keys: h home arm, g home
hand, s stop, q quit`. Add `--mock --local` for the fake arm.

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
| `fault: safety gate rejected waypoint ...` | a move outside the box (expected after the console's `out_of_box`); Ctrl-C, `run` again; if it happens with the laptop, check the box |
| `stopped` during a laptop episode, `stopped: watchdog` in the log | the laptop went quiet for 1 s (Wi-Fi stall, laptop closed): the arm holds; the laptop just reruns |
| `realtime WARN` and late cycles or reflexes while holding | expected without a real-time (PREEMPT_RT) kernel; install one when possible, then set `realtime: enforce` in `station.yaml` |
| build: `Could not find a configuration file for package "Eigen3" ... version "3.4"` | Ubuntu 20.04's Eigen is 3.3. Install 3.4 into `~/franka` without `sudo`, then build again with it (below) |
| build: `No tests were found!!!` | step 4, "If not" |

Eigen 3.4 without `sudo`:

```bash
mkdir -p ~/franka/src && cd ~/franka/src
wget https://gitlab.com/libeigen/eigen/-/archive/3.4.0/eigen-3.4.0.tar.gz && tar xf eigen-3.4.0.tar.gz
cmake -S eigen-3.4.0 -B eigen-3.4.0/build -DCMAKE_INSTALL_PREFIX=$HOME/franka/eigen-3.4.0 && cmake --install eigen-3.4.0/build
cd ~/Capstone/deployment && CMAKE_PREFIX_PATH=$HOME/franka/eigen-3.4.0 control_deploy/franka_control/build.sh --test
```

## Tests (optional)

```bash
(cd control_deploy && PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 ~/envs/control_deploy/bin/python -m pytest tests -q)
```

**Good:** `... passed` and no `failed` (about 2 minutes). The C++ tests run with `build.sh --test` (step 4).

## The files you edit

| File | What's in it |
|---|---|
| `control_deploy/station.yaml` | this PC's settings and the measured arm values (step 8) |
| `contracts/link.yaml` | the PC's and the laptop's addresses: the same on both machines (step 6) |
| `control_deploy/franka_control/config.yaml` | every arm setting with its default (change values through `station.yaml` instead) |

How it is built and why: `deployment/PLAN.md` and `control_deploy/STRUCTURE.md`.
