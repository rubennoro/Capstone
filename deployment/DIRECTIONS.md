# Deployment: start here

A policy runs on the **laptop**, which sees through two cameras on the robot's hand (streamed by a **Raspberry Pi**)
and sends timed moves to the **PC**, which is the only machine that drives the **Franka Panda**. This page says how to
set it up physically, which directions to follow on which machine, and in what order. The details are in the two
machine directions; this page only points to their steps.

| Machine | Its directions | What runs on it |
|---|---|---|
| The PC (Ubuntu 20.04, cabled to the arm) | [control_deploy/DIRECTIONS.md](control_deploy/DIRECTIONS.md) | `control_deploy run`: the arm, its safety checks, its keyboard |
| The laptop (Ubuntu 22.04) | [policy_deploy/DIRECTIONS.md](policy_deploy/DIRECTIONS.md) | `policy_deploy run`: cameras, policy, episodes, its keyboard |
| Any computer (optional) | [policy_deploy/TEST_FRAMES.md](policy_deploy/TEST_FRAMES.md) | saves camera images from the Pi, to check them |

## 1. The physical setup

```
                         Wi-Fi: lab router ASUS_5g
          ┌───────────────────────┬───────────────────────┐
          │                       │                       │
   ┌──────┴──────┐         ┌──────┴──────┐         ┌──────┴───────────────┐
   │   LAPTOP    │         │     PC      │         │  SENSOR RIG (on hand)│
   │ policy,     │         │ arm control │         │  Raspberry Pi 5      │
   │ cameras in  │         │ + Desk page │         │  main + ultrawide cam│
   └─────────────┘         └──────┬──────┘         └──────────────────────┘
                                  │ Ethernet cable (172.16.0.x)
                           ┌──────┴──────┐   ┌──────────────┐
                           │ Franka      │───│ Panda arm +  │──── user stop button
                           │ controller  │   │ Franka Hand  │     (within reach)
                           └─────────────┘   └──────────────┘
```

- **Router:** ASUS_5g powered on. The laptop, the PC and the Pi all join it.
- **Arm:** the Panda with the Franka Hand, the printed fingers and the sensor rig mounted. The controller is on. The
  **user stop button** is plugged in and within reach of whoever runs it. The table around the arm is clear.
- **PC:** an Ethernet cable from the PC **straight to the Franka controller** (not to the router), and the PC on
  ASUS_5g Wi-Fi as well. Both are needed.
- **Laptop:** on ASUS_5g, plugged into power, sleep turned off.
- **Pi:** powered. It joins ASUS_5g by itself. It starts streaming when you press the rig's **C button** (LED
  **blue**); press it again to stop.

## 2. Software setup (once, in this order)

Each line is a step in that machine's directions. Each step has its own **Good** check: don't move on until it passes.

| # | Machine | Steps | What it gives you |
|---|---|---|---|
| 1 | PC | 1–5 | code, Python environment, `franka_control` built, the cable to the arm checked |
| 2 | Laptop | 1–3 | code (the **same commit** as the PC), Python environment |
| 3 | PC + laptop | PC 6, laptop 4 | the addresses: `contracts/link.yaml` **identical on both** (commit it once, `git pull` on the other); the Pi's address in the laptop's `station.yaml` |
| 4 | PC | 7 (optional) | firewall: only the laptop can command the arm |
| 5 | PC | 8 | the workspace box and the rig's weight |
| 6 | PC | 9 | a run with a fake arm (checks the install) |
| 7 | PC | 10 | **the arm alone:** first real run, homing, the console tests, measuring the start pose. Keep a hand on the user stop |
| 8 | Laptop | 5 | the Pi's stream arrives (press the C button first) |
| 9 | Laptop | 6 | the link to the PC (with `control_deploy run` running on the PC) |
| 10 | Laptop | 7 | **the first trial run** (below) |

## 3. A trial run (every time, once set up)

The policy is `hold_still` (the arm stays where the episode started) until you choose a trained one, once, on the
laptop: `policy_deploy use <run folder>` (laptop directions, "A trained policy").

1. **Physical:** everything in section 1 on and connected. User stop within reach.
2. **PC, Desk:** open `https://172.16.0.2` in the PC's browser. Unlock the brakes, activate FCI, release the user stop.
3. **PC:** start the arm side, and wait for `control_deploy: ready`:

   ```bash
   cd ~/Capstone/deployment && unset PYTHONPATH
   ~/envs/control_deploy/bin/control_deploy run
   ```

4. **Pi:** press the C button; the LED turns blue.
5. **Laptop** (only after the PC says `ready`):

   ```bash
   cd ~/Capstone/deployment/policy_deploy && unset PYTHONPATH
   ~/envs/policy_deploy/bin/policy_deploy check
   ~/envs/policy_deploy/bin/policy_deploy run
   ```

6. **Each episode:** on the **PC**, `h` (home), then `p` (start pose: the arm goes to where the training demos
   started and the hand opens); then on the **laptop**, `c` (start), and `y` / `n` (end it as success / failure).
   Repeat from the PC's `h`. No laptop key moves the arm to a pose. The start pose is in the PC's `station.yaml` (a first
   value from the simulation; checked on the arm in PC directions step 10, "The start pose"); `c` is refused until the
   arm is there.

**Always the PC first, then the laptop.** Stop the arm at any time with `s` on either machine, or the user stop.

### Ending

1. Laptop: `q`.
2. Pi: press the C button (LED off).
3. PC: `q`, then lock the brakes in Desk.

If anything fails, the run prints one line saying what and where: look it up in "When something fails" at the end of
that machine's directions.
