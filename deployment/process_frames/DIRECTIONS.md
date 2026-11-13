# process_frames: how to rerun

Linux, PC or Jetson. Run every command from **inside `deployment/process_frames/`**; from anywhere else the imports break.

## 1. Environment (once per machine)

The code needs **Python 3.10**.

```bash
cd deployment
# Python 3.10 available (e.g. Ubuntu 22.04):
python3.10 -m venv ~/envs/deploy
# No Python 3.10 (Jetson / Ubuntu 24.04): get it with uv, no sudo
python3 -m venv ~/envs/uv-tool && ~/envs/uv-tool/bin/pip install uv
~/envs/uv-tool/bin/uv python install 3.10
~/envs/uv-tool/bin/uv venv --python 3.10 --seed ~/envs/deploy

~/envs/deploy/bin/pip install -e './contracts[test]' -e './deploy[test]' -e './process_frames[test]' \
                              -e './message_comms[test]' -e './policy[test]'
```

## 2. Tests

```bash
cd deployment/process_frames
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 ~/envs/deploy/bin/python -m pytest tests -q
```

`PYTEST_DISABLE_PLUGIN_AUTOLOAD=1` stops ROS's pytest plugins from crashing pytest. The real-episode tests skip
unless `~/Downloads/episode_000045` exists (a symlink is fine: `ln -s /data/incoming/episode_000045 ~/Downloads/`).

## 3. Find the Pi

The Pi has to be on the same network as this machine.

- **Jetson hotspot** (`CapstoneGripper`, 10.42.0.0/24): `iw dev wlP1p1s0 station dump`, then `ip neigh show dev wlP1p1s0`.
- **Any network:** run `hostname -I` on the Pi.
- **Check that the sender is up:** `timeout 2 bash -c '</dev/tcp/<PI_IP>/5555' && echo open`.

Put the address in `deployment/station.yaml` → `endpoints.pi_frames: tcp://<PI_IP>:5555`.

## 4. Collect from the Pi (its current 3-part stream)

Start the Pi sender first, then:

```bash
~/envs/deploy/bin/python -m process_frames.pi_collect --station ../station.yaml --seconds 30
```

- `--endpoint tcp://IP:PORT` overrides `station.yaml`. `--bind --endpoint tcp://0.0.0.0:PORT` is for a Pi that
  connects to this machine.
- It only receives; nothing is sent to the Pi.
- **Good:** about `30.0 pkt/s`, `gaps 0`, `bad 0`, both cameras `640x480`.
- **Output:** `~/deploy_sessions/<time>_pi_collect/`, with every header in `headers.jsonl` and sample JPEGs in `jpeg/`.

## 5. Pipeline check without the Pi (C0, fake Pi)

```bash
~/envs/deploy/bin/python -m process_frames --station ../station.yaml --mock       # terminal 1, wait for "ready"
~/envs/deploy/bin/python -m process_frames.probe --station ../station.yaml --save /tmp/probe   # terminal 2
```

This needs `~/Downloads/episode_000045`. **Good:** stats lines with both cameras `ok 30.0fps`, `drop 0`,
`packets lost 0`.

## Notes

- **`process_frames` without `--mock` won't work with the Pi yet:** it expects the C0 format (`contracts/PI_PAYLOAD.md`)
  and stops with `FATAL` on the Pi's current packets. Use `pi_collect` until the two formats are joined.
- **`rigs/rig_pi_v2_640.yaml`** is the rig file for the 640×480 stream. `station.yaml` still uses `rig_pi_v2.yaml`,
  because `--mock` replays a 1600×1200 recording.
- **On a PC with `franka_control` built,** also run `cd ../deploy && PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 ~/envs/deploy/bin/python -m pytest tests -q`.
