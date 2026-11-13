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

## 3. Start the Pi

Power the Pi (`admin@RaspberryPi5`) and start its camera sender. The Pi and this machine must be on the same network,
for example the Jetson hotspot `CapstoneGripper`, or a Wi-Fi network both are on.

## 4. Find the Pi

Try these in order and stop at the first that works. What it gives you is `<PI>` below. The IP can change between
days, but the name and the hardware address don't.

- **By name.** If it answers, `<PI>` is `RaspberryPi5.local`.

  ```bash
  ping -c 2 RaspberryPi5.local
  ```

- **By the Pi's Wi-Fi hardware address.** This pings every address on this machine's networks, then prints the Pi's
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

Then check that the sender is up:

```bash
timeout 2 bash -c '</dev/tcp/<PI>/5555' && echo "sender is up"
```

## 5. Collect packets into a folder

```bash
cd deployment/process_frames
~/envs/deploy/bin/python -m process_frames.pi_collect --station ../station.yaml --endpoint tcp://<PI>:5555 \
    --out ~/pi_runs/run_01 --seconds 60 --save-every 0
```

- **`--out`:** the folder to write into. Use a new name for each run.
- **`--seconds`:** how long to collect. Leave it out and press Ctrl-C to stop.
- **`--save-every`:** which packets get their images (main + ultrawide JPEG) saved in `jpeg/`. Every packet is still
  received, checked and logged in `headers.jsonl`, whatever this is set to.
  - **Left out (default 5):** the first packet's images, then one pair every 5 seconds. A 60 s run gives about 13
    pairs (about 1.5 MB).
  - **`0`:** every packet's images, 30 pairs a second. A 60 s run gives about 1,800 pairs (about 220 MB).
  - **Any other number N:** one pair every N seconds.
- **Good:** about `30.0 pkt/s`, `gaps 0`, `bad 0`, both cameras `640x480`. It only receives; nothing is sent to the
  Pi.

In the folder:

```
~/pi_runs/run_01/
├── headers.jsonl        # one line per packet: the Pi's header, arrival time, sizes, any problems
└── jpeg/                # packet_00000800_main.jpg, packet_00000800_ultrawide.jpg, ...
```

To view the images, run `eog ~/pi_runs/run_01/jpeg/`, or open the folder in VS Code.

## 6. Optional: the full pipeline

Set `endpoints.pi_frames: tcp://<PI>:5555` in `deployment/station.yaml` first.

```bash
~/envs/deploy/bin/python -m process_frames --station ../station.yaml                           # terminal 1
~/envs/deploy/bin/python -m process_frames.probe --station ../station.yaml --save /tmp/probe   # terminal 2
```

- **Good:** `clock synced ... (one-way, from t_send_ns)`, then `ready`, then every 10 s a stats line showing both
  cameras `ok 30.0fps`, `lost 0`, `unusable 0`. A small ultrawide `drop` count is normal.
- **Probe:** `/tmp/probe` gets the four 224×224 policy images.
- **Without the Pi:** add `--mock` (needs `~/Downloads/episode_000045`).
- **The whole deployment:** `deployment/DIRECTIONS.md`, starting with `deploy check` and `deploy run`.
- **Contract:** `contracts/PI_PAYLOAD.md` §0. If the Pi's frame size changes, it needs a matching rig file in
  `rigs/` (see `rig_pi_v2_640.yaml`).
