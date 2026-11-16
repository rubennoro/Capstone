# Testing Frame Quality - On Local Computer

## Requirements

- The laptop (Python 3.10), internet, about 10 GB free.
- Wi-Fi on the lab router **ASUS-5G** (the PC and the Pi are on it too).

## Steps

### 1 - Get the Repo with the proper git branch
```bash
cd ~
git clone https://github.com/rubennoro/Capstone.git
cd Capstone
git checkout rnoro/final-deployment
cd deployment
```

### 2 - Setup the environment 

Run the command below. In the event that python3.10 fails, you'll need to do some installation, which depending on device, is either:
MACOS-
```bash
brew install python@3.10
```

WSL/Linux-
```bash
uv python install 3.10
```

```bash
unset PYTHONPATH
python3.10 -m venv ~/envs/policy_deploy
~/envs/policy_deploy/bin/pip install --upgrade pip
~/envs/policy_deploy/bin/pip install -e contracts -e "policy_deploy[light,test]"
cd policy_deploy
~/envs/policy_deploy/bin/policy_deploy --help
```

### 2 - Power on the Pi

Connect the Pi to power, and since this is a deployment test on the Franka hand, do your best to get everything setup as it would be on AI Model deployment day. 

The Pi should automatically find the ASUS-5G wifi and connect to it.

### 3 - Find the pi

By name:
```bash
ping -c 2 RaspberryPi5.local
```

The address in brackets on the first line is <PI>. 

You can also find the name on the Pi itself:
```bash
hostname -I
```

### 4 - Start the Pi streaming

To start the Pi's streaming, ensure that it's connected to ASUS-5G wifi, which should be automatic. 

You should press the **C Button** and see the LED light up
**blue** on the Sensor Rig. Once this is working, you can let it go until you've collected the frames with the next step. 

When done, press the **C Button** to stop things. 

For additional verification, you can check that the Port is open:

```bash
timeout 2 bash -c "</dev/tcp/<PI>/5555" && echo "port 5555 open" || echo "nothing on 5555"
```


### 5 - Collecting camera output from Pi to local machine

Fill in the <PI> below with the PI's address you found ahead of time:

```bash
~/envs/policy_deploy/bin/python -m policy_deploy.frames.pi_collect \
    --endpoint tcp://<PI>:5555 \
    --seconds 30 \
    --out ~/pi_runs/test_01
```

The ```--out``` arg lets you specify the location of the 
folder where the images will go. 

By default, this will save one image pair every 5s, so a 30s run
from this command provides about 7 camera image pairs. 

#### Camera Output you get:

For the output folder, in this case ```~/pi_runs/test_01```, you should see the following format:

```
test_01/
├── jpeg/
│   ├── packet_00000137_main.jpg
│   ├── packet_00000137_ultrawide.jpg
│   ├── packet_00000288_main.jpg
│   ├── packet_00000288_ultrawide.jpg
│   └── ...
├── headers.jsonl
└── pair_gap.json
```

The rig has two cameras. Every moment the Pi sends a *pair*: one image from each.

| File ends in | Camera | What it should look like |
|---|---|---|
| `_main.jpg` | IMX519, the normal lens | A regular, undistorted view of what's in front of the gripper |
| `_ultrawide.jpg` | IMX219, the 160° fisheye | A much wider view, bent strongly at the edges |

- Both are 640×480 and saved exactly as the Pi sent them.
- The number in the name is the packet number. The Pi sends 30 packets a second, so `packet_00000137` and
  `packet_00000288` are about 5 s apart.
- Two files with the same number were taken at (almost) the same moment.
- Only one pair every 5 s is saved by default. Add `--save-every 0` to save every pair: 900 pairs, about 45 MB per
  30 s.

**`headers.jsonl`: the timing record of every packet (including ones whose images weren't saved).**

One line per packet received. You normally don't need to read it. The useful parts:

- `problems`: should be `[]`. Anything else means that packet arrived broken.
- `image_wh`: the image size of each camera, should be `[640, 480]`.
- `header.main.frame_no` / `header.ultrawide.frame_no`: each camera's frame counter. Normally it goes up by one per
  packet. If the ultrawide number repeats, `uw_reused` is `true`: the Pi had no newer ultrawide frame and sent the
  last one again.
- `header.*.t_mid_ns`: when each image was taken (middle of the exposure, Pi clock, nanoseconds).

**`pair_gap.json`: how far apart in time the two images of a pair were taken.**

- Values in ms are `main` time minus `ultrawide` time. Negative means the ultrawide image is the newer one.
- On the real Pi expect around −16 ms.
- `max_ms` must stay below 60.
- `longest_same_pair_run` should be 1 or 2. Higher means the ultrawide camera froze and kept sending the same frame.

**The terminal output** prints one line per second while running. A healthy line looks like:

```
 30.0 pkt/s | packet_no gaps 0 | main 640x480 18 kB frame_no gaps 0 | ultrawide 640x480 35 kB frame_no gaps 0 reused 0 | bad 0
```

| You see | Meaning |
|---|---|
| ~30 pkt/s, all gaps 0, bad 0 | Healthy |
| `waiting for the first packet ...` repeating | Nothing is arriving. Check that the LED is blue, you're on ASUS-5G, and `<PI>` is right (step 3) |
| `packet_no gaps` above 0 | Packets lost on Wi-Fi. A few are fine; many mean a weak signal |
| `bad` above 0 | Broken packets. The reason is in `problems` in `headers.jsonl` |

