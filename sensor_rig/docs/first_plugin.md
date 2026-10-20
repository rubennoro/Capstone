# First plug-in: from a bare Jetson to the cluster pipeline

The procedure for the day the parts arrive. Do the steps in order; each one says what PASS looks like and what to do
when it fails. Everything up to step 10 runs on the Jetson (over SSH, inside `tmux`); steps 11–13 run on the laptop.

Hardware: Jetson Orin Nano (JetPack 6) with NVMe; 2× IMX219 (main: Arducam B0180 + 2.1 mm lens; ultrawide: Arducam B0179, fixed 175° fisheye) on the CSI
connectors (CAM0 = main, CAM1 = ultrawide); ICM-42688-P on SPI1 (pins 19 MOSI, 21 MISO, 23 SCK, 24 CS0, 3.3 V,
GND); for step 7, an LED with a 330 Ω resistor on header pin 7.

The `sensor-rig` commands in the marked blocks below are run by `tests/test_first_plugin.py` (with `--mock`, fakes and
temporary paths), so they stay correct. Variables used throughout:

<!-- test: vars -->
```bash
RIG=~/Capstone/sensor_rig/sensor_rig/configs/rig_jetson_imx219_2p1_b0179.yaml   # nominal template for these lenses
RIG_V1=~/rig_v1.yaml                   # calibrated rig.yaml (step 9)
RAW=/mnt/nvme/raw                      # raw root on the Jetson
LAPTOP_RAW=~/Capstone/data/rig_raw     # copy of the raw root on the laptop (step 11)
EXPORT=~/Capstone/data/rig_export/rig1020
```

## 1. JetPack

```bash
cat /etc/nv_tegra_release          # PASS: "R36 (release), REVISION: 4.x" or later (JetPack 6.x)
df -h /mnt/nvme                    # PASS: the NVMe is mounted, >= 100 GB free
```

FAIL: flash JetPack 6 with SDK Manager; mount the NVMe at `/mnt/nvme` (fstab).

## 2. jetson-io

```bash
sudo /opt/nvidia/jetson-io/jetson-io.py   # CSI: "Camera IMX219 Dual"; 40-pin header: enable spi1; save, reboot
ls /dev/video0 /dev/video1 /dev/spidev0.0
```

PASS: all three device nodes exist. FAIL: re-run jetson-io (the overlay was not saved), check the CSI ribbon
orientation (contacts towards the board on the Orin Nano dev kit), `sudo modprobe spidev`.

## 3. Software setup

```bash
cd ~ && git clone <repo> Capstone && cd Capstone
sensor_rig/scripts/jetson_setup.sh --dry-run      # read what it will do
sensor_rig/scripts/jetson_setup.sh --yes
source ~/sensor_rig_venv/bin/activate              # then log out and in once (groups video, spi)
```

PASS: the import check prints `ok` for numpy, cv2, yaml, scipy, spidev, gi, sensor_rig. FAIL: the `MISSING` line names
the package (`--pip-spidev` if apt has no python3-spidev).

## 4. Doctor

<!-- test: append=--mock -->
```bash
sensor-rig doctor --rig $RIG --out $RAW
```

PASS (exit 0): `PASS` for python deps, gstreamer, nvargus-daemon, `/dev/video0 (main)`, `/dev/video1 (ultrawide)`
(names `vi-output, imx219 ...`), argus mode main/ultrawide (`mode 3 = 1640x1232@30.00`), `spi /dev/spidev0.0`,
`imu WHO_AM_I 0x47`, capture main/ultrawide (`30.0/30 fps`, latency positive, 0 drops), imu rate (`1000/1000 Hz`),
imu gravity (`|a| = 9.81`, rig still), wall clock, disk free/write (≥ 60 MB/s). Expected WARNs: `rig placeholders`
(until step 9). Any FAIL line ends with `->` and the fix; HARDWARE_CHECKLIST §1–§2 has the details.

## 5. JPEG budget

<!-- test: append=--n 4 --workers 1 -->
```bash
sensor-rig bench-encode --width 1640 --height 1232
```

PASS: with 2 workers, ≥ 75 frames/s (2 cameras × 30 fps with 25 % headroom). FAIL: try `--encoder turbojpeg`, then
quality 85 (`capture.jpeg.quality`), then `sudo nvpmodel -m <max>; sudo jetson_clocks`.

## 6. 60 s test session

<!-- test: append=--mock; stdin=q -->
```bash
sensor-rig collect --rig $RIG --out $RAW --session rigtest01
```

Record the calibration (Enter, open/close the fingers 2–3 times, Enter), then one 60 s demo, `s`, `q`.
PASS: `CAL KEPT` with `tags: both visible`, `DEMO KEPT` with `30.0 fps`, `drops 0` for both cameras and
`imu 1000 Hz`; `tegrastats` shows no throttling. FAIL: `CAL REJECTED (no ultrawide frame shows both finger tags)` →
lighting / tag visibility; drops → step 5; aborted `frame_gap` → CSI cable / power.

## 7. Stamp offsets (LED test, under full load)

<!-- test: replace=--led gpio:7=>--mock --duration 10 --mock-speed 8 -->
```bash
sensor-rig stamp-offset --rig $RIG --led gpio:7 --json stamp_offset.json
```

Point both cameras at the LED (it must be in both views). PASS: both cameras report ≥ 200 edges, jitter ≤ 1 ms rms,
|drift| ≤ 2 ms/min, and an offset that is stable across two runs (±1 ms): about −30 ms with the `argus` backend
(appsink delivery stamps), a few ms positive with `argus_sof`. Paste the printed `offset_prior_ns` patch into `$RIG`.
FAIL: `no LED` → pass `--roi main=x,y,w,h` (pixel box around the LED); jitter > 2 ms → build the start-of-frame
helper (`sensor_rig/native/argus_sof/build.sh`, backend `argus_sof`) and repeat.

## 8. Kalibr (laptop, Docker)

Record two collect sessions with the Aprilgrid (cameras: board moved slowly through both FOVs, 90 s; camera–IMU:
all 6 axes excited, 90 s) and convert them to bags as in [calibration.md](calibration.md) §3–4. Then:

```bash
kalibr_calibrate_cameras --target aprilgrid.yaml --models pinhole-radtan pinhole-equi \
    --topics /cam_main/image_raw /cam_ultrawide/image_raw --bag cams.bag --bag-freq 4
kalibr_calibrate_imu_camera --target aprilgrid.yaml --cam camchain.yaml --imu imu.yaml --bag camimu.bag
```

PASS: reprojection < 0.3 px (main) / < 0.5 px (ultrawide); |timeshift_cam_imu| < 10 ms; two runs agree within
0.2°, 3 mm, 1 ms. FAIL: see calibration.md (wrong sensor mode, blur, too little excitation).

## 9. Import + validate

<!-- test: kalibr -->
```bash
sensor-rig calib-import --rig $RIG --camchain camchain-imucam.yaml --imu imu.yaml --out $RIG_V1
sensor-rig rig-validate $RIG_V1
```

PASS: `coverage main = 1.0000`, `coverage ultrawide ≥ 0.99`, no placeholder warnings. Then align the virtual views
(`sensor-rig calib-align`, calibration.md §6) and write the gripper TCP entry (`sensor-rig calib-tcp`, §7; the
`JetsonRig1,1` entry in gripper_calibration.yaml is a copy of the iPhone's until then). FAIL: coverage < 0.99 → lens
or `calib-align`; resolution mismatch → recalibrate in mode 3.

## 10. Real session

<!-- test: append=--mock; stdin=q -->
```bash
sensor-rig collect --rig $RIG_V1 --out $RAW --session rig1020 --task-label block_raise
```

One calibration, then ≥ 45 demos (Enter / Enter per demo, `d` to discard a bad one). PASS: the summary
(`$RAW/.sensor_rig/sessions/rig1020_*.json`) shows the calibration kept and ≥ 45 demos kept, aborted ≤ 10 %.

## 11. Copy to the laptop

```bash
rsync -a --info=progress2 jetson:/mnt/nvme/raw/ ~/Capstone/data/rig_raw/      # on the laptop
```

PASS: same number of episode folders on both sides (`ls $LAPTOP_RAW/*/ | wc -l`).

## 12. Process the session (laptop)

<!-- test: fake-docker -->
```bash
sensor-rig process-session --raw $LAPTOP_RAW --session rig1020 --out $EXPORT --vio orbslam3-docker --rig $RIG_V1 --workers 4
```

Needs the ORB-SLAM3 image (`sensor_rig/docker/orbslam3/README.md`; ≥ 20 GB free to build). PASS: exit 0,
`calibration exported: True`, ≥ 90 % of demos exported, lag |mean| < 2 ms; failures are listed per episode in
`$EXPORT/.sensor_rig/process_rig1020.json` (resume by re-running; `--force-episode NAME` only after looking at it).
FAIL: many `pose_import` refusals (`translation jump`, `gyro ... residual`) → ORB-SLAM3 lost tracking: re-record
with slower motion / more texture, or use map-then-localize (docker README).

## 13. Cluster

The last line of step 12 is the exact command, e.g.:

```bash
scripts/run_pipeline.sh all --task block_raise --dataset rig-rig1020-single --sessions rig1020 \
    --raw-dir data/rig_export/rig1020 --smoke-first --note "rig session rig1020 (sensor_rig)"
```

It needs the sensor_rig work and the measured `JetsonRig1,1` entry committed and pushed (the cluster pulls from
GitHub). PASS: the smoke run finishes and the full run's val action MSE is of the same order as run_001.
