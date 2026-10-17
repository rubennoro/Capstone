# Hardware bring-up checklist (drivers + recorder, WP-B)

The Argus and ICM-42688-P backends have never run on a Jetson. Everything else (FIFO parsing, timestamp unwrapping,
clock mapping, pipeline strings, recorder, abort rules) is unit-tested on a laptop with mocks. Work through these steps
on the Orin Nano (JetPack 6.x, nvpmodel 15 W, NVMe for the raw root), in this order.

The step-by-step day-one procedure (setup → doctor → test session → stamp-offset → Kalibr → real session →
process-session → cluster) is [docs/first_plugin.md](../../docs/first_plugin.md). This checklist holds the detailed
follow-ups for its steps.

## 0. Setup
- `sensor_rig/scripts/jetson_setup.sh` (run `--dry-run` first to see every command). It installs the apt packages
  (`python3-gi`, GStreamer, `python3-numpy/-opencv/-scipy/-yaml/-spidev`) and creates `~/sensor_rig_venv` with
  `--system-site-packages` plus `pip install --no-deps -e sensor_rig`. It also adds the `video`/`spi` groups and the
  spidev udev rule, and loads spidev at boot. Never pip-install OpenCV into the venv: it would shadow JetPack's build.
- The script prints the manual steps:
  - `sudo /opt/nvidia/jetson-io/jetson-io.py`: enable "Camera IMX219 Dual" and SPI1 on the 40-pin header (pins 19
    MOSI, 21 MISO, 23 SCK, 24 CS0). Reboot. You should see `/dev/video0`, `/dev/video1` and `/dev/spidev0.0`.
  - Power mode (`sudo nvpmodel -q`, then `sudo jetson_clocks`).
  - The clock (`sudo timedatectl set-ntp true`; the Jetson has no RTC battery by default and episode names use the
    wall clock).
  - Log out and in for the group changes.
- `source ~/sensor_rig_venv/bin/activate && sensor-rig doctor --rig rig.yaml --out /mnt/nvme/raw`. It must end with
  `OK` (WARNs allowed). Each FAIL line names its fix. The sections below are the detailed follow-ups when a doctor
  check fails or needs a number written down.
- `sensor-rig bench-encode --width 1640 --height 1232` (and `--width 1280 --height 720`). Write down
  ms/frame and frames/s for 1-3 workers. Target: 2 workers per camera handle 30 fps at 1640x1232 with headroom
  (P1 §3.4 estimate: 15-22 ms/frame on one core). If not: `--encoder turbojpeg`, Q85, or UW at 15 fps.

## 1. Cameras (ArgusCamera)
- List the sensor modes: `gst-launch-1.0 nvarguscamerasrc sensor-id=0 num-buffers=1 ! fakesink` prints them. Check
  that `sensor_mode` / `resolution` / `fps` in rig.yaml match a mode exactly (default: mode 3 = 1640x1232@30).
- Check the launch string by hand: `python -c "from sensor_rig.drivers.argus_logic import *; from sensor_rig.rig_config
  import load_rig; print(build_argus_pipeline(**argus_params_from_camera(load_rig('rig.yaml').cameras['main'])))"`,
  then run it with `gst-launch-1.0 <string with appsink replaced by fakesink>` for both sensor ids at the same time.
- Python check: open `make_camera(rig, "main", backend="argus")`, read 300 frames and confirm that
  (a) `seq` has no gaps and `stats["internal_drops"] == 0`, (b) `t_ns` steps are 33.3 ms +- jitter,
  (c) `t_arrival_ns - t_ns` (`LatencyTracker.summary()`) is positive and stable (expect a few ms to ~40 ms;
  `latency_negative` must be 0). A negative or drifting latency means the pipeline clock is not MONOTONIC, so check
  `use_clock` in `argus_camera.py`.
- Clock-domain check (R4 §1.5/§10): `sensor-rig stamp-offset --rig rig.yaml --led gpio:7` blinks an LED from a GPIO
  at logged `time.monotonic_ns()` times and compares them with the frame `t_ns`. That gives the true capture-to-stamp offset. Put it in
  `cameras.<c>.time.offset_prior_ns` in rig.yaml (sign: `t_capture = t_stamp + offset`, so it is negative when stamps
  are late).
- Start from `configs/rig_jetson_imx219_2p1_1p55.yaml` (the charter lenses; nominal intrinsics, coverage 1.0). It
  locks AE/AWB with a fixed exposure (3 ms) and gain (4.0): tune these two numbers once under the lab lighting
  (brightest exposure <= 4 ms without clipping the tags), then keep them for all sessions.
- Confirm the exposure/gain locks: with `aelock`/`awblock` true and a fixed `exposure_time_range_ns`, brightness must
  not change when you cover half the lens.
- Two cameras + the IMU together: CAM1 has known second-instance issues on JP 6.2.1. Restart the recorder 5 times and
  check that both cameras come up each time.

## 2. IMU (Icm42688Spi)
- WHO_AM_I: `Icm42688Spi.from_rig(rig).start()` must not raise. It reads 0x75 and expects 0x47. If it reads 0x00 or
  0xFF, check wiring, CS, `spi_mode` (0 or 3), and lower `spi_speed_hz` to 1 MHz.
- Register values are from DS-000347 **rev 1.6**. Re-check them against the current TDK revision (1.8). In particular:
  INTF_CONFIG0 = 0x33, TMST_CONFIG = 0x21, FIFO_CONFIG1 = 0x57 (hi-res) / 0x47, PWR_MGMT0 = 0x0F. You can read them
  back over SPI after start.
- Static test, 60 s on a table: `|mean(accel)|` must be 9.81 +- 0.1 m/s^2. This also confirms the 20-bit (hi-res)
  scale: the raw 20-bit word is decoded with 32768 LSB/g and 262.144 LSB/dps (16x the 16-bit +-16 g / +-2000 dps
  sensitivities, i.e. the datasheet's 8192 LSB/g / 131 LSB/dps on the 18-/19-bit values). If it reads 2.45 or 39.2, fix
  `HIRES_*` in `icm42688_proto.py`, or set `fifo.packet: 16byte`. Gyro mean must be < 0.01 rad/s, the sample rate
  must be ODR +- 1 %, and there must be no `fifo_overflows` and no `invalid_samples` after the first few ms.
- Clock fit: `info()["clock_fit"]`. `slope_ppm` is the chip clock error (expect |slope| < 10,000 ppm with the PLL;
  the clamp is +-20,000). `resid_us_p99` shows the host read jitter (expect < 1-2 ms). If the slope changes by more
  than ~100 ppm over minutes, the chip is probably on the RC oscillator.
- Overflow recovery: pause the reader for 200 ms (e.g. `kill -STOP` / `-CONT` the process). `fifo_overflows`
  increments, and the IMU timestamps stay continuous (no 65.5 ms jumps) and strictly increasing.
- Polling in Python at 5 ms is expected to work at 1 kHz (20 B x 5 = 100 B per read). If `rate_measured` drops or
  overflows appear under recording load, raise `poll_period_ms` to 2-3, or move the reader to C++ (same decoder logic).

## 3. Recorder
- The session command first: `sensor-rig collect --rig rig.yaml --out /mnt/nvme/raw --session hwtest` (inside tmux).
  Record the calibration (it must say `CAL KEPT`), then 3 demos of ~20 s, then check one `d` discard and one `a`
  abort. Close the terminal mid-demo once: the summary in `/mnt/nvme/raw/.sensor_rig/sessions/` must say `SIGHUP`
  and no `.*.recording` folder may be left. With a button wired, check short and long presses.
- Long single-kind runs for the numbers: `sensor-rig record --rig rig.yaml --out /mnt/nvme/raw --session hw-test --side left --kind
  gripper_calibration --duration 60`, then the same with a demonstration. Requirements: zero `queue_full`,
  `sensor_gap` and `internal_drops` in `meta.json` / `drops.csv`, `fps_measured` within 1 % of nominal, IMU
  `rate_measured` within 1 % of ODR, and `latency_ms_median` logged.
- Disk: about 20 MB/s at 2x 1640x1232@30 Q90. Check that free space is reported and that the 2 GB abort works (fill a
  test partition).
- `gravity_check` on a static recording (pose stage, WP-A) passes after pose-import with a trivial trajectory, and
  the camera-IMU `time_offset_check` lag (after a wiggle recording + ORB-SLAM3) is below 5 ms once `offset_prior_ns` is
  set.
- Thermal: a 10-minute session at 15 W with no throttling (`tegrastats`) and no rising drops.

## 4. Known upgrade paths
- Camera stamps: a small C++ libargus helper emitting `(frame_no, ICaptureMetadata::getSensorTimestamp())`
  (CLOCK_MONOTONIC_RAW, converted with `clock.RawToMono`) behind the `CameraBackend` interface. Gives SOF stamps
  instead of appsink delivery times.
- IMU: a C++ reader thread with the same register sequence / `ImuFifoDecoder` logic. Optionally wire the camera strobe
  to the ICM-42688 FSYNC pin (`FIFO_TMST_FSYNC_EN`) to tag exposure times in the IMU stream.

## Raspberry Pi 5 (picamera2 backend)

The `picamera2` backend and the Pi wiring have never run on a Pi either. Setup and background:
[docs/raspberry_pi.md](../../docs/raspberry_pi.md). Laptop coverage: `test_picamera2_logic.py` and
`test_picamera2_camera.py`, which use a fake `picamera2` module. Do the steps on the Pi 5 (Bookworm 64-bit, active
cooler, 5 A supply, SSD for the raw root), in this order.

### P0. Setup
- SPI on (`dtparam=spi=on`), both cameras on CAM0/CAM1 with Pi 5 cables, reboot. `rpicam-hello --list-cameras` must
  list both. Write down each mode's size, crop and fps, and check them against `KNOWN_SENSOR_MODES` in
  `picamera2_logic.py`:
  - IMX477: 2028x1520 with crop (0,0)/4056x3040.
  - IMX219: 1640x1232 with crop (0,0)/3280x2464.
  - IMX708: 2304x1296 with crop (0,0)/4608x2592.
  If a crop or fps differs, fix the table.
- `ls /dev/spidev0.0`; `python -c "import picamera2, spidev, cv2, av"` inside the `--system-site-packages` venv.
- `sensor-rig rig-validate rig.yaml`: coverage 1.0 for both cameras, no `picamera2` warnings.
- `sensor-rig bench-encode --width 1014 --height 760` and `--width 1640 --height 1232`. Both must run at
  >= 3x 30 fps on one worker (estimate: ~5 ms and ~12 ms per frame). If not, use `--encoder turbojpeg` or Q85.

### P1. Cameras (Picamera2Camera)
- Python check, per camera and then both at once: `make_camera(rig, "main")`, read 300 frames, then confirm:
  - (a) `seq` has no gaps and `stats["sensor_gaps"] == stats["internal_drops"] == 0`;
  - (b) `t_ns` steps are 33.333 ms +- a few µs (SOF interrupt stamps are much tighter than the Argus appsink stamps);
  - (c) `info()["latency_ms_median"]` (`t_arrival - SensorTimestamp`) is positive and stable, at about one readout
    plus ISP time (expect ~25-45 ms), and `latency_negative == 0`.
  A negative or huge latency means the stamp is not CLOCK_MONOTONIC: try `timestamp_clock: boottime` and report it.
- Check what libcamera actually configured: `info()` must show the expected `sensor_mode`, `sensor_crop` = the full
  array and `scaler_crop` = the full mode. The backend already refuses a changed main-stream size or format.
- Clock-domain / offset check: film an LED driven from a GPIO toggled at known `time.monotonic_ns()` (for example with
  gpiod) and compare with the frame `t_ns`. The expected result is that the LED shows up `readout/2 - exposure/2`
  after the SOF stamp, on average over rows. Put the measured value in `cameras.<c>.time.offset_prior_ns`. It is
  positive for SOF stamps: about +9.5 ms for the IMX477 at 4 ms exposure.
- Locks:
  - With `exposure_us` / `analogue_gain` / `colour_gains` set, covering half the lens must change neither brightness
    nor colour. Frame metadata `ExposureTime` / `AnalogueGain` (`frames_meta.csv`) must stay constant.
  - Focus: for a CM3, `LensPosition` in the metadata must stay at the configured dioptres and never move. For M12
    lenses, focus at the working distance (~0.3 m) with `rpicam-hello`, then glue the thread.
- Colour order: a red object must be red in a recorded JPEG (`cam_main/data/*.jpg`). The backend assumes "RGB888" =
  [B, G, R].
- Two cameras + IMU: restart the recorder 5 times. Both cameras must come up each time, with the `model` check
  passing (`camera_num` order is stable).

### P2. IMU (Icm42688Spi on SPI0)
- Same as section 2 above (WHO_AM_I 0x47, register read-back, 60 s static gravity check 9.81 +- 0.1 m/s^2, clock fit,
  overflow recovery). On the Pi it is on header pins 19/21/23/24, `/dev/spidev0.0`.
- Under recording load, IMU `rate_measured` must stay within 1 % of the ODR. The Python reader shares the GIL with two
  camera pumps and the encoders. If it does not hold, raise `poll_period_ms` to 2-3 or move the reader to C++.

### P3. Recorder, 60 s, two cameras + IMU
- `sensor-rig record --rig rig.yaml --out /mnt/ssd/raw --session hw-test --side left --kind
  gripper_calibration --duration 60`, then a demonstration. Requirements:
  - zero `queue_full`, `sensor_gap` and `internal_drops` in `meta.json` / `drops.csv`;
  - `fps_measured` within 1 % of 30;
  - IMU `rate_measured` within 1 % of the ODR;
  - `host.board` says Raspberry Pi 5.
- Thermal: during a 10-minute session, watch `watch -n 5 'vcgencmd measure_temp; vcgencmd get_throttled'`.
  Temperature must stay < 80 C, `get_throttled` must stay `throttled=0x0` (no under-voltage or throttling), and the
  drops must not grow.
- `timedatectl` reports a synchronised clock (the episode names use wall time; the Pi has no RTC battery by default).
- Copy to the laptop (`rsync --exclude '.*'`), run pose-prep / pose-import / export / validate on one episode, and
  check the virtual views with `calib-check --episode`.
