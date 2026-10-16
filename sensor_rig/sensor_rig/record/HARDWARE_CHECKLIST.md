# Hardware bring-up checklist (drivers + recorder, WP-B)

The Argus and ICM-42688-P backends have never run on a Jetson. Everything else (FIFO parsing, timestamp unwrapping,
clock mapping, pipeline strings, recorder, abort rules) is unit-tested on a laptop with mocks. Work through these steps
on the Orin Nano (JetPack 6.x, nvpmodel 15 W, NVMe for the raw root), in this order.

## 0. Setup
- `sudo /opt/nvidia/jetson-io/jetson-io.py`: enable "Camera IMX219 Dual" and SPI1 on the 40-pin header (pins 19 MOSI,
  21 MISO, 23 SCK, 24 CS0). Reboot. You should see `/dev/video0`, `/dev/video1` and `/dev/spidev0.0`.
- `sudo apt install python3-gi gir1.2-gstreamer-1.0`, then `pip install spidev` into the environment (or
  `pip install -e sensor_rig[jetson]`). Add the user to the `video` group and give it access to `/dev/spidev0.0`.
- `python -m sensor_rig bench-encode --width 1640 --height 1232` (and `--width 1280 --height 720`). Write down
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
- Clock-domain check (R4 §1.5/§10): film a phone stopwatch or blink an LED from a GPIO toggled at known
  `time.monotonic_ns()` and compare with the frame `t_ns`. That gives the true capture-to-stamp offset. Put it in
  `cameras.<c>.time.offset_prior_ns` in rig.yaml (sign: `t_capture = t_stamp + offset`, so it is negative when stamps
  are late).
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
- `python -m sensor_rig record --rig rig.yaml --out /mnt/nvme/raw --session hw-test --side left --kind
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
