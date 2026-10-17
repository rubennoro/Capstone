# Recording on a Raspberry Pi 5

The rig can collect episodes on a Raspberry Pi 5 instead of the Jetson. Only the camera backend changes: the
`picamera2` backend (`sensor_rig/drivers/picamera2_camera.py`, logic in `picamera2_logic.py`) sits behind the same
`CameraBackend` interface as Argus. The ICM-42688-P driver (`Icm42688Spi`, userspace spidev) runs unchanged. The raw
episode format, pose stage, export and deployment are the same on both boards, so a Pi episode exports exactly like a
Jetson episode (`tests/test_picamera2_camera.py::test_pi_episode_records_and_exports` checks this end to end with a
fake `picamera2` module).

**Status: not run on a Pi yet.** Work through the Raspberry Pi section of
[HARDWARE_CHECKLIST.md](../sensor_rig/record/HARDWARE_CHECKLIST.md) before the first real recording.

## 1. Cameras: what to buy

The rig reproduces two virtual iPhone 16 Pro cameras, both 4:3: main 68.6 x 54.2 deg and ultrawide 102 x 86 deg. A
remap can rotate and undistort, but it cannot add field of view. Coverage of each virtual camera (the fraction of its
pixels the physical lens actually sees; `rig-validate` prints it):

| physical camera (sensor mode) | role | coverage |
|---|---|---|
| **HQ Camera IMX477, M12, ~4.5 mm lens** (2028x1520 binned, 4:3, ~70 x 55 deg) | main | **1.000**, closest match: almost no unused pixels |
| Camera Module 3 Wide IMX708 (2304x1296 binned, 16:9, 102 x 67 deg) | main | 1.000, but about half the sensor width is unused |
| Camera Module 3 IMX708 (16:9, 66 x 41 deg) | main | 0.743 |
| Camera Module 2 IMX219 (62 x 49 deg) | main | 0.784 |
| **IMX219 + ~160 deg M12 fisheye** (1640x1232 binned, 4:3), equidistant | ultrawide | **1.000** |
| IMX219 + 140 deg fisheye | ultrawide | 0.995 |
| IMX219 + 120 deg fisheye | ultrawide | 0.833 |
| IMX708 (16:9) + 160 deg fisheye | ultrawide | 0.955 |
| IMX708 (16:9) + 120 deg lens | ultrawide | 0.690 |
| Camera Module 3 Wide as ultrawide | ultrawide | 0.67 to 0.77 (depends on distortion) |

Below about 0.98, exported images get black borders that the policy never saw in iPhone data. The default
[`configs/rig_pi5.yaml`](../sensor_rig/configs/rig_pi5.yaml) therefore uses **IMX477 + 4.5 mm** as main and
**IMX219 + 160 deg fisheye** as ultrawide (`rig-validate rig_pi5.yaml`: coverage 1.0000 / 1.0000). The yaml also has a
commented Camera Module 3 Wide block. The 4.5 mm lens leaves only about 0.5 deg of vertical margin on each side
(55.3 vs 54.2 deg). If the calibrated intrinsics or principal point come out less favourable, a 4.0 to 4.2 mm lens
restores the margin, at the cost of about 10 % of the pixels.

The driver is generic libcamera code. It has no per-sensor logic, only the mode tables in
`picamera2_logic.KNOWN_SENSOR_MODES` (IMX708, IMX219, IMX477, IMX296), which are used to validate configs without
hardware. At start, the camera's own `Picamera2.sensor_modes` are what count. Any other libcamera sensor works if its
mode list has a full-FOV mode of the right aspect ratio.

### Full field of view: how the backend guarantees it

- Every sensor has binned modes that read the whole pixel array and cropped high-fps modes. Picamera2 reports each
  mode's `crop_limits`. Examples: IMX708 1536x864 is a 3072x1728 centre crop, IMX477 2028x1080 and 1332x990 are
  crops, IMX219 1920x1080 and 640x480 are crops.
- `choose_sensor_mode` accepts a mode only if all of the following hold. Otherwise `rig-validate` warns and the camera
  refuses to start.
  - Its crop is the whole pixel array.
  - The output has the same aspect ratio as the mode. If it differs, the ISP crops to match.
  - The output is not larger than the mode.
  - The mode reaches the requested fps.
- The mode is forced with `sensor={"output_size", "bit_depth"}` (`picamera2.sensor_size` / `bit_depth` in rig.yaml).
  `ScalerCrop` is then set to the mode's maximum crop.
- The main stream is configured at exactly `cameras.<c>.resolution`, which is also the Kalibr calibration resolution.
  `align_configuration` is never called. After `configure()`, the backend checks the size and format libcamera really
  chose, and fails if libcamera changed them.

### Output sizes and the CPU JPEG budget

The Pi 5 has no hardware JPEG or H.264 encoder: JPEG is in software on every Pi, and on the Pi 5 even H.264 is
software (Picamera2 manual, "Encoders"). The recorder JPEG-encodes every frame with OpenCV (libjpeg-turbo) on
`workers_per_camera` threads. Measured on the laptop (i7-12700H, Q90, one thread); the Cortex-A76 at 2.4 GHz is
assumed to be ~2.5-3x slower:

| stream | laptop ms/frame | KB/frame | est. Pi 5 core load at 30 fps | disk |
|---|---|---|---|---|
| main 1014x760 (2028x1520 / 2) | 1.7 | ~190 | ~0.15 core | ~6 MB/s |
| main 2028x1520 (unscaled) | 6.8 | ~750 | ~0.6 core | ~23 MB/s |
| ultrawide 1640x1232 | 4.5 | ~490 | ~0.4 core | ~15 MB/s |
| (alt) CM3 Wide 1536x864 | 2.8 | ~320 | ~0.25 core | ~10 MB/s |

Default: **main 1014x760**, an exact 1/2 ISP downscale of the full-FOV 2028x1520 mode, so there is no fractional
resampling and both sizes are even. It is essentially the deployment resolution (the policy sees 960x720 renders), it
leaves most of the 4 cores for the 1 kHz IMU reader (Python, shares the GIL) and the pumps, and it keeps the disk rate
at about 21 MB/s for both cameras together. The **ultrawide stays at its native 1640x1232**: VIO uses it and needs the
fisheye periphery. If `bench-encode` on the Pi shows plenty of headroom, `resolution: [2028, 1520]` (with intrinsics
x2) gives Kalibr more pixels. Either way, recalibrate at the size you record. Write to an NVMe (M.2 HAT+) or a USB 3
SSD, not the SD card.

## 2. Setup from a fresh Raspberry Pi OS (Bookworm, 64-bit)

```bash
sudo apt update && sudo apt full-upgrade -y
sudo raspi-config            # Interface Options -> SPI -> enable   (or: dtparam=spi=on in /boot/firmware/config.txt)
# cameras: official modules (HQ, CM3, CM2/IMX219) are auto-detected (camera_auto_detect=1). For boards that are not:
#   camera_auto_detect=0
#   dtoverlay=imx477,cam0
#   dtoverlay=imx219            # without ",cam0" an overlay targets CAM1
sudo reboot
rpicam-hello --list-cameras  # both cameras, with their modes and crops
ls /dev/spidev0.*            # /dev/spidev0.0 (CE0) and /dev/spidev0.1

sudo apt install -y python3-picamera2 --no-install-recommends \
    python3-numpy python3-opencv python3-scipy python3-yaml python3-av python3-spidev
python3 -m venv --system-site-packages ~/rigenv     # system site-packages: picamera2/libcamera come from apt
source ~/rigenv/bin/activate
git clone <repo> ~/Capstone && cd ~/Capstone
pip install --no-deps -e sensor_rig                 # --no-deps: keep apt's numpy/opencv (picamera2 is built against them)
python -c "import picamera2, spidev, cv2, av; print('ok')"
sudo usermod -aG video,spi,gpio $USER               # then log out/in
```

- Use the Pi 5 camera cables (22-pin, 0.5 mm pitch on the board side). CAM0 and CAM1 are the two MIPI connectors.
- Picamera2 must come from apt, not pip: the Raspberry Pi docs say to uninstall a pip copy. If pip pulled in its own
  numpy, Picamera2 or simplejpeg can fail with numpy ABI errors.
- Fit the official active cooler and use the 5 V / 5 A supply: two cameras plus an SSD plus continuous encoding is a
  sustained load.
- Clock: the episode NAME and the iPhone time strings come from CLOCK_REALTIME at record start, and the Pi has no RTC
  battery by default. Check `timedatectl` (`System clock synchronized: yes`) before recording.

### IMU wiring (ICM-42688-P on SPI0, same header pins as the Jetson)

| ICM-42688-P breakout | Pi 5 header pin | signal |
|---|---|---|
| VCC (3.3 V) | 1 | 3V3 |
| GND | 6 | GND |
| SCLK / SCL | 23 | GPIO11 SPI0 SCLK |
| SDI / SDA (MOSI) | 19 | GPIO10 SPI0 MOSI |
| SDO / AD0 (MISO) | 21 | GPIO9 SPI0 MISO |
| CS | 24 | GPIO8 SPI0 CE0 -> `/dev/spidev0.0` |
| INT1 (optional) | any free GPIO | not used by the driver (FIFO polling) |

Keep the wires short (< 10 cm) at 10 MHz. On the Pi 5, SPI is provided by the RP1 I/O controller (DesignWare
`spi_dw` driver). `imu.spi` in rig.yaml is unchanged: `/dev/spidev0.0`, 10 MHz, mode 0. The driver's FIFO reads are
chunked below spidev's default 4096-byte `bufsiz`, so no module parameter is needed.

## 3. rig.yaml: `cameras.<c>.picamera2`

```yaml
picamera2:
  camera_num: 0            # default: sensor_id. Checked against Picamera2.global_camera_info()
  model: imx477            # optional "Model" prefix; if camera_num points elsewhere, the unique match is used
  sensor_size: [2028, 1520]
  bit_depth: 12
  exposure_us: 4000        # with analogue_gain: manual exposure (AeEnable false). Omit both for auto (warned)
  analogue_gain: 4.0
  colour_gains: [2.0, 1.8] # red, blue: AWB off. Omit for auto white balance (warned)
  af: none                 # fixed-focus lens. Autofocus lenses (CM3) need lens_position (dioptres) => AfMode manual
  lens_position: 3.0       # 1 / focus distance in m (3.0 = 33 cm, 0 = infinity)
  buffer_count: 6
  hflip: false             # ISP transform; calibrate with the same flips
  vflip: false
  noise_reduction: fast    # off | fast | high_quality | minimal
  timestamp_clock: monotonic   # or boottime (see section 5)
```

- The frame rate is fixed through `FrameDurationLimits = (1e6 / fps, 1e6 / fps)` µs.
- Values are validated when rig.yaml loads. Unknown keys only warn, as with `argus`.
- `rig.capture.backends.<cam>: picamera2` selects the backend. `make_camera(rig, cam, backend="picamera2")` and
  `RigFrameSource(..., backend="picamera2")` work the same way.
- A camera with an autofocus lens refuses to start without `lens_position`, so the lens never hunts and the
  calibration stays valid. Controls a camera does not have (for example `lens_position` on an IMX219) are refused.

## 4. Record and copy to the laptop

```bash
sensor-rig rig-validate rig_pi5.yaml            # coverage + full-FOV mode check (offline)
sensor-rig bench-encode --width 1014 --height 760   # and 1640 1232: JPEG budget on this Pi
sensor-rig record --rig ~/rig_v1.yaml --out /mnt/ssd/raw --session firstpi --side left \
       --kind gripper_calibration --duration 20
sensor-rig record --rig ~/rig_v1.yaml --out /mnt/ssd/raw --session firstpi --side left   # see --help
# laptop: hidden folders (.<NAME>.recording / .aborted) and the session state file are skipped
rsync -av --exclude '.*' pi@<pi-host>:/mnt/ssd/raw/ ~/Capstone/data/rig_raw/
```

Then follow the normal data path (pose-prep -> VIO -> pose-import -> export -> validate). `meta.json` records
`sensors.<cam>.backend = "picamera2"`, the sensor mode, crop, locks and latency, plus `host.board`.

## 5. Research findings (with sources)

**`SensorTimestamp`: what it means and which clock it uses.** The backend uses `t_ns = SensorTimestamp` (Pi 5).

- *Semantics.* The Picamera2 manual (controls appendix) says it is "the time this frame was produced by the sensor,
  measured in nanoseconds since the system booted. The time is sampled on the camera start of frame interrupt, which
  occurs as the first pixel of the new frame is written out by the sensor". Section 6.4.1 says it is "the time ... that
  the first pixel was read out of the sensor. So we have to further subtract the image exposure time to find out when
  the first pixel started being exposed". Picamera2's own `wait_for_timestamp_` computes
  `SensorTimestamp - 1000 * ExposureTime` for exactly that. libcamera's `control_ids_core.yaml` describes it as "the
  time when the first row of the image sensor active array is exposed". On the Raspberry Pi pipeline it is really the
  start of frame (SOF), i.e. the end of the first row's exposure.
  [Picamera2 manual](https://pip.raspberrypi.com/documents/RP-008156-DS-picamera2-manual.pdf),
  [libcamera control_ids_core.yaml](https://github.com/raspberrypi/libcamera/blob/main/src/libcamera/control_ids_core.yaml)
- *Clock.* libcamera documents CLOCK_BOOTTIME. On the Pi 5, however, the value comes from the hardware path:
  - The PiSP pipeline handler sets `SensorTimestamp = buffer->metadata().timestamp` of the CSI-2 receiver (CFE)
    buffer ([pisp.cpp](https://github.com/raspberrypi/libcamera/blob/main/src/libcamera/pipeline/rpi/pisp/pisp.cpp),
    around `ctrl.set(controls::SensorTimestamp, ...)`).
  - That is the V4L2 buffer timestamp
    ([v4l2_videodevice.cpp](https://github.com/raspberrypi/libcamera/blob/main/src/libcamera/v4l2_videodevice.cpp)).
  - The rp1-cfe kernel driver sets it in `cfe_sof_isr_handler()` with `ktime_get_ns()` and declares
    `V4L2_BUF_FLAG_TIMESTAMP_MONOTONIC`
    ([rp1_cfe/cfe.c](https://github.com/raspberrypi/linux/blob/rpi-6.6.y/drivers/media/platform/raspberrypi/rp1_cfe/cfe.c)).
  - So it is **CLOCK_MONOTONIC, taken at the start-of-frame interrupt**.
  - Picamera2 agrees: `capture_request(flush=True)` compares SensorTimestamp with `time.monotonic_ns()`
    ([picamera2.py](https://github.com/raspberrypi/picamera2/blob/main/picamera2/picamera2.py)).
  - BOOTTIME and MONOTONIC differ only by time spent in suspend, which a recording rig never does.
  - `picamera2.timestamp_clock: boottime` subtracts `clock.boot_minus_mono_ns()`, in case a future kernel switches
    clocks.
  - The checklist verifies the domain on the device: `t_arrival - t_ns` must be small and positive, and an LED test
    gives the absolute value.
- *Consequence for rig.yaml.* SOF stamps are early relative to mid-exposure. For a rolling shutter, the middle row's
  mid-exposure is `t_SOF + readout/2 - exposure/2`, so `time.offset_prior_ns` is **positive** (Jetson appsink stamps
  were late, hence negative there). Readout times from the kernel mode tables:
  - IMX477 2028x1520: 1520 x 12740 px / 840 MHz = 23.1 ms, so +9.5 ms at 4 ms exposure.
  - IMX219 1640x1232: 1232 x 3448 px / 182.4 MHz = 23.3 ms, so +9.65 ms.
  - IMX708 2304x1296: 1296 x 7824 px / 585.6 MHz = 17.3 ms, so +6.66 ms.
  - The IMX296 global shutter has no rolling readout: use about `-exposure/2`.

  Kalibr's `timeshift_cam_imu` refines all of these.
  ([imx477.c](https://github.com/raspberrypi/linux/blob/rpi-6.6.y/drivers/media/i2c/imx477.c),
  [imx219.c](https://github.com/raspberrypi/linux/blob/rpi-6.6.y/drivers/media/i2c/imx219.c),
  [imx708.c](https://github.com/raspberrypi/linux/blob/rpi-6.6.y/drivers/media/i2c/imx708.c),
  [imx296.c](https://github.com/raspberrypi/linux/blob/rpi-6.6.y/drivers/media/i2c/imx296.c))

**Pixel byte order.** The Picamera2 manual, in its "Image formats" table, says "RGB888 - 24 bits per pixel, ordered
[B, G, R]" and "BGR888 - as above, but ordered [R, G, B]". It also says OpenCV users should choose RGB888, not BGR888.
The backend requests `"RGB888"` and uses the array as-is: it is already OpenCV BGR.

**Sensor modes / field of view.**
- Picamera2 `sensor_modes` reports `crop_limits` ("the exact field of view of this mode within the full resolution
  sensor output") and max `fps`.
- A mode is selected with `sensor={"output_size": ..., "bit_depth": ...}`. `ScalerCrop` is in full-sensor coordinates
  (Picamera2 manual 4.2.2.3 and the controls appendix).
- Kernel mode tables (links above):
  - IMX708: 4608x2592 (full), 2304x1296 (binned full FOV, 56.03 fps), 1536x864 (crop 3072x1728 @ (768, 432),
    120.13 fps).
  - IMX477: 4056x3040, 2028x1520 (binned full FOV, 40.01 fps), 2028x1080 (crop), 1332x990 (crop).
  - IMX219: 3280x2464, 1640x1232 (binned full FOV, ~41.9 fps), 1920x1080 and 640x480 (crops).
  - IMX296: 1456x1088, 60.38 fps.
- Camera Module 3 Wide: f = 2.75 mm, 102 x 67 deg; Camera Module 3: 66 x 41 deg
  ([camera hardware specs](https://www.raspberrypi.com/documentation/accessories/camera.html#hardware-specification)).

**Controls for fixed imaging.** From the Picamera2 manual and `control_ids_core.yaml`:
- `FrameDurationLimits` is (min, max) in µs; equal values give a fixed frame rate.
- `ExposureTime` is in µs; `AnalogueGain` is >= 1.0. They are manual when `AeEnable` is false.
- `ColourGains` is (red, blue) and applies only when `AwbEnable` is false.
- `AfMode` Manual means "the lens will never move spontaneously". `LensPosition` is in dioptres (0 = infinity) and is
  read only in manual mode (manual, autofocus section: `picam2.set_controls({"AfMode": controls.AfModeEnum.Manual,
  "LensPosition": 0.0})`).

**Two cameras, buffers, capture loop.**
- `Picamera2(camera_num)` per camera; `Picamera2.global_camera_info()` lists Model / Id / Num. The manual warns that
  numbering can change, hence the `model` check.
- The Pi 5 has two MIPI connectors that can run at the same time
  ([multicam](https://www.raspberrypi.com/documentation/computers/camera_software.html#multiple-cameras)).
  rpicam-apps need one process per camera, but Picamera2 can open both in one process (manual, "Multiple cameras" example).
- Video configurations default to 6 buffers, and more buffers mean fewer drops (manual "More on the buffer_count").
- `post_callback` runs in the camera event loop for every frame, and the manual recommends that it "does not take too
  long". The backend only stamps arrival, copies the array (`make_array` returns a copy) and the metadata, and hands
  them to its reader thread. `queue=False` stops Picamera2 from holding an extra frame back.

**SPI.**
- `dtparam=spi=on` (or raspi-config) enables SPI0 with CE0/CE1, which gives `/dev/spidev0.0` and `/dev/spidev0.1`.
- SPI0 pins: MOSI GPIO10, MISO GPIO9, SCLK GPIO11, CE0 GPIO8.
- On the Pi 5, SPI is provided by RP1 (`spi_dw`) and DDR mode is not supported.
- The Python `spidev` package (pip or apt) works.
- ([SPI docs source](https://github.com/raspberrypi/documentation/blob/master/documentation/asciidoc/computers/raspberry-pi/spi-bus-on-raspberry-pi.adoc))

**Install.** `sudo apt install -y python3-picamera2` (Raspberry Pi docs, "Picamera2"), and don't mix in a pip copy. A
venv needs `--system-site-packages` to see it.
([libcamera_python.adoc](https://github.com/raspberrypi/documentation/blob/master/documentation/asciidoc/computers/camera/libcamera_python.adoc),
[camera overlays](https://github.com/raspberrypi/documentation/blob/master/documentation/asciidoc/computers/camera/rpicam_configuration.adoc))

**Encoders.** "On a Raspberry Pi 5 ... these codecs are implemented in software using FFmpeg libraries ... The JPEG
encoder runs in software on all platforms" (Picamera2 manual, "Encoders"). So JPEG is a CPU budget.

## 6. Not verified without the Pi

- The exact mode lists, `crop_limits` and maximum fps on the installed kernel, and whether the PiSP back end accepts
  1014x760 RGB888 without changing the size (the backend fails loudly if it does).
- The SensorTimestamp clock on the installed kernel (source-verified for rpi-6.6.y rp1-cfe) and the true SOF ->
  mid-exposure offset (LED test, then Kalibr).
- Two cameras + 1 kHz SPI polling + encoding under the GIL: frame drops, IMU `rate_measured`, thermals.
- Whether the fisheye IMX219 board is auto-detected or needs `dtoverlay=imx219`, and which of CAM0/CAM1 maps to
  camera_num 0 (libcamera orders cameras by ID; the `model` check covers a swap between different sensors).
