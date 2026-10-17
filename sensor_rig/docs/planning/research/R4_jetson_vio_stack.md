# R4: Replacing the iPhone on the iPhUMI gripper with Jetson Orin Nano + cameras + IMU + VIO

Researched 2026-10-15. Target: Jetson Orin Nano Super dev kit, JetPack 6.x (L4T 36.x), Ubuntu 22.04.
Confidence labels: **[V]** confirmed from a vendor doc or README, **[F]** from forum or community reports, **[I]** my own inference or
from memory, not re-checked.

---

## 0. TL;DR recommendation

**The key design decision is that data collection does not need online VIO.** The policy only needs the gripper pose
trajectory *after* recording. UMI itself records GoPro fisheye video plus the GoPro IMU, then runs **ORB-SLAM3
monocular-inertial offline** (build a map, then localize each demo against it). Do the same: on the Jetson, log
timestamped raw frames and IMU samples, and run VIO offline on the laptop or cluster. This removes the Orin Nano's CPU
budget, thermal throttling and real-time deadlines from the critical path. At Franka deployment the IMU and VIO aren't
used at all: pose comes from FK, and the Jetson just serves images.

| Item | Recommendation |
|---|---|
| Cameras | 2x **Sony IMX219** CSI modules. The IMX219 is the only sensor with an in-tree, jetson-io-selectable driver on Orin Nano JP6 (supports "Camera IMX219 Dual"). Main: ~77-80° lens. Ultrawide: 120°-160° M12 or fixed lens. |
| VIO camera | The **ultrawide** camera, running in the 1640x1232@30 or 1280x720@60 mode with fixed short exposure. Wider FOV gives more features and more robustness, which is why UMI uses a fisheye. |
| IMU | **TDK ICM-42688-P** on SPI (40-pin header SPI, 1 kHz ODR, FIFO with on-chip timestamps). Fallback: Bosch **BMI088** (NVIDIA ships an IIO driver, but it's I2C and needs AON GPIOs for hardware timestamps). |
| IMU driver | Easiest robust path: **userspace spidev reader in C/C++** that drains the FIFO and maps chip timestamps to CLOCK_MONOTONIC. Kernel IIO `inv_icm42600` + overlay is the "proper" path (RidgeRun recipe) but needs a kernel module build. |
| Capture | Small **C++ libargus** program (or GStreamer `nvarguscamerasrc` + appsink) that records each frame's Argus sensor start-of-frame timestamp. Write to NVMe. |
| VIO/SLAM | **ORB-SLAM3 mono-inertial (KannalaBrandt8 fisheye model)**, offline, as in UMI. Second choice: **OpenVINS** (mono+IMU, online time-offset and extrinsic calibration, ROS-free build possible). **Not cuVSLAM**, which has no mono+IMU mode. |
| Calibration | Kalibr (Docker): `kalibr_calibrate_cameras` (models `pinhole-equi` or `ds-none` for the ultrawide), `kalibr_calibrate_imu_camera` (extrinsics + time offset), with **allan_variance_ros** noise parameters. |
| Logging format | Plain files (no ROS on the Jetson): per-camera frame files or MJPEG/raw + CSV of timestamps, IMU CSV/NPY. Convert to a ROS bag only for Kalibr (`rosbags` Python lib). MCAP is optional. |
| Stereo? | Only worth it with **hardware-synced global-shutter** stereo (RealSense D435i ~$334, OAK-D family) and if online VIO is needed, e.g. cuVSLAM stereo-inertial. Two IMX219s on the two CSI ports can't be hardware-synced (see 1.6), so they make poor stereo. |

Sensor BOM: about **$60-110** (section 8).

---

## 1. Cameras on Jetson Orin Nano

### 1.1 Connectors
- Dev kit carrier: two 22-pin 0.5 mm CSI connectors (J20 = CAM0, J21 = CAM1). **CAM0 is 2-lane only, CAM1 is 2 or 4 lane** [V]
  ([NVIDIA hardware layout](https://docs.nvidia.com/jetson/orin-nano-devkit/user-guide/latest/hardware_layout.html),
  [forum on 4-lane](https://forums.developer.nvidia.com/t/jetson-orin-nano-developer-kit-4gb-8gb-mipi-csi-2-capabilities-sd-card/249627)).
  Raspberry Pi-style 15-pin modules need a 15-to-22-pin FFC cable.
- Plain FFC cables are reliable to about 20-30 cm [I]. If the Jetson is *tethered* rather than mounted on the gripper, budget for
  a CSI extension kit (Arducam sells CSI-to-HDMI-cable extenders), or mount the Jetson on the gripper or forearm.

### 1.2 Sensor driver support on JetPack 6 (Orin Nano)
| Sensor | Status on Orin Nano JP6 | Notes |
|---|---|---|
| **IMX219** (RPi Cam v2, Arducam, Waveshare clones) | **In-tree, works.** Select "Camera IMX219-A/B/Dual" in `jetson-io` [V/F] ([JetsonHacks 2025](https://jetsonhacks.com/2025/03/04/jetson-orin-nano-super-with-the-raspberry-pi-v2-and-v3-cameras/amp/), [dual IMX219 forum](https://forums.developer.nvidia.com/t/how-to-enable-dual-imx219-camera-connection-on-jetson-orin-nano/273643)) | Rolling shutter, 8 MP. Sensor modes via Argus: 3280x2464@21, 3280x1848@28, 1920x1080@30 (center crop, so less FOV), **1640x1232@30 (2x2 binned, full FOV)**, **1280x720@60** (binned, 16:9) [F] ([mode list](https://forums.developer.nvidia.com/t/imx219-77-module-can-not-work-on-orin-nano-dev-kit/329929)). A request must match a driver mode exactly. Some reports of CAM1 or second-instance issues on JP 6.2.1 ([forum](https://forums.developer.nvidia.com/t/jetson-orin-nano-developer-kit-jetpack-6-2-1-36-4-7-nvarguscamerasrc-fails-to-load-the-camera-with-id-1-when-launched-for-the-second-time/358915)). |
| **IMX477** (RPi HQ) | In-tree overlay exists (jetson-io "IMX477") [I/F] | 12 MP, C/CS mount, so any lens incl. fisheye. Big and heavy for a gripper. |
| **IMX708** (RPi Cam Module 3, incl. 120° "Wide") | **No reliable JP6 driver.** Arducam and RidgeRun drivers are tied to specific L4T versions, and users report kernel-mismatch and Argus errors on 36.4.x [F] ([Arducam forum](https://forum.arducam.com/t/imx708-on-jetpack-6-2-l4t36-4-4/8395), [RidgeRun](https://developer.ridgerun.com/wiki/index.php/NVIDIA_Jetson_Orin_Nano/Camera_Sensors_Support/IMX708), [NVIDIA forum](https://forums.developer.nvidia.com/t/raspberry-pi-camera-module-3-imx708-linux-driver-for-jetson/354026)) | Avoid unless you want to port kernel drivers. |
| **OV9281** (1 MP mono global shutter, Arducam Jetvariety) | Arducam installer per exact L4T. Many JP 6.2 failure reports (missing overlay, probe errors, broken after kernel update) [F] ([1](https://forum.arducam.com/t/arducam-ov9281-not-detected-on-jetson-orin-nano-with-jetpack-6/9274), [2](https://forum.arducam.com/t/ov9281-is-not-starting-on-jetson-orin-nano-jetvariety-driver/8456)) | Global shutter is good for VIO, but monochrome (bad for the policy camera) and driver-fragile. |
| **IMX296** (RPi Global Shutter cam, 1.6 MP color) | Community driver only, reported working on JP 6.2.2 [F] ([forum](https://forums.developer.nvidia.com/t/imx296-driver-for-nvidia-jetson-orin-nano-nx/381940)) | Promising global-shutter color option if you accept a community kernel module. C/CS mount, so any lens. |

### 1.3 Ultrawide lens options (~120°)
- **Waveshare IMX219-120** (120° FOV, fixed lens) and **IMX219-160** (160°), about $20-30 [F]
  ([listing](https://thinkrobotics.com/ar/products/imx219-120-camera-for-jetson-nano-online)). Same IMX219 driver.
- **Arducam B0180** IMX219 with 175° M12 lens (~$10-25), "drop-in replacement for RPi V2 / Jetson Orin" [V]
  ([Arducam](https://www.arducam.com/b0180-arducam-imx219-wide-angle-camera-module-drop-replacement-raspberry-pi-v2-nvidia-jetson-camera.html)).
  It's an M12 mount, so you can swap in a ~120° M12 lens to match the iPhone ultrawide.
- For more than ~120°, use a fisheye model in calibration (`pinhole-equi` / Kannala-Brandt, or `ds` double sphere).

### 1.4 Global vs rolling shutter for VIO
- IMX219 full-frame readout is roughly 20-30 ms [I]. During fast hand rotation (a few rad/s) the image skews by many pixels. Neither
  OpenVINS nor ORB-SLAM3 models rolling shutter [F]
  ([multi-VIS paper noting OpenVINS lacks RS](https://arxiv.org/pdf/2308.05303)). Basalt has no RS model either [I].
- That said, **UMI's GoPro is also rolling shutter** and ORB-SLAM3 mono-inertial worked well enough for UMI. The iPhone cameras
  are rolling shutter too. Mitigations: short exposure, binned modes (shorter readout), ultrawide (pixel motion per rad is smaller),
  60 fps, and slow-ish demonstrations.
- Global-shutter options if needed: OV9281 (mono), IMX296 (community driver), a USB OV9782 color global shutter
  ([Arducam B0385, ~$72-100, UVC, 1280x800@100-120 fps MJPG, 70° lens](https://core-electronics.com.au/120fps-global-shutter-color-usb-camera-board.html.md)),
  or a stereo module (RealSense D435i, OAK-D).

### 1.5 Capture in Python/C++ and timestamps
- **Argus / nvarguscamerasrc** (ISP path, gives debayered NV12): `nvarguscamerasrc sensor-id=0 sensor-mode=4 ! video/x-raw(memory:NVMM),width=1280,height=720,framerate=60/1 ! nvvidconv ! video/x-raw,format=BGRx ! appsink`.
  Exposure and gain can be locked via `exposuretimerange`, `gainrange`, `aelock=true`, `wbmode`.
- **Hardware timestamps:**
  - Argus `ICaptureMetadata::getSensorTimestamp()` returns the kernel SOF timestamp. Forum tests on JP 6.2 indicate it's in the
    **CLOCK_MONOTONIC_RAW** domain [F].
  - `Ext::ISensorTimestampTsc::getSensorSofTimestampTsc()` / `getSensorEofTimestampTsc()` give raw VI hardware SOF/EOF in the Tegra-wide TSC
    counter [V] ([API ref](https://docs.nvidia.com/jetson/l4t-multimedia/SensorTimestampTsc_8h_source.html)).
  - Conversion formula from NVIDIA: `MONOTONIC_RAW = TSC_ns - offset_ns` (offset in `/sys/devices/system/clocksource/clocksource0/offset_ns`) [F]
    ([forum](https://forums.developer.nvidia.com/t/argus-getsensortimestamp-clock-domain-in-l4t-32-4-4/159220),
    [JP6.2 thread](https://forums.developer.nvidia.com/t/getsensortimestamp-getsensortimestamptsc-and-v4l2-timestamp/340004)).
- **V4L2 raw path** (`v4l2-ctl` / `v4l2src` on `/dev/videoN`, Bayer RG10, no ISP): the VI driver writes the SOF **TSC** timestamp into
  the vb2 buffer [F] ([forum](https://forums.developer.nvidia.com/t/where-is-the-image-frame-timestamp-set-in-the-vi-module-driver/276921)).
  That domain isn't standard V4L2 MONOTONIC, so `v4l2src` may discard it
  ([Orin NX JP6.1 thread](https://forums.developer.nvidia.com/t/jetson-orgin-agx-camera-frames-time-sync/343701)). You also have to debayer yourself.
- **Practical guidance:** GStreamer PTS in `nvarguscamerasrc` is pipeline running-time, not sensor SOF [I]. For VIO-grade stamps, write a ~200-line
  C++ libargus capture (based on the `argus_camera`/`oneShot` samples) that outputs `(frame_number, sensor_ts_monoraw, image)`, or a
  Python wrapper around it. **Verify the clock domain empirically on your JetPack** by comparing against `clock_gettime(CLOCK_MONOTONIC_RAW)` at
  delivery. The difference should be a stable ~20-40 ms latency.
- **USB/UVC cameras:** `uvcvideo` stamps buffers with CLOCK_MONOTONIC at USB-packet arrival, optionally corrected using the UVC
  SOF/PTS hardware-clock interpolation. This is known to be noisy and still being patched in 2026 [V]
  ([LKML 2026 patches](https://lkml.iu.edu/hypermail/linux/kernel/2603.2/15052.html)). Expect millisecond-level jitter, with a constant offset estimated
  by Kalibr/OpenVINS.
- **OAK-D:** device clock synced to host at under 200 µs (USB, 1σ), with IMU packets stamped in the same clock [V]
  ([Luxonis](https://discuss.luxonis.com/d/3270-host-clock-syncing-improved)). One BMI270 latency anomaly has been reported
  ([forum](https://discuss.luxonis.com/d/1775-bmi270-latency-issue)). Note the **OAK-D Lite price rose from $169 to $269 in July 2026**
  ([Luxonis](https://shop.luxonis.com/pages/pricing-change-oak-lite-oak-1)).

### 1.6 Can two CSI cameras be hardware-synchronized on Orin Nano?
- **Not with off-the-shelf IMX219 modules on the dev kit.** The RPi-style 15/22-pin camera pinout doesn't carry XVS/XHS/FSIN sync
  lines. Jetson has no generic "frame sync" for CSI. Sync needs sensor-board-level wiring (one master's XVS to slave) plus driver
  support for slave mode [I]. Arducam's old synchronized IMX219 stereo kit used a proprietary bridge chip and is discontinued
  ([Arducam](https://www.arducam.com/blog/?p=21775)). No NVIDIA doc confirms XVS routing on Orin Nano (search gap).
- Software result: both cameras free-run at the same nominal rate, with a phase offset that drifts slowly. Each frame carries its own SOF
  timestamp, which is fine for **mono VIO** plus a separately timestamped policy camera.
- Hardware-synced options: OAK-D (internal sync), RealSense D435i, ZED Mini (~$449) / ZED X Mini (GMSL, needs a capture card, too expensive),
  or Vision Components / e-con boards with trigger inputs (expensive).

---

## 2. IMU options

Approximate datasheet values from memory [I]. Verify against current datasheets before ordering.

| IMU | Gyro noise density | Accel noise density | Max ODR | Interface | Notes |
|---|---|---|---|---|---|
| **TDK ICM-42688-P** | ~2.8 mdps/√Hz | ~70 µg/√Hz | 32 kHz | SPI (24 MHz) / I2C / I3C | Low noise, **FIFO with 16-bit on-chip timestamps**, upstream Linux driver `inv_icm42600` (ICM-42688 ID added ~kernel 6.10, backport needed for L4T 5.15). Bare chip ~$3.4 ([LCSC](https://fat.lcsc.com/product-image/C1850418.html)). Breakouts vary from ~$10 (generic) to $88 (Elecrow). |
| **Bosch BMI088** | ~14 mdps/√Hz | ~160-190 µg/√Hz | gyro 2 kHz, accel 1.6 kHz | SPI/I2C (separate accel and gyro dies) | Vibration-robust (used in PX4 autopilots). **NVIDIA ships an IIO driver** (I2C, data-ready IRQs on **AON GPIOs** with GTE hardware timestamps) [V] ([NVIDIA r36.2 doc](https://docs.nvidia.com/jetson/archives/r36.2/DeveloperGuide/SD/Kernel/Bmi088ImuIioDriver.html)). Seeed Grove BMI088 ~$15-20 [I]. |
| **ST ISM330DHCX** | ~5 mdps/√Hz | ~60-80 µg/√Hz | 6.7 kHz | SPI/I2C | Industrial, good temperature stability, FIFO with timestamps. Linux `st_lsm6dsx` IIO driver is in mainline [I]. **SparkFun SEN-19764 $25.50** [V] ([SparkFun](https://www.sparkfun.com/sparkfun-6dof-imu-breakout-ism330dhcx-qwiic.html)). Default I2C, SPI via pads. |
| **BNO085/086** | — | — | gyro ~400 Hz | I2C/SPI/UART (SH-2 protocol) | **Not recommended for VIO.** Its on-chip fusion firmware gives processed outputs with jitter and latency, and raw rates are limited. |
| BMI270 / MPU-6050 | worse | worse | | | Fine for toys; noisier and with worse bias stability. |

**Rates:** 200 Hz is the VIO minimum. 400-1000 Hz is typical (UMI's GoPro IMU is ~200 Hz). Set ODR to 1 kHz with internal low-pass filtering,
and optionally downsample to 200-400 Hz for the estimator.

**Jetson 40-pin header:** SPI1 on pins 19 (MOSI), 21 (MISO), 23 (SCK), 24 (CS0), 26 (CS1). Enable it with `sudo /opt/nvidia/jetson-io/jetson-io.py`
→ "Configure header pins manually" → spi1. That creates `/dev/spidev0.0` [V/F]
([RidgeRun ICM42688 setup on Orin Nano JP 6.2](https://developer.ridgerun.com/wiki/index.php/RidgeRun_Video_Stabilization_Library/Useful_Links/ICM42688_Setup)).
RidgeRun's pin table and DT naming are slightly inconsistent, so check against the carrier-board spec.
I2C: pins 3/5 (I2C bus 7) and 27/28 (bus 1).

**Driver paths (pick one):**
1. **Userspace spidev (recommended for a capstone).** A C/C++ thread configures ICM-42688-P registers, enables FIFO with the
   20-byte packets incl. timestamp (TMST, 1 µs or 16 µs resolution), and polls the FIFO every ~5 ms. For each drained batch, record
   `clock_gettime(CLOCK_MONOTONIC_RAW)`. Reconstruct each sample's host time from the on-chip timestamps, and fit a linear clock model
   (offset + drift) over a sliding window. Python `spidev` works for 200-400 Hz but has more jitter, so prefer C++.
   No kernel rebuild needed.
2. **Kernel IIO driver (`inv_icm42600_spi`).** Requires building the module for L4T 36.x plus RidgeRun's ICM42688 patch and a DT overlay with an IRQ GPIO.
   IIO buffers deliver `(x, y, z, timestamp)` scans. The timestamp clock is selectable via
   `/sys/bus/iio/devices/iio:deviceN/current_timestamp_clock` (e.g. `monotonic_raw`, `monotonic`, `boottime`) [I]. The driver also uses the
   chip's FIFO timestamps to de-jitter.
3. **BMI088 with NVIDIA's driver.** Gets GTE hardware timestamps of the data-ready IRQs, but only if the IRQs go to **AON GPIOs**, which may not be broken out on the
   dev kit's 40-pin header [V/I]. Without AON you lose the HW-timestamp benefit.
4. Microcontroller bridge (RP2040/Teensy reads IMU, streams over USB). Simple but adds a second clock. You'd need a sync scheme (e.g. MCU also
   timestamps a Jetson GPIO pulse). Not recommended unless SPI on the Jetson becomes a blocker.

**Mount the IMU rigidly** on the same bracket as the VIO camera, close to it, with no foam. Kalibr assumes a rigid camera-IMU body.

---

## 3. VIO/SLAM stacks

| Stack | Sensor configs | Runs on Orin Nano? | ROS | Maintenance | Notes |
|---|---|---|---|---|---|
| **NVIDIA cuVSLAM** (Isaac ROS Visual SLAM; PyCuVSLAM) | Mono (visual only, scale-ambiguous), RGB-D, multi-camera (needs at least one overlapping stereo pair, up to 32 cams), **Inertial = stereo pair + IMU**, Multisensor (experimental). **No mono+IMU** [V] ([PyCuVSLAM README](https://github.com/NVlabs/PyCuVSLAM), [arXiv 2506.04359](https://arxiv.org/pdf/2506.04359)) | Yes, GPU-accelerated. Wheels for Jetson Orin JP 6 / Py 3.10. Isaac ROS **3.x** is the Orin Nano line; 4.x targets Thor/Jazzy [F] ([forum](https://forums.developer.nvidia.com/t/does-ros-issac-4-1-0-work-on-the-jetson-orin-nano/359612)) | ROS 2 Humble (Isaac ROS) or standalone Python/C++ | Active (repo updated Sept 2026, package renamed `isaac_ros_cuvslam`) | Isaac ROS wrapper Apache-2.0. cuVSLAM library under the **NVIDIA Community License** (check terms; fine for academic use [I]). Best choice **if** you buy a synced stereo+IMU module. |
| **OpenVINS** (ov_msckf) | Mono, stereo, "binocular" (multiple cameras tracked independently, non-overlapping OK), + IMU. Online calibration of cam-IMU extrinsics, **time offset**, intrinsics [V] ([docs](https://docs.openvins.com/)) | Yes, CPU (MSCKF, lightweight). Shown running on Orin Nano with a GS camera at 24 fps + 200 Hz IMU, ~1.7% drift on a 25 m loop [F] ([ROS Discourse](https://discourse.openrobotics.org/t/first-vio-demo-global-shutter-camera-imu-streamed-over-one-gmsl2-coax-mapped-live-in-ros-2/57700)) | ROS 1, ROS 2, and **ROS-free build** (v2.x) | Maintained (RPNG) but slow cadence | No rolling-shutter model. Multi-cam expects synchronized frames [I]. Good for an online check on the Jetson, or offline replay. Odometry only, no loop closure or relocalization against a map. |
| **ORB-SLAM3** | Mono, stereo, RGB-D, **mono-inertial**, stereo-inertial. Pinhole + **KannalaBrandt8 fisheye** | CPU, heavy (~2-4 cores). A 2026 GPU port reports 32 FPS mono-inertial on Orin Nano [F] ([arXiv 2608.17874](https://arxiv.org/pdf/2608.17874)) | Standalone C++ (community ROS 2 wrappers) | Upstream largely dormant since ~2022 [I]; many forks | **Used by UMI** (offline mono-inertial on GoPro fisheye + IMU, with a mapping pass then per-demo localization against the map, multi-map Atlas). Inertial init can fail on short or slow clips. UMI handles this with a dedicated mapping video. GPLv3. |
| **Basalt** | Stereo + IMU (upstream). Forks (Monado/mateosss, Luxonis BasaltVIO) add features. Mono not established [F] ([Luxonis BasaltVIO = stereo+IMU](https://docs.luxonis.com/software-v3/depthai/depthai-components/host_nodes/basalt_vio/)) | Yes, CPU-efficient, fast | Standalone; `mola_basalt_vio` ROS wrapper | Upstream slow; Monado fork active | Excellent calibration tools (ds camera model, time offset). Only relevant with stereo. |
| **VINS-Mono / VINS-Fusion** | Mono+IMU, stereo(+IMU) | CPU. VINS-Fusion-gpu variant for Jetson [F] ([benchmark](https://arxiv.org/pdf/2103.01655)) | ROS 1 (community ROS 2 ports) | Unmaintained since ~2020 [I] | Failed on drone-racing sequences in a 2025 Orin NX comparison where OpenVINS and LARVIO worked [F] ([arXiv 2512.20475](https://arxiv.org/pdf/2512.20475)). |
| **Kimera-VIO** | Stereo+IMU (mono limited) | CPU, heavy | ROS 1 mainly | Low activity [I] | Not recommended here. |

**Recommendation:**
- **Primary: ORB-SLAM3 mono-inertial, offline, on the ultrawide camera + IMU.** This mirrors the UMI reference pipeline, so the team can reuse UMI's
  scripts (mapping video → map → per-demo localization → gripper TCP via camera-to-TCP extrinsic). Fisheye/ultrawide is the supported, proven case.
- **Secondary / sanity check: OpenVINS mono+IMU.** Run it offline on the same logs, or online on the Jetson for a live "tracking OK?" indicator during
  recording. Its online time-offset estimate is a useful cross-check of the timestamp pipeline.
- **Main + ultrawide as a non-overlapping multi-cam rig:** not worth it. The cameras aren't synchronized (1.6). OpenVINS binocular assumes synced frames,
  and the main camera adds little to VIO beyond what a wide FOV already gives. Use the main camera only as a policy input.
- **Stereo pair?** Wiser *only* if (a) online pose is required on the gripper, or (b) mono-inertial initialization or scale proves unreliable on short demos.
  In that case buy a **hardware-synced global-shutter stereo+IMU module** (RealSense D435i ~$334, [price source](https://roboticscenter.ai/learn/robot-camera-depth-sensor-comparison);
  OAK-D Pro W) and use cuVSLAM stereo-inertial. Don't build stereo from two unsynced IMX219s.

---

## 4. Calibration workflow

1. **IMU noise (once):** record **≥3 hours** (ideally longer) of the IMU sitting still at the operating rate, as a ROS 1 bag. Run
   [allan_variance_ros](https://github.com/ori-drs/allan_variance_ros) to get `accelerometer_noise_density`, `accelerometer_random_walk`, `gyroscope_noise_density`,
   `gyroscope_random_walk`. **Inflate them ~5-10x** for Kalibr and the VIO, which is standard practice [I].
2. **Camera intrinsics:** Kalibr `kalibr_calibrate_cameras` with an **Aprilgrid** target (printed flat on foam board or aluminium) [V]
   ([Kalibr wiki](https://github.com/ethz-asl/kalibr/wiki/camera-IMU-calibration)).
   - Models: `pinhole-radtan` for the ~77° main camera. For the ultrawide, `pinhole-equi` (Kannala-Brandt, matches ORB-SLAM3 KB8 and OpenVINS `equidistant`)
     or `ds-none` (double sphere, which matches Basalt).
   - Use short exposure and good light, and cover the whole FOV including corners.
3. **Camera-IMU extrinsic + time offset:** `kalibr_calibrate_imu_camera` (temporal calibration on by default) [V]. Excite all 6 axes for 60-90 s, without blur.
   Kalibr uses header timestamps verbatim. **Bad timestamps are its most common failure mode** [V], so check IMU `dt` histograms first.
4. **Rolling shutter:** Kalibr has a separate RS calibrator (single camera, no IMU) [V]
   ([wiki](https://github.com/ethz-asl/kalibr/wiki/Rolling-Shutter-Camera-calibration)). Mainly useful for knowing the line delay, since the VIO won't use it.
5. **Camera-to-TCP (gripper tip) extrinsic:** a fixed-geometry measurement from CAD, as in UMI. At Franka deployment, also do a hand-eye calibration
   (camera to flange) so FK poses and training VIO poses refer to the same gripper frame.
6. **Tooling:** Kalibr is ROS 1 and runs fine in the official **Docker** image on the laptop. Convert your raw logs to ROS 1 bags with the pure-Python
   [`rosbags`](https://gitlab.com/ternaris/rosbags) library, so the Jetson never needs ROS.
   OpenCV alternatives: `cv2.fisheye.calibrate` / `cv2.omnidir` for intrinsics, but OpenCV has **no camera-IMU calibration**. Alternatives for that are Basalt's
   `basalt_calibrate_imu` or OpenVINS online calibration (good refinement but not a replacement).
7. Re-run steps 2-3 whenever a camera or IMU is re-mounted or a lens refocused. Keep calibration YAMLs versioned per rig build.

---

## 5. Time synchronization on one Jetson

- **Use one clock: `CLOCK_MONOTONIC_RAW`** (or `CLOCK_MONOTONIC`; pick one and convert everything to it). Argus's sensor timestamp appears to already be
  MONOTONIC_RAW [F]. TSC stamps convert with `offset_ns`. IIO can be told to stamp in `monotonic_raw`. Never use wall-clock (REALTIME) for sensor fusion.
- **PTP/NTP aren't needed** for a single host. On Franka deployment, robot state comes over the network via libfranka. Stamp robot state on arrival in the same
  monotonic clock on whichever machine runs the policy, or run chrony between the Franka PC and the Jetson if they're different hosts.
- **Camera latency / offset:** stamp at **SOF** (exposure start + reset offset). For VIO the ideal is mid-exposure, so add `exposure/2 + readout/2` for a rolling shutter.
  Then let Kalibr (`timeshift_cam_imu`) and OpenVINS (`calib_cam_timeoffset`) estimate the residual constant offset. Expect a few ms if stamps are SOF-based,
  versus 20-60 ms if you stamp on userspace delivery.
- **Validate the timestamp pipeline:**
  - Point the camera at an LED blinked by a Jetson GPIO whose toggle time you log, and measure the delay.
  - Or rotate the rig sharply and cross-correlate gyro rate with image optical-flow magnitude.
  - Check frame `dt` and IMU `dt` histograms for drops and jitter.
- **Hardware trigger:** not available for IMX219 on the dev kit (1.6). If a future camera has a trigger input, drive it from a Jetson GPIO or the IMU's
  data-ready / FSYNC pin. The ICM-42688-P has an **FSYNC input** that can tag the IMU sample nearest a camera strobe, which is a cheap way to tie camera exposure to IMU
  time if the camera exposes a strobe/XVS output [I].
- Run a `chrony`/NTP client only for wall-clock file naming. It doesn't affect MONOTONIC_RAW.

---

## 6. Recording formats and tools

- **No ROS on the Jetson is practical and recommended** for this offline-VIO design. A C++ capture daemon for cameras (libargus) and IMU (spidev), with a thin Python
  controller (start/stop, demo IDs, gripper-width sensor), writes:
  - `cam_main/` and `cam_uw/`: frames as **JPEG** (Orin Nano has a hardware JPEG engine via `nvjpegenc` [I, verify]) or raw NV12 to NVMe, plus
    `frames.csv` (`frame_idx, sensor_sof_ns, exposure_ns, gain`).
  - `imu.csv`/`.npy` (`t_ns, ax, ay, az, gx, gy, gz, temp`).
  - Then offline convert to whatever ORB-SLAM3/OpenVINS/Kalibr need (EuRoC-style folders or ROS bags via `rosbags`).
- **Orin Nano has no hardware video encoder (NVENC)** [V]
  ([NVIDIA forum](https://forums.developer.nvidia.com/t/jetson-orin-nano-video-encoding/352823)). Software x264 copes with about one 1080p30 stream, or 2x 720p30.
  So **don't H.264-encode two 60 fps streams live**. Store JPEG or raw to an **NVMe SSD** (raw 1280x720 NV12 at 60 fps is about 83 MB/s per camera) and transcode
  offline to whatever the UMI/iPhUMI dataset converter expects.
- **ROS 2 + rosbag2 with MCAP** (`ros-humble-rosbag2-storage-mcap`) is the standard choice if you need live visualization (Foxglove) or plan to run Isaac ROS.
  Image topics are big, though, so you'd use `CompressedImage`. It adds overhead without clear benefit for offline processing.
- At **Franka deployment** the same capture daemon serves the latest frames (with SOF timestamps) to the policy process over shared memory, ZeroMQ, or a ROS 2 topic.
  The IMU is ignored and pose comes from libfranka FK, stamped in the same monotonic clock.

---

## 7. Known pitfalls

1. **Rolling shutter + fast hand motion:** geometric skew that VIO doesn't model. Use the ultrawide, binned 60 fps mode, short exposure, and demonstrations
   that avoid violent wrist flicks. Watch tracking-loss rates in ORB-SLAM3.
2. **Auto-exposure / motion blur:** lock exposure (e.g. 2-5 ms) and gain, and lock AWB (`aelock`, `awblock`, `exposuretimerange`). AE changes break feature
   tracking and photometric consistency, and long exposure smears features. Indoors you may need extra light on the workspace.
3. **IMU vibration / mounting:** a printed-plastic flex mount or gripper motor/servo vibration aliases into the IMU. Use a stiff mount and the chip's internal
   anti-alias filter, and sample at 1 kHz then low-pass and decimate. Keep the IMU away from the gripper motor. Temperature drift: let the rig warm up, or log temperature.
4. **Inertial initialization (mono):** monocular VIO needs translational excitation in the first seconds to get scale. Start each demo with a short
   "wiggle", or use UMI's approach of one mapping video then localize-only.
5. **Thermal / power throttling on Orin Nano Super:** at 25 W / MAXN SUPER, forum users report *over-current* throttling that cuts performance by about 50% [F]
   ([forum](https://forums.developer.nvidia.com/t/question-regarding-jetson-orin-nano-super-system-throttling-due-to-overcurrent-undercurrent/371007),
   [NVIDIA blog](https://developer.nvidia.com/blog/nvidia-jetpack-6-2-brings-super-mode-to-nvidia-jetson-orin-nano-and-jetson-orin-nx-modules/)).
   For a capture-only workload, run the **15 W mode** (plenty for capture) and monitor with `tegrastats`/`jtop`. Keep the fan; an enclosed gripper mount
   will overheat. Battery power for a handheld rig: a 19 V or USB-PD-to-barrel supply rated ≥25 W [I].
6. **Driver fragility across JetPack updates:** anything not in-tree (OV9281, IMX708, IMX296, ICM-42688 module) breaks on `apt upgrade` of the kernel
   [F]. **Pin the L4T version** (`apt-mark hold nvidia-l4t-kernel*`) once things work.
7. **Argus quirks:** sensor-mode resolution must match exactly. The second camera instance sometimes fails to reopen on JP 6.2.1 [F]. Restart
   `nvargus-daemon` when a capture crashes.
8. **CSI cable length and EMI** if the Jetson is tethered rather than mounted on the gripper.
9. **Camera-to-TCP consistency between training and deployment:** VIO poses are of the IMU/camera body. Apply the same fixed extrinsic used at deployment
   (camera to flange from hand-eye), or the policy's action frame will be offset.

---

## 8. Recommended bill of materials (sensors only; Jetson + fan already owned)

| Qty | Part | Purpose | Approx. price | Link |
|---|---|---|---|---|
| 1 | Arducam/Waveshare **IMX219 ~77-80°** module (Jetson Orin compatible, 22-pin cable) | Main policy camera | $15-27 | [Arducam IMX219 for Orin](https://www.arducam.com/arducam-imx219-pdaf-cdaf-autofocus-camera-module-with-case-for-raspberry-pi-nvidiar-jetson-orin-series.html) |
| 1 | **Waveshare IMX219-120** (120°) *or* **Arducam B0180** (175°, M12, swap lens to ~120°) | Ultrawide camera, also the VIO camera | $10-30 | [Arducam B0180](https://www.arducam.com/b0180-arducam-imx219-wide-angle-camera-module-drop-replacement-raspberry-pi-v2-nvidia-jetson-camera.html), [IMX219-120](https://thinkrobotics.com/ar/products/imx219-120-camera-for-jetson-nano-online) |
| 2 | 22-pin to 15-pin FFC, 15-30 cm (often included) | CSI cabling | $3-6 each | Arducam/Waveshare |
| 1 | **ICM-42688-P breakout with SPI pins exposed** | VIO IMU | $10-20 (generic) | e.g. [open-hardware module](https://github.com/SUSTAINA-OP/ICM-42688-P-Module). Search "ICM42688P SPI breakout" |
| (alt) | SparkFun **ISM330DHCX** breakout (SEN-19764) | VIO IMU, mainline Linux driver | $25.50 | [SparkFun](https://www.sparkfun.com/sparkfun-6dof-imu-breakout-ism330dhcx-qwiic.html) |
| 1 | NVMe M.2 2280 SSD, 256-512 GB (if not already installed) | Raw frame logging | $25-40 | any |
| 1 | Aprilgrid target (print A2/A1 on foam board or Dibond) | Kalibr | $15-30 | [Kalibr targets](https://github.com/ethz-asl/kalibr/wiki/calibration-targets) |
| — | 3D-printed rigid bracket: both cameras + IMU on one plate | Rigid camera-IMU body | filament | — |

**Total: about $60-110.**

Optional upgrades if mono-inertial proves inadequate:
- **IMX296 global shutter** (~$50, community driver).
- **Arducam B0385 OV9782 USB global-shutter color** (~$72-100).
- A **synced stereo+IMU module** (RealSense D435i ~$334; OAK-D Lite now $269) with cuVSLAM stereo-inertial.

---

## 9. Recommended software stack

```
Jetson (JetPack 6.2.x, L4T pinned, nvpmodel 15W, no ROS)
├── capture_cams (C++, libargus): 2x IMX219, fixed exposure/gain/AWB, SOF timestamps (MONOTONIC_RAW), JPEG/raw → NVMe
├── capture_imu  (C++, spidev): ICM-42688-P @1 kHz, FIFO + chip timestamps → host MONOTONIC_RAW (linear fit)
├── recorder.py: session/demo control, metadata JSON (calib version, L4T, exposure), gripper-width channel
└── (optional) OpenVINS ROS-free build, mono+IMU, live "tracking OK" indicator

Laptop / cluster (offline)
├── rosbags (Python): raw → ROS1 bag for Kalibr; raw → EuRoC folders for ORB-SLAM3/OpenVINS
├── Kalibr (Docker): intrinsics (pinhole-radtan main, pinhole-equi ultrawide) + cam-IMU extrinsics/time offset
├── allan_variance_ros: IMU noise params
├── ORB-SLAM3 mono-inertial (UMI's docker/scripts): mapping pass + per-demo localization → camera trajectory
├── OpenVINS (cross-check, time-offset sanity)
└── existing iPhUMI → zarr dataset converter (swap ARKit pose for VIO pose × camera-to-TCP extrinsic)

Franka deployment
└── same capture_cams daemon → policy; pose from libfranka FK; IMU unused
```

---

## 10. Uncertainties / things to verify on hardware

- **Exact clock domain** of Argus `getSensorTimestamp()` on your L4T 36.4.x. Forum evidence says MONOTONIC_RAW, but threads conflict. Verify empirically (5.).
- Whether **nvarguscamerasrc** exposes the sensor timestamp to GStreamer/Python apps. I assume not, hence the C++ libargus recommendation.
- Whether ORB-SLAM3 mono-inertial initializes reliably on **IMX219 rolling-shutter, 30-60 fps, ~120°** footage of short tabletop demos. UMI used a 155° GoPro.
  Pilot this before committing (record 5 demos plus a mapping video, run the UMI SLAM pipeline).
- IMU noise figures in section 2 are from memory. Check the datasheets. ICM-42688 kernel support needs a backport (upstream ID added in ~6.10).
- Whether any **AON GPIO** is on the dev kit's 40-pin header (needed for the BMI088 GTE-timestamp path).
- Whether the Orin Nano has a hardware JPEG encoder usable via `nvjpegenc`. If not, use CPU libjpeg-turbo (fine at 2x 720p60 with a few cores), or store raw.
- CAM0 is 2-lane only. IMX219 needs only 2 lanes, so this is fine for the recommended parts, but it rules out 4-lane sensors on CAM0.
- cuVSLAM license (NVIDIA Community License) terms weren't read in full. Irrelevant if you follow the mono-inertial recommendation.
