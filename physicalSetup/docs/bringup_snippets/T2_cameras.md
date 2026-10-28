# T2 bring-up snippet: cameras (BRINGUP step 10; acceptance test H2)

For T8 to merge into BRINGUP.md. Hardware: Orin Nano dev kit, JetPack 6.2 / L4T 36.4.3; CAM0 = Arducam IMX519
(main), CAM1 = Arducam IMX219 160° (ultrawide). Preconditions: BRINGUP 8 (Arducam IMX519 driver) and 9 (mixed
overlay) are done; physicalSetup is built with `-DRIG_WITH_ARGUS=ON` (needs `nvidia-l4t-jetson-multimedia-api`).
Every value marked [I] in the code is checked here.

## 10.1 Drivers, daemon and nodes

| # | Command | Expected | If not |
|---|---|---|---|
| 1 | `cat /etc/nv_tegra_release` | `# R36 (release), REVISION: 4.3` | stop: the Arducam driver is kernel-exact |
| 2 | `sudo dmesg \| grep -i -E "imx519\|imx219"` | both sensors probe; IMX519 at `9-001a` or `10-001a`, IMX219 at `10-0010` or `9-0010` | `error during i2c read probe (-121)` = wrong overlay/port or cable (Arducam forum 8481) |
| 3 | `ls /dev/video*` | `/dev/video0 /dev/video1` | overlay not applied (BRINGUP 9) |
| 4 | `sudo systemctl status nvargus-daemon` | `active (running)` | `sudo systemctl restart nvargus-daemon`; check `journalctl -u nvargus-daemon` |
| 5 | `v4l2-ctl -d /dev/video0 --list-formats-ext` | `RG10`, sizes 4656x3496, 3840x2160, 1920x1080, 1280x720 (IMX519) | record the real list in the worklog; the T2 table must match |
| 6 | `v4l2-ctl -d /dev/video1 --list-formats-ext` | `RG10`, 3280x2464, 3280x1848, 1920x1080, 1640x1232, 1280x720 (IMX219) | as above |
| 7 | `v4l2-ctl -d /dev/video0 --list-ctrls` | sensor controls (gain, exposure, frame_rate, sensor_mode, ...); note whether `focus_absolute` exists | no `focus_absolute`: the V4L2 focus fallback is unavailable (fine if 0x0c is free) |

## 10.2 Focus address (`UU` or free)

| # | Command | Expected | Meaning |
|---|---|---|---|
| 8 | `sudo i2cdetect -y -r 9` and `-r 10` | `1a` shows `UU` (sensor driver) on the IMX519's bus; `0c` shows `0c` or `UU` | `0c` = userspace VCM write (`vcm_i2c`); `UU` = kernel owns it: V4L2 fallback or `power_up` |
| 9 | `ls -l /sys/bus/i2c/devices/9-000c/driver` | absent (free), or a link to e.g. `ak7375` | same fact as `UU`; Imx519Focus checks both |
| 10 | Set `cameras.main.driver.focus.i2c_bus` to the bus where the IMX519 is (step 2) | | the yaml's `9` is [I] |

## 10.3 Mode table and GStreamer sanity

```bash
gst-inspect-1.0 nvarguscamerasrc | grep -E "exposuretimerange|gainrange|ispdigitalgainrange|aelock|awblock|wbmode|tnr-mode|ee-mode|aeantibanding|sensor-mode"
gst-launch-1.0 nvarguscamerasrc sensor-id=0 num-buffers=120 ! 'video/x-raw(memory:NVMM),width=1920,height=1080,framerate=60/1' ! fakesink
gst-launch-1.0 nvarguscamerasrc sensor-id=1 num-buffers=60 ! 'video/x-raw(memory:NVMM),width=1640,height=1232,framerate=30/1' ! fakesink
```

Expected: every property above exists (they are what `build_gst_pipeline` emits); the `GST_ARGUS:` lines list the
modes with `FR =` and `Duration =`; IMX519 mode 2 = `1920 x 1080 FR = 59.999999`, IMX219 mode 3 = `1640 x 1232 FR =
29.999999`. A missing property: tell the orchestrator (fix in `capture_plan.cpp`).

## 10.4 ec-cam-probe

```bash
RIG=episode_collection/configs/rig_jetson_imx519_imx219w160_lsm6dsox.yaml
ec-cam-probe --config $RIG --list                              # driver-reported modes + the mode each camera selects
ec-cam-probe --config $RIG --camera main --focus-sweep 0:4095:256   # then narrow, e.g. 300:900:32
ec-cam-probe --config $RIG --camera main --seconds 60 --save-frames 5 --out /data/probe
ec-cam-probe --config $RIG --camera uw   --seconds 60 --save-frames 5 --out /data/probe
ec-cam-probe --config $RIG --seconds 300 --together --json /data/probe/h2.json   # H2 (5 min, both cameras)
```

Expected output per camera (H2 passes when every line is `[PASS]`):

```
camera main: PASS
  frames 18000 over 300.00 s -> 60.0000 fps; mode 2; time source argus:getSensorTimestamp in monotonic_raw ...
  t_mid - t_sof 15.x ms (exposure/2 + readout/2); readout ~2xxxxxxx ns
  TSC cross-check (TSC SOF - offset_ns) - sensor_ts: p50 ~0 us
  [PASS] fps / jitter_p99 (< 2 ms) / drops (total 0) / latency / exposure_locked / gain_locked / focus
```

What to look at, and what to do if it differs:

| Line | Expected | If not |
|---|---|---|
| `--list` | `configured 1920x1080@60 -> 2: ...` (main), `-> 3: 1640x1232` (uw) | a size/fps mismatch prints all modes: fix the yaml or the T2 table |
| `time source` | `getSensorTimestamp in monotonic_raw` (NVIDIA forum 268916: "kernel timestamp (MONOTONIC_RAW)") | `tsc-offset_ns` or `realtime`/`monotonic`: record it; the conversion is applied in one place (`SensorClockResolver`), but tell the orchestrator |
| TSC cross-check | \|p50\| < 100 µs, min/max within ±1 ms | a constant offset = the two stamps are different events or domains: report the number |
| `latency` | t_arrival - t_sof positive and stable; main roughly exposure + readout + 5..40 ms | negative = wrong domain (start() would normally refuse); drifting = clock problem |
| `exposure_locked` / `gain_locked` | readback constant, within 5 % of the yaml (line quantization) | not constant: AE lock failed; check the log line "AE lock ... applied" |
| ISP digital gain (in `gain_locked`) | `1..1` | > 1: `setIspDigitalGainRange` ignored |
| `focus` | `vcm_i2c` (or `v4l2`) with the configured position | `power_up` = no control path (0x0c is UU and no V4L2 control): focus is wherever the lens powered up; decide with the orchestrator. `failed` = write NACK: wrong `focus.i2c_bus` (step 10) |
| `drops` | 0 | `backend_queue` > 0 = the probe/CPU is too slow; `seq_gap` = Argus/driver drops; `time_gap` = sensor stalls |
| log `TSC EOF - SOF = ... getFrameReadoutTime = ...` | EOF - SOF ≈ readout (+ one line) | a large mismatch: the readout used in t_mid is wrong; tell the orchestrator |
| warnings `IDenoiseSettings unavailable` / `IEdgeEnhanceSettings unavailable` | absent | denoise/edge may still be on; the images will look sharpened/smoothed |

Sample frames: check focus at the working distance, colour (AWB locked after 1 s of auto), and no sharpening halos.
Colour range (R1 F16): Argus frames are full-range NV12 (`NVBUF_COLOR_FORMAT_NV12_ER`, `FrameMeta::color_range =
full`), so a JPEG encoded straight from them must look identical to the sample JPEGs (no washed-out blacks or
crushed whites). `--focus-sweep` refuses mock / replay / gst `source_override` cameras and exits 1 if no step moved
the lens (every step `power_up`/`failed`).

## 10.5 GStreamer fallback (only if the Argus backend fails)

```bash
ec-cam-probe --backend gst --sensor imx519 --sensor-id 0 --width 1920 --height 1080 --fps 60 \
             --exposure-ns 8000000 --gain 2 --seconds 30
```

Expected: PASS for fps/jitter/drops/latency; no exposure/gain readback on this path (checks skipped); time source
`gst:pts(monotonic->raw)`; latency is about 30 ms and jittery. That is why gst is bring-up only. Colour range on this
path is whatever the nvvidconv caps state (`colorimetry`); check with `GST_DEBUG=2` / the frame meta and record it.

## Hardware-day verification list (T2)

1. Mode indices, sizes and fps vs `known_sensor_modes()` (steps 5, 6, 10.3), especially that IMX519 mode 2 is the
   full 16:9 field. Compare the FOV of 1920x1080 and 1280x720 in sample frames: per the upstream imx519.c crops,
   1280x720 sees only 2/3 of the width.
2. Timestamp domain chosen by `SensorClockResolver`, TSC cross-check ≈ 0, latency positive and stable.
3. `t_mid` = getSensorTimestamp - exposure/2 + readout/2 (getSensorTimestamp = first data = end of row-0 exposure):
   LED / stamp-offset test (BRINGUP 13); a residual goes into `cameras.<id>.time.offset_ns`.
4. Readout time: `getFrameReadoutTime` vs TSC EOF - SOF (log line).
5. Locks: exposure and gain readback constant and equal to the yaml; ISP digital gain 1; AWB stable over 5 min after
   the deferred lock (Argus re-submits the repeating request; gst sets `awblock` at runtime, which is unverified).
6. Denoise / edge enhancement off (no "unavailable" warnings; no halos).
7. Focus: which path won (`vcm_i2c` / `v4l2` / `power_up`), the VCM readback, the sweep result, and that the
   sharpness is constant over a 10 min session (no autofocus anywhere).
8. `IFrame::getNumber()` increments by 1 per frame (seq gaps only on real drops).
9. Both cameras together on one CameraProvider; CPU cost of the NV12 copies (`top`, `tegrastats`) during H2/H3.
10. nvarguscamerasrc property names (10.3), and whether its GST_BUFFER_OFFSET is a frame counter.
11. CAM0/CAM1 I2C bus numbers for `focus.i2c_bus` (step 2/8).
12. Colour range: Argus NV12_ER really is full range (a black/white target: Y spans ~0..255, not 16..235); the
    nvvidconv NV12 range stated in its caps on the gst path.
13. Future-stamped frames: `timestamp_anomalies` in the probe JSON should be 0 or tiny (stamps up to 2 ms after
    arrival are accepted and counted, R1 F3).
