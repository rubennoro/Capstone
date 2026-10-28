# When the hardware arrives

What you can do with the rig, depending on how far the software is. Written 2026-10-28, when **Waves 0–1 are built**
(sensor drivers + probes) and **Wave 2 is not** (no recording programs yet). Check
[STATUS.md](STATUS.md) for the current state before starting.

Hardware: Jetson Orin Nano dev kit (JetPack 6.2 / L4T 36.4.3), Arducam IMX519 (main, CAM0), Arducam IMX219 160°
(ultrawide, CAM1), Arduino Modulino Movement (LSM6DSOX IMU), monitor + keyboard on the Jetson, NVMe SSD, internet.

Detailed command tables live in [bringup_snippets/T1_imu.md](bringup_snippets/T1_imu.md) and
[bringup_snippets/T2_cameras.md](bringup_snippets/T2_cameras.md). This page is the order to do things in and what
"working" looks like.

---

## 1. What is possible today vs after Wave 2

| | Today (Waves 0–1) | After Wave 2 |
|---|---|---|
| Build the code on the Jetson | yes; first real build of the Argus camera code | yes |
| **Live view of both cameras + IMU readings on the monitor (`ec-live`, §9)** | **yes** | yes (and `ec-viz` with live SLAM) |
| Test the IMU (`ec-imu-probe`): register check, 833 Hz stream, 10-min acceptance test H1 | **yes** | yes |
| Test each camera (`ec-cam-probe`): modes, fps, timestamps, exposure/gain lock, focus, sample frames, acceptance test H2 | **yes**, once the camera driver + dual-camera device tree are set up by hand | yes |
| Written step-by-step setup (`BRINGUP.md`), `jetson_setup.sh`, `ec-doctor` all-system check | no (use this page + the snippets) | yes (T8) |
| Gripper calibration run (`ec-calibrate`) | no | yes (T7) |
| Record episodes with live checks (`ec-record`) | no | yes (T5) |
| See cameras / live SLAM on the monitor (`ec-viz`, `ec-slam-live`) | no | yes (T6; ORB-SLAM3 first built on the Jetson) |
| 15-min back-to-back thermal test (H5) | no | yes |
| Turn episodes into training data (`ec-process`) | no | no: Wave 3, plus the Kalibr calibration (TODO) |

**So today is a sensor bring-up day**: prove the IMU and both cameras work on the Jetson, and measure the things the
code can only guess (bus numbers, timestamp clock, focus behaviour, colour range). That is the riskiest part of the
whole rig, and every finding feeds back into the code.

---

## 2. Before you plug anything in

1. **Don't upgrade the Jetson's system packages.** The Arducam IMX519 driver is built for the exact kernel of
   L4T 36.4.3; an `apt upgrade` of the kernel/L4T packages can break it.
   ```bash
   cat /etc/nv_tegra_release          # expect: R36 (release), REVISION: 4.3
   uname -r                           # note it down
   sudo apt-mark hold $(dpkg -l | awk '/^ii  nvidia-l4t-/ {print $2}')   # keep L4T/kernel packages fixed
   ```
2. **Network + SSH**, so the laptop can help (you set this up).
3. **NVMe:** know where it is mounted (e.g. `/data`); recordings go there later.
4. **Get the code** on the Jetson: you handle the git side (push the branch `rnoro/jetson-sensors`, then clone or
   pull it on the Jetson).

---

## 3. Build on the Jetson

Install the build dependencies (names for Ubuntu 22.04 / JetPack 6.2; T8's `jetson_setup.sh` will automate and
verify this later):

```bash
sudo apt install cmake ninja-build g++ pkg-config \
     libopencv-dev libeigen3-dev libyaml-cpp-dev libspdlog-dev nlohmann-json3-dev libjpeg-dev libgtest-dev \
     libgstreamer1.0-dev libgstreamer-plugins-base1.0-dev \
     nvidia-l4t-jetson-multimedia-api i2c-tools v4l-utils
```

Build with the Argus camera backend (on the NVMe, not the SD card):

```bash
cd ~/Capstone            # your clone, on branch rnoro/jetson-sensors
physicalSetup/scripts/check.sh /data/rigbuild --jobs 4 -- -DRIG_WITH_ARGUS=ON
```

- Expected: it builds and the unit tests pass (about 398). The Argus code has only ever been compiled against
  NVIDIA's headers on the laptop, so **a compile error here is a real finding**: copy the error back.
- The programs end up in `/data/rigbuild/episode_collection/` (`ec-imu-probe`, `ec-cam-probe`).

---

## 4. IMU first (no driver install needed, lowest risk)

**Wiring** (Modulino Qwiic → Jetson 40-pin header):

| Modulino | Header pin |
|---|---|
| 3V3 | 1 |
| SDA | 3 |
| SCL | 5 |
| GND | 6 |

**Leave the Modulino's INT1 pin unconnected.** If INT1 is high at power-up, the chip starts in a mode (I3C-only) where
it doesn't answer on I2C.

Then, in order (full expected outputs in [T1_imu.md](bringup_snippets/T1_imu.md)):

```bash
sudo i2cdetect -y -r 7                         # expect "6a"
sudo i2cget -y 7 0x6a 0x0f                     # expect 0x6c (the LSM6DSOX chip id)
xxd /sys/bus/i2c/devices/i2c-7/of_node/clock-frequency   # want 400 kHz (0006 1a80) or more; 100 kHz is too slow
cd /data/rigbuild
./episode_collection/ec-imu-probe --raw 30     # registers written + read back, FIFO layout
./episode_collection/ec-imu-probe --duration 600 --record /data/imu_h1   # H1: 10 min, rig still on the bench
```

**Working looks like:** every line `[PASS]`, then `== ALL PASS`. Sample rate within 0.5 % of
833 × (1 + 0.0015 × FREQ_FINE) Hz, 0 FIFO overruns, gravity at rest 9.81 ± 0.2 m/s², clock residual p99 < 100 µs.

**Report back:** the full `--raw 30` output and the H1 summary. Several FIFO details were taken from ST's and
Linux's driver code because ST's application note (AN5272) couldn't be downloaded. The probe output confirms or
refutes them (slot order, burst-read wrap, timestamp trim).

---

## 5. Cameras (one at a time, then both)

The camera driver and device-tree setup is **not yet written down step by step** (that's T8, Wave 2). Do it in the
order with the least risk, and **check with me before changing the device tree or `/boot/extlinux/extlinux.conf`**,
since a bad boot config can stop the Jetson from booting (the plan adds a fallback boot entry first).

1. **IMX219 ultrawide alone** (stock JetPack has an IMX219 driver): enable it with `sudo /opt/nvidia/jetson-io/jetson-io.py`
   on its CSI port, reboot, then `ls /dev/video*` and `v4l2-ctl -d /dev/video0 --list-formats-ext`
   (expect 1640x1232 among the sizes).
2. **IMX519 main:** Arducam's installer for L4T 36.4.3 (`install_full.sh -m imx519`, from Arducam's
   Jetson camera docs). Arducam's overlay assumes two IMX519s, so running IMX519 + IMX219 together needs a
   custom mixed overlay. Get the IMX519 working alone first if the mixed one isn't ready.
3. **Both together** with the mixed overlay.

Then the checks (full tables in [T2_cameras.md](bringup_snippets/T2_cameras.md)):

```bash
sudo dmesg | grep -i -E "imx519|imx219"        # both sensors probe; note which I2C bus each is on (9 or 10)
sudo systemctl status nvargus-daemon           # active (running)
sudo i2cdetect -y -r 9; sudo i2cdetect -y -r 10   # where the IMX519 focus chip (0c) is, and whether it shows UU
RIG=physicalSetup/episode_collection/configs/rig_jetson_imx519_imx219w160_lsm6dsox.yaml
./episode_collection/ec-cam-probe --config $RIG --list                    # modes; main -> 1920x1080@60, uw -> 1640x1232@30
./episode_collection/ec-cam-probe --config $RIG --camera main --focus-sweep 0:4095:256
./episode_collection/ec-cam-probe --config $RIG --camera main --seconds 60 --save-frames 5 --out /data/probe
./episode_collection/ec-cam-probe --config $RIG --camera uw   --seconds 60 --save-frames 5 --out /data/probe
./episode_collection/ec-cam-probe --config $RIG --seconds 300 --together --json /data/probe/h2.json   # H2
```

Update the yaml as you learn things (e.g. `cameras.main.driver.focus.i2c_bus` = the IMX519's bus from `dmesg`).

**Working looks like:**
- main 60 fps, ultrawide 30 fps;
- timing jitter p99 < 2 ms;
- 0 drops;
- exposure and gain read back constant;
- latency positive and stable;
- `time source ... monotonic_raw`;
- focus `vcm_i2c` (or `v4l2`).

**Report back:**
- the `--list` output;
- the H2 summary + `h2.json`;
- the time source line and the TSC cross-check number;
- which focus path won;
- the I2C bus numbers;
- the sample frames: are they in focus, with natural colour (no washed-out blacks), no sharpening halos?

If the Argus backend won't start at all, the GStreamer fallback is in section 10.5 of the camera snippet. It's for
bring-up only, because its timestamps are about 30 ms late.

---

## 6. Day-one checklist (today's software)

- [ ] Code builds on the Jetson with `-DRIG_WITH_ARGUS=ON`; unit tests pass.
- [ ] IMU: `6a` on bus 7, chip id 0x6c, bus ≥ 400 kHz, `ec-imu-probe` ALL PASS, H1 10-min PASS.
- [ ] IMX219 ultrawide alone streams 1640x1232@30.
- [ ] IMX519 main alone streams 1920x1080@60; focus path known.
- [ ] Both cameras together; H2 5-min PASS.
- [ ] Recorded and sent back: probe outputs, bus numbers, timestamp clock, focus result, sample frames.
- [ ] The nine hardware-day verification items in [TODO.md](../../TODO.md) ("physicalSetup/") ticked or reported.

---

## 7. Once Wave 2 is built (the full flow)

These programs don't exist yet; this is what the plan delivers ([PIPELINE.md](PIPELINE.md) §4).

1. `BRINGUP.md` + `jetson_setup.sh`: the written setup, including the dual-camera device tree with a fallback boot
   entry.
2. `ec-doctor`: one command, PASS/FAIL for the whole Jetson (versions, IMU, cameras, modes, focus, NVMe speed,
   thermal, display, ORB-SLAM3 files).
3. Build ORB-SLAM3 + Pangolin on the Jetson's NVMe (script from T6).
4. Per session:
   ```bash
   ec-calibrate --session block-raise-2026-10-30        # open/close the gripper ~3x, Enter; PASS/FAIL
   ec-record --session block-raise-2026-10-30 --task block_raise   # move the gripper until SLAM is ready,
                                                                    # recording starts by itself; Enter to stop
   ```
   Repeat `ec-record` for each episode. Live checks abort a bad episode with the reason shown, as the iPhone does.
   The monitor shows both cameras and the live SLAM view.
5. Acceptance tests H3–H7 ([IMPLEMENTATION_PLAN.md](IMPLEMENTATION_PLAN.md) §5): all sensors together without drops,
   live SLAM ≥ 28 Hz, **15 minutes of back-to-back episodes without overheating**, first real calibration, first five
   real episodes.

Session names use letters, digits and `-` only (no `_`; iPhUMI's tools split names on `_`).

---

## 8. What still can't be done after Wave 2

- **Training data:** `ec-process` (offline SLAM → 60 Hz pose → gripper width → 224 images → zarr) is Wave 3.
- **Training-grade poses** also need the rig calibration (Kalibr), which is a separate discussion (TODO.md).
- **Deployment** on the Jetson is Wave 5.

Recorded episodes are kept as packages, so they can be processed later once Wave 3 and the calibration exist.

---

## 9. `ec-live`: see the sensors live (side task L1)

One program that shows both cameras side by side plus a live IMU strip on the Jetson's monitor. It is built with
everything else (`/data/rigbuild/episode_collection/ec-live`). Use it alongside the probes in sections 4–5: the
probes give PASS/FAIL numbers; `ec-live` lets you see that the picture, the focus and the IMU respond.

**Run it in a terminal on the Jetson's desktop** (monitor + keyboard), so `$DISPLAY` is set. Over SSH there is no
display: it says so and prints only the once-per-second status line (or `export DISPLAY=:0` if the desktop is
logged in as you). Don't run it at the same time as `ec-cam-probe` / `ec-imu-probe` (same devices).

```bash
cd /data/rigbuild
RIG=~/Capstone/physicalSetup/episode_collection/configs/rig_jetson_imx519_imx219w160_lsm6dsox.yaml
./episode_collection/ec-live --config $RIG --no-cameras            # 1. after ec-imu-probe passes: IMU only
./episode_collection/ec-live --config $RIG --camera uw   --no-imu  # 2. after the IMX219 streams alone
./episode_collection/ec-live --config $RIG --camera main --no-imu  # 3. after the IMX519 streams alone
./episode_collection/ec-live --config $RIG                         # 4. everything (both cameras + IMU)
```

**What it should look like:**
- Top: main (left) and uw (right), each scaled to fit with its aspect ratio kept. Each tile's overlay shows:
  - id, mode and backend, e.g. `main (main) 1920x1080@60 [argus]`;
  - measured fps over 1 s (expect 60 / 30), frame count and drops (expect 0; turns yellow otherwise);
  - exposure ms and gain read back from the frame (expect the yaml's 8.00 ms / 2.00 and 4.00 ms / 4.00);
  - `focus vcm_i2c @512` (or `v4l2`) on main;
  - `range full` (`unknown` is shown in yellow);
  - frame age in ms (now − mid-exposure time). Expect tens of ms and stable; negative or growing is a finding.
- Bottom strip (IMU):
  - accel x/y/z and |a| (at rest about 9.81 m/s²) and gyro x/y/z in deg/s (about 0 at rest);
  - temperature, rate (expect about 833 Hz), FIFO max level / watermark, overruns (expect 0, red otherwise), I2C
    errors, and the clock-fit residual;
  - a rolling 5 s plot of accel and gyro (x red, y green, z blue): tilt or shake the rig and the lines move.
- A camera that stops delivering shows a red **NO FRAMES for N s** on its tile. A sensor that fails to start shows
  **FAILED TO START** plus the reason. At start the terminal prints `[FAIL] ...` with the error and a hint, then
  continues with the sensors that did start (`--strict` exits with code 1 instead).
- The terminal prints one status line per second (rates, drops, frame age, |a|, overruns, `ui` = render ms per tick).
  A `thread placement not applied ... SCHED_FIFO` warning without root is harmless for viewing.

**Keys:** `q` or `Esc` quits; `s` saves a PNG of the whole window to the current directory (or `--out DIR`). Ctrl-C
also stops cleanly (a second Ctrl-C exits at once).

**Other flags:** `--width 1200` (smaller window), `--display-hz 10`, `--no-window` (terminal only), `--seconds N`
(stop after N s), `--stall-s 2` (NO FRAMES threshold), `--mock` (laptop: mock cameras + simulated IMU, sizes from
the yaml), `--headless-snapshot out.png --seconds 3` (no window; writes the final picture; used by the tests).

**Report back:** a snapshot (`s`) of step 4 with the rig still and one while moving it, plus the status lines.
