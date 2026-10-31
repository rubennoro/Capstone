# Running the IMX219 + IMU on the current Jetson (JetPack 7.2.1)

What works on the rig's Jetson today and how to run it. Written 2026-10-31, after the first hardware session. The
IMX519 is **not** part of this setup (section 5).

Jetson as found: Orin Nano Super dev kit, **JetPack 7.2.1 / L4T R39.2.1**, kernel `6.8.12-1021-tegra`, Ubuntu 24.04.
This is not the version the plan pins (JetPack 6.2 / L4T 36.4.3, [PIPELINE.md](PIPELINE.md) D9).

---

## 1. Hardware

| Sensor | Connection | Shows up as |
|---|---|---|
| IMX219 160° ultrawide | camera connector on I2C bus 10 | `imx219 10-0010`, `/dev/video0`, Argus sensor 0 |
| Modulino Movement (LSM6DSOX) | 40-pin header: 3V3 pin 1, SDA pin 3, SCL pin 5, GND pin 6 | `6a` on I2C bus 7, chip id `0x6c` |
| IMX519 | **unplugged** | (section 5) |

- Leave the Modulino's INT1 pin unconnected ([HARDWARE_ARRIVAL.md](HARDWARE_ARRIVAL.md) §4).
- Boot entry: `JetsonIO` with NVIDIA's stock `tegra234-p3767-camera-p3768-imx219-dual.dtbo` (set by `jetson-io.py`).
- Only plug or unplug camera ribbons with the Jetson powered off.

Check:
```bash
grep -n "^DEFAULT" /boot/extlinux/extlinux.conf   # DEFAULT JetsonIO
v4l2-ctl --list-devices                           # vi-output, imx219 10-0010 ... /dev/video0
i2cget -y 7 0x6a 0x0f                             # 0x6c
```

## 2. Build

The build lives on the NVMe at `/data/rigbuild`, with the Argus backend on:
```bash
cd ~/Capstone            # branch rnoro/jetson-sensors
physicalSetup/scripts/check.sh /data/rigbuild --jobs 4 -- -DRIG_WITH_ARGUS=ON
```
After a code change, rebuilding only the programs is faster:
```bash
cd /data/rigbuild && ninja ec-live ec-imu-probe ec-cam-probe
```

The build needs the IMU driver fix from 2026-10-31 (`lsm6dsox_driver.cpp`, post-reset check). Without it, only the
first IMU start after power-up works; later starts fail with
`after reset CTRL9_XL (0x18) = 0xe2, expected default 0xe0`. The chip keeps its I3C_disable bit through a software
reset.

## 3. Config: IMX219 as Argus sensor 0

With only the IMX219 connected, Argus numbers it 0, but the rig config expects it at 1 (with the IMX519 at 0).
Make a copy with the two sensor ids swapped. The config check rejects two cameras on the same id, so `main` moves
to 1; it isn't started with `--camera uw`.
```bash
RIG=~/Capstone/physicalSetup/episode_collection/configs/rig_jetson_imx519_imx219w160_lsm6dsox.yaml
sed -e '31s/sensor_id: 0 /sensor_id: 1 /' -e '106s/sensor_id: 1 /sensor_id: 0 /' $RIG > ~/rig_uw_only.yaml
grep -n "sensor_id" ~/rig_uw_only.yaml            # 31: sensor_id: 1 (main), 106: sensor_id: 0 (uw)
```
The line numbers match the yaml as of commit `0237513c`. If the yaml changes, edit the two `sensor_id` lines by
hand.

## 4. Run

### Live view (`ec-live`): IMX219 + IMU

Run it from SSH; the window opens on the Jetson's monitor. The desktop session is display `:1` here, not `:0`.
```bash
cd /data/rigbuild
export DISPLAY=:1
./episode_collection/ec-live --config ~/rig_uw_only.yaml --camera uw
```
- Top: the IMX219 image. Bottom: IMU readings and a rolling 5 s plot. Tilt or shake the rig and the lines move.
- Keys: `s` saves a PNG to the current directory; `q` / `Esc` / Ctrl-C quits.
- The SSH terminal prints one status line per second.

Without a window (works over SSH with no display):
```bash
./episode_collection/ec-live --config ~/rig_uw_only.yaml --camera uw --no-window --seconds 5
```

What it should look like (measured 2026-10-31):
```
[ OK ] imu imu0 (lsm6dsox_i2c) started, ODR 833 Hz
[ OK ] camera uw (argus 1640x1232@30) started
[    1s] | uw 30.0fps drops 0 age 39ms | imu0 842.2Hz |a| 9.81 ovr 0 i2c 0
...
ec-live: stopped cleanly
```
- uw 30 fps, 0 drops, exposure 4 ms / gain 4 (locked).
- Frame age about 40–80 ms.
- IMU about 842 Hz, 0 overruns, 0 I2C errors.
- |a| at rest 9.8–10.0 m/s². The target is 9.81 ± 0.2; a reading just above 10.0 has been seen once, so watch it.
- A `thread placement not applied ... SCHED_FIFO` warning is harmless without root.

IMU only, or the camera only:
```bash
./episode_collection/ec-live --config ~/rig_uw_only.yaml --no-cameras
./episode_collection/ec-live --config ~/rig_uw_only.yaml --camera uw --no-imu
```

### Saving data (probes)

Recording programs (`ec-record`, Wave 2) don't exist yet. The probes can write data today:
```bash
cd /data/rigbuild
# IMU: 60 s to disk + PASS/FAIL summary (H1 is --duration 600)
./episode_collection/ec-imu-probe --duration 60 --record ~/probe/imu_test
# IMX219: 60 s timing check + 5 sample frames
./episode_collection/ec-cam-probe --config ~/rig_uw_only.yaml --camera uw --seconds 60 --save-frames 5 --out ~/probe
```
- Don't run a probe while `ec-live` is open; they use the same devices.
- Copy results off the Jetson (from the laptop): `scp -r capstone-gripper@<jetson-ip>:~/probe .`

## 5. Why the IMX519 is not included

Tried on 2026-10-31 and rolled back:
- **Driver:** works. Arducam has no package for L4T 39.2.1, but their 39.2.0 beta `imx519.ko`
  ([forum thread](https://forum.arducam.com/t/official-arducam-driver-status-update-jetpack-7-2-l4t-39-2-kernel-6-8/9297))
  matches all 35 kernel symbols on 39.2.1. Loaded on the stock kernel with Arducam's dual-IMX519 overlay, the sensor
  probed (`imx519 9-001a`, i.e. bus 9: modes 4656x3496@9, 3840x2160@17, 1920x1080@60, 1280x720@120).
- **Argus: blocked.** Argus enumerates 0 cameras, for two reasons only Arducam/NVIDIA can fix:
  1. JetPack 7 requires a `.nito` tuning file per sensor. There is no `imx519.nito`. The legacy
     `camera_overrides.isp` (`NVCAMERA_NITO_PATH=CONFIG`) is rejected ("not allowed anymore"). Converting it needs
     NVIDIA's partner-only Camera Partner Toolkit.
  2. Argus looks the tuning file up from the module badge. It can't map Arducam's `jakku_front_arducam` ("Could not
     map module to ISP config string"), so even a copied `imx519.nito` is never opened.
- **Don't run Arducam's installer** (`install_full.sh` / `dpkg -i`). Its package replaces the kernel `Image`,
  overwrites `/boot/extlinux/extlinux.conf`, and sets both ports to IMX519, which disables the IMX219.

Everything from that attempt was removed: the module, overlay, boot entry, tuning files and the nvargus-daemon
drop-in.

**Plan:** reflash to JetPack 6.2, the version the plan pins, where Arducam supports the IMX519. Then follow
[HARDWARE_ARRIVAL.md](HARDWARE_ARRIVAL.md) and T8 ([IMPLEMENTATION_PLAN.md](IMPLEMENTATION_PLAN.md)). The reflash
wipes the NVMe; push code and copy data off first.
