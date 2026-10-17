# Purchasing list: Jetson sensor rig + Franka deployment

What to buy so that the software in `sensor_rig/` (data collection) and `controlSW/franka_deploy/` (deployment) can run
on real hardware.

- **Where the specs come from:** derived in [sensor_rig/docs/planning/SPEC_sensors.md](sensor_rig/docs/planning/SPEC_sensors.md)
  §3 and §5. The software assumes exactly these specs (`configs/rig_jetson_imx219_2p1_1p55.yaml`).
- **Prices** are approximate list prices from the spec research (2026). **Check stock, price and compatibility
  before ordering.**
- **Already owned?** Skip it. Items we may already have are marked *(if not owned)*.

## 1. Data-collection rig (required)

| Qty | Item | Must satisfy | Why | ~Price |
|---|---|---|---|---|
| 1 | NVIDIA **Jetson Orin Nano Super Developer Kit** *(if not owned)* | JetPack 6.x; includes the power supply | the single computer: recording, and inference at deployment | ~$249 |
| 1 | **NVMe SSD**, M.2 2280, 256–512 GB *(if not owned)* | PCIe; mounts in the dev kit | boots JetPack; records ~20 MB/s of JPEG | $25–40 |
| 2 | **Arducam B0180** IMX219 camera module (M12 lens mount, RPi-v2 pinout) | IMX219 (stock JetPack 6 driver); **M12 mount so the lens can be swapped** | main + ultrawide cameras. Stick to IMX219: other sensors have fragile JetPack 6 drivers | $15–25 each |
| 1 | **M12 2.1 mm low-distortion lens**, 1/4" image circle, IR-cut (Arducam 1/4" 2.1 mm < 1 % series, or Commonlands CIL821) | focal length **2.0–2.3 mm**, never > 2.3 mm | main camera: covers the iPhone main view 100 % with mount tolerance (82°×67°) | $10–30 |
| 1 | **M12 ~1.55 mm fisheye lens** (f-theta), IR-cut (Commonlands **CIL216** 1.55 mm f/2.2, or The Pi Hut 1.56 mm M12 fisheye) | focal length **1.45–1.69 mm**, never > 1.75 mm; image circle ≥ sensor height | ultrawide camera: finger tags (gripper width) + pose tracking; covers the iPhone ultrawide view 100 % (~145°×105°) | $15–40 |
| 2 | **CSI ribbon cable, 15-pin (camera) → 22-pin (Jetson), 0.5 mm pitch, 15–30 cm** | **≤ 30 cm** (longer is unreliable for MIPI) | cameras to the dev kit's CAM0/CAM1 | $3–6 each |
| 1 | **ICM-42688-P IMU breakout** with SPI pins exposed (e.g. the open-hardware SUSTAINA-OP module or any SPI breakout) | **SPI** (not I²C-only), 3.3 V logic; INT1 and FSYNC broken out preferred | pose tracking (VIO) and timing. The driver is written for this exact chip | $10–20 |
| 1 set | Female-female jumper wires (Dupont), 10–20 cm, and a short pin header | 3.3 V, SCK/MOSI/MISO/CS, GND to header pins 17/19/21/23/24/25 | IMU wiring | $5 |
| 1 | **Rigid camera + IMU bracket** (3D-printed PETG, or machined aluminium) | **lens centres within ±3 mm of the iPhone 16 Pro lens positions** on the gripper; IMU rigidly attached | a remap can fix rotation but not position, so this decides whether the views match the training data | filament / shop time |
| 1 | Lens-focus locking: low-strength thread locker or UV glue | | lock the focus once set; focus must not drift | $5 |

**Powering the gripper-mounted Jetson while collecting (pick one, decision D7 in [TODO.md](TODO.md)):**

| Option | Item | Notes | ~Price |
|---|---|---|---|
| a. Tethered (simplest) | the dev kit's own supply plus a 2–3 m DC extension cable | the operator carries the gripper with a cable | $10 |
| b. Battery | USB-C PD power bank (≥ 65 W, 20 V PD profile) plus a USB-C PD-trigger → 5.5×2.5 mm barrel cable | **verify the dev kit's DC input voltage range first.** The capture power mode is 15 W | $50–90 |

## 2. Deployment on the Franka (required)

| Qty | Item | Must satisfy | Why | ~Price |
|---|---|---|---|---|
| 1 | **Feetech STS3215** serial bus servo (or **Dynamixel XM430-W350** as the pricier alternative) | position read-back **and a current/torque limit**. Both driver stubs exist in `franka_deploy/gripper/driver.py` | motorizes the iPhUMI gripper, so the fingers and tags look like training. The Franka Hand would look different to the policy | STS3215 $15–25 · XM430 ~$290 |
| 1 | Servo USB adapter: Feetech/Waveshare **bus servo adapter** (or **U2D2** for Dynamixel) | USB serial at 1 Mbaud (Feetech) / 57.6 kbaud+ (Dynamixel) | Jetson USB → servo bus | $10 (Feetech) · ~$35 (U2D2) |
| 1 | Servo power supply, **12 V** (STS3215 12 V version) or as rated, ≥ 3 A | matches the servo voltage | servo power, separate from the Jetson | $15 |
| 1 | **Gripper actuation mechanism** (3D-printed parts coupling the servo to the iPhUMI finger linkage) | keep the original fingers and ArUco tags; the full width range must reach ≥ 0.085 m | | filament |
| 1 | **Wrist mount** (3D-printed or aluminium) holding the gripper + rig on the Franka flange | ISO 9409-1-31.5-4-M5 flange pattern; known mass for the payload setting | hand-eye and TCP calibration assume a rigid mount | filament / shop |
| 1 | **Cat6 Ethernet cable**, length to reach the Franka control box | direct link (no switch) for the 1 kHz loop | Jetson → Franka FCI | $10 |
| 1 | **USB 3 → gigabit Ethernet adapter** | Linux-supported (ASIX / Realtek r8152) | the Jetson's only RJ45 port goes to the Franka, so this one gives development/network access | $15 |
| 1 | **3Dconnexion SpaceMouse** (Compact or Wireless) *(if not owned)* | works with real-env's `SpacemouseClient` | start, teleop and reset in the real-env task flow | ~$150 |

## 3. Calibration and test aids (required for hardware day)

| Qty | Item | Spec | Used by | ~Price |
|---|---|---|---|---|
| 1 | **Kalibr Aprilgrid** 6×6, 30 mm tags, printed and glued flat on a rigid board (foam board / dibond) | measure the real tag size after printing | camera intrinsics + camera–IMU calibration (`first_plugin.md` step 8) | $20–30 |
| 1 | **ChArUco board** 9×7, 30 mm squares, 22 mm markers, DICT_5X5_100, rigid and flat | measure the square size after printing | hand-eye calibration on the Franka (`hardware_day.md` 4.2) | $10–20 |
| 1 | **LED + 330 Ω resistor** (plus 2 jumper wires) | on Jetson header pin 7 + GND | camera timestamp test `sensor-rig stamp-offset` (`first_plugin.md` step 7) | $2 |
| 1 | The eraser / task objects used in the `firstreal` demos, plus tape marks for start-pose buckets | same objects as training | evaluation protocol | — |
| 1 | Grey card or colour checker *(optional)* | | locking exposure/white balance to a reference | $10–40 |
| 1 | 64 GB microSD card *(optional)* | | alternative first boot / recovery for JetPack | $10 |

## 4. Fallbacks (only if a hardware test fails; don't buy up front)

| Item | Buy only if | ~Price |
|---|---|---|
| Used x86 mini-PC with an Intel NIC (e.g. NUC i5) for a PREEMPT_RT Franka bridge | the Jetson fails the 30-min go/no-go load test (`hardware_day.md` step 1.3) | $150–250 |
| IMX296 global-shutter camera module, or a RealSense D435i / OAK-D (stereo + IMU) | ORB-SLAM3 mono-inertial can't track reliably on the rolling-shutter ultrawide footage | $50–330 |
| External SSD, 1 TB | the laptop disk is ~90 % full, and raw sessions + ORB-SLAM3 Docker need ≥ 20 GB | $60–80 |

## Rough totals

| Group | If the Jetson, NVMe and SpaceMouse are already owned | Buying everything |
|---|---|---|
| Data-collection rig | ~$90–180 | ~$355–460 |
| Deployment (Feetech path) | ~$65–75 | ~$215–225 |
| Calibration and test aids | ~$30–50 | ~$40–100 |
| **Total** | **~$185–305** | **~$610–785** |

Mechanical parts (brackets, mounts, gripper actuation) are extra: 3D printing or machine-shop time.

## Don't buy

- **Raspberry Pi Camera Module 3 / IMX708** for the Jetson: fragile JetPack 6 drivers, and its 16:9 sensor can't cover
  the iPhone ultrawide view (coverage ≤ 0.75).
- **Narrow lenses**: an IMX219 "77°" module (coverage 0.83), a 120° ultrawide (0.84), or a main lens longer than 2.3 mm.
- **The Franka Hand** as the deployment gripper: it looks nothing like the training gripper, so the policy would see
  unfamiliar images.
