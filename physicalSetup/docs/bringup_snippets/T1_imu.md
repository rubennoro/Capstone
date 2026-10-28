# BRINGUP snippet T1: LSM6DSOX IMU (for T8 to merge into BRINGUP.md step 7 and step 13)

Sources: DS12814 Rev 4 (the LSM6DSOX datasheet), ST `lsm6dsox-pid` (vendored in `third_party/st_lsm6dsox/`) and
Linux `st_lsm6dsx` facts. **AN5272 was not available when the driver was written**: the items marked [AN] below are
taken from the datasheet, ST's driver or the Linux driver, and are checked here on real silicon.

Wiring (Modulino Movement Qwiic to the 40-pin header): 3V3 to pin 1, SDA to pin 3, SCL to pin 5, GND to pin 6.
Leave the module's INT1 unconnected: if INT1 is high at power-up the chip starts in I3C-only mode and does not answer
on I2C (DS §5.3).

## 7a. The bus and the address

```bash
ls -l /dev/i2c-7                      # expect: crw-rw---- root i2c ... /dev/i2c-7
groups | grep -w i2c                  # expect: i2c listed. If not: sudo usermod -aG i2c $USER, then log in again
sudo i2cdetect -y -r 7                # expect: "6a" in row 60, column a. Nothing else is required on this bus
```

- No `6a`: check the wiring and 3V3; try `sudo i2cdetect -y -r 1` (pins 27/28) in case of a mislabelled bus.
- `6b` instead of `6a`: SDO/SA0 is pulled high on this board. Set `imu.driver.i2c_addr: 0x6B`.
- `UU`: a kernel driver (`st_lsm6dsx`) has claimed the chip. `lsmod | grep st_lsm6dsx`, then unbind it, or blacklist
  the module, before using the userspace driver.

```bash
xxd /sys/bus/i2c/devices/i2c-7/of_node/clock-frequency   # expect: 0006 1a80 (400000 Hz) or 000f 4240 (1 MHz)
```

**This value matters.** At 833 Hz, with a timestamp word per sample, the FIFO needs about 160 kbit/s (3 words x
7 bytes x 9 bits x 833 Hz). That is about 40 % of a 400 kHz bus, and it does not fit on 100 kHz (`0001 86a0`). On a
100 kHz bus, either raise `clock-frequency` in the device tree (T8 overlay), or use 416 Hz.
`ec-imu-probe` prints this value too.

## 7b. Register-level checks with i2c-tools (before any of our code)

```bash
sudo i2cget -y 7 0x6a 0x0f            # WHO_AM_I. Expect 0x6c (DS §9.14)
sudo i2cget -y 7 0x6a 0x12            # CTRL3_C after power-up. Expect 0x04 (IF_INC=1, DS Table 21)
sudo i2cget -y 7 0x6a 0x18            # CTRL9_XL after power-up. Expect 0xe0 (DS Table 21)
sudo i2cget -y 7 0x6a 0x63            # INTERNAL_FREQ_FINE (signed 0.15 %/LSB). Note the value; expect small, |v| < 20
```

## 7c. `ec-imu-probe` (run as the user, from the build dir)

```bash
./episode_collection/ec-imu-probe --raw 30
```

What to check in the output:
- `[PASS] WHO_AM_I`, then every configuration register `wrote X read X` PASS:

  | Register | Value |
  |---|---|
  | CTRL3_C | 0x44 |
  | CTRL9_XL | 0xe2 |
  | FIFO_CTRL1 | 0x00 |
  | FIFO_CTRL2 | 0x01 |
  | FIFO_CTRL3 | 0x77 |
  | CTRL1_XL | 0x7c |
  | CTRL2_G | 0x7c |
  | CTRL10_C | 0x20 |
  | FIFO_CTRL4 | 0x66 |

- **FIFO slot layout [AN].** The repeating pattern should be `TS cnt k`, `GYRO cnt k`, `ACCEL cnt k`, then `k+1`,
  and so on. A `TEMP` word appears about every 67 slots.
  - The summary line should report `different TAG_CNT: 0`.
  - If timestamp words carry a different TAG_CNT than their data, the default `pair_by_tag_counter: true` is wrong.
    In that case, rerun with `--pairing ts` and compare the summaries, then report back.
- `ticks` should step by about 48 per slot (25 us ticks at 833 Hz).
- The accel word should read about 4100 LSB on the axis pointing up (1 g at +-8 g: 1/0.000244).
- `burst read of 8 words from 0x78: 8/8 valid tags`. This confirms that the address wraps from 0x7e back to 0x78
  [AN]. If it is less than 8/8, set `fifo_burst_words: 1` (in code, ImuConfig) and report back.
- The odd-parity count tells us the definition of TAG_PARITY, which is not documented. Note it down; the driver
  does not use it.

```bash
./episode_collection/ec-imu-probe --duration 600 --record /tmp/imu_h1     # H1: 10 min at rest, rig on the bench
```

Expected output: every check `[PASS]`, then `== ALL PASS` (exit code 0). The H1 thresholds are:

| Check | Pass |
|---|---|
| sample rate | 833 x (1 + 0.0015 x FREQ_FINE) Hz, within 0.5 % |
| FIFO overruns | 0 |
| \|g\| at rest | 9.81 +- 0.2 m/s^2 |
| clock residual p99 | < 100 us |

Also watch the following:
- **`slope ... ppm`:** the fitted tick versus the trimmed tick (exact AN5192 formula 25 us / (1 + 0.0015 x
  FREQ_FINE)). Expect a few hundred ppm at most.
  - Thousands of ppm, or a `slope not clamped` FAIL, means the trim formula or its sign is wrong [AN]. Report the
    slope together with the `FREQ_FINE` value.
- **`partial slots`, `slots without timestamp word`, `unknown tags`:** all 0.
- **`torn reads`:** expect 0 or a few. Each one is a TIMESTAMP0..3 read that crossed a byte carry; it is detected
  and dropped. Many of them means BDU does not latch the timestamp registers. That is still handled, but report it.
- **`fifo max`:** about 25 to 35 words at a 10 ms poll. Much higher means the poll thread is starved: check the CPU
  pinning and priority of the IMU thread.
- **`bracket width p50`:** the duration of one I2C read of 4 registers. Expect about 150 to 400 us at 400 kHz.
  - The clock offset has a constant bias of about half the asymmetry of the transaction (latch after the address
    phase). It is absorbed by the camera-IMU time offset (Kalibr) and only needs to be stable.
- **gyro mean at rest:** within +-1 dps (DS Table 2 zero-rate level). **accel per-axis std:** about 1 to 3 mg at
  833 Hz (80 ug/rtHz at +-8 g, DS Table 2).

Deliberate overrun (checks that detection works, and shows the FIFO capacity of 512 words [AN]):

```bash
./episode_collection/ec-imu-probe --duration 5 --poll-ms 400 --no-rest-check
```

Expected: `FIFO overruns` FAIL with a count of about 12, `fifo max 512`, and `sample gaps` FAIL. Any other capacity
than 512 means the [AN] assumption is wrong: report it.

## 13. Timestamp domain (IMU part)

`/tmp/imu_h1/chip.csv` holds the raw brackets and `clock_fit.json` holds the fit. Two things to check:
- `residual_ns.p99 < 100000`;
- `bracket_width_ns.p50` is stable between runs.

`sample_meta.csv` (`t_ns,chip_ticks,temp_c,flags`, the package layout) keeps the chip ticks of every sample, so the mapping can be refitted offline if needed.

## Laptop dry run (no hardware)

```bash
./episode_collection/ec-imu-probe --sim --duration 3      # the same driver against the register simulator: ALL PASS
./episode_collection/ec-imu-probe --sim --raw 12          # shows the simulator's assumed slot layout
```
