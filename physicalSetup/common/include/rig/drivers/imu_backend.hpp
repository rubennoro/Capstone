// SPDX-License-Identifier: MIT
/// @file rig/drivers/imu_backend.hpp
/// @brief IMU driver interface: LSM6DSOX over i2c-dev (day one), register-level simulator, replay.
/// FROZEN CONTRACT (T0); implemented by T1 in common/drivers/imu/.
///
/// Lifecycle: make_imu(config) -> start() -> read()... -> stop(). Restartable after stop().
/// Threading:
/// - start(), stop() from the owning thread. start() spawns the backend's internal poll thread (absolute-time loop
///   every poll_period_ms; PLAN T1); samples are queued internally (bounded; overflow is counted as a host overrun).
/// - read() and read_brackets() from exactly one consumer thread (the IMU capture thread).
/// - stats(), status(), config() thread-safe at any time. stop() unblocks a concurrent read().
#pragma once

#include <chrono>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "rig/drivers/i2c_bus.hpp"
#include "rig/status.hpp"
#include "rig/time/clock.hpp"
#include "rig/types.hpp"

namespace rig {

/// YAML strings in parentheses.
enum class ImuBackendKind : uint8_t {
  kLsm6dsoxI2c = 0,  ///< ("lsm6dsox_i2c") userspace FIFO driver on /dev/i2c-<i2c_bus>.
  kSim = 1,          ///< ("sim") the same driver code against T1's Lsm6dsoxSim register simulator (real-time paced).
  kReplay = 2,       ///< ("replay") from a package raw/imu0 (data.csv + chip.csv) or EuRoC imu0/data.csv.
};

/// Simulator options (backend "sim" and T1's tests). Mirrors `imu.driver.sim`. Additive fields allowed (T1).
struct ImuSimConfig {
  double chip_clock_drift_ppm = 0.0;  ///< Chip oscillator error versus host [ppm] (tests use up to +/-20000 = 2 %).
  int64_t chip_clock_offset_ticks = 0;  ///< Initial chip counter value.
  double accel_noise_std_mps2 = 0.0;  ///< White noise per sample [m/s^2].
  double gyro_noise_std_rps = 0.0;    ///< White noise per sample [rad/s].
  bool motion = false;                ///< Add sinusoidal motion (axis/scale checks) on top of gravity.
  double i2c_error_rate = 0.0;        ///< Probability a transaction NACKs [0..1].
  uint64_t seed = 1;
};

/// Replay options. Mirrors `imu.driver.replay`.
struct ImuReplayConfig {
  std::string path;      ///< Package raw/imu0 dir or EuRoC imu0 dir.
  bool realtime = true;  ///< Pace by timestamps.
  bool retime = true;    ///< Shift so the first sample is at now_raw_ns() at start().
};

/// Driver configuration. Mirrors rig.yaml `imu.driver`. Defaults = the rig's LSM6DSOX settings (PLAN T1).
struct ImuConfig {
  std::string id = "imu0";          ///< Stream id; package dir raw/<id>/.
  ImuBackendKind backend = ImuBackendKind::kLsm6dsoxI2c;
  int i2c_bus = 7;                  ///< Linux bus number (40-pin pins 3/5 = i2c-7 on the Orin Nano dev kit).
  int i2c_addr = 0x6A;              ///< 7-bit address (SDO/SA0 low; Modulino Movement default).
  int expected_who_am_i = 0x6C;     ///< WHO_AM_I value for the LSM6DSOX (DS12814).
  double odr_hz = 833.0;            ///< Accel and gyro ODR [Hz]; one of 12.5, 26, 52, 104, 208, 416, 833, 1666,
                                    ///< 3332, 6667 (nominal values; actual = nominal * trim, AN5272).
  int accel_fs_g = 8;               ///< Accel full scale [g]: 2 | 4 | 8 | 16.
  int gyro_fs_dps = 2000;           ///< Gyro full scale [dps]: 125 | 250 | 500 | 1000 | 2000.
  int fifo_watermark_words = 256;   ///< FIFO watermark [7-byte words]; FIFO holds 512 (about 50 %).
  int poll_period_ms = 10;          ///< FIFO poll period [ms] (absolute-time loop).
  int bracket_period_ms = 100;      ///< Period of bracketed TIMESTAMP register reads for the clock fit [ms].
  int i2c_retries = 3;              ///< Retries per failed transaction before it counts as an error.
  int i2c_error_budget = 20;        ///< Errors allowed per 10 s before status() turns fatal.
  double gap_factor = 2.0;          ///< A sample interval > gap_factor / odr counts as a gap in ImuStats.
  std::size_t queue_capacity = 8192;  ///< Internal sample queue between poll thread and read() [samples].
  // --- T1 additive fields (no yaml key yet; defaults = the PLAN T1 configuration) ---
  int timestamp_decimation = 1;     ///< FIFO timestamp word every 1 | 8 | 32 slots (FIFO_CTRL4.DEC_TS_BATCH). 1 = one
                                    ///< per sample (PLAN T1); 8 cuts I2C load by ~30 % if the bus is slow.
  double temperature_batch_hz = 12.5;  ///< Temperature words in the FIFO: 0 (off) | 1.6 | 12.5 | 52 [Hz].
  int fifo_burst_words = 32;        ///< Max FIFO words per I2C read transaction (address wraps 0x7E -> 0x78).
  bool pair_by_tag_counter = true;  ///< Group gyro/accel/timestamp words into one sample by TAG_CNT (DS Table 195);
                                    ///< false = a timestamp word starts a new sample (Linux st_lsm6dsx order).
  ChipClockConfig clock;            ///< `imu.driver.clock`.
  ImuSimConfig sim;                 ///< `imu.driver.sim` (backend sim only).
  ImuReplayConfig replay;           ///< `imu.driver.replay` (backend replay only).
};

/// Running counters since start(). Copy returned by stats(); thread-safe.
struct ImuStats {
  uint64_t samples = 0;             ///< Samples delivered by read().
  double rate_hz_estimate = kNaN;   ///< Sample rate over the last 1 s of t_ns [Hz].
  int fifo_level_max = 0;           ///< Highest FIFO fill seen [words].
  int fifo_watermark_words = 0;     ///< Configured watermark (for display).
  uint64_t fifo_overruns = 0;       ///< FIFO_STATUS2 overrun flags seen (chip lost data). An abort rule (T5).
  uint64_t queue_overruns = 0;      ///< Samples lost because the internal host queue was full.
  uint64_t i2c_errors = 0;          ///< Transactions failed after all retries.
  uint64_t i2c_retries = 0;         ///< Retries performed.
  uint64_t unknown_tags = 0;        ///< FIFO words with tags other than gyro/accel/timestamp/temperature.
  uint64_t timestamp_regressions = 0;  ///< Chip timestamp words going backwards (not wrap).
  uint64_t gaps = 0;                ///< Intervals > gap_factor / odr.
  int64_t max_gap_ns = 0;           ///< Largest sample interval seen [ns].
  uint64_t acc_saturations = 0;     ///< Samples with kImuFlagAccSaturated.
  uint64_t gyro_saturations = 0;    ///< Samples with kImuFlagGyroSaturated.
  double nominal_tick_ns = kNaN;    ///< Trimmed tick period used by the clock mapper [ns].
  // --- T1 additive fields ---
  int freq_fine = 0;                ///< INTERNAL_FREQ_FINE read at start() (int8; 0.15 %/LSB).
  uint64_t fifo_words = 0;          ///< FIFO words read (all tags).
  uint64_t fifo_full_flags = 0;     ///< Polls with FIFO_STATUS2.FIFO_FULL_IA set (an overrun is imminent).
  uint64_t partial_slots = 0;       ///< Samples missing gyro or accel (stale value repeated; flagged).
  uint64_t slots_without_timestamp = 0;  ///< Samples whose time was extrapolated from the previous timestamp word
                                         ///< (expected 0 with timestamp_decimation = 1).
  uint64_t torn_timestamp_reads = 0;  ///< Bracketed TIMESTAMP reads rejected by the double-read consistency check.
  uint64_t temperature_words = 0;   ///< Temperature FIFO words decoded.
  ClockFitStats clock;              ///< ChipClockMapper diagnostics.
  TimeNs last_t_ns = kInvalidTime;
};

/// Abstract IMU. See the file comment for lifecycle and threading.
class ImuBackend {
 public:
  virtual ~ImuBackend() = default;

  virtual const ImuConfig& config() const = 0;

  /// Probe (WHO_AM_I), soft reset, configure, verify read-back, start the poll thread. kDeviceError on a
  /// WHO_AM_I mismatch or read-back mismatch (message names the register and both values).
  virtual Status start() = 0;

  /// Stops the poll thread and puts the chip in power-down. Idempotent.
  virtual Status stop() = 0;

  /// Appends available samples (time order, t_ns strictly increasing) to `out`, waiting up to `timeout` for at least
  /// one. Returns the number appended; 0 on timeout, after stop() or on a fatal error (then status() is not OK).
  virtual std::size_t read(std::vector<ImuSample>& out, std::chrono::nanoseconds timeout) = 0;

  /// Appends the raw clock brackets collected since the last call (non-blocking), for raw/imu0/chip.csv.
  virtual std::size_t read_brackets(std::vector<ClockBracket>& out) = 0;

  virtual ImuStats stats() const = 0;

  /// OK while healthy; the first fatal error otherwise (error budget exceeded, device gone).
  virtual Status status() const = 0;
};

/// Creates the backend selected by `config.backend`; does not start it.
/// @param bus optional bus injection for lsm6dsox_i2c / sim (tests): if null, lsm6dsox_i2c opens
///            make_linux_i2c_bus(config.i2c_bus) and sim creates T1's Lsm6dsoxSim. Ignored by replay.
/// Implemented by T1 (common/drivers/imu/imu_factory.cpp; T0 ships a stub).
Result<std::unique_ptr<ImuBackend>> make_imu(const ImuConfig& config, std::unique_ptr<I2cBus> bus = nullptr);

}  // namespace rig
