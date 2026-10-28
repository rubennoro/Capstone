// SPDX-License-Identifier: MIT
/// @file rig/drivers/imu/lsm6dsox_sim.hpp
/// @brief Lsm6dsoxSim: a register-level LSM6DSOX on an I2cBus, for tests and the "sim" backend (PLAN T1).
///
/// Models (sources as in lsm6dsox_regs.hpp):
/// - the register file with reset defaults (DS Table 21), read-only registers, IF_INC auto-increment, SW_RESET
///   and BOOT (busy for a while; transactions NACK while booting), WHO_AM_I (configurable for the mismatch test);
/// - the FIFO: 512 words, continuous/bypass modes, a timestamp word + gyro + accel per slot with one TAG_CNT,
///   temperature words, overrun (oldest overwritten, FIFO_OVR_IA / FIFO_OVR_LATCHED, cleared by reading
///   FIFO_STATUS2), watermark and full flags, DIFF_FIFO level; a burst read from 0x78 wraps 0x7E -> 0x78 and pops
///   one word per 7 bytes (Linux st_lsm6dsx reads up to 4 words per transaction this way);
/// - the 32-bit timestamp counter at the trimmed tick (INTERNAL_FREQ_FINE) with an extra oscillator error
///   (drift_ppm) and an initial value; optional byte-by-byte sampling of TIMESTAMP0..3 (torn reads);
/// - I2C timing on a virtual clock: per transaction a host pre-latency (fixed + exponential + rare preemption
///   spikes), the bytes at bus_hz, and a post-latency. On the real clock transactions are instantaneous;
/// - faults: random NACKs, scheduled failures, device absent, a register that ignores writes;
/// - synthetic motion: gravity along a configurable body vector + sinusoids + white noise; the exact generated
///   values are recorded as ground truth.
/// SIMULATOR ASSUMPTIONS (not chip facts; [H] items in BRINGUP): the slot period is an integer number of ticks
/// (OdrEntry::sim_period_ticks), the timestamp word comes first in a slot, temperature bytes 0..1 = OUT_TEMP.
/// Threading: the I2cBus methods are called from one thread; test hooks and truth access are mutex-protected.
#pragma once

#include <cstdint>
#include <memory>
#include <mutex>
#include <random>
#include <string>
#include <vector>

#include <Eigen/Core>

#include "rig/drivers/i2c_bus.hpp"
#include "rig/drivers/imu/imu_clock.hpp"

namespace rig {

struct ImuSimConfig;

struct Lsm6dsoxSimOptions {
  uint8_t addr7 = 0x6A;
  uint8_t who_am_i = 0x6C;
  int8_t freq_fine = 0;               ///< INTERNAL_FREQ_FINE value.
  double drift_ppm = 0.0;             ///< Oscillator error beyond the trim [ppm]; + = chip clock fast.
  uint32_t counter_start = 0;         ///< Counter value when TIMESTAMP_EN is set.
  double bus_hz = 400000.0;
  // Host latency model (virtual clock only) [ns].
  int64_t pre_latency_ns = 40000;
  int64_t post_latency_ns = 30000;
  double latency_exp_mean_ns = 15000.0;
  double spike_probability = 0.005;
  int64_t spike_max_ns = 1500000;
  // Data.
  Eigen::Vector3d gravity_body{0.0, 0.0, 9.80665};  ///< Specific force at rest in chip axes [m/s^2].
  Eigen::Vector3d gyro_bias_rps{0.0, 0.0, 0.0};
  bool motion = false;
  Eigen::Vector3d accel_amp_mps2{1.5, 1.0, 0.5};
  Eigen::Vector3d gyro_amp_rps{1.0, 0.6, 0.3};
  Eigen::Vector3d motion_freq_hz{0.7, 1.1, 1.9};
  double accel_noise_std_mps2 = 0.0;
  double gyro_noise_std_rps = 0.0;
  double temperature_c = 31.5;
  uint64_t seed = 1;
  // Faults / variants.
  double nack_rate = 0.0;             ///< Probability a transaction NACKs at the address phase.
  bool tear_timestamp_reads = false;  ///< TIMESTAMP0..3 bytes sampled one byte time apart (no latch).
  bool timestamp_after_data = false;  ///< Slot order gyro, accel, timestamp (pairing robustness test).
  int fifo_capacity_words = 512;
  int64_t boot_busy_ns = 10000000;    ///< BOOT takes this long; transactions NACK meanwhile.
  int64_t reset_busy_ns = 50000;      ///< SW_RESET reads back 1 this long.
  int stuck_register = -1;            ///< Register address whose writes are ignored (readback test); -1 none.
  bool record_truth = true;  ///< Keep ground truth (grows ~50 KB/s at 833 Hz until drain_truth()).

  /// From the rig.yaml `imu.driver.sim` block. Sets record_truth = false (long-running "sim" backend).
  static Lsm6dsoxSimOptions from_config(const ImuSimConfig& c, int i2c_addr);
};

/// Ground truth of one generated slot.
struct Lsm6dsoxSimTruth {
  int64_t ticks = 0;        ///< Unwrapped counter value of the slot (= the timestamp word value, unwrapped).
  TimeNs t_host_ns = 0;     ///< Host time of the slot.
  Eigen::Vector3d gyro_rps = Eigen::Vector3d::Zero();   ///< Value before quantization.
  Eigen::Vector3d acc_mps2 = Eigen::Vector3d::Zero();
};

class Lsm6dsoxSim final : public I2cBus {
 public:
  Lsm6dsoxSim(Lsm6dsoxSimOptions opt, std::shared_ptr<ImuClock> clock);
  ~Lsm6dsoxSim() override;

  Status write_read(uint8_t addr7, const uint8_t* w, std::size_t wn, uint8_t* r, std::size_t rn) override;
  Status write(uint8_t addr7, const uint8_t* w, std::size_t wn) override;
  std::string name() const override { return "sim:lsm6dsox"; }

  // ---- test hooks (thread-safe) ----
  void fail_next(int n);               ///< The next n transactions NACK.
  void set_nack_rate(double p);
  void set_present(bool present);      ///< false: every transaction NACKs (device unplugged).
  void inject_unknown_tag();           ///< Next generated slot gets an extra word with an unused tag (0x1F).
  uint8_t peek(uint8_t reg) const;     ///< Register file content (no side effects).
  std::vector<Lsm6dsoxSimTruth> drain_truth();
  /// Host time [ns] at which the unwrapped counter reaches `ticks` (exact model), for clock-mapping tests.
  double host_time_of_ticks(double ticks) const;
  uint64_t overwritten_words() const;
  uint64_t transactions() const;
  uint64_t nacks() const;
  int fifo_level() const;
  /// Duration of one counter tick in host ns (trim and drift applied).
  double true_tick_ns() const;

 private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

}  // namespace rig
