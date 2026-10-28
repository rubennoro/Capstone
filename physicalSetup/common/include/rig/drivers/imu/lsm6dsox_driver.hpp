// SPDX-License-Identifier: MIT
/// @file rig/drivers/imu/lsm6dsox_driver.hpp
/// @brief LSM6DSOX userspace FIFO driver over an I2cBus (PLAN T1), plus the pieces ec-imu-probe reuses.
///
/// start(): probe WHO_AM_I -> FIFO to bypass -> SW_RESET (wait 50 ms, poll until clear) -> BOOT (wait 50 ms, poll)
/// -> verify reset defaults -> read INTERNAL_FREQ_FINE -> write the configuration (lsm6dsox_config_sequence) ->
/// read back every written register and compare -> clear the latched overrun -> two clock brackets -> start the
/// poll thread. (Reset order and waits: Linux st_lsm6dsx_reset_device; register facts: lsm6dsox_regs.hpp.)
///
/// Poll thread (absolute-time loop, poll_period_ms): every poll reads FIFO_STATUS1/2 then the FIFO words in bursts
/// of fifo_burst_words from 0x78; every bracket_period_ms (every poll during the first 2 s) a bracketed read of
/// TIMESTAMP0..3, done twice back to back to detect torn multi-byte reads. Chip ticks -> MONOTONIC_RAW through
/// ChipClockMapper. Errors: every transaction is retried i2c_retries times; failures count against
/// i2c_error_budget per 10 s, beyond which status() turns fatal and read() returns 0.
/// Thread placement: the poll thread inherits the CPU affinity / scheduling of the thread that calls start()
/// (Linux pthread semantics), so CaptureService should call start() from the pinned IMU thread.
#pragma once

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "rig/drivers/i2c_bus.hpp"
#include "rig/drivers/imu/imu_clock.hpp"
#include "rig/drivers/imu_backend.hpp"
#include "rig/status.hpp"

namespace rig {

/// One register write of the configuration, with what it means.
struct Lsm6dsoxRegWrite {
  uint8_t reg = 0;
  uint8_t value = 0;
  const char* name = "";
  std::string meaning;
};

/// Result of reading back one configured register.
struct Lsm6dsoxReadback {
  uint8_t reg = 0;
  const char* name = "";
  uint8_t expected = 0;
  uint8_t actual = 0;
  bool ok() const { return expected == actual; }
};

/// Validates `c` (ODR, full scales, watermark, decimation, periods) and returns the ordered configuration writes
/// (applied after reset; FIFO_CTRL4 last, which starts batching). kInvalidArgument names the field.
Result<std::vector<Lsm6dsoxRegWrite>> lsm6dsox_config_sequence(const ImuConfig& c);

/// Reset + configure + readback on `bus` (the start() sequence without the thread). Fills `readback` and
/// `freq_fine` when non-null. kDeviceError on a WHO_AM_I mismatch, a reset that does not complete, or a readback
/// mismatch (message: register name, address, expected and actual values).
Status lsm6dsox_reset_and_configure(I2cBus& bus, const ImuConfig& c, ImuClock& clock,
                                    std::vector<Lsm6dsoxReadback>* readback, int8_t* freq_fine);

/// Powers the chip down (CTRL1_XL = CTRL2_G = 0, FIFO to bypass). Best effort.
Status lsm6dsox_power_down(I2cBus& bus, uint8_t addr7);

/// The LSM6DSOX backend. Construct through make_lsm6dsox() (or make_imu()).
class Lsm6dsoxDriver;

/// Creates the driver on `bus` with `clock` (nullptr = real clock). Validates the config; does not touch the bus.
Result<std::unique_ptr<ImuBackend>> make_lsm6dsox(const ImuConfig& c, std::unique_ptr<I2cBus> bus,
                                                   std::shared_ptr<ImuClock> clock = nullptr);

/// Readback table of the last successful start() of an LSM6DSOX backend (empty for other backends).
std::vector<Lsm6dsoxReadback> lsm6dsox_last_readback(const ImuBackend& backend);

/// Decoder statistics (tag counts, odd parity count) of an LSM6DSOX backend, for ec-imu-probe; zeros otherwise.
struct Lsm6dsoxDiagnostics {
  uint64_t gyro_words = 0, accel_words = 0, timestamp_words = 0, temperature_words = 0, cfg_change_words = 0;
  uint64_t odd_parity_words = 0, lost_slots = 0, untimed_slots = 0;
  double learned_period_ticks = 0.0;
  uint64_t polls = 0, poll_overruns = 0;  ///< Poll loop iterations; iterations that started > 1 period late.
};
Lsm6dsoxDiagnostics lsm6dsox_diagnostics(const ImuBackend& backend);

/// clock_fit.json content for raw/<id>/clock_fit.json (T5 writes the file): fit, residual and width stats,
/// trim, counts. Stable keys (schema 1).
std::string clock_fit_json(const ImuConfig& c, const ImuStats& s);

/// Euclidean norm helper used by probes: mean |acc| of samples [m/s^2].
double mean_acc_norm(const std::vector<ImuSample>& v);

}  // namespace rig
