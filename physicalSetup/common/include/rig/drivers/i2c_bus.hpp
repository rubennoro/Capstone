// SPDX-License-Identifier: MIT
/// @file rig/drivers/i2c_bus.hpp
/// @brief Abstract I2C master, so the IMU driver (T1) and the IMX519 focuser (T2) run against a register-level
/// simulator in tests and `/dev/i2c-N` on the Jetson. FROZEN CONTRACT (T0).
///
/// Addresses are 7-bit (e.g. LSM6DSOX 0x6A, IMX519 VCM 0x0C). Register addresses are 8-bit (both devices).
/// Threading: an I2cBus instance is NOT thread-safe; it is owned by one thread (the driver thread). Two drivers on
/// the same physical bus each open their own instance (the kernel serializes transfers).
/// Errors: kDeviceError = NACK / arbitration lost / EREMOTEIO; kTimeout; kIoError = other errno (text in message);
/// kInvalidArgument = bad sizes or address > 0x7F. Retries are the caller's policy (T1's counted error budget).
#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>

#include "rig/status.hpp"

namespace rig {

class I2cBus {
 public:
  virtual ~I2cBus() = default;

  /// Combined transaction with a repeated start: write `wn` bytes, then read `rn` bytes, no STOP in between
  /// (I2C_RDWR with two messages on Linux). `wn` = 0 performs a plain read. @param r buffer of at least rn bytes.
  virtual Status write_read(uint8_t addr7, const uint8_t* w, std::size_t wn, uint8_t* r, std::size_t rn) = 0;

  /// Plain write transaction (START, addr+W, bytes, STOP). For register writes, w[0] is the register address.
  virtual Status write(uint8_t addr7, const uint8_t* w, std::size_t wn) = 0;

  /// Human-readable identity for logs, e.g. "/dev/i2c-7" or "sim:lsm6dsox".
  virtual std::string name() const = 0;

  /// Reads `n` consecutive registers starting at `reg` (device auto-increment must be enabled, CTRL3_C.IF_INC on
  /// the LSM6DSOX).
  Status read_regs(uint8_t addr7, uint8_t reg, uint8_t* out, std::size_t n) { return write_read(addr7, &reg, 1, out, n); }

  /// Writes one 8-bit register.
  Status write_reg(uint8_t addr7, uint8_t reg, uint8_t value) {
    const uint8_t buf[2] = {reg, value};
    return write(addr7, buf, 2);
  }
};

/// Opens `/dev/i2c-<bus_number>` (implemented by T1 in common/drivers/i2c/linux_i2c_bus.cpp; I2C_RDWR).
/// Fails with kNotFound if the device node is absent and kIoError if it cannot be opened (permissions: user in group
/// `i2c`, see BRINGUP).
Result<std::unique_ptr<I2cBus>> make_linux_i2c_bus(int bus_number);

}  // namespace rig
