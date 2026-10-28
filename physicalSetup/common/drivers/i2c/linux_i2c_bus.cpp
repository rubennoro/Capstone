// SPDX-License-Identifier: MIT
// T0 STUB of make_linux_i2c_bus(). T1 replaces this file with the I2C_RDWR implementation.
#include <string>

#include "rig/drivers/i2c_bus.hpp"

namespace rig {

Result<std::unique_ptr<I2cBus>> make_linux_i2c_bus(int bus_number) {
  return unimplemented("/dev/i2c-" + std::to_string(bus_number) + ": LinuxI2cBus not implemented yet (T1)");
}

}  // namespace rig
