// SPDX-License-Identifier: MIT
// T0 STUB of make_imu(). T1 replaces this file with the real factory (lsm6dsox_i2c/sim/replay).
#include "rig/drivers/imu_backend.hpp"

namespace rig {

Result<std::unique_ptr<ImuBackend>> make_imu(const ImuConfig& config, std::unique_ptr<I2cBus> /*bus*/) {
  return unimplemented("make_imu(" + config.id + "): IMU backends not implemented yet (T1)");
}

}  // namespace rig
