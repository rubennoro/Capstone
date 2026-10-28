// SPDX-License-Identifier: MIT
// make_imu() (rig/drivers/imu_backend.hpp), T1: lsm6dsox_i2c | sim | replay.
#include "rig/drivers/imu/imu_replay.hpp"
#include "rig/drivers/imu/lsm6dsox_driver.hpp"
#include "rig/drivers/imu/lsm6dsox_sim.hpp"
#include "rig/drivers/imu_backend.hpp"

namespace rig {

Result<std::unique_ptr<ImuBackend>> make_imu(const ImuConfig& config, std::unique_ptr<I2cBus> bus) {
  switch (config.backend) {
    case ImuBackendKind::kLsm6dsoxI2c: {
      if (!bus) {
        RIG_ASSIGN_OR_RETURN(bus, make_linux_i2c_bus(config.i2c_bus));
      }
      return make_lsm6dsox(config, std::move(bus), real_imu_clock());
    }
    case ImuBackendKind::kSim: {
      // The same driver code against the register simulator, paced by the real clock.
      if (!bus)
        bus = std::make_unique<Lsm6dsoxSim>(Lsm6dsoxSimOptions::from_config(config.sim, config.i2c_addr),
                                            real_imu_clock());
      return make_lsm6dsox(config, std::move(bus), real_imu_clock());
    }
    case ImuBackendKind::kReplay:
      return make_imu_replay(config);
  }
  return invalid_argument("make_imu(" + config.id + "): unknown backend");
}

}  // namespace rig
