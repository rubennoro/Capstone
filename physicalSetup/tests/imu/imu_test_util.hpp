// SPDX-License-Identifier: MIT
// Helpers for tests/imu: the real driver against Lsm6dsoxSim on a virtual clock.
#pragma once

#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <chrono>
#include <cstdint>
#include <cstring>
#include <memory>
#include <vector>

#include "rig/drivers/imu/imu_clock.hpp"
#include "rig/drivers/imu/lsm6dsox_driver.hpp"
#include "rig/drivers/imu/lsm6dsox_regs.hpp"
#include "rig/drivers/imu/lsm6dsox_sim.hpp"
#include "rig/drivers/imu_backend.hpp"

namespace rig::test {

inline constexpr int64_t kSec = 1000000000;

/// A driver + simulator pair on a virtual clock. `sim` stays valid while `imu` lives (the driver owns it).
struct SimRig {
  std::shared_ptr<VirtualImuClock> clock;
  Lsm6dsoxSim* sim = nullptr;
  std::unique_ptr<ImuBackend> imu;
};

inline ImuConfig test_config() {
  ImuConfig c;
  c.backend = ImuBackendKind::kSim;
  c.queue_capacity = 200000;  // the virtual clock runs much faster than real time
  return c;
}

inline SimRig make_rig(const ImuConfig& c, Lsm6dsoxSimOptions o = {}) {
  SimRig r;
  r.clock = std::make_shared<VirtualImuClock>();
  // The poll thread parks 1 s after creation until run_for() raises the limit (start() needs ~0.1 s of sleeps),
  // so runs are deterministic regardless of how the test thread is scheduled.
  r.clock->set_limit(r.clock->now_ns() + kSec);
  auto sim = std::make_unique<Lsm6dsoxSim>(o, r.clock);
  r.sim = sim.get();
  auto imu = make_lsm6dsox(c, std::move(sim), r.clock);
  EXPECT_TRUE(imu.ok()) << imu.status().to_string();
  if (imu.ok()) r.imu = std::move(imu).value();
  return r;
}

/// Lets the driver run for `seconds` of virtual time (from now) and appends everything it produced.
inline void run_for(SimRig& r, double seconds, std::vector<ImuSample>& out) {
  r.clock->set_limit(std::max(r.clock->limit(), r.clock->now_ns()) + static_cast<int64_t>(seconds * 1e9));
  const auto wall_deadline = std::chrono::steady_clock::now() + std::chrono::seconds(300);
  for (;;) {
    const std::size_t n = r.imu->read(out, std::chrono::milliseconds(20));
    if (n == 0 && (r.clock->wait_until_parked(0) || !r.imu->status().ok())) {
      r.imu->read(out, std::chrono::milliseconds(0));
      break;
    }
    if (std::chrono::steady_clock::now() > wall_deadline) {
      ADD_FAILURE() << "run_for: wall-clock timeout";
      break;
    }
  }
}

/// Builds one FIFO word.
inline std::array<uint8_t, 7> word(uint8_t sensor, uint8_t cnt, int16_t x = 0, int16_t y = 0, int16_t z = 0) {
  std::array<uint8_t, 7> w{};
  w[0] = static_cast<uint8_t>((sensor << 3) | ((cnt & 3) << 1));
  const int16_t v[3] = {x, y, z};
  for (int i = 0; i < 3; ++i) {
    const uint16_t u = static_cast<uint16_t>(v[i]);
    w[static_cast<std::size_t>(1 + 2 * i)] = static_cast<uint8_t>(u & 0xFF);
    w[static_cast<std::size_t>(2 + 2 * i)] = static_cast<uint8_t>(u >> 8);
  }
  return w;
}
inline std::array<uint8_t, 7> ts_word(uint8_t cnt, uint32_t ticks) {
  std::array<uint8_t, 7> w{};
  w[0] = static_cast<uint8_t>((lsm6dsox::kTagTimestamp << 3) | ((cnt & 3) << 1));
  for (int i = 0; i < 4; ++i) w[static_cast<std::size_t>(1 + i)] = static_cast<uint8_t>(ticks >> (8 * i));
  return w;
}
inline std::vector<uint8_t> flat(const std::vector<std::array<uint8_t, 7>>& ws) {
  std::vector<uint8_t> b;
  for (const auto& w : ws) b.insert(b.end(), w.begin(), w.end());
  return b;
}

}  // namespace rig::test
