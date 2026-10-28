// SPDX-License-Identifier: MIT
// make_imu(): sim backend in real time, replay backend, lsm6dsox_i2c without a device node, clock_fit.json.
#include <gtest/gtest.h>

#include <nlohmann/json.hpp>

#include <filesystem>
#include <fstream>
#include <thread>

#include "rig/drivers/imu/imu_replay.hpp"
#include "rig/drivers/imu/lsm6dsox_driver.hpp"
#include "rig/drivers/imu/lsm6dsox_sim.hpp"
#include "rig/drivers/imu_backend.hpp"
#include "rig/time/clock.hpp"

namespace rig {
namespace {

TEST(ImuFactory, SimBackendRealTime) {
  ImuConfig c;
  c.backend = ImuBackendKind::kSim;
  c.sim.chip_clock_drift_ppm = 300;
  auto imu = make_imu(c);
  ASSERT_TRUE(imu.ok()) << imu.status().to_string();
  ASSERT_TRUE((*imu)->start().ok()) << (*imu)->status().to_string();
  std::vector<ImuSample> v;
  const auto end = std::chrono::steady_clock::now() + std::chrono::milliseconds(1500);
  while (std::chrono::steady_clock::now() < end) (*imu)->read(v, std::chrono::milliseconds(50));
  const ImuStats s = (*imu)->stats();
  ASSERT_TRUE((*imu)->stop().ok());
  EXPECT_GT(v.size(), 900u);  // ~1.4 s x 833 Hz after the reset waits
  for (std::size_t i = 1; i < v.size(); ++i) ASSERT_GT(v[i].t_ns, v[i - 1].t_ns);
  // Real clock: samples are recent (MONOTONIC_RAW domain).
  const TimeNs now = now_raw_ns();
  EXPECT_LT(now - v.back().t_ns, 200000000);
  EXPECT_GT(now - v.back().t_ns, -5000000);
  EXPECT_EQ(s.fifo_overruns, 0u);
  const auto j = nlohmann::json::parse(clock_fit_json(c, s));
  EXPECT_EQ(j["schema"], 1);
  EXPECT_EQ(j["method"], "lower_envelope");
  EXPECT_TRUE(j["fit"]["valid"].get<bool>());
  EXPECT_GT(j["brackets_added"].get<int>(), 10);
}

TEST(ImuFactory, ReplayBackend) {
  const std::string dir = std::string(RIG_TEST_TMP_ROOT) + "/imu_replay";
  std::filesystem::create_directories(dir);
  {
    std::ofstream f(dir + "/data.csv");
    f << "#timestamp [ns],w_RS_S_x [rad s^-1],w_RS_S_y [rad s^-1],w_RS_S_z [rad s^-1],a_RS_S_x [m s^-2],"
         "a_RS_S_y [m s^-2],a_RS_S_z [m s^-2]\n";
    for (int i = 0; i < 100; ++i) f << 1000000000LL + i * 1200000LL << ",0.1,0.2,0.3,0,0," << 9.8 << "\n";
  }
  ImuConfig c;
  c.backend = ImuBackendKind::kReplay;
  c.replay.path = dir;
  c.replay.realtime = false;
  c.replay.retime = false;
  auto imu = make_imu(c);
  ASSERT_TRUE(imu.ok()) << imu.status().to_string();
  ASSERT_TRUE((*imu)->start().ok());
  std::vector<ImuSample> v;
  while ((*imu)->read(v, std::chrono::milliseconds(10)) > 0) {
  }
  ASSERT_EQ(v.size(), 100u);
  EXPECT_EQ(v.front().t_ns, 1000000000LL);
  EXPECT_DOUBLE_EQ(v[5].gyro_rps.y(), 0.2);
  EXPECT_DOUBLE_EQ(v[5].acc_mps2.z(), 9.8);
  ASSERT_TRUE((*imu)->stop().ok());

  c.replay.realtime = true;
  c.replay.retime = true;
  auto rt = make_imu(c);
  ASSERT_TRUE(rt.ok());
  ASSERT_TRUE((*rt)->start().ok());
  const TimeNs t0 = now_raw_ns();
  v.clear();
  while (v.size() < 100 && now_raw_ns() - t0 < 2000000000) (*rt)->read(v, std::chrono::milliseconds(20));
  EXPECT_EQ(v.size(), 100u);
  EXPECT_GE(now_raw_ns() - t0, 100000000);  // 100 samples x 1.2 ms paced in real time
  EXPECT_NEAR(static_cast<double>(v.front().t_ns - t0), 0.0, 5e6);
  ASSERT_TRUE((*rt)->stop().ok());

  c.replay.path = dir + "/missing";
  EXPECT_EQ(make_imu(c).status().code(), StatusCode::kNotFound);
}

TEST(ImuFactory, LinuxBusMissingDevice) {
  ImuConfig c;
  c.backend = ImuBackendKind::kLsm6dsoxI2c;
  c.i2c_bus = 250;  // no /dev/i2c-250
  auto imu = make_imu(c);
  ASSERT_FALSE(imu.ok());
  EXPECT_EQ(imu.status().code(), StatusCode::kNotFound);
  EXPECT_NE(imu.status().message().find("/dev/i2c-250"), std::string::npos);
  EXPECT_EQ(make_linux_i2c_bus(-1).status().code(), StatusCode::kInvalidArgument);
}

TEST(ImuFactory, SimBackendDoesNotKeepTruth) {
  // R1 F2: the long-running "sim" backend must not grow the ground-truth log.
  ImuSimConfig sc;
  EXPECT_FALSE(Lsm6dsoxSimOptions::from_config(sc, 0x6A).record_truth);
  EXPECT_TRUE(Lsm6dsoxSimOptions{}.record_truth);  // tests opt in by default construction
}

TEST(ImuFactory, InvalidConfigRejectedBeforeTouchingTheBus) {
  ImuConfig c;
  c.backend = ImuBackendKind::kSim;
  c.odr_hz = 1000;
  EXPECT_EQ(make_imu(c).status().code(), StatusCode::kInvalidArgument);
}

}  // namespace
}  // namespace rig
