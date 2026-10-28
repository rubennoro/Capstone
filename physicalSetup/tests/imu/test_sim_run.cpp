// SPDX-License-Identifier: MIT
// The real driver against Lsm6dsoxSim on a virtual clock: 60 s at 833 Hz (zero loss, rate, monotonic, 32-bit
// counter wrap), clock mapping under +-2 % drift with jitter, scale and axes for every full scale, saturation,
// timestamp decimation, slot-order robustness.
#include <gtest/gtest.h>

#include <cmath>
#include <map>
#include <unordered_map>

#include "imu_test_util.hpp"

namespace rig {
namespace {

using test::make_rig;
using test::run_for;
using test::test_config;

constexpr double kPi = 3.14159265358979323846;

struct MapErr {
  double max_us = 0, mean_us = 0;
  std::size_t n = 0;
};

// |t_ns - truth| for samples later than `skip_s` after the first sample (truth matched by chip ticks).
MapErr mapping_error(const std::vector<ImuSample>& v, const std::vector<Lsm6dsoxSimTruth>& truth, double skip_s) {
  std::unordered_map<int64_t, TimeNs> t_of;
  for (const auto& tr : truth) t_of[tr.ticks] = tr.t_host_ns;
  MapErr e;
  double sum = 0;
  for (const ImuSample& s : v) {
    if (static_cast<double>(s.t_ns - v.front().t_ns) < skip_s * 1e9) continue;
    auto it = t_of.find(s.chip_ticks);
    if (it == t_of.end()) continue;
    const double err = std::abs(static_cast<double>(s.t_ns - it->second)) / 1000.0;
    e.max_us = std::max(e.max_us, err);
    sum += err;
    ++e.n;
  }
  e.mean_us = e.n ? sum / static_cast<double>(e.n) : 0;
  return e;
}

TEST(SimRun, SixtySecondsAt833HzZeroLoss) {
  ImuConfig c = test_config();
  Lsm6dsoxSimOptions o;
  o.counter_start = 0xFFFFFFFFu - 40000u * 10u;  // the 32-bit counter wraps 10 s into the run
  o.accel_noise_std_mps2 = 0.01;
  o.gyro_noise_std_rps = 0.001;
  auto r = make_rig(c, o);
  ASSERT_TRUE(r.imu);
  ASSERT_TRUE(r.imu->start().ok()) << r.imu->status().to_string();
  std::vector<ImuSample> v;
  run_for(r, 60.0, v);
  const ImuStats s = r.imu->stats();
  ASSERT_TRUE(r.imu->stop().ok());
  const auto truth = r.sim->drain_truth();

  ASSERT_GT(v.size(), 49000u);
  // Zero loss: every generated slot up to the last delivered one was delivered exactly once, in order.
  std::size_t j = 0;
  while (j < truth.size() && truth[j].ticks < v.front().chip_ticks) ++j;
  EXPECT_LE(j, 2u) << "samples lost at the start";
  for (std::size_t i = 0; i < v.size(); ++i, ++j) {
    ASSERT_LT(j, truth.size());
    ASSERT_EQ(v[i].chip_ticks, truth[j].ticks) << "sample " << i;
  }
  EXPECT_LE(truth.size() - j, 40u) << "only what is still in the FIFO may be missing at the end";
  // Monotonic, no gaps, flags clean.
  for (std::size_t i = 1; i < v.size(); ++i) ASSERT_GT(v[i].t_ns, v[i - 1].t_ns) << i;
  for (const ImuSample& x : v) ASSERT_EQ(x.flags & ~kImuFlagTimeExtrapolated, 0u);
  // Rate within 0.1 % of the configured 833 Hz (sim: 40 kHz / 48 = 833.33 Hz).
  const double rate = static_cast<double>(v.size() - 1) * 1e9 / static_cast<double>(v.back().t_ns - v.front().t_ns);
  EXPECT_NEAR(rate / 833.0, 1.0, 0.001) << rate;
  EXPECT_NEAR(s.rate_hz_estimate / 833.0, 1.0, 0.002);
  EXPECT_EQ(s.samples, v.size());
  EXPECT_EQ(s.fifo_overruns, 0u);
  EXPECT_EQ(s.queue_overruns, 0u);
  EXPECT_EQ(s.gaps, 0u);
  EXPECT_EQ(s.partial_slots, 0u);
  EXPECT_EQ(s.slots_without_timestamp, 0u);
  EXPECT_EQ(s.unknown_tags, 0u);
  EXPECT_EQ(s.timestamp_regressions, 0u);
  EXPECT_EQ(s.i2c_errors, 0u);
  EXPECT_GT(s.fifo_level_max, 10);
  EXPECT_LT(s.fifo_level_max, 100);
  EXPECT_GT(s.temperature_words, 700u);
  EXPECT_NEAR(v.back().temp_c, 31.5, 0.01);
  EXPECT_GT(v.back().chip_ticks, int64_t{1} << 32);  // unwrapped past the 32-bit wrap
  const MapErr e = mapping_error(v, truth, 1.0);
  EXPECT_GT(e.n, 45000u);
  EXPECT_LT(e.max_us, 50.0);
  std::printf("60 s run: %zu samples, rate %.4f Hz, map error max %.1f us mean %.1f us, fifo max %d, residual "
              "p99 %.1f us\n",
              v.size(), rate, e.max_us, e.mean_us, s.fifo_level_max, s.clock.residual_ns.p99 / 1000.0);
  // Gravity at rest within the accel noise.
  Eigen::Vector3d mean = Eigen::Vector3d::Zero();
  for (const ImuSample& x : v) mean += x.acc_mps2;
  mean /= static_cast<double>(v.size());
  EXPECT_NEAR(mean.z(), kStandardGravity, 0.003);
  EXPECT_NEAR(mean.x(), 0.0, 0.003);
  // Brackets for chip.csv.
  std::vector<ClockBracket> br;
  EXPECT_GT(r.imu->read_brackets(br), 700u);
}

class SimDrift : public ::testing::TestWithParam<double> {};

TEST_P(SimDrift, ClockMappingBelow50us) {
  ImuConfig c = test_config();
  Lsm6dsoxSimOptions o;
  o.drift_ppm = GetParam();
  o.freq_fine = -7;  // trim applied by the driver; drift is the residual error on top
  auto r = make_rig(c, o);
  ASSERT_TRUE(r.imu->start().ok());
  std::vector<ImuSample> v;
  run_for(r, 40.0, v);
  const ImuStats s = r.imu->stats();
  ASSERT_TRUE(r.imu->stop().ok());
  const MapErr e = mapping_error(v, r.sim->drain_truth(), 1.0);
  EXPECT_GT(e.n, 30000u);
  EXPECT_LT(e.max_us, 50.0);
  EXPECT_FALSE(s.clock.fit.slope_clamped);
  EXPECT_NEAR(s.clock.fit.slope_ns_per_tick / r.sim->true_tick_ns(), 1.0, 5e-6);
  EXPECT_EQ(s.freq_fine, -7);
  EXPECT_NEAR(s.nominal_tick_ns, 25000.0 / (1.0 - 7 * 0.0015), 1e-6);
  EXPECT_EQ(s.gaps, 0u);  // a 2 % drift is not a gap
  std::printf("drift %+.0f ppm: map error max %.1f us mean %.1f us (n=%zu), residual p99 %.1f us\n", GetParam(),
              e.max_us, e.mean_us, e.n, s.clock.residual_ns.p99 / 1000.0);
}

INSTANTIATE_TEST_SUITE_P(Drift, SimDrift, ::testing::Values(-20000.0, 20000.0));

TEST(SimRun, TimestampAfterDataStillPairedByTagCounter) {
  ImuConfig c = test_config();
  Lsm6dsoxSimOptions o;
  o.timestamp_after_data = true;
  auto r = make_rig(c, o);
  ASSERT_TRUE(r.imu->start().ok());
  std::vector<ImuSample> v;
  run_for(r, 5.0, v);
  ASSERT_TRUE(r.imu->stop().ok());
  const MapErr e = mapping_error(v, r.sim->drain_truth(), 1.0);
  EXPECT_GT(e.n, 3000u);
  EXPECT_LT(e.max_us, 50.0);
  EXPECT_EQ(r.imu->stats().partial_slots, 0u);
}

TEST(SimRun, TimestampDecimation8) {
  ImuConfig c = test_config();
  c.timestamp_decimation = 8;
  auto r = make_rig(c, {});
  ASSERT_TRUE(r.imu->start().ok());
  std::vector<ImuSample> v;
  run_for(r, 10.0, v);
  const ImuStats s = r.imu->stats();
  ASSERT_TRUE(r.imu->stop().ok());
  const MapErr e = mapping_error(v, r.sim->drain_truth(), 1.0);
  EXPECT_GT(e.n, 7000u);
  EXPECT_LT(e.max_us, 50.0);
  EXPECT_GT(s.slots_without_timestamp, 6000u);
  EXPECT_EQ(s.gaps, 0u);
}

TEST(SimRun, MotionMatchesTruthPerAxis) {
  ImuConfig c = test_config();
  Lsm6dsoxSimOptions o;
  o.motion = true;
  o.gravity_body = Eigen::Vector3d(1.0, -9.5, 2.0);
  auto r = make_rig(c, o);
  ASSERT_TRUE(r.imu->start().ok());
  std::vector<ImuSample> v;
  run_for(r, 5.0, v);
  ASSERT_TRUE(r.imu->stop().ok());
  std::unordered_map<int64_t, Lsm6dsoxSimTruth> by;
  for (const auto& t : r.sim->drain_truth()) by[t.ticks] = t;
  const double acc_lsb = 0.244e-3 * kStandardGravity, gyro_lsb = 70e-3 * kPi / 180.0;
  std::size_t n = 0;
  for (const ImuSample& x : v) {
    auto it = by.find(x.chip_ticks);
    ASSERT_NE(it, by.end());
    for (int i = 0; i < 3; ++i) {
      ASSERT_NEAR(x.acc_mps2[i], it->second.acc_mps2[i], 0.51 * acc_lsb) << "axis " << i;
      ASSERT_NEAR(x.gyro_rps[i], it->second.gyro_rps[i], 0.51 * gyro_lsb) << "axis " << i;
    }
    ++n;
  }
  EXPECT_GT(n, 4000u);
}

class FullScale : public ::testing::TestWithParam<std::pair<int, int>> {};

TEST_P(FullScale, ScaleWithinOneLsb) {
  const auto [g, dps] = GetParam();
  ImuConfig c = test_config();
  c.accel_fs_g = g;
  c.gyro_fs_dps = dps;
  Lsm6dsoxSimOptions o;
  o.gravity_body = Eigen::Vector3d(0.3 * g * kStandardGravity, -0.5 * g * kStandardGravity, 0.9 * g * kStandardGravity);
  o.gyro_bias_rps = Eigen::Vector3d(0.25, -0.5, 0.9) * dps * kPi / 180.0;
  auto r = make_rig(c, o);
  ASSERT_TRUE(r.imu->start().ok());
  std::vector<ImuSample> v;
  run_for(r, 0.5, v);
  ASSERT_TRUE(r.imu->stop().ok());
  ASSERT_GT(v.size(), 200u);
  const double acc_lsb = lsm6dsox::accel_fs(g)->mg_per_lsb * 1e-3 * kStandardGravity;
  const double gyro_lsb = lsm6dsox::gyro_fs(dps)->mdps_per_lsb * 1e-3 * kPi / 180.0;
  for (int i = 0; i < 3; ++i) {
    EXPECT_NEAR(v.back().acc_mps2[i], o.gravity_body[i], acc_lsb) << g << " g axis " << i;
    EXPECT_NEAR(v.back().gyro_rps[i], o.gyro_bias_rps[i], gyro_lsb) << dps << " dps axis " << i;
  }
  EXPECT_EQ(v.back().flags & (kImuFlagAccSaturated | kImuFlagGyroSaturated), 0u);
}

INSTANTIATE_TEST_SUITE_P(All, FullScale,
                         ::testing::Values(std::make_pair(2, 125), std::make_pair(4, 250), std::make_pair(8, 500),
                                           std::make_pair(16, 1000), std::make_pair(8, 2000)));

TEST(SimRun, SaturationDetected) {
  ImuConfig c = test_config();
  c.accel_fs_g = 2;
  c.gyro_fs_dps = 125;
  Lsm6dsoxSimOptions o;
  o.gravity_body = Eigen::Vector3d(25.0, 0.0, 0.0);   // > 2 g
  o.gyro_bias_rps = Eigen::Vector3d(0.0, -3.0, 0.0);  // > 125 dps = 2.18 rad/s
  auto r = make_rig(c, o);
  ASSERT_TRUE(r.imu->start().ok());
  std::vector<ImuSample> v;
  run_for(r, 0.3, v);
  ASSERT_TRUE(r.imu->stop().ok());
  ASSERT_FALSE(v.empty());
  EXPECT_TRUE(v.back().flags & kImuFlagAccSaturated);
  EXPECT_TRUE(v.back().flags & kImuFlagGyroSaturated);
  EXPECT_EQ(r.imu->stats().acc_saturations, v.size());
  EXPECT_EQ(r.imu->stats().gyro_saturations, v.size());
  EXPECT_NEAR(v.back().acc_mps2.x(), 32767 * 0.061e-3 * kStandardGravity, 1e-9);
}

}  // namespace
}  // namespace rig
