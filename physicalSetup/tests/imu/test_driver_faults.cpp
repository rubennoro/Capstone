// SPDX-License-Identifier: MIT
// Driver fault handling against the simulator: FIFO overrun, I2C NACK recovery and the error budget, WHO_AM_I
// mismatch, reset + configuration readback verification, a reset that never completes, torn counter reads,
// unknown tags, stop/restart.
#include <gtest/gtest.h>

#include <algorithm>
#include <unordered_map>

#include "imu_test_util.hpp"

namespace rig {
namespace {

using namespace lsm6dsox;  // NOLINT
using test::make_rig;
using test::run_for;
using test::test_config;

void expect_monotonic(const std::vector<ImuSample>& v) {
  for (std::size_t i = 1; i < v.size(); ++i) ASSERT_GT(v[i].t_ns, v[i - 1].t_ns) << i;
}

TEST(DriverFaults, FifoOverrunDetectedAndCounted) {
  ImuConfig c = test_config();
  c.poll_period_ms = 400;  // 3 words x 833 Hz x 0.4 s = 1000 words per poll > 512: the chip overwrites
  auto r = make_rig(c, {});
  ASSERT_TRUE(r.imu->start().ok());
  std::vector<ImuSample> v;
  run_for(r, 5.0, v);
  const ImuStats s = r.imu->stats();
  ASSERT_TRUE(r.imu->stop().ok());
  EXPECT_GE(s.fifo_overruns, 10u);
  EXPECT_GT(r.sim->overwritten_words(), 0u);
  EXPECT_EQ(s.fifo_level_max, kFifoCapacityWords);
  EXPECT_GT(s.gaps, 0u);
  EXPECT_GT(s.max_gap_ns, 100000000);
  std::size_t flagged = 0;
  for (const ImuSample& x : v) flagged += (x.flags & kImuFlagAfterGap) != 0;
  EXPECT_GE(flagged, s.fifo_overruns - 1);
  expect_monotonic(v);
  EXPECT_TRUE(r.imu->status().ok());  // overrun is reported upward (T5 aborts), not fatal to the driver
}

TEST(DriverFaults, RandomNacksAreRetriedWithoutLoss) {
  ImuConfig c = test_config();
  Lsm6dsoxSimOptions o;
  o.nack_rate = 0.05;
  auto r = make_rig(c, o);
  ASSERT_TRUE(r.imu->start().ok()) << r.imu->status().to_string();
  std::vector<ImuSample> v;
  run_for(r, 10.0, v);
  const ImuStats s = r.imu->stats();
  ASSERT_TRUE(r.imu->stop().ok());
  EXPECT_GT(s.i2c_retries, 100u);
  EXPECT_LE(s.i2c_errors, 2u);  // 0.05^4 per transaction
  EXPECT_TRUE(r.imu->status().ok());
  EXPECT_EQ(s.gaps, 0u);
  EXPECT_EQ(s.fifo_overruns, 0u);
  EXPECT_GT(v.size(), 8000u);
  expect_monotonic(v);
}

TEST(DriverFaults, FailedTransactionRecovers) {
  ImuConfig c = test_config();
  auto r = make_rig(c, {});
  ASSERT_TRUE(r.imu->start().ok());
  std::vector<ImuSample> v;
  run_for(r, 1.0, v);
  r.sim->fail_next(1 + c.i2c_retries);  // exactly one transaction fails after all retries
  run_for(r, 1.0, v);
  const ImuStats s = r.imu->stats();
  ASSERT_TRUE(r.imu->stop().ok());
  EXPECT_EQ(s.i2c_errors, 1u);
  EXPECT_EQ(s.i2c_retries, static_cast<uint64_t>(c.i2c_retries));
  EXPECT_TRUE(r.imu->status().ok());
  EXPECT_EQ(s.gaps, 0u);
  EXPECT_GT(v.size(), 1500u);
  expect_monotonic(v);
}

TEST(DriverFaults, DeviceGoneExceedsBudgetAndTurnsFatal) {
  ImuConfig c = test_config();
  auto r = make_rig(c, {});
  ASSERT_TRUE(r.imu->start().ok());
  std::vector<ImuSample> v;
  run_for(r, 1.0, v);
  r.sim->set_present(false);
  run_for(r, 2.0, v);
  const Status st = r.imu->status();
  EXPECT_EQ(st.code(), StatusCode::kDeviceError);
  EXPECT_NE(st.message().find("budget"), std::string::npos) << st.message();
  EXPECT_EQ(r.imu->stats().i2c_errors, static_cast<uint64_t>(c.i2c_error_budget + 1));
  std::vector<ImuSample> more;
  EXPECT_EQ(r.imu->read(more, std::chrono::milliseconds(10)), 0u);  // fatal: read() returns 0
  EXPECT_FALSE(r.imu->stop().ok());  // power-down cannot reach the chip
  EXPECT_TRUE(r.imu->stop().ok());   // idempotent
}

TEST(DriverFaults, WhoAmIMismatch) {
  Lsm6dsoxSimOptions o;
  o.who_am_i = 0x6B;  // e.g. an LSM6DSR / ISM330DHCX on the bus
  auto r = make_rig(test_config(), o);
  const Status st = r.imu->start();
  EXPECT_EQ(st.code(), StatusCode::kDeviceError);
  EXPECT_NE(st.message().find("WHO_AM_I"), std::string::npos);
  EXPECT_NE(st.message().find("0x6b"), std::string::npos) << st.message();
  EXPECT_NE(st.message().find("0x6c"), std::string::npos) << st.message();
  EXPECT_FALSE(r.imu->status().ok());
}

TEST(DriverFaults, WrongAddressIsNack) {
  ImuConfig c = test_config();
  c.i2c_addr = 0x6B;
  auto r = make_rig(c, {});  // sim answers at 0x6a only
  const Status st = r.imu->start();
  EXPECT_EQ(st.code(), StatusCode::kDeviceError);
  EXPECT_NE(st.message().find("WHO_AM_I read failed"), std::string::npos) << st.message();
}

TEST(DriverFaults, ResetAndReadbackVerified) {
  ImuConfig c = test_config();
  auto r = make_rig(c, {});
  // Leave junk from a previous user in registers the driver never writes: the soft reset must clear it.
  ASSERT_TRUE(r.sim->write_reg(0x6A, kRegCtrl6C, 0x10).ok());  // XL_HM_MODE = 1 (accel high-performance off)
  ASSERT_TRUE(r.sim->write_reg(0x6A, kRegCtrl7G, 0x80).ok());  // G_HM_MODE = 1
  ASSERT_EQ(r.sim->peek(kRegCtrl6C), 0x10);
  ASSERT_TRUE(r.imu->start().ok());
  EXPECT_EQ(r.sim->peek(kRegCtrl6C), 0x00);
  EXPECT_EQ(r.sim->peek(kRegCtrl7G), 0x00);
  auto seq = lsm6dsox_config_sequence(c);
  ASSERT_TRUE(seq.ok());
  for (const auto& w : *seq) EXPECT_EQ(r.sim->peek(w.reg), w.value) << w.name;
  const auto rb = lsm6dsox_last_readback(*r.imu);
  ASSERT_EQ(rb.size(), seq->size());
  for (const auto& x : rb) EXPECT_TRUE(x.ok()) << x.name;
  ASSERT_TRUE(r.imu->stop().ok());
  EXPECT_EQ(r.sim->peek(kRegCtrl1Xl), 0x00);  // powered down
  EXPECT_EQ(r.sim->peek(kRegCtrl2G), 0x00);
  EXPECT_EQ(r.sim->peek(kRegFifoCtrl4), 0x00);
}

TEST(DriverFaults, ReadbackMismatchNamesTheRegister) {
  Lsm6dsoxSimOptions o;
  o.stuck_register = kRegCtrl1Xl;
  auto r = make_rig(test_config(), o);
  const Status st = r.imu->start();
  EXPECT_EQ(st.code(), StatusCode::kDeviceError);
  EXPECT_NE(st.message().find("CTRL1_XL (0x10): wrote 0x7c, read 0x00"), std::string::npos) << st.message();
}

TEST(DriverFaults, BootThatNeverCompletes) {
  Lsm6dsoxSimOptions o;
  o.boot_busy_ns = 2000000000;  // 2 s; the driver waits 50 ms + polls for 200 ms
  auto r = make_rig(test_config(), o);
  const Status st = r.imu->start();
  EXPECT_EQ(st.code(), StatusCode::kDeviceError);
  EXPECT_NE(st.message().find("BOOT did not complete"), std::string::npos) << st.message();
}

TEST(DriverFaults, TornTimestampReadsRejected) {
  ImuConfig c = test_config();
  c.odr_hz = 416;  // a 100 kHz bus: slower byte times make carries during a read ~4x more likely
  Lsm6dsoxSimOptions o;
  o.tear_timestamp_reads = true;
  o.bus_hz = 100000;
  auto r = make_rig(c, o);
  ASSERT_TRUE(r.imu->start().ok());
  std::vector<ImuSample> v;
  run_for(r, 60.0, v);
  const ImuStats s = r.imu->stats();
  ASSERT_TRUE(r.imu->stop().ok());
  EXPECT_GT(s.torn_timestamp_reads, 3u);
  EXPECT_EQ(s.fifo_overruns, 0u);
  std::unordered_map<int64_t, TimeNs> t_of;
  for (const auto& tr : r.sim->drain_truth()) t_of[tr.ticks] = tr.t_host_ns;
  std::vector<double> err;
  for (const ImuSample& x : v)
    if (auto it = t_of.find(x.chip_ticks); it != t_of.end() && x.t_ns - v.front().t_ns > 1000000000)
      err.push_back(static_cast<double>(x.t_ns - it->second));
  ASSERT_GT(err.size(), 20000u);
  std::vector<double> sorted = err;
  std::sort(sorted.begin(), sorted.end());
  const double median = sorted[sorted.size() / 2];
  double worst = 0;
  for (double e : err) worst = std::max(worst, std::abs(e - median));
  // On a 100 kHz bus the latch sits ~50 us off the bracket centre: a constant offset (absorbed by the camera-IMU
  // time-offset calibration). What a torn read would break is the spread, so check that.
  EXPECT_LT(worst / 1000.0, 50.0);
  std::printf("torn reads rejected: %llu of %llu brackets, spread max %.1f us around median %.1f us\n",
              static_cast<unsigned long long>(s.torn_timestamp_reads),
              static_cast<unsigned long long>(s.clock.brackets_added), worst / 1000.0, median / 1000.0);
}

TEST(DriverFaults, UnknownTagCountedNoLoss) {
  auto r = make_rig(test_config(), {});
  ASSERT_TRUE(r.imu->start().ok());
  std::vector<ImuSample> v;
  run_for(r, 1.0, v);
  r.sim->inject_unknown_tag();
  run_for(r, 1.0, v);
  const ImuStats s = r.imu->stats();
  ASSERT_TRUE(r.imu->stop().ok());
  EXPECT_EQ(s.unknown_tags, 1u);
  EXPECT_EQ(s.gaps, 0u);
  EXPECT_EQ(s.partial_slots, 0u);
}

TEST(DriverFaults, StopRestartAndDoubleStart) {
  auto r = make_rig(test_config(), {});
  ASSERT_TRUE(r.imu->start().ok());
  EXPECT_EQ(r.imu->start().code(), StatusCode::kFailedPrecondition);
  std::vector<ImuSample> v;
  run_for(r, 0.5, v);
  ASSERT_TRUE(r.imu->stop().ok());
  ASSERT_TRUE(r.imu->stop().ok());
  const std::size_t first = v.size();
  EXPECT_GT(first, 300u);
  v.clear();
  r.clock->set_limit(r.clock->now_ns() + test::kSec);  // room for start()'s reset waits
  ASSERT_TRUE(r.imu->start().ok());
  run_for(r, 0.5, v);
  ASSERT_TRUE(r.imu->stop().ok());
  EXPECT_GT(v.size(), 300u);
  EXPECT_EQ(r.imu->stats().samples, v.size());  // counters restart
  expect_monotonic(v);
}

TEST(DriverFaults, HostQueueOverflowCounted) {
  ImuConfig c = test_config();
  c.queue_capacity = 100;
  auto r = make_rig(c, {});
  ASSERT_TRUE(r.imu->start().ok());
  r.clock->set_limit(r.clock->limit() + 2 * test::kSec);  // nobody reads while 2 s pass
  ASSERT_TRUE(r.clock->wait_until_parked(30000));
  const ImuStats s = r.imu->stats();
  EXPECT_GT(s.queue_overruns, 1000u);
  std::vector<ImuSample> v;
  r.imu->read(v, std::chrono::milliseconds(0));
  EXPECT_EQ(v.size(), 100u);
  ASSERT_TRUE(r.imu->stop().ok());
}

}  // namespace
}  // namespace rig
