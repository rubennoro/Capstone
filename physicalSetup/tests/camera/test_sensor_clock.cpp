// SPDX-License-Identifier: MIT
// Timestamp semantics and clock-domain resolution (rig/drivers/camera/sensor_clock.hpp).
#include <gtest/gtest.h>

#include <filesystem>
#include <fstream>

#include "rig/drivers/camera/sensor_clock.hpp"
#include "rig/drivers/camera_backend.hpp"

namespace rig {
namespace {

// A host where MONOTONIC_RAW = 1000 s, MONOTONIC = 1000.3 s (NTP slewed 300 ms over a long uptime), REALTIME =
// 1.8e9 s since the epoch.
ClockAnchor anchor(TimeNs raw) {
  ClockAnchor a;
  a.raw_ns = raw;
  a.mono_ns = raw + 300'000'000;
  a.unix_ns = raw + 1'800'000'000'000'000'000LL;
  a.bracket_ns = 100;
  return a;
}
constexpr TimeNs kRaw = 1'000'000'000'000;

TEST(SensorClock, Classify) {
  const ClockAnchor a = anchor(kRaw);
  EXPECT_EQ(classify_sensor_clock(kRaw - 20'000'000, a), SensorClockDomain::kMonotonicRaw);
  EXPECT_EQ(classify_sensor_clock(a.mono_ns - 20'000'000, a, 100'000'000), SensorClockDomain::kMonotonic);
  EXPECT_EQ(classify_sensor_clock(a.unix_ns - 20'000'000, a), SensorClockDomain::kRealtime);
  EXPECT_EQ(classify_sensor_clock(kRaw + 5'000'000, a, 100'000'000), SensorClockDomain::kUnknown);  // future
  // ... but with the default 1 s window it is a plausible MONOTONIC stamp (the 300 ms slew): inherent ambiguity.
  EXPECT_EQ(classify_sensor_clock(kRaw + 5'000'000, a), SensorClockDomain::kMonotonic);
  EXPECT_EQ(classify_sensor_clock(12345, a), SensorClockDomain::kUnknown);
  EXPECT_EQ(classify_sensor_clock(0, a), SensorClockDomain::kUnknown);
  // Ambiguous raw/mono (offset 300 ms < 1 s max latency): RAW wins.
  EXPECT_EQ(classify_sensor_clock(a.mono_ns - 400'000'000, a), SensorClockDomain::kMonotonicRaw);
}

// R1 F3: stamps slightly in the future of arrival (r36.4.0 offset_ns precision, forum 329038) stay plausible.
TEST(SensorClock, SlightlyFutureStampIsPlausible) {
  const ClockAnchor a = anchor(kRaw);
  EXPECT_EQ(kMaxFutureCameraStampNs, 2'000'000);
  EXPECT_EQ(classify_sensor_clock(kRaw + 500'000, a, 100'000'000), SensorClockDomain::kMonotonicRaw);
  EXPECT_EQ(classify_sensor_clock(kRaw + 2'000'000, a, 100'000'000), SensorClockDomain::kMonotonicRaw);
  EXPECT_EQ(classify_sensor_clock(kRaw + 2'000'001, a, 100'000'000), SensorClockDomain::kUnknown);
  // Configurable: strict (0) and generous (5 ms).
  EXPECT_EQ(classify_sensor_clock(kRaw + 1, a, 100'000'000, 0), SensorClockDomain::kUnknown);
  EXPECT_EQ(classify_sensor_clock(kRaw + 4'000'000, a, 100'000'000, 5'000'000), SensorClockDomain::kMonotonicRaw);
}

TEST(SensorClockResolver, SlightlyFutureStampsStillResolveToRaw) {
  // Every frame 0.3 ms "in the future", TSC agreeing: start() must not fail (it did before F3).
  SensorClockResolver r(5, 100'000'000);
  for (int i = 0; i < 5; ++i) {
    const TimeNs arr = kRaw + i * 16'666'667;
    r.add(static_cast<uint64_t>(arr + 300'000), static_cast<uint64_t>(arr + 300'000 + 99), 99, anchor(arr));
  }
  auto d = r.decide();
  ASSERT_TRUE(d.ok()) << d.status().to_string();
  EXPECT_EQ(d->domain, SensorClockDomain::kMonotonicRaw);
  EXPECT_FALSE(d->use_tsc);
  SensorClockResolver strict(1, 100'000'000, 0);
  strict.add(static_cast<uint64_t>(kRaw + 300'000), std::nullopt, std::nullopt, anchor(kRaw));
  EXPECT_EQ(strict.decide().status().code(), StatusCode::kDeviceError);
}

TEST(SensorClock, Convert) {
  const ClockAnchor a = anchor(kRaw);
  EXPECT_EQ(sensor_clock_to_raw_ns(500, SensorClockDomain::kMonotonicRaw, a), 500);
  EXPECT_EQ(sensor_clock_to_raw_ns(static_cast<uint64_t>(a.mono_ns - 7), SensorClockDomain::kMonotonic, a), kRaw - 7);
  EXPECT_EQ(sensor_clock_to_raw_ns(static_cast<uint64_t>(a.unix_ns - 7), SensorClockDomain::kRealtime, a), kRaw - 7);
  EXPECT_EQ(sensor_clock_to_raw_ns(5000, SensorClockDomain::kTscOffset, a, 1200), 3800);
  EXPECT_EQ(sensor_clock_to_raw_ns(5000, SensorClockDomain::kTscOffset, a), kInvalidTime);
  EXPECT_EQ(sensor_clock_to_raw_ns(5000, SensorClockDomain::kUnknown, a), kInvalidTime);
  EXPECT_EQ(gst_clock_time_to_raw_ns(100'000, 50'000, a), 150'000 - 300'000'000);
}

// The Argus first-data stamp is the END of row 0's exposure; the image time is the middle row's mid-exposure.
TEST(SensorClock, ArgusMidExposureMath) {
  const TimeNs first_data = 10'000'000'000;
  const int64_t exposure = 8'000'000, readout = 15'000'000;
  const TimeNs sof = argus_start_of_exposure(first_data, exposure);
  EXPECT_EQ(sof, first_data - exposure);
  // mid_exposure_time (frozen header) applied to the start of exposure = ts - E/2 + R/2.
  EXPECT_EQ(mid_exposure_time(sof, exposure, readout), first_data - exposure / 2 + readout / 2);
  EXPECT_EQ(mid_exposure_time(sof, exposure, readout), 10'003'500'000);
  EXPECT_EQ(argus_start_of_exposure(first_data, -1), first_data);
  EXPECT_EQ(argus_start_of_exposure(kInvalidTime, 5), kInvalidTime);
}

TEST(SensorClockResolver, RawDomain) {
  SensorClockResolver r(3);
  EXPECT_EQ(r.decide().status().code(), StatusCode::kFailedPrecondition);
  for (int i = 0; i < 3; ++i) {
    const TimeNs arr = kRaw + i * 16'666'667;
    r.add(static_cast<uint64_t>(arr - 25'000'000), static_cast<uint64_t>(arr - 25'000'000 + 777), 777, anchor(arr));
  }
  ASSERT_TRUE(r.ready());
  auto d = r.decide();
  ASSERT_TRUE(d.ok()) << d.status().to_string();
  EXPECT_EQ(d->domain, SensorClockDomain::kMonotonicRaw);
  EXPECT_FALSE(d->use_tsc);
  EXPECT_TRUE(d->raw_mono_ambiguous);
  EXPECT_NE(d->description.find("tsc cross-check available"), std::string::npos);
  EXPECT_EQ(r.to_raw_ns(42, std::nullopt, std::nullopt, anchor(kRaw)), 42);
}

TEST(SensorClockResolver, RealtimeStampPrefersTsc) {
  SensorClockResolver r(2);
  for (int i = 0; i < 2; ++i) {
    const TimeNs arr = kRaw + i * 33'000'000;
    const ClockAnchor a = anchor(arr);
    r.add(static_cast<uint64_t>(a.unix_ns - 30'000'000), static_cast<uint64_t>(arr - 30'000'000 + 5'000), 5'000, a);
  }
  auto d = r.decide();
  ASSERT_TRUE(d.ok());
  EXPECT_TRUE(d->use_tsc);
  EXPECT_EQ(d->domain, SensorClockDomain::kTscOffset);
  EXPECT_NE(d->description.find("classified as realtime"), std::string::npos) << d->description;
  EXPECT_EQ(r.to_raw_ns(1, static_cast<uint64_t>(kRaw + 5'000), 5'000, anchor(kRaw)), kRaw);
  EXPECT_EQ(r.to_raw_ns(1, std::nullopt, 5'000, anchor(kRaw)), kInvalidTime);
}

TEST(SensorClockResolver, RealtimeWithoutTscConverts) {
  SensorClockResolver r(2);
  for (int i = 0; i < 2; ++i) {
    const ClockAnchor a = anchor(kRaw + i * 33'000'000);
    r.add(static_cast<uint64_t>(a.unix_ns - 30'000'000), std::nullopt, std::nullopt, a);
  }
  auto d = r.decide();
  ASSERT_TRUE(d.ok());
  EXPECT_EQ(d->domain, SensorClockDomain::kRealtime);
  const ClockAnchor a = anchor(kRaw);
  EXPECT_EQ(r.to_raw_ns(static_cast<uint64_t>(a.unix_ns - 30), std::nullopt, std::nullopt, a), kRaw - 30);
}

TEST(SensorClockResolver, InconsistentFailsLoudly) {
  SensorClockResolver r(2);
  r.add(static_cast<uint64_t>(kRaw - 1'000'000), std::nullopt, std::nullopt, anchor(kRaw));
  r.add(12345, std::nullopt, std::nullopt, anchor(kRaw + 1));
  auto d = r.decide();
  ASSERT_FALSE(d.ok());
  EXPECT_EQ(d.status().code(), StatusCode::kDeviceError);
  EXPECT_NE(d.status().message().find("no plausible timestamp domain"), std::string::npos);
}

TEST(SensorClock, ReadTscOffset) {
  namespace fs = std::filesystem;
  const fs::path root = fs::path(RIG_TEST_TMP_ROOT) / "t2_tsc_sysroot";
  fs::remove_all(root);
  EXPECT_EQ(read_tsc_offset_ns(root.string()).status().code(), StatusCode::kNotFound);
  fs::create_directories(root / "sys/devices/system/clocksource/clocksource0");
  std::ofstream(root / "sys/devices/system/clocksource/clocksource0/offset_ns") << "-123456789\n";
  auto v = read_tsc_offset_ns(root.string());
  ASSERT_TRUE(v.ok()) << v.status().to_string();
  EXPECT_EQ(*v, -123456789);
  std::ofstream(root / "sys/devices/system/clocksource/clocksource0/offset_ns") << "garbage\n";
  EXPECT_EQ(read_tsc_offset_ns(root.string()).status().code(), StatusCode::kIoError);
  fs::remove_all(root);
}

}  // namespace
}  // namespace rig
