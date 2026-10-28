// SPDX-License-Identifier: MIT
// ChipClockMapper on synthetic brackets: drift +-2 %, realistic jitter with preemption spikes, torn reads,
// rejection rules, clamp, single-bracket behaviour.
#include <gtest/gtest.h>

#include <cmath>
#include <random>

#include "rig/time/clock.hpp"

namespace rig {
namespace {

struct Truth {
  double t0 = 5e9;        // host time of tick 0 [ns]
  double tick = 25000.0;  // true host ns per tick
  double at(double k) const { return t0 + k * tick; }
};

// One bracket like an I2C register read: the counter latches inside the transaction.
ClockBracket make_bracket(const Truth& tr, double t_latch, std::mt19937_64& rng, bool spikes = true) {
  std::exponential_distribution<double> ex(1.0 / 15000.0);
  std::uniform_real_distribution<double> u(0, 1);
  double pre = 107500 + ex(rng), post = 120000 + ex(rng);
  if (spikes && u(rng) < 0.01) pre += u(rng) * 1.5e6;
  if (spikes && u(rng) < 0.01) post += u(rng) * 1.5e6;
  ClockBracket b;
  b.chip_ticks = static_cast<int64_t>(std::floor((t_latch - tr.t0) / tr.tick));
  b.t_host_before_ns = static_cast<TimeNs>(t_latch - pre);
  b.t_host_after_ns = static_cast<TimeNs>(t_latch + post);
  return b;
}

// Max |map(k) - true(k)| over sample ticks spread through the last `span_s` seconds before `t_end`.
double max_error_us(const ChipClockMapper& m, const Truth& tr, double t_end, double span_s) {
  double worst = 0;
  for (int i = 0; i <= 200; ++i) {
    const double t = t_end - span_s * 1e9 * i / 200.0;
    const double k = std::floor((t - tr.t0) / tr.tick);
    const double err = static_cast<double>(m.map(static_cast<int64_t>(k))) - tr.at(k);
    worst = std::max(worst, std::abs(err));
  }
  return worst / 1000.0;
}

class MapperDrift : public ::testing::TestWithParam<std::tuple<double, ClockFitMethod>> {};

TEST_P(MapperDrift, ErrorBelow50usWithDriftAndJitter) {
  const auto [drift, method] = GetParam();
  Truth tr;
  tr.tick = 25000.0 / (1.0 + drift);
  ChipClockConfig cfg;
  cfg.method = method;
  ChipClockMapper m(cfg);
  std::mt19937_64 rng(7);
  double t = tr.t0 + 1e9;
  double worst_live = 0;
  for (int i = 0; i < 1200; ++i) {  // 120 s at 10 Hz
    t += 1e8 + std::uniform_real_distribution<double>(-2e6, 2e6)(rng);
    (void)m.add_bracket(make_bracket(tr, t, rng));
    if (i >= 20) {
      // Live use: samples up to one poll (10 ms) after the newest bracket and back to one bracket period before.
      for (double dt : {-1e8, -5e7, 0.0, 1e7, 2e7}) {
        const double k = std::floor((t + dt - tr.t0) / tr.tick);
        worst_live = std::max(worst_live, std::abs(static_cast<double>(m.map(static_cast<int64_t>(k))) - tr.at(k)));
      }
    }
  }
  ASSERT_TRUE(m.ready());
  const ClockFitStats s = m.stats();
  EXPECT_FALSE(s.fit.slope_clamped);
  EXPECT_NEAR(s.fit.slope_ns_per_tick / tr.tick - 1.0, 0.0, 2e-6);  // < 2 ppm slope error
  EXPECT_LT(max_error_us(m, tr, t, 30.0), 50.0);
  EXPECT_LT(worst_live / 1000.0, 50.0);
  EXPECT_GT(s.residual_ns.count, 0u);
  EXPECT_LT(std::abs(s.residual_ns.p50), 50000.0);
  RecordProperty("worst_live_us", std::to_string(worst_live / 1000.0));
  std::printf("drift %+.3f method %d: worst live error %.1f us, window error %.1f us, slope err %.3f ppm\n", drift,
              static_cast<int>(method), worst_live / 1000.0, max_error_us(m, tr, t, 30.0),
              (s.fit.slope_ns_per_tick / tr.tick - 1.0) * 1e6);
}

INSTANTIATE_TEST_SUITE_P(All, MapperDrift,
                         ::testing::Combine(::testing::Values(-0.02, 0.0, 0.02),
                                            ::testing::Values(ClockFitMethod::kLowerEnvelope,
                                                              ClockFitMethod::kRobustLinear)));

TEST(Mapper, SingleBracketUsesNominalSlopeThroughMidpoint) {
  ChipClockMapper m;
  EXPECT_EQ(m.map(5), kInvalidTime);
  EXPECT_FALSE(m.ready());
  ASSERT_TRUE(m.add_bracket(1000000, 100, 1200000).ok());
  EXPECT_FALSE(m.ready());
  EXPECT_EQ(m.map(100), 1100000);
  EXPECT_EQ(m.map(104), 1100000 + 4 * 25000);
  ASSERT_TRUE(m.add_bracket(101000000, 4100, 101200000).ok());
  EXPECT_TRUE(m.ready());
}

TEST(Mapper, RejectionRules) {
  ChipClockConfig cfg;
  cfg.max_bracket_width_ns = 1000000;
  ChipClockMapper m(cfg);
  EXPECT_EQ(m.add_bracket(2000, 10, 1000).code(), StatusCode::kInvalidArgument);        // after < before
  EXPECT_EQ(m.add_bracket(0, 10, 1000001).code(), StatusCode::kInvalidArgument);        // too wide
  ASSERT_TRUE(m.add_bracket(1000000, 100, 1100000).ok());
  EXPECT_EQ(m.add_bracket(2000000, 99, 2100000).code(), StatusCode::kInvalidArgument);  // ticks backwards
  EXPECT_EQ(m.add_bracket(900000, 200, 1150000).code(), StatusCode::kInvalidArgument);  // host backwards
  EXPECT_EQ(m.stats().brackets_rejected, 4u);
  EXPECT_EQ(m.stats().brackets_added, 1u);
  m.reset();
  EXPECT_EQ(m.map(100), kInvalidTime);
  EXPECT_TRUE(m.add_bracket(5, 1, 10).ok());  // after reset anything goes again
}

TEST(Mapper, TornReadOutlierIsGatedAndRestartWorks) {
  Truth tr;
  ChipClockMapper m;
  std::mt19937_64 rng(3);
  double t = tr.t0 + 1e9;
  for (int i = 0; i < 100; ++i) {
    t += 1e8;
    ASSERT_TRUE(m.add_bracket(make_bracket(tr, t, rng, false)).ok());
  }
  t += 1e8;
  ClockBracket torn = make_bracket(tr, t, rng, false);
  torn.chip_ticks += 255;  // LSB read before a carry, next byte after it
  EXPECT_EQ(m.add_bracket(torn).code(), StatusCode::kOutOfRange);
  EXPECT_LT(max_error_us(m, tr, t, 5.0), 50.0);
  t += 1e8;
  ASSERT_TRUE(m.add_bracket(make_bracket(tr, t, rng, false)).ok());  // a good one resets the outlier count
  // A real relation change (e.g. the host misread every time): 5 consecutive outliers restart the window.
  int accepted = 0;
  for (int i = 0; i < 5; ++i) {
    t += 1e8;
    ClockBracket b = make_bracket(tr, t, rng, false);
    b.chip_ticks += 100000;
    accepted += m.add_bracket(b).ok();
  }
  EXPECT_EQ(accepted, 1);  // the 5th restarts the window
  EXPECT_EQ(m.stats().brackets_added, 102u);
  EXPECT_EQ(m.stats().residual_ns.count, 1u);
}

TEST(Mapper, SlopeClampEngages) {
  Truth tr;
  tr.tick = 25000.0 / 1.05;  // 5 % fast chip, clamp is 3 %
  ChipClockMapper m;
  std::mt19937_64 rng(1);
  double t = tr.t0 + 1e9;
  for (int i = 0; i < 50; ++i) {
    t += 1e8;
    (void)m.add_bracket(make_bracket(tr, t, rng, false));
  }
  const ClockFit f = m.fit();
  EXPECT_TRUE(f.slope_clamped);
  EXPECT_NEAR(f.slope_ns_per_tick, 25000.0 * 0.97, 1e-6);
  EXPECT_GT(m.stats().slope_clamp_events, 0u);
}

TEST(Mapper, WindowSlides) {
  ChipClockConfig cfg;
  cfg.window_brackets = 10;
  ChipClockMapper m(cfg);
  Truth tr;
  std::mt19937_64 rng(2);
  double t = tr.t0 + 1e9;
  for (int i = 0; i < 50; ++i) {
    t += 1e8;
    ASSERT_TRUE(m.add_bracket(make_bracket(tr, t, rng, false)).ok());
  }
  EXPECT_EQ(m.stats().residual_ns.count, 10u);
  EXPECT_EQ(m.stats().brackets_added, 50u);
}

}  // namespace
}  // namespace rig
