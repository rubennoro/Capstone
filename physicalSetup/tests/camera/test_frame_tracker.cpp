// SPDX-License-Identifier: MIT
// Drop detection from seq and timestamp gaps, and timing windows (rig/drivers/camera/frame_tracker.hpp).
#include <gtest/gtest.h>

#include "rig/drivers/camera/frame_tracker.hpp"

namespace rig {
namespace {

constexpr int64_t P = 33'333'333;  // 30 fps

TEST(Summarize, Percentiles) {
  SummaryStats s = summarize({});
  EXPECT_EQ(s.count, 0u);
  EXPECT_TRUE(std::isnan(s.p50));
  std::vector<double> v;
  for (int i = 1; i <= 101; ++i) v.push_back(i);
  s = summarize(v);
  EXPECT_EQ(s.count, 101u);
  EXPECT_DOUBLE_EQ(s.min, 1);
  EXPECT_DOUBLE_EQ(s.max, 101);
  EXPECT_DOUBLE_EQ(s.mean, 51);
  EXPECT_DOUBLE_EQ(s.p50, 51);
  EXPECT_DOUBLE_EQ(s.p99, 100);
  s = summarize({4, 1, 3, 2});
  EXPECT_DOUBLE_EQ(s.p50, 2.5);
}

TEST(FrameTracker, SteadyStream) {
  FrameTracker t(30.0, 1.5);
  for (uint64_t k = 0; k < 90; ++k) {
    DropEvent e;
    ASSERT_TRUE(t.observe(k, 1'000'000'000 + static_cast<int64_t>(k) * P, 1'000'000'000 + static_cast<int64_t>(k) * P + 20'000'000, &e));
    EXPECT_FALSE(e.any());
  }
  CameraStats s;
  t.fill(s);
  EXPECT_EQ(s.drops_seq_gap, 0u);
  EXPECT_EQ(s.drops_time_gap, 0u);
  EXPECT_NEAR(s.fps_estimate, 30.0, 0.01);
  EXPECT_EQ(s.inter_frame_ns.count, 60u);  // window = max(32, 2 s) = 60
  EXPECT_NEAR(s.inter_frame_ns.p50, P, 1);
  EXPECT_NEAR(s.latency_ns.p99, 20'000'000, 1);
  EXPECT_EQ(s.last_t_sof_ns, 1'000'000'000 + 89 * P);
}

TEST(FrameTracker, SeqGapCountsOnceAndNotAsTimeGap) {
  FrameTracker t(30.0, 1.5);
  ASSERT_TRUE(t.observe(10, 0, 1));
  DropEvent e;
  ASSERT_TRUE(t.observe(13, 3 * P, 3 * P + 1, &e));  // 2 frames missing in both seq and time
  EXPECT_EQ(e.seq_missing, 2u);
  EXPECT_EQ(e.time_missing, 0u);
  EXPECT_NE(e.reason.find("2 missing seq"), std::string::npos) << e.reason;
  EXPECT_EQ(t.drops_seq_gap(), 2u);
  EXPECT_EQ(t.drops_time_gap(), 0u);
}

TEST(FrameTracker, TimeGapWithoutSeqGap) {
  FrameTracker t(30.0, 1.5);
  ASSERT_TRUE(t.observe(0, 0, 1));
  DropEvent e;
  ASSERT_TRUE(t.observe(1, 4 * P, 4 * P + 1, &e));  // stall: 3 slots missing, seq contiguous
  EXPECT_EQ(e.seq_missing, 0u);
  EXPECT_EQ(e.time_missing, 3u);
  // A gap just under the factor (1.4 periods) is jitter, not a drop.
  ASSERT_TRUE(t.observe(2, 4 * P + 14 * P / 10, 0, &e));
  EXPECT_FALSE(e.any());
  // Time gap larger than the seq gap: the excess is unexplained.
  ASSERT_TRUE(t.observe(4, 4 * P + 14 * P / 10 + 6 * P, 0, &e));
  EXPECT_EQ(e.seq_missing, 1u);
  EXPECT_EQ(e.time_missing, 4u);
  EXPECT_EQ(t.drops_time_gap(), 7u);
}

TEST(FrameTracker, NoDriverSeqUsesTimeOnly) {
  FrameTracker t(60.0, 1.5);
  ASSERT_TRUE(t.observe(std::nullopt, 0, 0));
  DropEvent e;
  ASSERT_TRUE(t.observe(std::nullopt, 33'333'334, 0, &e));
  EXPECT_EQ(e.time_missing, 1u);
}

TEST(FrameTracker, RejectsNonMonotonic) {
  FrameTracker t(30.0, 1.5);
  ASSERT_TRUE(t.observe(0, 100, 0));
  DropEvent e;
  EXPECT_FALSE(t.observe(1, 100, 0, &e));
  EXPECT_TRUE(e.rejected);
  EXPECT_NE(e.reason.find("non-monotonic SOF"), std::string::npos);
  EXPECT_FALSE(t.observe(0, 200, 0, &e));
  EXPECT_NE(e.reason.find("non-monotonic driver seq"), std::string::npos);
  EXPECT_FALSE(t.observe(2, kInvalidTime, 0, &e));
  EXPECT_EQ(t.rejected(), 3u);
  EXPECT_TRUE(t.observe(1, 200, 0, &e));  // state unchanged by rejections
  EXPECT_EQ(t.accepted(), 2u);
  t.reset();
  EXPECT_EQ(t.accepted(), 0u);
  EXPECT_TRUE(t.observe(0, 50, 0));
}

}  // namespace
}  // namespace rig
