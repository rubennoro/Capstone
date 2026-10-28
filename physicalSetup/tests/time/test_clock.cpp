// SPDX-License-Identifier: MIT
// T3: host clocks and the session clock anchor (rig/time/clock.hpp, common/time/clock.cpp).
#include <gtest/gtest.h>

#include <time.h>

#include <atomic>
#include <cstdlib>
#include <thread>
#include <vector>

#include "rig/time/clock.hpp"

namespace rig {
namespace {

TEST(Clock, ClocksMatchClockGettime) {
  struct timespec ts {};
  for (auto [id, fn] : std::vector<std::pair<clockid_t, TimeNs (*)()>>{
           {CLOCK_MONOTONIC_RAW, &now_raw_ns}, {CLOCK_MONOTONIC, &now_mono_ns}, {CLOCK_REALTIME, &now_unix_ns}}) {
    clock_gettime(id, &ts);
    const TimeNs a = static_cast<TimeNs>(ts.tv_sec) * 1000000000LL + ts.tv_nsec;
    const TimeNs b = fn();
    clock_gettime(id, &ts);
    const TimeNs c = static_cast<TimeNs>(ts.tv_sec) * 1000000000LL + ts.tv_nsec;
    EXPECT_LE(a, b);
    EXPECT_LE(b, c);
  }
}

TEST(Clock, RawIsMonotonicAcrossThreads) {
  std::vector<std::thread> ts;
  std::atomic<int> bad{0};
  for (int t = 0; t < 4; ++t)
    ts.emplace_back([&] {
      TimeNs last = now_raw_ns();
      for (int i = 0; i < 100000; ++i) {
        const TimeNs n = now_raw_ns();
        if (n < last) bad++;
        last = n;
      }
    });
  for (auto& t : ts) t.join();
  EXPECT_EQ(bad.load(), 0);
}

TEST(Clock, AnchorIsConsistent) {
  const TimeNs before = now_raw_ns();
  const ClockAnchor a = sample_clock_anchor(9);
  const TimeNs after = now_raw_ns();
  EXPECT_GE(a.raw_ns, before);
  EXPECT_LE(a.raw_ns, after);
  EXPECT_GE(a.bracket_ns, 0);
  EXPECT_LT(a.bracket_ns, 1000000);
  // raw and mono differ only by NTP slewing since boot (both exclude suspend): well under a minute.
  EXPECT_LT(std::llabs(a.mono_ns - a.raw_ns), 60LL * 1000000000LL);
  EXPECT_GT(a.unix_ns, 1700000000LL * 1000000000LL);
  const ClockAnchor b = sample_clock_anchor(0);  // treated as 1 round
  EXPECT_GE(b.bracket_ns, 0);
  // Two anchors give the same wall<->raw relation within a few ms (no NTP step in between).
  const TimeNs off_a = a.unix_ns - a.raw_ns, off_b = b.unix_ns - b.raw_ns;
  EXPECT_LT(std::llabs(off_a - off_b), 50000000LL);
}

}  // namespace
}  // namespace rig
