// SPDX-License-Identifier: MIT
// Host clocks (rig/time/clock.hpp). Implemented in T0 because every task needs them; owned by T3 from here on.
#include "rig/time/clock.hpp"

#include <time.h>

namespace rig {
namespace {
inline TimeNs read_clock(clockid_t id) noexcept {
  struct timespec ts {};
  clock_gettime(id, &ts);  // Cannot fail for these clock ids with a valid pointer (clock_gettime(2)).
  return static_cast<TimeNs>(ts.tv_sec) * 1000000000LL + static_cast<TimeNs>(ts.tv_nsec);
}
}  // namespace

TimeNs now_raw_ns() noexcept { return read_clock(CLOCK_MONOTONIC_RAW); }
TimeNs now_mono_ns() noexcept { return read_clock(CLOCK_MONOTONIC); }
TimeNs now_unix_ns() noexcept { return read_clock(CLOCK_REALTIME); }

ClockAnchor sample_clock_anchor(int rounds) noexcept {
  if (rounds < 1) rounds = 1;
  ClockAnchor best;
  for (int i = 0; i < rounds; ++i) {
    const TimeNs r0 = now_raw_ns();
    const TimeNs mono = now_mono_ns();
    const TimeNs unix_ns = now_unix_ns();
    const TimeNs r1 = now_raw_ns();
    const TimeNs width = r1 - r0;
    if (best.bracket_ns < 0 || width < best.bracket_ns) {
      best.raw_ns = r0 + width / 2;
      best.mono_ns = mono;
      best.unix_ns = unix_ns;
      best.bracket_ns = width;
    }
  }
  return best;
}

}  // namespace rig
