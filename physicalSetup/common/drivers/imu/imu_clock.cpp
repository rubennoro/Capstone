// SPDX-License-Identifier: MIT
#include "rig/drivers/imu/imu_clock.hpp"

#include <time.h>

#include <algorithm>
#include <cerrno>

#include "rig/time/clock.hpp"

namespace rig {

namespace {

class RealImuClock final : public ImuClock {
 public:
  TimeNs now_ns() override { return now_raw_ns(); }

  // clock_nanosleep() does not accept CLOCK_MONOTONIC_RAW, so sleep relative on CLOCK_MONOTONIC in slices of at
  // most 10 ms (cancel latency). The deadline itself stays absolute in the RAW domain, so the loop schedule does
  // not drift; NTP slew of CLOCK_MONOTONIC (<= 500 ppm) only shifts a wake-up by nanoseconds.
  void sleep_until(TimeNs t, const std::atomic<bool>* cancel) override {
    constexpr int64_t kSliceNs = 10000000;
    for (;;) {
      if (cancel != nullptr && cancel->load()) return;
      const int64_t d = t - now_raw_ns();
      if (d <= 0) return;
      const int64_t s = std::min(d, kSliceNs);
      timespec ts{static_cast<time_t>(s / 1000000000), static_cast<long>(s % 1000000000)};
      while (clock_nanosleep(CLOCK_MONOTONIC, 0, &ts, &ts) == EINTR) {
      }
    }
  }
};

}  // namespace

std::shared_ptr<ImuClock> real_imu_clock() {
  static const std::shared_ptr<ImuClock> clock = std::make_shared<RealImuClock>();
  return clock;
}

}  // namespace rig
