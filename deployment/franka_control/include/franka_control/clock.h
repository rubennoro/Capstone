// The clock every time-dependent class takes, so tests can drive time (deploy STRUCTURE.md 7.6).
#pragma once

#include <cstdint>
#include <ctime>
#include <functional>

namespace franka_control {

// Returns CLOCK_MONOTONIC in ns (the PC time every contract t_ns uses).
using Clock = std::function<int64_t()>;

inline int64_t monotonicNowNs() {
  timespec ts{};
  clock_gettime(CLOCK_MONOTONIC, &ts);
  return static_cast<int64_t>(ts.tv_sec) * 1000000000LL + static_cast<int64_t>(ts.tv_nsec);
}

constexpr int64_t kNsPerS = 1000000000LL;

inline int64_t secondsToNs(double s) { return static_cast<int64_t>(s * 1e9); }
inline double nsToSeconds(int64_t ns) { return static_cast<double>(ns) * 1e-9; }

}  // namespace franka_control
