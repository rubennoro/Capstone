// "No C3 request for limit" (STRUCTURE.md 7.1). Pure; used by both threads with their own limit.
#pragma once

#include <cstdint>

namespace franka_control {

class Watchdog {
 public:
  explicit Watchdog(int64_t limit_ns) : limit_ns_(limit_ns) {}
  bool expired(int64_t now_ns, int64_t last_alive_ns) const { return now_ns - last_alive_ns > limit_ns_; }
  int64_t ageNs(int64_t now_ns, int64_t last_alive_ns) const { return now_ns - last_alive_ns; }
  int64_t limitNs() const { return limit_ns_; }

 private:
  int64_t limit_ns_;
};

}  // namespace franka_control
