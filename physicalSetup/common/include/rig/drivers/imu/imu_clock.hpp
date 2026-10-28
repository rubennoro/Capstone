// SPDX-License-Identifier: MIT
/// @file rig/drivers/imu/imu_clock.hpp
/// @brief Time source of the IMU driver poll loop and of the LSM6DSOX simulator (T1).
///
/// The real clock is CLOCK_MONOTONIC_RAW (rig/time/clock.hpp). The virtual clock lets tests run a 60 s, 833 Hz
/// session in a fraction of a second: sleeping jumps time forward, and the simulator advances it by the duration
/// of each simulated I2C transaction, so bracket widths and FIFO fill are realistic.
#pragma once

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <limits>
#include <memory>
#include <mutex>

#include "rig/types.hpp"

namespace rig {

class ImuClock {
 public:
  virtual ~ImuClock() = default;
  /// Current time [ns], CLOCK_MONOTONIC_RAW domain (or virtual). Thread-safe.
  virtual TimeNs now_ns() = 0;
  /// Sleeps until now_ns() >= t, or returns early once `*cancel` is true (checked at least every 10 ms on the real
  /// clock; immediately on the virtual clock after wake()).
  virtual void sleep_until(TimeNs t, const std::atomic<bool>* cancel) = 0;
  /// Wakes sleepers so that they re-check their cancel flag.
  virtual void wake() {}
  /// True for the virtual clock.
  virtual bool is_virtual() const { return false; }
  /// Virtual clock only: consume `ns` of time (a simulated bus transaction). No-op on the real clock.
  virtual void advance(int64_t /*ns*/) {}
};

/// Shared real clock (CLOCK_MONOTONIC_RAW; sleeps with clock_nanosleep on CLOCK_MONOTONIC, relative).
std::shared_ptr<ImuClock> real_imu_clock();

/// Deterministic virtual clock. Time only moves through sleep_until() and advance(). An optional limit makes
/// sleep_until() block (in real time) at the limit, so a test can run "exactly 60 s" and then inspect results.
class VirtualImuClock final : public ImuClock {
 public:
  explicit VirtualImuClock(TimeNs start_ns = 1000000000) : t_(start_ns) {}

  TimeNs now_ns() override { return t_.load(std::memory_order_acquire); }

  void sleep_until(TimeNs t, const std::atomic<bool>* cancel) override {
    std::unique_lock<std::mutex> lk(mu_);
    if (t > limit_) {
      waiting_at_limit_ = true;
      cv_.notify_all();
      cv_.wait(lk, [&] { return t <= limit_ || (cancel != nullptr && cancel->load()); });
      waiting_at_limit_ = false;
      if (t > limit_) return;  // cancelled
    }
    TimeNs cur = t_.load(std::memory_order_acquire);
    if (t > cur) t_.store(t, std::memory_order_release);
  }

  void wake() override {
    std::lock_guard<std::mutex> lk(mu_);
    cv_.notify_all();
  }

  bool is_virtual() const override { return true; }
  void advance(int64_t ns) override {
    if (ns > 0) t_.fetch_add(ns, std::memory_order_acq_rel);
  }

  /// Sleeps past `limit` block until the limit is raised (or the sleeper is cancelled).
  void set_limit(TimeNs limit) {
    std::lock_guard<std::mutex> lk(mu_);
    limit_ = limit;
    cv_.notify_all();
  }
  TimeNs limit() {
    std::lock_guard<std::mutex> lk(mu_);
    return limit_;
  }
  /// Blocks (real time) until a sleeper is parked at the limit; false after `timeout_ms`.
  bool wait_until_parked(int timeout_ms) {
    std::unique_lock<std::mutex> lk(mu_);
    return cv_.wait_for(lk, std::chrono::milliseconds(timeout_ms), [&] { return waiting_at_limit_; });
  }

 private:
  std::atomic<TimeNs> t_;
  std::mutex mu_;
  std::condition_variable cv_;
  TimeNs limit_ = std::numeric_limits<TimeNs>::max();
  bool waiting_at_limit_ = false;
};

}  // namespace rig
