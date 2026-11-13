#include "franka_control/gripper_worker.h"

#include <cmath>
#include <ctime>

#include "franka_control/log.h"

namespace franka_control {
namespace {

// Sleeps until the next period on CLOCK_MONOTONIC. After a long blocking call (a finger move) it continues at once
// instead of catching up with a burst of steps.
void waitNextPeriod(int64_t& next, int64_t period, int64_t now) {
  next += period;
  if (next < now) next = now;
  timespec ts{};
  ts.tv_sec = static_cast<time_t>(next / kNsPerS);
  ts.tv_nsec = static_cast<long>(next % kNsPerS);
  clock_nanosleep(CLOCK_MONOTONIC, TIMER_ABSTIME, &ts, nullptr);
}

}  // namespace

GripperWorker::GripperWorker(Gripper& g, SharedState& shared, const GripperConfig& cfg, Clock clock)
    : gripper_(g), shared_(shared), cfg_(cfg), clock_(std::move(clock)), last_sent_m_(shared.gripperWidth()) {}

bool GripperWorker::fail(const GripperError& e) {
  Log::error(std::string("gripper: ") + e.what());
  shared_.setGripperFault(std::string("gripper: ") + e.what());
  failed_.store(true);
  return false;
}

bool GripperWorker::stepOnce() {
  try {
    const double target = shared_.gripperTarget();
    if (std::isfinite(target) && std::abs(target - last_sent_m_) > cfg_.deadband_m) {
      gripper_.moveTo(target);   // blocks while the fingers move; the reader thread keeps C2's width current
      last_sent_m_ = target;
    }
    return true;
  } catch (const GripperError& e) {
    return fail(e);
  }
}

bool GripperWorker::readStep() {
  try {
    shared_.setGripperWidth(gripper_.widthM());
    return true;
  } catch (const GripperError& e) {
    return fail(e);
  }
}

void GripperWorker::run() {
  const int64_t period = secondsToNs(cfg_.poll_period_s);
  int64_t next = clock_();
  while (!shared_.shutdownRequested() && !failed_.load()) {
    if (!stepOnce()) break;
    waitNextPeriod(next, period, clock_());
  }
  gripper_.stop();
}

void GripperWorker::runReader() {
  const int64_t period = secondsToNs(1.0 / cfg_.read_rate_hz);
  int64_t next = clock_();
  while (!shared_.shutdownRequested() && !failed_.load()) {
    if (!readStep()) break;
    waitNextPeriod(next, period, clock_());
  }
}

}  // namespace franka_control
