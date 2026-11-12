#include "franka_control/gripper_worker.h"

#include <cmath>
#include <ctime>

#include "franka_control/log.h"

namespace franka_control {

GripperWorker::GripperWorker(Gripper& g, SharedState& shared, const GripperConfig& cfg, Clock clock)
    : gripper_(g), shared_(shared), cfg_(cfg), clock_(std::move(clock)), last_sent_m_(shared.gripperWidth()) {}

bool GripperWorker::stepOnce() {
  try {
    const double target = shared_.gripperTarget();
    if (std::isfinite(target) && std::abs(target - last_sent_m_) > cfg_.deadband_m) {
      gripper_.moveTo(target);   // blocks while the fingers move; the width in C2 is stale meanwhile
      last_sent_m_ = target;
    }
    shared_.setGripperWidth(gripper_.widthM());
    return true;
  } catch (const GripperError& e) {
    Log::error(std::string("gripper: ") + e.what());
    shared_.setGripperFault(std::string("gripper: ") + e.what());
    return false;
  }
}

void GripperWorker::run() {
  const int64_t period = secondsToNs(cfg_.poll_period_s);
  int64_t next = clock_();
  while (!shared_.shutdownRequested()) {
    if (!stepOnce()) break;
    next += period;
    const int64_t now = clock_();
    if (next < now) next = now;   // after a long finger move: no burst of steps
    timespec ts{};
    ts.tv_sec = static_cast<time_t>(next / kNsPerS);
    ts.tv_nsec = static_cast<long>(next % kNsPerS);
    clock_nanosleep(CLOCK_MONOTONIC, TIMER_ABSTIME, &ts, nullptr);
  }
  gripper_.stop();
}

}  // namespace franka_control
