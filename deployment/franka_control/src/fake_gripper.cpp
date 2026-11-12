#include "franka_control/fake_gripper.h"

#include <cmath>

#include "franka_control/clock.h"

namespace franka_control {

FakeGripper::FakeGripper(const FakeGripperConfig& cfg, std::function<void(int64_t)> sleep_ns)
    : cfg_(cfg), sleep_ns_(std::move(sleep_ns)), width_m_(cfg.start_width_m) {}

void FakeGripper::prepare() {}

double FakeGripper::widthM() { return width_m_.load(); }

void FakeGripper::moveTo(double width_m) {
  if (fail_next_.exchange(false)) throw GripperError("FakeGripper: injected gripper error");
  const double from = width_m_.load();
  double to = width_m;
  if (cfg_.object_width_m && to < from && to < *cfg_.object_width_m && from >= *cfg_.object_width_m) {
    to = *cfg_.object_width_m;   // closing past the object: the fingers stop on it
  }
  sleep_ns_(secondsToNs(std::abs(to - from) / cfg_.speed_m_per_s));
  width_m_.store(to);
}

void FakeGripper::stop() {}

void FakeGripper::failNextMove(bool on) { fail_next_.store(on); }

}  // namespace franka_control
