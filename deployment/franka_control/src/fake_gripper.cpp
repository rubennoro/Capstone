#include "franka_control/fake_gripper.h"

#include <algorithm>
#include <cmath>

#include "franka_control/clock.h"

namespace franka_control {

FakeGripper::FakeGripper(const FakeGripperConfig& cfg, std::function<void(int64_t)> sleep_ns)
    : cfg_(cfg), sleep_ns_(std::move(sleep_ns)), width_m_(cfg.start_width_m) {}

void FakeGripper::prepare() {}

double FakeGripper::widthM() {
  if (fail_next_read_.exchange(false)) throw GripperError("FakeGripper: injected gripper read error");
  return width_m_.load();
}

void FakeGripper::moveTo(double width_m) {
  if (fail_next_.exchange(false)) throw GripperError("FakeGripper: injected gripper error");
  stop_.store(false);   // like the hand: stop() ends a running move only
  const double from = width_m_.load();
  double to = width_m;
  if (cfg_.object_width_m && to < from && to < *cfg_.object_width_m && from >= *cfg_.object_width_m) {
    to = *cfg_.object_width_m;   // closing past the object: the fingers stop on it
  }
  const int64_t total_ns = secondsToNs(std::abs(to - from) / cfg_.speed_m_per_s);
  for (int64_t done_ns = 0; done_ns < total_ns;) {
    if (stop_.load()) return;   // stopped: the fingers stay where they are
    const int64_t step = std::min(kStepNs, total_ns - done_ns);
    sleep_ns_(step);
    done_ns += step;
    width_m_.store(from + (to - from) * static_cast<double>(done_ns) / static_cast<double>(total_ns));
  }
  width_m_.store(to);
}

void FakeGripper::stop() { stop_.store(true); }

void FakeGripper::failNextMove(bool on) { fail_next_.store(on); }

void FakeGripper::failNextRead(bool on) { fail_next_read_.store(on); }

}  // namespace franka_control
