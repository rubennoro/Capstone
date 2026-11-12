// Simulated fingers (STRUCTURE.md 9.3): the width moves toward the target at fake_gripper.speed_m_per_s and stops
// at fake_gripper.object_width_m when closing past it.
#pragma once

#include <atomic>
#include <cstdint>
#include <functional>

#include "franka_control/config.h"
#include "franka_control/gripper.h"

namespace franka_control {

class FakeGripper : public Gripper {
 public:
  FakeGripper(const FakeGripperConfig& cfg, std::function<void(int64_t)> sleep_ns);   // tests pass a no-op sleep
  void prepare() override;
  double widthM() override;
  void moveTo(double width_m) override;
  void stop() override;
  void failNextMove(bool on);                      // tests: the next moveTo throws GripperError

 private:
  FakeGripperConfig cfg_;
  std::function<void(int64_t)> sleep_ns_;
  std::atomic<double> width_m_;
  std::atomic<bool> fail_next_{false};
};

}  // namespace franka_control
