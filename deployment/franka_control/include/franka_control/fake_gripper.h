// Simulated fingers (STRUCTURE.md 9.3): the width moves toward the target at fake_gripper.speed_m_per_s, in 5 ms
// steps (so widthM() from another thread sees it change during a move), and stops at fake_gripper.object_width_m when
// closing past it. stop() ends a running move where the fingers are.
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
  void failNextRead(bool on);                      // tests: the next widthM throws GripperError

  static constexpr int64_t kStepNs = 5000000;      // one simulated step of a move

 private:
  FakeGripperConfig cfg_;
  std::function<void(int64_t)> sleep_ns_;
  std::atomic<double> width_m_;
  std::atomic<bool> fail_next_{false};
  std::atomic<bool> fail_next_read_{false};
  std::atomic<bool> stop_{false};
};

}  // namespace franka_control
