// The gripper interface (STRUCTURE.md 9.3): FrankaHandGripper (the real hand) and FakeGripper.
#pragma once

#include <stdexcept>
#include <string>

namespace franka_control {

struct GripperError : std::runtime_error {
  using std::runtime_error::runtime_error;
};

class Gripper {
 public:
  virtual ~Gripper() = default;
  virtual void prepare() = 0;                      // start-up: connect, optional finger homing
  virtual double widthM() = 0;                     // fingertip opening now
  virtual void moveTo(double width_m) = 0;         // blocks until reached, or stopped by an object
  virtual void stop() = 0;                         // any thread: ends a running moveTo
};
// All but stop throw GripperError.

}  // namespace franka_control
