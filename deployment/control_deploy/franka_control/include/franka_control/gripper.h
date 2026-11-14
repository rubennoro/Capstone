// from rnoro/deployment:deployment/franka_control/include/franka_control/gripper.h @ 91b178b
// changes: + home(), the hand's own homing (key g) (control_deploy/STRUCTURE.md 3.3).
// The gripper interface: FrankaHandGripper (the real hand) and FakeGripper.
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
  virtual void stop() = 0;                         // any thread: ends a running moveTo (or home)
  // Blocks for the hand's own homing (it closes and reopens to calibrate). True when done, false when ended by stop().
  virtual bool home() = 0;
};
// All but stop throw GripperError.

}  // namespace franka_control
