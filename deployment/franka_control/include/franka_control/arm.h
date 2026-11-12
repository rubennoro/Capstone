// The arm interface (STRUCTURE.md 9.2): LibfrankaArm (the real Panda) and FakeArm (--mock, tests).
#pragma once

#include <functional>
#include <stdexcept>
#include <string>

#include "franka_control/arm_status.h"
#include "franka_control/homing_motion.h"
#include "franka_control/types.h"

namespace franka_control {

struct ArmError : std::runtime_error {
  enum class Kind { kReflex, kNetwork, kRealtime, kCommand, kStopRequested, kOther };
  ArmError(Kind kind_in, const std::string& what) : std::runtime_error(what), kind(kind_in) {}
  Kind kind;
};

class Arm {
 public:
  using TorqueStep = std::function<TorqueCommand(const ArmReading&)>;
  using Observer = std::function<void(const ArmReading&)>;
  virtual ~Arm() = default;
  virtual ArmReading readOnce() = 0;                                   // start-up only, outside control
  virtual ArmStatusFlags status() = 0;                                 // robot mode + error flags (start-up check)
  virtual void runTorqueControl(const TorqueStep& step) = 0;           // blocks; step every 1 ms until finished
  virtual void moveToJoints(const HomingMotion& m, const Observer& observe) = 0;   // blocks
  virtual void requestStop() = 0;                                      // any thread; the running call throws kStopRequested
  virtual std::string describe() const = 0;                            // for the log and deploy check
};
// All methods but requestStop/describe throw ArmError.

}  // namespace franka_control
