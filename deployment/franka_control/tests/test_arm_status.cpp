// describeArmProblem: the port of main-sim's derive_panda_error (mode and current_errors).
#include <gtest/gtest.h>

#include "franka_control/arm_status.h"

namespace franka_control {
namespace {

TEST(ArmStatus, ReflexModeWinsOverErrorFlags) {
  EXPECT_EQ(describeArmProblem({RobotMode::kReflex, "[cartesian_reflex]"}),
            "reflex (robot_mode Reflex, current_errors: [cartesian_reflex])");
  EXPECT_EQ(describeArmProblem({RobotMode::kReflex, "[]"}),
            "reflex (robot_mode Reflex, current_errors: no error flags)");
}

TEST(ArmStatus, CurrentErrorsText) {
  EXPECT_EQ(describeArmProblem({RobotMode::kIdle, "[joint_position_limits_violation]"}),
            "error flags (current_errors: [joint_position_limits_violation])");
}

TEST(ArmStatus, EmptyAndBracketsMeanNone) {
  EXPECT_EQ(describeArmProblem({RobotMode::kIdle, "[]"}), "");
  EXPECT_EQ(describeArmProblem({RobotMode::kIdle, ""}), "");
  EXPECT_EQ(describeArmProblem({RobotMode::kIdle, "  []  "}), "");
}

TEST(ArmStatus, UserStopPressed) {
  EXPECT_EQ(describeArmProblem({RobotMode::kUserStopped, "[]"}), "user stop pressed (robot_mode UserStopped)");
}

TEST(ArmStatus, OtherModesAreNotReady) {
  EXPECT_NE(describeArmProblem({RobotMode::kOther, "[]"}), "");
  EXPECT_NE(describeArmProblem({RobotMode::kGuiding, "[]"}), "");
  EXPECT_NE(describeArmProblem({RobotMode::kMove, "[]"}), "");
  EXPECT_NE(describeArmProblem({RobotMode::kAutomaticErrorRecovery, "[]"}), "");
}

}  // namespace
}  // namespace franka_control
