// With J_F = [I6 | 0], F_T_TCP = I and dq = 0, the torques are the TCP wrench itself (joints 1-6 = force, moment).
#include <gtest/gtest.h>

#include "franka_control/impedance_controller.h"
#include "support/test_support.h"

namespace franka_control {
namespace {

using test::rotvec;

ControlConfig cfgWith(const std::function<void(ControlConfig&)>& f = {}) {
  ControlConfig c;   // config.yaml defaults: K 800 / 40, ratio 1, k_ns 0.5, lambda 0.2, clip 0.03 m / 0.2 rad
  c.torque_rate_max_nm_per_s = 1e9;   // the rate limit has its own test
  if (f) f(c);
  return c;
}

ArmReading reading(const Pose& O_T_F) {
  ArmReading r;
  r.O_T_F = O_T_F;
  r.J_F.setZero();
  r.J_F.leftCols<6>().setIdentity();
  return r;
}

const Pose kTarget = test::makePose({0.5, 0.0, 0.3}, Eigen::Quaterniond::Identity());

Pose shifted(const Eigen::Vector3d& d) {
  Pose p = kTarget;
  p.translation() += d;
  return p;
}

TEST(ImpedanceController, OneCentimetreErrorPushesBackOnThatAxisOnly) {
  const ImpedanceController c(cfgWith(), Pose::Identity());
  const TorqueResult r = c.compute(reading(shifted({0.01, 0, 0})), kTarget);
  ASSERT_TRUE(r.finite);
  EXPECT_NEAR(r.tau(0), -800.0 * 0.01, 1e-9);   // toward the target
  for (int i = 1; i < 7; ++i) EXPECT_NEAR(r.tau(i), 0.0, 1e-9) << i;
  EXPECT_NEAR(r.position_error_m, 0.01, 1e-12);
  EXPECT_TRUE(r.O_T_TCP.isApprox(shifted({0.01, 0, 0})));
}

TEST(ImpedanceController, RotationErrorAboutZFollowsTheLibfrankaFormula) {
  const ImpedanceController c(cfgWith(), Pose::Identity());
  Pose measured = kTarget;
  measured.linear() = rotvec({0, 0, 0.1}).toRotationMatrix();
  const TorqueResult r = c.compute(reading(measured), kTarget);
  // q_e = q^-1 q_d = Rz(-0.1): vec = (0, 0, -sin 0.05); e_o = -R vec = (0, 0, sin 0.05); F = -K e_o
  EXPECT_NEAR(r.tau(5), -40.0 * std::sin(0.05), 1e-9);   // turns back toward the target
  for (int i = 0; i < 7; ++i) {
    if (i != 5) {
      EXPECT_NEAR(r.tau(i), 0.0, 1e-9) << i;
    }
  }
  EXPECT_NEAR(r.rotation_error_rad, 0.1, 1e-12);
}

TEST(ImpedanceController, DampingOpposesTheTcpVelocity) {
  const ImpedanceController c(cfgWith([](ControlConfig& k) { k.nullspace_stiffness = 0.0; }), Pose::Identity());
  ArmReading r = reading(kTarget);
  r.dq(1) = 0.1;
  const TorqueResult t = c.compute(r, kTarget);
  EXPECT_NEAR(t.tau(1), -2.0 * std::sqrt(800.0) * 0.1, 1e-9);
  EXPECT_NEAR(t.tau(0), 0.0, 1e-12);
}

TEST(ImpedanceController, ErrorsAreClipped) {
  const ImpedanceController c(cfgWith(), Pose::Identity());
  auto r = c.compute(reading(shifted({0, 0, 0.05})), kTarget);
  EXPECT_NEAR(r.tau(2), -800.0 * 0.03, 1e-9);
  EXPECT_NEAR(r.position_error_m, 0.05, 1e-12);
  Pose measured = kTarget;
  measured.linear() = rotvec({0.5, 0, 0}).toRotationMatrix();
  r = c.compute(reading(measured), kTarget);
  EXPECT_NEAR(r.tau(3), -40.0 * std::sin(0.1), 1e-9);   // clipped to 0.2 rad
  EXPECT_NEAR(r.rotation_error_rad, 0.5, 1e-12);
}

TEST(ImpedanceController, NullspaceTermMovesOnlyJoint7ForThisJacobian) {
  const ImpedanceController exact(cfgWith([](ControlConfig& k) { k.jacobian_pinv_damping = 1e-6; }), Pose::Identity());
  ImpedanceController c = exact;
  c.setNullspaceTarget(Vec7::Constant(0.1));
  const TorqueResult r = c.compute(reading(kTarget), kTarget);
  for (int i = 0; i < 6; ++i) EXPECT_NEAR(r.tau(i), 0.0, 1e-9) << i;
  EXPECT_NEAR(r.tau(6), 0.5 * 0.1, 1e-12);
  // with the default damping lambda = 0.2 the projector lets 1 - 1/(1 + 0.04) through on joints 1-6
  ImpedanceController d(cfgWith(), Pose::Identity());
  d.setNullspaceTarget(Vec7::Constant(0.1));
  const TorqueResult rd = d.compute(reading(kTarget), kTarget);
  EXPECT_NEAR(rd.tau(0), (1.0 - 1.0 / 1.04) * 0.05, 1e-12);
  EXPECT_NEAR(rd.tau(6), 0.05, 1e-12);
}

TEST(ImpedanceController, CoriolisIsAddedNoGravity) {
  const ImpedanceController c(cfgWith(), Pose::Identity());
  ArmReading r = reading(kTarget);
  r.coriolis << 1, 2, 3, 4, 5, 6, 7;
  EXPECT_TRUE(c.compute(r, kTarget).tau.isApprox(r.coriolis, 1e-12));
}

TEST(ImpedanceController, TorqueClampAndRateLimit) {
  const ImpedanceController c(cfgWith(), Pose::Identity());
  ArmReading r = reading(kTarget);
  r.coriolis = Vec7::Constant(100.0);
  r.tau_J_d = Vec7::Constant(100.0);
  const Vec7 tau = c.compute(r, kTarget).tau;
  for (int i = 0; i < 4; ++i) EXPECT_DOUBLE_EQ(tau(i), 87.0);
  for (int i = 4; i < 7; ++i) EXPECT_DOUBLE_EQ(tau(i), 12.0);

  const ImpedanceController limited(cfgWith([](ControlConfig& k) { k.torque_rate_max_nm_per_s = 1000.0; }),
                                    Pose::Identity());
  ArmReading s = reading(shifted({0.01, 0, 0}));   // wants -8 N
  EXPECT_NEAR(limited.compute(s, kTarget).tau(0), -1.0, 1e-12);   // 1000 Nm/s * 1 ms from tau_J_d = 0
  s.tau_J_d(0) = -7.5;
  EXPECT_NEAR(limited.compute(s, kTarget).tau(0), -8.0, 1e-9);
}

TEST(ImpedanceController, NonFiniteInputIsFlagged) {
  const ImpedanceController c(cfgWith(), Pose::Identity());
  ArmReading r = reading(kTarget);
  r.q(2) = std::nan("");
  const TorqueResult t = c.compute(r, kTarget);
  EXPECT_FALSE(t.finite);
  EXPECT_TRUE(t.tau.isZero());
  ArmReading s = reading(kTarget);
  s.dq(0) = std::numeric_limits<double>::infinity();
  EXPECT_FALSE(c.compute(s, kTarget).finite);
}

TEST(ImpedanceController, TheTcpOffsetIsUsed) {
  Pose F_T_TCP = Pose::Identity();
  F_T_TCP.translation() = Eigen::Vector3d(0, 0, 0.15);
  const ImpedanceController c(cfgWith([](ControlConfig& k) { k.nullspace_stiffness = 0.0; }), F_T_TCP);
  const Pose O_T_F = test::makePose({0.5, 0.0, 0.3}, Eigen::Quaterniond::Identity());
  Pose target = O_T_F * F_T_TCP;
  target.translation().x() -= 0.01;   // the TCP is 1 cm past the target in x
  const TorqueResult r = c.compute(reading(O_T_F), target);
  EXPECT_NEAR(r.tau(0), -8.0, 1e-9);
  EXPECT_NEAR(r.tau(4), 0.15 * -8.0, 1e-9);   // the force at the TCP makes a moment about the flange: r x F
  EXPECT_NEAR(r.O_T_TCP.translation().z(), 0.45, 1e-12);
}

}  // namespace
}  // namespace franka_control
