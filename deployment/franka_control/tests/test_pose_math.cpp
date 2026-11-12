#include <gtest/gtest.h>

#include <random>

#include "franka_control/pose_math.h"
#include "support/test_support.h"

namespace franka_control {
namespace {

using test::makePose;
using test::rotvec;

// Port of main-sim's test_tcp_composition_against_independent_4x4.
TEST(PoseMath, TcpFromFlangeMatchesAnIndependent4x4Formula) {
  Eigen::Matrix3d R_ft;
  R_ft << 1, 0, 0, 0, 0.966, -0.259, 0, 0.259, 0.966;
  Pose F_T_TCP = Pose::Identity();
  F_T_TCP.linear() = Eigen::Quaterniond(R_ft).normalized().toRotationMatrix();
  F_T_TCP.translation() = Eigen::Vector3d(0.0, 0.0644, 0.3114);
  const Pose O_T_F = makePose({0.45, 0.1, 0.4}, rotvec({M_PI, 0.1, -0.2}));
  const Eigen::Vector3d p_ind = O_T_F.translation() + O_T_F.linear() * F_T_TCP.translation();
  const Eigen::Matrix3d R_ind = O_T_F.linear() * F_T_TCP.linear();
  const Pose got = tcpFromFlange(O_T_F, F_T_TCP);
  EXPECT_TRUE(got.translation().isApprox(p_ind, 1e-12));
  EXPECT_TRUE(got.linear().isApprox(R_ind, 1e-12));
  EXPECT_TRUE(flangeFromTcp(got, F_T_TCP).matrix().isApprox(O_T_F.matrix(), 1e-12));   // exactly once each way
  // The fixed hand F_T_TCP: -45 deg about the flange z axis, fingertips 150 mm out.
  const Config cfg = test::testConfig();
  EXPECT_NEAR(cfg.F_T_TCP.translation().z(), 0.150, 1e-12);
  EXPECT_TRUE(cfg.F_T_TCP.linear().isApprox(Eigen::AngleAxisd(-M_PI / 4, Eigen::Vector3d::UnitZ()).toRotationMatrix(),
                                            1e-12));
}

TEST(PoseMath, TcpJacobianGivesFlangeTwistPlusOmegaCrossR) {
  std::mt19937 rng(1);
  std::uniform_real_distribution<double> u(-1.0, 1.0);
  for (int k = 0; k < 50; ++k) {
    Mat67 J_F;
    for (int i = 0; i < 6; ++i)
      for (int j = 0; j < 7; ++j) J_F(i, j) = u(rng);
    Vec7 dq;
    for (int j = 0; j < 7; ++j) dq(j) = u(rng);
    const Eigen::Vector3d r(u(rng), u(rng), u(rng));
    const Vec6 twist_F = J_F * dq;
    const Vec6 twist_T = tcpJacobian(J_F, r) * dq;
    const Eigen::Vector3d v_F = twist_F.head<3>(), w = twist_F.tail<3>();
    EXPECT_TRUE(twist_T.head<3>().isApprox(v_F + w.cross(r), 1e-12));
    EXPECT_TRUE(twist_T.tail<3>().isApprox(w, 1e-12));
  }
}

TEST(PoseMath, RotationAngleBetweenQuaternions) {
  const Eigen::Quaterniond a = rotvec({0, 0, 0.3});
  EXPECT_NEAR(rotationAngle(a, rotvec({0, 0, 0.8})), 0.5, 1e-12);
  Eigen::Quaterniond neg = rotvec({0, 0, 0.8});
  neg.coeffs() = -neg.coeffs();   // the same rotation
  EXPECT_NEAR(rotationAngle(a, neg), 0.5, 1e-12);
  EXPECT_NEAR(rotationAngle(a, a), 0.0, 1e-12);
}

TEST(PoseMath, RigidCheckRefusesTransposedAndScaled) {
  const Pose pose = makePose({0.45, 0.1, 0.3}, rotvec({M_PI, 0.2, 0.1}));
  const auto m = rowMajorFromPose(pose);
  EXPECT_TRUE(isRigidRowMajor(m, 1e-6));
  std::array<double, 16> transposed{};
  for (int r = 0; r < 4; ++r)
    for (int c = 0; c < 4; ++c) transposed[4 * c + r] = m[4 * r + c];
  EXPECT_FALSE(isRigidRowMajor(transposed, 1e-6));   // translation lands in the bottom row
  auto scaled = m;
  for (int r = 0; r < 3; ++r)
    for (int c = 0; c < 3; ++c) scaled[4 * r + c] *= 1.01;
  EXPECT_FALSE(isRigidRowMajor(scaled, 1e-6));
  auto reflection = m;
  for (int r = 0; r < 3; ++r) reflection[4 * r] = -reflection[4 * r];
  EXPECT_FALSE(isRigidRowMajor(reflection, 1e-6));
}

TEST(PoseMath, RowMajorAndColumnMajorConversions) {
  const Pose pose = makePose({0.45, -0.1, 0.3}, rotvec({0.1, 0.2, 0.3}));
  const auto m = rowMajorFromPose(pose);
  EXPECT_DOUBLE_EQ(m[3], 0.45);   // row 0, column 3
  EXPECT_DOUBLE_EQ(m[7], -0.1);
  EXPECT_TRUE(poseFromRowMajor(m).matrix().isApprox(pose.matrix(), 1e-12));
  std::array<double, 16> col{};
  Eigen::Map<Eigen::Matrix4d>(col.data()) = pose.matrix();
  EXPECT_DOUBLE_EQ(col[12], 0.45);   // libfranka's layout: translation at 12..14
  EXPECT_TRUE(poseFromColumnMajor(col).matrix().isApprox(pose.matrix(), 1e-12));
}

}  // namespace
}  // namespace franka_control
