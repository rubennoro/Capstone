#include "franka_control/pose_math.h"

#include <algorithm>
#include <cmath>

namespace franka_control {

Pose tcpFromFlange(const Pose& O_T_F, const Pose& F_T_TCP) { return O_T_F * F_T_TCP; }

Pose flangeFromTcp(const Pose& O_T_TCP, const Pose& F_T_TCP) { return O_T_TCP * F_T_TCP.inverse(); }

Eigen::Matrix3d skew(const Eigen::Vector3d& r) {
  Eigen::Matrix3d s;
  s << 0.0, -r.z(), r.y(), r.z(), 0.0, -r.x(), -r.y(), r.x(), 0.0;
  return s;
}

Mat67 tcpJacobian(const Mat67& J_F, const Eigen::Vector3d& r) {
  Mat67 J = J_F;
  J.topRows<3>() = J_F.topRows<3>() - skew(r) * J_F.bottomRows<3>();
  return J;
}

double rotationAngle(const Eigen::Quaterniond& a, const Eigen::Quaterniond& b) {
  const Eigen::Quaterniond d = a.normalized().conjugate() * b.normalized();
  return 2.0 * std::atan2(d.vec().norm(), std::abs(d.w()));
}

Pose poseFromRowMajor(const std::array<double, 16>& m) {
  Pose pose = Pose::Identity();
  for (int r = 0; r < 3; ++r) {
    for (int c = 0; c < 3; ++c) pose.linear()(r, c) = m[static_cast<size_t>(4 * r + c)];
    pose.translation()(r) = m[static_cast<size_t>(4 * r + 3)];
  }
  return pose;
}

Pose normalized(const Pose& pose) {
  Pose out = Pose::Identity();
  out.linear() = Eigen::Quaterniond(pose.linear()).normalized().toRotationMatrix();
  out.translation() = pose.translation();
  return out;
}

std::array<double, 16> rowMajorFromPose(const Pose& pose) {
  const Pose n = normalized(pose);
  std::array<double, 16> m{};
  for (int r = 0; r < 3; ++r) {
    for (int c = 0; c < 3; ++c) m[static_cast<size_t>(4 * r + c)] = n.linear()(r, c);
    m[static_cast<size_t>(4 * r + 3)] = n.translation()(r);
  }
  m[12] = 0.0;
  m[13] = 0.0;
  m[14] = 0.0;
  m[15] = 1.0;
  return m;
}

Pose poseFromColumnMajor(const std::array<double, 16>& m) {
  Pose pose = Pose::Identity();
  pose.matrix() = Eigen::Map<const Eigen::Matrix4d>(m.data());  // Eigen's default storage is column-major
  pose.makeAffine();
  return pose;
}

bool isRigidRowMajor(const std::array<double, 16>& m, double tol) {
  for (double v : m) {
    if (!std::isfinite(v)) return false;
  }
  if (m[12] != 0.0 || m[13] != 0.0 || m[14] != 0.0 || m[15] != 1.0) return false;
  Eigen::Matrix3d R;
  for (int r = 0; r < 3; ++r) {
    for (int c = 0; c < 3; ++c) R(r, c) = m[static_cast<size_t>(4 * r + c)];
  }
  const double err = (R.transpose() * R - Eigen::Matrix3d::Identity()).cwiseAbs().maxCoeff();
  return err <= tol && R.determinant() > 0.0;
}

}  // namespace franka_control
