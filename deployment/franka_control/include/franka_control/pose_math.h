// Free functions on poses: flange <-> TCP, the TCP Jacobian, rotation angles, row/column-major conversion and the
// rigid-transform check of the contracts (STRUCTURE.md 5.4, 8.4).
#pragma once

#include <array>

#include "franka_control/types.h"

namespace franka_control {

// O_T_TCP = O_T_F * F_T_TCP.
Pose tcpFromFlange(const Pose& O_T_F, const Pose& F_T_TCP);
// O_T_F = O_T_TCP * F_T_TCP^-1.
Pose flangeFromTcp(const Pose& O_T_TCP, const Pose& F_T_TCP);

// [r]x: the matrix with skew(r) * v == r x v.
Eigen::Matrix3d skew(const Eigen::Vector3d& r);

// The TCP Jacobian from the flange Jacobian, r = p_TCP - p_flange (base frame):
// J_lin = J_F,lin - [r]x J_F,ang; J_ang = J_F,ang.
Mat67 tcpJacobian(const Mat67& J_F, const Eigen::Vector3d& r);

// Angle (rad, in [0, pi]) of the rotation between two orientations.
double rotationAngle(const Eigen::Quaterniond& a, const Eigen::Quaterniond& b);

// 4x4 row-major (the wire form) <-> Pose. The rotation is passed through a unit quaternion on the way out, so the
// result is orthonormal to machine precision.
Pose poseFromRowMajor(const std::array<double, 16>& m);
std::array<double, 16> rowMajorFromPose(const Pose& pose);
// libfranka gives 4x4 matrices column-major.
Pose poseFromColumnMajor(const std::array<double, 16>& m);

// The contracts' rigid check on a row-major 4x4: all finite, bottom row exactly [0, 0, 0, 1],
// max |R^T R - I| <= tol, det(R) > 0.
bool isRigidRowMajor(const std::array<double, 16>& m, double tol);

// A pose with an orthonormal rotation (through a unit quaternion).
Pose normalized(const Pose& pose);

}  // namespace franka_control
