#include "franka_control/impedance_controller.h"

#include <algorithm>
#include <cmath>

#include "franka_control/pose_math.h"

namespace franka_control {
namespace {
constexpr double kControlPeriodS = 1e-3;   // the FCI's 1 kHz cycle (libfranka kDeltaT)
using Mat77 = Eigen::Matrix<double, 7, 7>;
using Mat66 = Eigen::Matrix<double, 6, 6>;
}  // namespace

ImpedanceController::ImpedanceController(const ControlConfig& cfg, const Pose& F_T_TCP)
    : cfg_(cfg), F_T_TCP_(F_T_TCP) {
  K_ << cfg.stiffness_translational_n_per_m, cfg.stiffness_rotational_nm_per_rad;
  D_ = 2.0 * cfg.damping_ratio * K_.cwiseSqrt();
}

void ImpedanceController::setNullspaceTarget(const Vec7& q_ns) { q_ns_ = q_ns; }

TorqueResult ImpedanceController::compute(const ArmReading& r, const Pose& O_T_TCP_target) const {
  TorqueResult out;
  out.O_T_TCP = tcpFromFlange(r.O_T_F, F_T_TCP_);
  const Eigen::Vector3d p = out.O_T_TCP.translation();
  const Eigen::Matrix3d R = out.O_T_TCP.linear();
  const Mat67 J = tcpJacobian(r.J_F, p - r.O_T_F.translation());

  // position error, clipped
  Eigen::Vector3d e_p = p - O_T_TCP_target.translation();
  out.position_error_m = e_p.norm();
  if (out.position_error_m > cfg_.max_position_error_m) e_p *= cfg_.max_position_error_m / out.position_error_m;

  // orientation error as in the libfranka example: q_e = q^-1 q_d, e_o = -R vec(q_e); clipped by angle
  Eigen::Quaterniond q(R);
  const Eigen::Quaterniond q_d(O_T_TCP_target.linear());
  if (q_d.coeffs().dot(q.coeffs()) < 0.0) q.coeffs() = -q.coeffs();
  const Eigen::Quaterniond q_e = q.inverse() * q_d;
  Eigen::Vector3d v_e = q_e.vec();
  out.rotation_error_rad = 2.0 * std::atan2(v_e.norm(), std::abs(q_e.w()));
  if (out.rotation_error_rad > cfg_.max_rotation_error_rad) {
    v_e *= std::sin(0.5 * cfg_.max_rotation_error_rad) / v_e.norm();
  }
  Vec6 e;
  e << e_p, -R * v_e;

  // spring and damper at the TCP, task torque
  const Vec6 F = -K_.cwiseProduct(e) - D_.cwiseProduct(J * r.dq);
  const Vec7 tau_task = J.transpose() * F;

  // nullspace: N = I - J^T (J J^T + lambda^2 I)^-1 J
  const double lambda2 = cfg_.jacobian_pinv_damping * cfg_.jacobian_pinv_damping;
  const Mat66 JJt = J * J.transpose() + lambda2 * Mat66::Identity();
  const Mat67 JJt_inv_J = JJt.ldlt().solve(J);
  const Mat77 N = Mat77::Identity() - J.transpose() * JJt_inv_J;
  const double k_ns = cfg_.nullspace_stiffness;
  const Vec7 tau_null = N * (k_ns * (q_ns_ - r.q) - 2.0 * std::sqrt(k_ns) * r.dq);

  Vec7 tau = tau_task + tau_null + r.coriolis;   // gravity: the robot compensates it

  // Checked before any clamp: std::min/max would turn a NaN into a finite limit.
  out.finite = tau.allFinite() && r.tau_J_d.allFinite();
  if (!out.finite) {
    out.tau.setZero();
    return out;
  }
  const double max_step = cfg_.torque_rate_max_nm_per_s * kControlPeriodS;
  for (int i = 0; i < 7; ++i) {
    const double lim = cfg_.torque_max_nm(i);
    double t = std::clamp(tau(i), -lim, lim);
    t = std::clamp(t, r.tau_J_d(i) - max_step, r.tau_J_d(i) + max_step);
    tau(i) = std::clamp(t, -lim, lim);
  }
  out.tau = tau;
  return out;
}

}  // namespace franka_control
