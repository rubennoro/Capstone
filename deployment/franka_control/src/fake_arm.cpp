#include "franka_control/fake_arm.h"

#include <cstring>
#include <ctime>

#include "franka_control/pose_math.h"

namespace franka_control {
namespace {

constexpr int64_t kCycleNs = 1000000;           // the FCI's 1 kHz period
constexpr double kCycleS = 1e-3;
constexpr int64_t kCatchUpNs = 50000000;        // after a 50 ms stall (debugger, overloaded laptop) don't burst

}  // namespace

FakeArm::FakeArm(const FakeArmConfig& cfg, const Pose& F_T_TCP, const Vec7& q_home, Clock clock)
    : cfg_(cfg), F_T_TCP_(F_T_TCP), q_home_(q_home), clock_(std::move(clock)), q_(q_home) {
  const Pose O_T_F = flangeFromTcp(cfg.home_O_T_TCP, F_T_TCP);
  p_F_ = O_T_F.translation();
  R_F_ = O_T_F.linear();
}

ArmReading FakeArm::reading() const {
  ArmReading r;
  r.q = q_;
  r.dq = dq_;
  r.coriolis.setZero();
  r.tau_J_d = tau_last_;
  r.O_T_F = Pose::Identity();
  r.O_T_F.linear() = R_F_;
  r.O_T_F.translation() = p_F_;
  r.J_F.setZero();
  r.J_F.leftCols<6>().setIdentity();
  r.success_rate = 1.0;
  r.missed_packets = 0;
  return r;
}

ArmReading FakeArm::readOnce() { return reading(); }

ArmStatusFlags FakeArm::status() { return ArmStatusFlags{RobotMode::kIdle, "[]"}; }

std::string FakeArm::describe() const { return "FakeArm (simulated rigid body)"; }

void FakeArm::requestStop() { stop_requested_.store(true); }

void FakeArm::injectFault(const std::string& text) {
  const size_t n = std::min(text.size(), fault_text_.size() - 1);
  std::memcpy(fault_text_.data(), text.data(), n);
  fault_text_[n] = '\0';
  fault_pending_.store(true, std::memory_order_release);
}

void FakeArm::stallHoming(bool on) { stall_homing_.store(on); }

void FakeArm::ignoreFinish(bool on) { ignore_finish_.store(on); }

void FakeArm::waitNextCycle(int64_t& next_ns) {
  next_ns += kCycleNs;
  if (clock_() - next_ns > kCatchUpNs) next_ns = clock_();
  timespec ts{};
  ts.tv_sec = static_cast<time_t>(next_ns / kNsPerS);
  ts.tv_nsec = static_cast<long>(next_ns % kNsPerS);
  while (clock_nanosleep(CLOCK_MONOTONIC, TIMER_ABSTIME, &ts, nullptr) != 0) {
  }
}

void FakeArm::throwIfInterrupted() {
  if (stop_requested_.load()) throw ArmError(ArmError::Kind::kStopRequested, "FakeArm: stop requested");
  if (fault_pending_.exchange(false, std::memory_order_acquire)) {
    throw ArmError(ArmError::Kind::kReflex, std::string(fault_text_.data()));
  }
}

void FakeArm::integrate(const Vec7& tau, double dt_s) {
  const Eigen::Vector3d force = tau.head<3>();
  const Eigen::Vector3d moment = tau.segment<3>(3);
  v_ += force / cfg_.mass_kg * dt_s;
  w_ += moment / cfg_.inertia_kg_m2 * dt_s;
  p_F_ += v_ * dt_s;
  const double angle = w_.norm() * dt_s;
  if (angle > 0.0) {
    R_F_ = Eigen::AngleAxisd(angle, w_.normalized()).toRotationMatrix() * R_F_;
    R_F_ = Eigen::Quaterniond(R_F_).normalized().toRotationMatrix();
  }
  dq_.head<3>() = v_;
  dq_.segment<3>(3) = w_;
  dq_(6) = 0.0;
  q_ += dq_ * dt_s;
}

void FakeArm::runTorqueControl(const TorqueStep& step) {
  int64_t next = clock_();
  for (;;) {
    waitNextCycle(next);
    throwIfInterrupted();
    const TorqueCommand c = step(reading());
    integrate(c.tau, kCycleS);
    tau_last_ = c.tau;
    if (c.finished && !ignore_finish_.load()) return;
  }
}

void FakeArm::moveToJoints(const HomingMotion& m, const Observer& observe) {
  const Eigen::Vector3d p0 = p_F_;
  const Eigen::Quaterniond r0(R_F_);
  const Pose goal_F = flangeFromTcp(cfg_.home_O_T_TCP, F_T_TCP_);
  const Eigen::Vector3d p1 = goal_F.translation();
  const Eigen::Quaterniond r1(goal_F.linear());
  // The same minimum-jerk profile as m, as one number in [0, 1].
  const HomingMotion profile(Vec7::Zero(), Vec7::Ones(), 1e9, m.durationS());
  v_.setZero();
  w_.setZero();
  int64_t next = clock_();
  double t = 0.0;
  for (;;) {
    waitNextCycle(next);
    throwIfInterrupted();
    if (!stall_homing_.load()) t += kCycleS;
    const double s = profile.positionAt(t)(0);
    p_F_ = p0 + s * (p1 - p0);
    R_F_ = r0.slerp(s, r1).normalized().toRotationMatrix();
    q_ = m.positionAt(t);
    dq_ = m.velocityAt(t);
    observe(reading());
    if (!stall_homing_.load() && t >= m.durationS()) {
      q_ = m.goal();
      dq_.setZero();
      tau_last_.setZero();
      return;
    }
  }
}

}  // namespace franka_control
