#include "franka_control/safety_gate.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <optional>

#include "franka_control/clock.h"
#include "franka_control/pose_math.h"

namespace franka_control {
namespace {

// Floating-point slack for "on the boundary" / "exactly the allowed distance": far below anything physical.
constexpr double kBoundaryTol = 1e-12;
constexpr double kSpeedTol = 1e-9;
constexpr double kRadToDeg = 180.0 / M_PI;
const char* kAxes[3] = {"x", "y", "z"};

std::string fmt(double v, int precision) {
  char buf[64];
  std::snprintf(buf, sizeof(buf), "%.*f", precision, v);
  return buf;
}

}  // namespace

uint64_t GateResult::failedCount() const {
  return static_cast<uint64_t>(std::count_if(rejected.begin(), rejected.end(), [](const GateIssue& i) {
    return i.failed;
  }));
}

SafetyGate::SafetyGate(const GateConfig& cfg, const Pose& F_T_TCP, double gripper_width_max_m)
    : cfg_(cfg), q_F_TCP_(Eigen::Quaterniond(F_T_TCP.linear()).normalized()), width_max_m_(gripper_width_max_m) {
  cfg_.cone_axis.normalize();
}

bool SafetyGate::insideBox(const Eigen::Vector3d& p) const {
  for (int i = 0; i < 3; ++i) {
    if (!(p(i) >= cfg_.workspace_min_m(i) && p(i) <= cfg_.workspace_max_m(i))) return false;
  }
  return true;
}

double SafetyGate::tiltDeg(const Eigen::Quaterniond& q_tcp) const {
  const Eigen::Quaterniond q_F = q_tcp.normalized() * q_F_TCP_.conjugate();
  const Eigen::Vector3d z = q_F.toRotationMatrix().col(2);
  return std::acos(std::clamp(z.dot(cfg_.cone_axis), -1.0, 1.0)) * kRadToDeg;
}

GateResult SafetyGate::check(const TrajectoryPoint& start, const std::vector<TrajectoryPoint>& requested,
                             int64_t now_ns) const {
  struct Candidate {
    uint64_t index;
    TrajectoryPoint p;
  };
  GateResult res;
  std::vector<Candidate> candidates;
  std::vector<uint64_t> non_skipped;
  std::vector<GateIssue> failures;
  std::vector<GateIssue> pulled_back;

  for (size_t i = 0; i < requested.size(); ++i) {
    TrajectoryPoint wp = requested[i];
    if (wp.t_ns <= now_ns) {   // time already passed: skipped, not checked, not executed
      ++res.skipped;
      continue;
    }
    non_skipped.push_back(i);
    std::optional<std::string> reject;
    std::vector<std::string> pulls;

    // 1. finite
    if (!wp.p.allFinite() || !wp.q.coeffs().allFinite() || !std::isfinite(wp.width_m) || wp.q.norm() < 0.5) {
      reject = "not finite";
    }
    // 2. not too far ahead
    if (!reject) {
      const double lead_s = nsToSeconds(wp.t_ns - now_ns);
      if (lead_s > cfg_.max_lead_s) {
        reject = "time " + fmt(lead_s, 2) + " s ahead > max_lead_s " + fmt(cfg_.max_lead_s, 2) +
                 " s (is t_ns on CLOCK_MONOTONIC?)";
      }
    }
    // 3. gripper range
    if (!reject && !(wp.width_m >= 0.0 && wp.width_m <= width_max_m_)) {
      reject = "gripper width " + fmt(wp.width_m, 3) + " m outside [0, " + fmt(width_max_m_, 3) + "] m";
    }
    // 4. workspace box (TCP position): pull back within the margin, reject beyond
    if (!reject) {
      std::vector<std::string> beyond;
      Eigen::Vector3d p = wp.p;
      for (int a = 0; a < 3; ++a) {
        const double lo = cfg_.workspace_min_m(a), hi = cfg_.workspace_max_m(a);
        const double below = lo - wp.p(a), above = wp.p(a) - hi;
        const double out = std::max({below, above, 0.0});
        if (out <= 0.0) continue;
        if (out > cfg_.box_margin_m + kBoundaryTol) {
          const bool low = below > 0.0;
          beyond.push_back("outside workspace box by " + fmt(out, 3) + " m on " + kAxes[a] + " (" + kAxes[a] +
                           " = " + fmt(wp.p(a), 3) + " m " + (low ? "< min " : "> max ") + fmt(low ? lo : hi, 3) +
                           " m, margin " + fmt(cfg_.box_margin_m, 3) + " m)");
        } else {
          p(a) = below > 0.0 ? lo : hi;
          pulls.push_back(std::string(kAxes[a]) + " " + fmt(wp.p(a), 3) + " -> " + fmt(p(a), 3) + " m");
        }
      }
      if (!beyond.empty()) {
        std::string r = beyond[0];
        for (size_t k = 1; k < beyond.size(); ++k) r += "; " + beyond[k];
        reject = r;
      } else if (!pulls.empty()) {
        std::string r = "pulled back onto box: " + pulls[0];
        for (size_t k = 1; k < pulls.size(); ++k) r += ", " + pulls[k];
        pulls = {r};
        wp.p = p;
      }
    }
    // 5. tilt cone, measured on the flange's approach axis (flange = O_T_TCP * F_T_TCP^-1)
    if (!reject) {
      const Eigen::Quaterniond q_F = wp.q.normalized() * q_F_TCP_.conjugate();
      const Eigen::Vector3d z = q_F.toRotationMatrix().col(2);
      const double tilt_rad = std::acos(std::clamp(z.dot(cfg_.cone_axis), -1.0, 1.0));
      const double tilt_deg = tilt_rad * kRadToDeg;
      if (tilt_deg > cfg_.cone_max_deg) {
        if (tilt_deg > cfg_.cone_max_deg + cfg_.cone_margin_deg + kBoundaryTol) {
          reject = "tilt " + fmt(tilt_deg, 1) + " deg from approach axis > " + fmt(cfg_.cone_max_deg, 1) + " + " +
                   fmt(cfg_.cone_margin_deg, 1) + " deg";
        } else {
          // Turn back onto the boundary about the shortest axis (z x cone_axis), the TCP position unchanged.
          Eigen::Vector3d k = z.cross(cfg_.cone_axis);
          k = k.norm() > 1e-9 ? k.normalized() : z.unitOrthogonal();
          const Eigen::Quaterniond turn(Eigen::AngleAxisd(tilt_rad - cfg_.cone_max_deg / kRadToDeg, k));
          wp.q = ((turn * q_F) * q_F_TCP_).normalized();
          pulls.push_back("pulled back onto cone: tilt " + fmt(tilt_deg, 1) + " -> " + fmt(cfg_.cone_max_deg, 1) +
                          " deg");
        }
      }
    }

    if (reject) {
      failures.push_back(GateIssue{i, *reject, true});
      continue;
    }
    if (!pulls.empty()) {
      std::string r = pulls[0];
      for (size_t k = 1; k < pulls.size(); ++k) r += "; " + pulls[k];
      pulled_back.push_back(GateIssue{i, r, true});
    }
    candidates.push_back(Candidate{i, wp});
  }

  if (!failures.empty()) {   // the whole request is refused; every non-skipped waypoint is accounted for
    const uint64_t first = failures.front().index;
    size_t f = 0;
    for (uint64_t i : non_skipped) {
      if (f < failures.size() && failures[f].index == i) {
        res.rejected.push_back(failures[f++]);
      } else {
        res.rejected.push_back(
            GateIssue{i, "not used: waypoint " + std::to_string(first) + " of this request was rejected", false});
      }
    }
    return res;
  }

  // 6. speed (leaky bucket), position and rotation each, on the pulled-back poses (6.2)
  const double v = cfg_.v_max_m_per_s, w = cfg_.w_max_rad_per_s;
  double left_p = v * cfg_.slack_s, left_r = w * cfg_.slack_s;
  TrajectoryPoint prev = start;
  for (auto& c : candidates) {
    const double dt = std::max(0.0, nsToSeconds(c.p.t_ns - prev.t_ns));
    std::vector<std::string> why;
    const double allowed_p = std::min(left_p, v * cfg_.slack_s) + v * dt;
    const double d = (c.p.p - prev.p).norm();
    if (d > allowed_p + kSpeedTol) {
      c.p.p = prev.p + (c.p.p - prev.p) * (allowed_p / d);
      why.push_back(fmt(d, 4) + " m in " + fmt(dt, 3) + " s -> " + fmt(allowed_p, 4) + " m (v_max " + fmt(v, 2) +
                    " m/s)");
    }
    left_p = allowed_p - std::min(d, allowed_p);
    const double allowed_r = std::min(left_r, w * cfg_.slack_s) + w * dt;
    const double ang = rotationAngle(prev.q, c.p.q);
    if (ang > allowed_r + kSpeedTol) {
      c.p.q = prev.q.slerp(allowed_r / ang, c.p.q).normalized();
      why.push_back(fmt(ang, 4) + " rad in " + fmt(dt, 3) + " s -> " + fmt(allowed_r, 4) + " rad (w_max " +
                    fmt(w, 2) + " rad/s)");
    }
    left_r = allowed_r - std::min(ang, allowed_r);
    if (!why.empty()) {
      std::string r = "shortened: " + why[0];
      for (size_t k = 1; k < why.size(); ++k) r += "; " + why[k];
      res.shortened.push_back(GateIssue{c.index, r, true});
    }
    res.accepted.push_back(c.p);
    prev = c.p;
  }
  res.pulled_back = std::move(pulled_back);
  return res;
}

}  // namespace franka_control
