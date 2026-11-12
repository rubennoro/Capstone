// Safety gate: every case of main-sim's tests/test_bridge_safety.py ported (STRUCTURE.md section 13), plus the new
// rules. Poses are TCP poses; the default F_T_TCP here is identity unless a test says otherwise.
#include <gtest/gtest.h>

#include <random>

#include "franka_control/pose_math.h"
#include "franka_control/safety_gate.h"
#include "support/test_support.h"

namespace franka_control {
namespace {

using test::point;
using test::rotvec;

const Eigen::Quaterniond kDown = rotvec({M_PI, 0, 0});   // TCP z straight down (main-sim's T() default)
constexpr int64_t kNow = 1000 * kNsPerS;
constexpr int64_t kMs = 1000000;

int64_t at(double s) { return kNow + secondsToNs(s); }

GateConfig cfgWith(const std::function<void(GateConfig&)>& f = {}) {
  GateConfig c;   // the config.yaml defaults
  if (f) f(c);
  return c;
}

SafetyGate gate(const GateConfig& c = GateConfig(), const Pose& F_T_TCP = Pose::Identity()) {
  return SafetyGate(c, F_T_TCP, 0.08);
}

TrajectoryPoint wp(double t_s, const Eigen::Vector3d& p, const Eigen::Quaterniond& q = kDown, double w = 0.04) {
  return point(at(t_s), p, q, w);
}

TrajectoryPoint startAt(const Eigen::Vector3d& p, const Eigen::Quaterniond& q = kDown) { return point(kNow, p, q, 0.04); }

// Every reply accounts for each waypoint exactly once.
void expectAccounted(const GateResult& r, size_t n) {
  EXPECT_EQ(r.accepted.size() + r.skipped + r.rejected.size(), n);
}

GateResult checkOne(const SafetyGate& g, const TrajectoryPoint& start, const TrajectoryPoint& w) {
  const GateResult r = g.check(start, {w}, kNow);
  expectAccounted(r, 1);
  return r;
}

// ---- ported from main-sim ----

TEST(SafetyGate, NonFiniteRejected) {   // test_nonfinite_rejected_and_last_kept
  const auto g = gate();
  const auto start = startAt({0.5, 0, 0.3});
  auto bad = wp(0.01, {std::nan(""), 0, 0.3});
  auto r = checkOne(g, start, bad);
  ASSERT_EQ(r.rejected.size(), 1u);
  EXPECT_EQ(r.rejected[0].reason, "not finite");
  EXPECT_TRUE(r.accepted.empty());
  Eigen::Quaterniond inf_q = kDown;
  inf_q.x() = std::numeric_limits<double>::infinity();
  r = checkOne(g, start, wp(0.02, {0.5, 0, 0.3}, inf_q));
  ASSERT_EQ(r.rejected.size(), 1u);
  EXPECT_EQ(r.rejected[0].reason, "not finite");
  r = checkOne(g, start, wp(0.02, {0.5, 0, 0.3}, kDown, std::nan("")));
  EXPECT_EQ(r.failedCount(), 1u);
}

TEST(SafetyGate, WorkspaceBoxPullBackWithinTheMarginRejectBeyond) {   // test_workspace_box_clamp
  const auto g = gate(cfgWith([](GateConfig& c) { c.v_max_m_per_s = 100; }));
  // each 5 mm out: pulled back onto the box and counted
  auto r = checkOne(g, startAt({0.80, -0.40, 0.05}), wp(0.1, {0.805, -0.405, 0.045}));
  ASSERT_EQ(r.accepted.size(), 1u);
  EXPECT_TRUE(r.accepted[0].p.isApprox(Eigen::Vector3d(0.80, -0.40, 0.05), 1e-15));
  ASSERT_EQ(r.pulled_back.size(), 1u);
  EXPECT_NE(r.pulled_back[0].reason.find("x 0.805 -> 0.800 m"), std::string::npos) << r.pulled_back[0].reason;
  EXPECT_NE(r.pulled_back[0].reason.find("z 0.045 -> 0.050 m"), std::string::npos);
  // 11 mm out: rejected, the reason names the axis and the numbers
  r = checkOne(g, startAt({0.8, 0, 0.3}), wp(0.1, {0.811, 0, 0.3}));
  ASSERT_EQ(r.rejected.size(), 1u);
  EXPECT_EQ(r.rejected[0].reason, "outside workspace box by 0.011 m on x (x = 0.811 m > max 0.800 m, margin 0.010 m)");
  EXPECT_TRUE(r.accepted.empty());
  // far out on every axis: each axis named
  r = checkOne(g, startAt({0.8, 0, 0.3}), wp(0.1, {2.0, -1.0, -0.5}));
  ASSERT_EQ(r.rejected.size(), 1u);
  for (const char* axis : {"on x", "on y", "on z"}) {
    EXPECT_NE(r.rejected[0].reason.find(axis), std::string::npos) << r.rejected[0].reason;
  }
  EXPECT_NE(r.rejected[0].reason.find("z = -0.500 m < min 0.050 m"), std::string::npos);
  // exactly on the boundary: accepted unchanged
  r = checkOne(g, startAt({0.8, 0.4, 0.7}), wp(0.1, {0.80, 0.40, 0.70}));
  ASSERT_EQ(r.accepted.size(), 1u);
  EXPECT_TRUE(r.pulled_back.empty());
  EXPECT_EQ(r.accepted[0].p, Eigen::Vector3d(0.80, 0.40, 0.70));
  // each face separately
  const GateConfig c;
  for (int axis = 0; axis < 3; ++axis) {
    for (int side = 0; side < 2; ++side) {
      Eigen::Vector3d p(0.5, 0.0, 0.3);
      const double limit = side ? c.workspace_max_m(axis) : c.workspace_min_m(axis);
      p(axis) = limit + (side ? 0.004 : -0.004);
      r = checkOne(g, startAt(p), wp(0.1, p));
      ASSERT_EQ(r.accepted.size(), 1u) << axis << side;
      EXPECT_NEAR(r.accepted[0].p(axis), limit, 1e-15);
      EXPECT_EQ(r.pulled_back.size(), 1u);
      p(axis) = limit + (side ? 0.02 : -0.02);
      r = checkOne(g, startAt({0.5, 0, 0.3}), wp(0.1, p));
      EXPECT_EQ(r.rejected.size(), 1u) << axis << side;
    }
  }
}

Eigen::Quaterniond tilted(double deg) { return rotvec({0, deg * M_PI / 180.0, 0}) * kDown; }

TEST(SafetyGate, OrientationConePullBackWithinTheMarginRejectBeyond) {   // test_orientation_cone_clamp
  const auto g = gate(cfgWith([](GateConfig& c) {
    c.cone_max_deg = 30;
    c.w_max_rad_per_s = 100;
  }));
  const Eigen::Vector3d p(0.5, 0, 0.3);
  auto r = checkOne(g, startAt(p, tilted(29.9)), wp(0.1, p, tilted(29.9)));
  ASSERT_EQ(r.accepted.size(), 1u);
  EXPECT_TRUE(r.pulled_back.empty());
  EXPECT_NEAR(rotationAngle(r.accepted[0].q, tilted(29.9)), 0.0, 1e-12);

  r = checkOne(g, startAt(p, tilted(31.5)), wp(0.1, p, tilted(31.5)));
  ASSERT_EQ(r.accepted.size(), 1u);
  ASSERT_EQ(r.pulled_back.size(), 1u);
  EXPECT_NEAR(g.tiltDeg(r.accepted[0].q), 30.0, 1e-6);
  const Eigen::Matrix3d R = r.accepted[0].q.toRotationMatrix();
  EXPECT_TRUE((R * R.transpose()).isApprox(Eigen::Matrix3d::Identity(), 1e-9));
  EXPECT_EQ(r.accepted[0].p, p);   // the TCP position unchanged
  EXPECT_EQ(r.pulled_back[0].reason, "pulled back onto cone: tilt 31.5 -> 30.0 deg");

  r = checkOne(g, startAt(p), wp(0.1, p, tilted(32.1)));
  ASSERT_EQ(r.rejected.size(), 1u);
  EXPECT_EQ(r.rejected[0].reason, "tilt 32.1 deg from approach axis > 30.0 + 2.0 deg");
  r = checkOne(g, startAt(p), wp(0.1, p, rotvec({0, M_PI / 2, 0})));   // z horizontal (+x): 90 deg
  EXPECT_EQ(r.rejected.size(), 1u);
}

TEST(SafetyGate, StepPositionIsShortenedNotRejected) {   // test_step_clamp_position_is_clamp_not_reject
  const auto g = gate();
  auto r = checkOne(g, startAt({0.5, 0, 0.3}), wp(0.005, {0.6, 0, 0.3}));
  ASSERT_EQ(r.accepted.size(), 1u);
  ASSERT_EQ(r.shortened.size(), 1u);
  EXPECT_NEAR(r.accepted[0].p.x(), 0.5 + 0.25 * (0.005 + 0.010), 1e-12);   // 3.75 mm
  // waypoints every 5 ms toward the same far point converge to it (executed slower, not frozen)
  std::vector<TrajectoryPoint> many;
  for (int k = 1; k <= 200; ++k) many.push_back(wp(0.005 * k, {0.6, 0, 0.3}));
  r = g.check(startAt({0.5, 0, 0.3}), many, kNow);
  expectAccounted(r, many.size());
  EXPECT_NEAR(r.accepted.back().p.x(), 0.6, 1e-9);
}

TEST(SafetyGate, StepRotationIsShortened) {   // test_step_clamp_rotation
  const auto g = gate();
  const auto r = checkOne(g, startAt({0.5, 0, 0.3}), wp(0.001, {0.5, 0, 0.3}, rotvec({0, 0, 0.5}) * kDown));
  ASSERT_EQ(r.shortened.size(), 1u);
  EXPECT_NEAR(rotationAngle(kDown, r.accepted[0].q), 1.0 * (0.001 + 0.010), 1e-9);   // 0.011 rad
}

TEST(SafetyGate, ALongGapGivesOnlyTheSlack) {   // test_long_gap_gives_only_the_slack
  const auto g = gate(cfgWith([](GateConfig& c) { c.max_lead_s = 100; }));
  // 10 s still, then 10 cm 5 ms later: only v_max * (5 ms + slack)
  auto r = g.check(startAt({0.5, 0, 0.3}), {wp(10.0, {0.5, 0, 0.3}), wp(10.005, {0.6, 0, 0.3})}, kNow);
  expectAccounted(r, 2);
  ASSERT_EQ(r.accepted.size(), 2u);
  EXPECT_NEAR(r.accepted[1].p.x(), 0.5 + 0.25 * (0.005 + 0.010), 1e-12);
  // 20 cm spread over 10 s is 0.02 m/s: accepted unchanged
  r = checkOne(g, startAt({0.5, 0, 0.3}), wp(10.0, {0.7, 0, 0.3}));
  EXPECT_TRUE(r.shortened.empty());
  EXPECT_NEAR(r.accepted[0].p.x(), 0.7, 1e-15);
}

TEST(SafetyGate, BoxPullBackRunsBeforeTheSpeedCheck) {   // test_order_box_before_step
  const auto g = gate();
  auto r = checkOne(g, startAt({0.795, 0, 0.3}), wp(0.010, {0.805, 0, 0.3}));
  ASSERT_EQ(r.accepted.size(), 1u);
  EXPECT_EQ(r.pulled_back.size(), 1u);
  EXPECT_TRUE(r.shortened.empty());   // the remaining 5 mm fits 0.25 * (0.010 + 0.010)
  EXPECT_NEAR(r.accepted[0].p.x(), 0.800, 1e-15);
  r = checkOne(g, startAt({0.805, 0, 0.3}), wp(1.0, {0.811, 0, 0.3}));   // slow, but beyond the margin
  EXPECT_EQ(r.rejected.size(), 1u);
  r = checkOne(g, startAt({0.5, 0, 0.3}), wp(0.010, {0.7, 0, 0.3}));     // inside, too fast: only shortened
  EXPECT_EQ(r.shortened.size(), 1u);
  EXPECT_TRUE(r.pulled_back.empty());
  EXPECT_TRUE(r.rejected.empty());
}

// The plan evaluated at 1 kHz (linear position between points, as the loop does); the largest x motion within any
// 100 ms window.
double maxWindowMotion(const TrajectoryPoint& start, const std::vector<TrajectoryPoint>& pts) {
  std::vector<TrajectoryPoint> plan = {start};
  plan.insert(plan.end(), pts.begin(), pts.end());
  std::vector<double> xs;
  size_t seg = 0;
  for (int64_t t = plan.front().t_ns; t <= plan.back().t_ns; t += kMs) {
    while (seg + 1 < plan.size() - 1 && t >= plan[seg + 1].t_ns) ++seg;
    const auto& a = plan[seg];
    const auto& b = plan[seg + 1];
    const double s = std::clamp(static_cast<double>(t - a.t_ns) / static_cast<double>(b.t_ns - a.t_ns), 0.0, 1.0);
    xs.push_back(a.p.x() + s * (b.p.x() - a.p.x()));
  }
  double best = 0.0;
  for (size_t i = 0; i + 100 < xs.size(); ++i) best = std::max(best, std::abs(xs[i + 100] - xs[i]));
  return best;
}

class SafetyGateRate : public ::testing::TestWithParam<std::string> {};

TEST_P(SafetyGateRate, SpeedBoundIndependentOfRequestRate) {   // test_speed_bound_independent_of_request_rate
  const auto g = gate(cfgWith([](GateConfig& c) {
    c.workspace_max_m = {5, 5, 5};
    c.max_lead_s = 100;
  }));
  std::vector<double> times;
  if (GetParam() == "200hz") {
    for (int k = 1; k <= 200; ++k) times.push_back(k * 0.005);
  } else if (GetParam() == "1khz") {
    for (int k = 1; k <= 1000; ++k) times.push_back(k * 0.001);
  } else {   // bursts of 10 within 0.2 ms, every 20 ms
    for (int k = 1; k <= 50; ++k)
      for (int j = 0; j < 10; ++j) times.push_back(k * 0.02 + j * 2e-5);
  }
  std::vector<TrajectoryPoint> pts;
  for (double t : times) pts.push_back(wp(t, {0.5 + 2.0 * t, 0, 0.3}));   // commanded 2 m/s
  const auto start = startAt({0.5, 0, 0.3});
  const GateResult r = g.check(start, pts, kNow);
  expectAccounted(r, pts.size());
  ASSERT_EQ(r.accepted.size(), pts.size());
  const double m = maxWindowMotion(start, r.accepted);
  EXPECT_LE(m, 0.25 * 0.1 * 1.1 + 1e-9) << GetParam();
  if (GetParam() != "bursts") {
    EXPECT_GE(m, 0.25 * 0.1 * 0.9) << GetParam();
  }
}

INSTANTIATE_TEST_SUITE_P(Patterns, SafetyGateRate, ::testing::Values("200hz", "1khz", "bursts"));

TEST(SafetyGate, ReplyJitterDoesNotShorten) {   // test_reply_jitter_does_not_clamp
  const auto g = gate(cfgWith([](GateConfig& c) {
    c.workspace_max_m = {5, 5, 5};
    c.max_lead_s = 100;
  }));
  std::mt19937 rng(0);
  std::uniform_real_distribution<double> jitter(-0.002, 0.002);
  const double v = 0.9 * 0.25;
  std::vector<TrajectoryPoint> pts;
  for (int k = 1; k < 400; ++k) {
    const double t_cmd = k * 0.005;
    pts.push_back(wp(t_cmd + jitter(rng), {0.5 + v * t_cmd, 0, 0.3}));
  }
  const auto r = g.check(startAt({0.5, 0, 0.3}), pts, kNow);
  expectAccounted(r, pts.size());
  EXPECT_TRUE(r.shortened.empty());
}

// ---- new rules ----

TEST(SafetyGate, GripperWidthOutsideItsRangeRejected) {
  const auto g = gate();
  auto r = checkOne(g, startAt({0.5, 0, 0.3}), wp(0.1, {0.5, 0, 0.3}, kDown, -0.001));
  ASSERT_EQ(r.rejected.size(), 1u);
  EXPECT_EQ(r.rejected[0].reason, "gripper width -0.001 m outside [0, 0.080] m");
  r = checkOne(g, startAt({0.5, 0, 0.3}), wp(0.1, {0.5, 0, 0.3}, kDown, 0.095));
  ASSERT_EQ(r.rejected.size(), 1u);
  EXPECT_EQ(r.rejected[0].reason, "gripper width 0.095 m outside [0, 0.080] m");
  r = checkOne(g, startAt({0.5, 0, 0.3}), wp(0.1, {0.5, 0, 0.3}, kDown, 0.08));
  EXPECT_EQ(r.accepted.size(), 1u);
}

TEST(SafetyGate, PastWaypointsAreSkippedUncheckedAndCounted) {
  const auto g = gate();
  auto past_bad = point(kNow, {5.0, 0, 0}, kDown, 1.0);   // t == now, and would be rejected: never checked
  auto past = point(kNow - 10 * kMs, {0.5, 0, 0.3}, kDown, 0.04);
  auto r = g.check(startAt({0.5, 0, 0.3}), {past, past_bad, wp(0.05, {0.5, 0, 0.3})}, kNow);
  expectAccounted(r, 3);
  EXPECT_EQ(r.skipped, 2u);
  EXPECT_EQ(r.accepted.size(), 1u);
  EXPECT_TRUE(r.rejected.empty());
  r = g.check(startAt({0.5, 0, 0.3}), {past, past_bad}, kNow);   // all past: accepted 0, skipped N, nothing rejected
  expectAccounted(r, 2);
  EXPECT_EQ(r.skipped, 2u);
  EXPECT_TRUE(r.accepted.empty());
  EXPECT_TRUE(r.rejected.empty());
}

TEST(SafetyGate, TooFarAheadRejected) {
  const auto g = gate();
  const auto r = checkOne(g, startAt({0.5, 0, 0.3}), wp(7.2, {0.5, 0, 0.3}));
  ASSERT_EQ(r.rejected.size(), 1u);
  EXPECT_EQ(r.rejected[0].reason, "time 7.20 s ahead > max_lead_s 5.00 s (is t_ns on CLOCK_MONOTONIC?)");
}

TEST(SafetyGate, AnyRejectionRefusesTheWholeRequest) {
  const auto g = gate();
  const auto past = point(kNow - kMs, {0.5, 0, 0.3}, kDown, 0.04);
  const auto r = g.check(startAt({0.5, 0, 0.3}),
                         {past, wp(0.05, {0.5, 0, 0.3}), wp(0.10, {0.5, 0, 0.9}), wp(0.15, {0.5, 0, 0.3}),
                          wp(0.20, {0.5, 0, 0.3}, kDown, 0.2)},
                         kNow);
  expectAccounted(r, 5);
  EXPECT_TRUE(r.accepted.empty());
  EXPECT_TRUE(r.shortened.empty());
  EXPECT_TRUE(r.pulled_back.empty());
  EXPECT_EQ(r.skipped, 1u);
  ASSERT_EQ(r.rejected.size(), 4u);   // every non-skipped one
  EXPECT_EQ(r.failedCount(), 2u);
  EXPECT_EQ(r.rejected[0].index, 1u);
  EXPECT_EQ(r.rejected[0].reason, "not used: waypoint 2 of this request was rejected");
  EXPECT_FALSE(r.rejected[0].failed);
  EXPECT_EQ(r.rejected[1].index, 2u);
  EXPECT_NE(r.rejected[1].reason.find("on z"), std::string::npos);
  EXPECT_EQ(r.rejected[3].index, 4u);
  EXPECT_NE(r.rejected[3].reason.find("gripper width"), std::string::npos);
}

TEST(SafetyGate, AWaypointCanBeBothPulledBackAndShortened) {
  const auto g = gate();
  const auto r = checkOne(g, startAt({0.5, 0, 0.3}), wp(0.005, {0.805, 0, 0.3}));
  EXPECT_EQ(r.pulled_back.size(), 1u);
  EXPECT_EQ(r.shortened.size(), 1u);
  ASSERT_EQ(r.accepted.size(), 1u);
  EXPECT_NEAR(r.accepted[0].p.x(), 0.5 + 0.25 * 0.015, 1e-12);
  EXPECT_NE(r.shortened[0].reason.find("v_max 0.25 m/s"), std::string::npos) << r.shortened[0].reason;
}

TEST(SafetyGate, TheFirstSegmentStartsAtTheStartPoint) {
  const auto g = gate();
  // the waypoint itself is near, but the start point is 10 cm away: shortened from the start point
  const auto r = checkOne(g, startAt({0.4, 0, 0.3}), wp(0.05, {0.5, 0, 0.3}));
  ASSERT_EQ(r.shortened.size(), 1u);
  EXPECT_NEAR(r.accepted[0].p.x(), 0.4 + 0.25 * (0.05 + 0.010), 1e-12);
}

TEST(SafetyGate, A50msChunkAt099VmaxIsNeverShortened) {
  const auto g = gate();
  std::vector<TrajectoryPoint> pts;
  const double v = 0.99 * 0.25;
  for (int k = 1; k <= 16; ++k) pts.push_back(wp(0.05 * k, {0.4 + v * 0.05 * k, 0, 0.3}));
  const auto r = g.check(startAt({0.4, 0, 0.3}), pts, kNow);
  expectAccounted(r, pts.size());
  EXPECT_TRUE(r.shortened.empty());
}

TEST(SafetyGate, TheConeIsMeasuredOnTheFlangeAxis) {
  // A TCP tilted 30 deg about the flange x axis: the flange (and so the fingers) points straight down.
  Pose F_T_TCP = Pose::Identity();
  F_T_TCP.linear() = Eigen::AngleAxisd(30.0 * M_PI / 180.0, Eigen::Vector3d::UnitX()).toRotationMatrix();
  F_T_TCP.translation() = Eigen::Vector3d(0, 0, 0.15);
  const auto g = gate(cfgWith([](GateConfig& c) { c.cone_max_deg = 20; }), F_T_TCP);
  const Eigen::Quaterniond flange_down = kDown;
  const Eigen::Quaterniond q_tcp = flange_down * Eigen::Quaterniond(F_T_TCP.linear());
  const Eigen::Vector3d p(0.5, 0, 0.3);
  auto r = checkOne(g, startAt(p, q_tcp), wp(0.1, p, q_tcp));
  EXPECT_EQ(r.accepted.size(), 1u);
  EXPECT_TRUE(r.pulled_back.empty());
  EXPECT_NEAR(g.tiltDeg(q_tcp), 0.0, 1e-9);
  // the TCP's own z pointing straight down means the flange is tilted 30 deg: rejected
  r = checkOne(g, startAt(p, kDown), wp(0.1, p, kDown));
  EXPECT_EQ(r.rejected.size(), 1u);
}

TEST(SafetyGate, InsideBoxHasNoMargin) {
  const auto g = gate();
  EXPECT_TRUE(g.insideBox({0.8, 0.4, 0.7}));
  EXPECT_FALSE(g.insideBox({0.8001, 0.0, 0.3}));
  EXPECT_FALSE(g.insideBox({0.31, 0.0, 0.03}));
}

}  // namespace
}  // namespace franka_control
