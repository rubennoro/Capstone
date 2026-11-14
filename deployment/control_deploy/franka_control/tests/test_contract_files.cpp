// New (control_deploy/STRUCTURE.md 9.1): ContractFrame (frames.yaml) and ContractLink (link.yaml).
#include <gtest/gtest.h>

#include <cmath>

#include "franka_control/contract_frame.h"
#include "franka_control/contract_link.h"
#include "support/test_support.h"

namespace franka_control {
namespace {

constexpr double kTol = 1e-6;

std::filesystem::path frames(const std::string& text) {
  const auto dir = test::tmpDir("frames");
  test::writeFile(dir / "frames.yaml", text);
  return dir / "frames.yaml";
}

std::vector<std::string> frameProblems(const std::string& text) {
  std::vector<std::string> problems;
  ContractFrame::load(frames(text), kTol, problems);
  return problems;
}

std::filesystem::path link(const std::string& text) {
  const auto dir = test::tmpDir("link");
  test::writeFile(dir / "link.yaml", text);
  return dir / "link.yaml";
}

std::string linkText(const std::string& address = "192.168.50.10", const std::string& state = "5620",
                     const std::string& commands = "5630", const std::string& watchdog = "1.0") {
  return "pc:\n  address: " + address + "\nlaptop:\n  addresses: [192.168.50.20]\nports:\n  state: " + state +
         "\n  commands: " + commands + "\ntiming:\n  heartbeat_period_s: 0.02\n  watchdog_limit_s: " + watchdog +
         "\n  c3_timeout_s: 0.3\n  state_stale_s: 0.1\n  state_lost_s: 2.0\nclock:\n  estimator: round_trip\n";
}

std::vector<std::string> linkProblems(const std::string& text) {
  std::vector<std::string> problems;
  ContractLink::load(link(text), problems);
  return problems;
}

TEST(ContractFrame, TheRealFramesYamlLoadsAndEqualsTheContractLiteral) {   // deployment/STRUCTURE.md 4.2
  std::vector<std::string> problems;
  const Pose f = ContractFrame::load(test::framesFile(), kTol, problems);
  ASSERT_TRUE(problems.empty()) << problems.front();
  Eigen::Matrix4d want;
  want << 0.7071067811865476, 0.7071067811865476, 0.0, 0.0, -0.7071067811865476, 0.7071067811865476, 0.0, 0.0, 0.0,
      0.0, 1.0, 0.150, 0.0, 0.0, 0.0, 1.0;
  EXPECT_TRUE(f.matrix().isApprox(want, 1e-15));
  // re-normalised through a unit quaternion: orthonormal to rounding
  EXPECT_LT((f.linear().transpose() * f.linear() - Eigen::Matrix3d::Identity()).cwiseAbs().maxCoeff(), 1e-15);
}

TEST(ContractFrame, BadMatricesAreRefused) {
  const std::string rows = "  - [1, 0, 0, 0]\n  - [0, 1, 0, 0]\n  - [0, 0, 1, 0.15]\n";
  EXPECT_FALSE(frameProblems("F_T_TCP:\n" + rows).empty());                                     // 3 rows
  EXPECT_FALSE(frameProblems("F_T_TCP:\n" + rows + "  - [0, 0, 0, 1, 0]\n").empty());           // 5 columns
  EXPECT_FALSE(frameProblems("F_T_TCP:\n" + rows + "  - [0, 0, 0, .nan]\n").empty());           // non-finite
  EXPECT_FALSE(frameProblems("F_T_TCP:\n" + rows + "  - [0, 0, \"0\", 1]\n").empty());          // a string
  const auto transposed = frameProblems(
      "F_T_TCP:\n  - [1, 0, 0, 0]\n  - [0, 1, 0, 0]\n  - [0, 0, 1, 0]\n  - [0, 0, 0.15, 1]\n");
  ASSERT_EQ(transposed.size(), 1u);
  EXPECT_NE(transposed[0].find("frames.yaml: F_T_TCP: not a rigid transform"), std::string::npos);
  const auto reflection = frameProblems(
      "F_T_TCP:\n  - [1, 0, 0, 0]\n  - [0, 1, 0, 0]\n  - [0, 0, -1, 0.15]\n  - [0, 0, 0, 1]\n");
  ASSERT_EQ(reflection.size(), 1u);
  EXPECT_NE(reflection[0].find("not a rigid transform"), std::string::npos);
}

TEST(ContractFrame, SettledAndUnknownKeysAreIgnoredAMissingMatrixIsNamed) {
  EXPECT_TRUE(frameProblems("F_T_TCP:\n  - [1, 0, 0, 0]\n  - [0, 1, 0, 0]\n  - [0, 0, 1, 0.15]\n  - [0, 0, 0, 1]\n"
                            "settled: maybe\nsomething_else: 3\n")
                  .empty());
  const auto p = frameProblems("settled: false\n");
  ASSERT_EQ(p.size(), 1u);
  EXPECT_EQ(p[0], "frames.yaml: F_T_TCP: missing");
}

TEST(ContractLink, TheRealLinkYamlLoads) {
  std::vector<std::string> problems;
  const ContractLink l = ContractLink::load(test::linkFile(), problems);
  ASSERT_TRUE(problems.empty()) << problems.front();
  EXPECT_EQ(l.pc_address, "192.168.50.10");
  EXPECT_EQ(l.state_port, 5620);
  EXPECT_EQ(l.commands_port, 5630);
  EXPECT_DOUBLE_EQ(l.watchdog_limit_s, 1.0);
  EXPECT_EQ(l.endpoint(5620, false), "tcp://192.168.50.10:5620");
  EXPECT_EQ(l.endpoint(5630, true), "tcp://127.0.0.1:5630");
}

TEST(ContractLink, AddressesThatAreNotThisPcsIPv4AreRefused) {
  for (const std::string bad : {"0.0.0.0", "pc.local", "256.1.1.1", "192.168.50", "::1", "\"\""}) {
    SCOPED_TRACE(bad);
    const auto p = linkProblems(linkText(bad));
    ASSERT_EQ(p.size(), 1u);
    EXPECT_EQ(p[0].rfind("link.yaml: pc.address: ", 0), 0u) << p[0];
  }
  EXPECT_NE(linkProblems(linkText("0.0.0.0"))[0].find("not 0.0.0.0"), std::string::npos);
}

TEST(ContractLink, PortsAndWatchdogAreChecked) {
  EXPECT_EQ(linkProblems(linkText("192.168.50.10", "80"))[0].rfind("link.yaml: ports.state: ", 0), 0u);
  EXPECT_EQ(linkProblems(linkText("192.168.50.10", "5620", "70000"))[0].rfind("link.yaml: ports.commands: ", 0), 0u);
  EXPECT_EQ(linkProblems(linkText("192.168.50.10", "5620", "5620"))[0],
            "link.yaml: ports.commands: must differ from ports.state (5620)");
  EXPECT_FALSE(linkProblems(linkText("192.168.50.10", "5620.5")).empty());
  EXPECT_FALSE(linkProblems(linkText("192.168.50.10", "true")).empty());
  EXPECT_FALSE(linkProblems(linkText("192.168.50.10", "\"5620\"")).empty());
  EXPECT_EQ(linkProblems(linkText("192.168.50.10", "5620", "5630", "0"))[0]
                .rfind("link.yaml: timing.watchdog_limit_s: ", 0),
            0u);
  EXPECT_FALSE(linkProblems(linkText("192.168.50.10", "5620", "5630", "-1")).empty());
  EXPECT_FALSE(linkProblems(linkText("192.168.50.10", "5620", "5630", ".inf")).empty());
}

TEST(ContractLink, OtherKeysAreIgnoredMissingOnesAreNamed) {
  EXPECT_TRUE(linkProblems("pc:\n  address: 10.0.0.1\nports:\n  state: 6000\n  commands: 6001\ntiming:\n"
                           "  watchdog_limit_s: 0.7\nsomething: [1, 2]\n")
                  .empty());
  const auto p = linkProblems("pc: {}\nports:\n  state: 6000\n");
  ASSERT_EQ(p.size(), 3u);
  EXPECT_EQ(p[0], "link.yaml: pc.address: missing");
  EXPECT_EQ(p[1], "link.yaml: ports.commands: missing");
  EXPECT_EQ(p[2], "link.yaml: timing.watchdog_limit_s: missing");
}

}  // namespace
}  // namespace franka_control
