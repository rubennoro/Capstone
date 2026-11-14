// from rnoro/deployment:deployment/franka_control/tests/test_config_loader.cpp @ 91b178b
// changes: EndpointsAndSessionRootComeFromTheStation -> EndpointsComeFromLinkYaml; a station endpoints: section and
// overrides of tcp.F_T_TCP / watchdog.limit_s refused with their hints; max_lead_s 1.5; F_T_TCP from frames.yaml;
// watchdog.limit_s from link.yaml and its floor; contract file problems (control_deploy/STRUCTURE.md 9.1).
#include <gtest/gtest.h>

#include <cmath>

#include "franka_control/config_loader.h"
#include "support/test_support.h"

namespace franka_control {
namespace {

using test::configFile;

std::vector<std::string> problemsFor(const std::string& overrides,
                                     const std::filesystem::path& limits = test::limitsFile()) {
  const auto dir = test::tmpDir("cfgbad");
  try {
    ConfigLoader::load(test::writeStation(dir, overrides), false, false, configFile(),
                       ContractFiles{limits, test::framesFile(), test::linkFile()});
  } catch (const ConfigError& e) {
    return e.problems;
  }
  return {};
}

bool hasProblem(const std::vector<std::string>& problems, const std::string& needle) {
  for (const auto& p : problems) {
    if (p.find(needle) != std::string::npos) return true;
  }
  return false;
}

TEST(ConfigLoader, TheRealConfigLoads) {
  const Config c = test::testConfig();
  EXPECT_EQ(c.robot_ip, "172.16.0.2");
  EXPECT_EQ(c.realtime, RealtimeMode::kEnforce);
  EXPECT_EQ(c.libfranka_version, "0.9.2");
  EXPECT_DOUBLE_EQ(c.safety.v_max_m_per_s, 0.25);
  EXPECT_DOUBLE_EQ(c.watchdog.limit_s, 1.0);   // link.yaml timing.watchdog_limit_s
  EXPECT_DOUBLE_EQ(c.safety.max_lead_s, 1.5);
  EXPECT_DOUBLE_EQ(c.homing.timeout_s, 30.0);
  EXPECT_DOUBLE_EQ(c.limits.gripper_width_max_m, 0.08);
  EXPECT_FALSE(c.control_cpu);
  EXPECT_FALSE(c.fake_gripper.object_width_m);
  EXPECT_TRUE(c.safety.cone_axis.isApprox(Eigen::Vector3d(0, 0, -1)));
  EXPECT_NEAR(c.fake_arm.home_O_T_TCP.translation().x(), 0.45, 1e-15);
  const std::string yaml = ConfigLoader::toYaml(c);
  EXPECT_NE(yaml.find("v_max_m_per_s"), std::string::npos);
  EXPECT_NE(yaml.find("endpoints"), std::string::npos);
  EXPECT_NE(yaml.find("contracts:"), std::string::npos);
  EXPECT_NE(yaml.find("link_file: " + test::linkFile().string()), std::string::npos);
}

TEST(ConfigLoader, StationOverridesMergeMapsKeyByKeyAndReplaceLists) {
  const Config c = test::testConfig(
      "  realtime: ignore\n"
      "  safety:\n"
      "    v_max_m_per_s: 0.1\n"
      "    workspace_min_m: [0.3, -0.3, 0.1]\n"
      "  control_cpu: 3\n");
  EXPECT_EQ(c.realtime, RealtimeMode::kIgnore);
  EXPECT_DOUBLE_EQ(c.safety.v_max_m_per_s, 0.1);
  EXPECT_DOUBLE_EQ(c.safety.w_max_rad_per_s, 1.0);   // the rest of the map stays
  EXPECT_TRUE(c.safety.workspace_min_m.isApprox(Eigen::Vector3d(0.3, -0.3, 0.1)));
  EXPECT_TRUE(c.safety.workspace_max_m.isApprox(Eigen::Vector3d(0.8, 0.4, 0.7)));
  EXPECT_EQ(*c.control_cpu, 3);
}

TEST(ConfigLoader, AnUnknownOverrideKeyIsRefusedWithItsPath) {
  const auto p = problemsFor("  safety:\n    v_maxx: 1\n");
  ASSERT_EQ(p.size(), 1u);
  EXPECT_EQ(p[0], "station.yaml: franka_control.safety.v_maxx: not a key of franka_control/config.yaml");
}

TEST(ConfigLoader, AMapOverriddenByAScalarIsRefused) {
  const auto p = problemsFor("  safety: 3\n");
  ASSERT_EQ(p.size(), 1u);
  EXPECT_EQ(p[0], "station.yaml: franka_control.safety: expected a map (as in franka_control/config.yaml)");
}

TEST(ConfigLoader, EachBadValueIsRefusedAndAllAreListedAtOnce) {
  const auto p = problemsFor(
      "  safety:\n"
      "    v_max_m_per_s: -1\n"
      "    workspace_min_m: [0.9, -0.4, 0.05]\n"
      "    box_margin_m: -0.01\n"
      "    cone_margin_deg: -1\n"
      "  tcp:\n"
      "    F_T_TCP: [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1.1, 0.15], [0, 0, 0, 1]]\n"
      "  homing:\n"
      "    q_home_rad: [0, 0, 0, 0, 0, 0]\n"
      "  realtime: sometimes\n"
      "  watchdog:\n"
      "    limit_s: 0.4\n");
  EXPECT_EQ(p.size(), 8u);
  EXPECT_TRUE(hasProblem(p, "station.yaml: franka_control.safety.v_max_m_per_s: must be > 0"));
  EXPECT_TRUE(hasProblem(p, "safety.workspace_min_m: workspace_min_m must be < workspace_max_m on x"));
  EXPECT_TRUE(hasProblem(p, "safety.box_margin_m: must be >= 0"));
  EXPECT_TRUE(hasProblem(p, "safety.cone_margin_deg: must be >= 0"));
  EXPECT_TRUE(hasProblem(p, "station.yaml: franka_control.tcp: not a key of franka_control/config.yaml: moved to "
                           "contracts/frames.yaml"));
  EXPECT_TRUE(hasProblem(p, "homing.q_home_rad: expected a list of 7 numbers"));
  EXPECT_TRUE(hasProblem(p, "realtime: expected enforce or ignore"));
  EXPECT_TRUE(hasProblem(p, "station.yaml: franka_control.watchdog.limit_s: not a key of franka_control/config.yaml: "
                           "moved to contracts/link.yaml timing.watchdog_limit_s"));
  try {
    ConfigLoader::load(test::writeStation(test::tmpDir("cfgbad2"), "  safety:\n    v_max_m_per_s: -1\n"), false,
                       false, configFile(), test::contractFiles());
    ADD_FAILURE();
  } catch (const ConfigError& e) {
    EXPECT_STREQ(e.what(), "station.yaml: franka_control.safety.v_max_m_per_s: must be > 0");
    EXPECT_EQ(e.hint(), "fix station.yaml");
  }
}

TEST(ConfigLoader, EndpointsComeFromLinkYaml) {
  const Config c = test::testConfig();
  EXPECT_EQ(c.pc_address, "192.168.50.10");
  EXPECT_EQ(c.endpoint_state, "tcp://192.168.50.10:5620");
  EXPECT_EQ(c.endpoint_commands, "tcp://192.168.50.10:5630");
  EXPECT_FALSE(c.local);
  EXPECT_EQ(c.session_root.filename(), "sessions");
  const auto dir = test::tmpDir("cfglocal");
  const Config l = ConfigLoader::load(test::writeStation(dir), true, true, configFile(), test::contractFiles());
  EXPECT_TRUE(l.local);
  EXPECT_EQ(l.endpoint_state, "tcp://127.0.0.1:5620");
  EXPECT_EQ(l.endpoint_commands, "tcp://127.0.0.1:5630");
  EXPECT_EQ(l.link_file, test::linkFile());
  EXPECT_EQ(l.contracts_dir, test::limitsFile().parent_path());
}

TEST(ConfigLoader, AStationEndpointsSectionIsRefusedAndSessionRootIsRequired) {
  const auto dir = test::tmpDir("cfgnoep");
  test::writeFile(dir / "station.yaml", "station_version: 1\nendpoints:\n  state: tcp://127.0.0.1:5620\n");
  try {
    ConfigLoader::load(dir / "station.yaml", true, false, configFile(), test::contractFiles());
    ADD_FAILURE();
  } catch (const ConfigError& e) {
    EXPECT_TRUE(hasProblem(e.problems, "station.yaml: endpoints: no longer used: franka_control binds "
                                       "contracts/link.yaml pc.address and ports"));
    EXPECT_TRUE(hasProblem(e.problems, "station.yaml: session_root: missing"));
    EXPECT_EQ(e.problems.size(), 2u);
  }
}

TEST(ConfigLoader, TcpAndWatchdogLimitOverridesAreRefusedWithTheirNewHome) {
  auto p = problemsFor("  tcp:\n    F_T_TCP: [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0.1], [0, 0, 0, 1]]\n");
  ASSERT_EQ(p.size(), 1u);
  EXPECT_EQ(p[0], "station.yaml: franka_control.tcp: not a key of franka_control/config.yaml: moved to "
                  "contracts/frames.yaml");
  p = problemsFor("  watchdog:\n    limit_s: 2.0\n");
  ASSERT_EQ(p.size(), 1u);
  EXPECT_EQ(p[0], "station.yaml: franka_control.watchdog.limit_s: not a key of franka_control/config.yaml: moved to "
                  "contracts/link.yaml timing.watchdog_limit_s");
  EXPECT_TRUE(problemsFor("  watchdog:\n    backstop_extra_s: 0.2\n").empty());   // the rest of watchdog stays
}

TEST(ConfigLoader, FTcpIsFramesYamlReNormalised) {
  const Config c = test::testConfig();
  const double h = std::sqrt(0.5);
  Eigen::Matrix4d want;
  want << h, h, 0, 0, -h, h, 0, 0, 0, 0, 1, 0.150, 0, 0, 0, 1;
  EXPECT_TRUE(c.F_T_TCP.matrix().isApprox(want, 1e-15));
  EXPECT_LT((c.F_T_TCP.linear().transpose() * c.F_T_TCP.linear() - Eigen::Matrix3d::Identity()).cwiseAbs().maxCoeff(),
            1e-15);
}

TEST(ConfigLoader, TheWatchdogLimitComesFromLinkYamlWithItsFloor) {
  const auto dir = test::tmpDir("cfglink");
  const auto station = test::writeStation(dir);
  const Config c = ConfigLoader::load(station, true, false, configFile(),
                                      ContractFiles{test::limitsFile(), test::framesFile(),
                                                    test::writeLink(dir, "127.0.0.1", 6001, 6002, 0.75)});
  EXPECT_DOUBLE_EQ(c.watchdog.limit_s, 0.75);
  EXPECT_EQ(c.endpoint_commands, "tcp://127.0.0.1:6002");
  try {
    ConfigLoader::load(station, true, false, configFile(),
                       ContractFiles{test::limitsFile(), test::framesFile(),
                                     test::writeLink(dir, "127.0.0.1", 6001, 6002, 0.4)});
    ADD_FAILURE();
  } catch (const ConfigError& e) {
    ASSERT_EQ(e.problems.size(), 1u);
    EXPECT_EQ(e.problems[0], "link.yaml: timing.watchdog_limit_s: must be >= 0.5 s (franka_control's floor)");
    EXPECT_EQ(e.hint(), "fix link.yaml");
  }
}

TEST(ConfigLoader, ABadLinkAddressIsAFatalNamingLinkYaml) {
  const auto dir = test::tmpDir("cfgbadlink");
  try {
    ConfigLoader::load(test::writeStation(dir), true, false, configFile(),
                       ContractFiles{test::limitsFile(), test::framesFile(),
                                     test::writeLink(dir, "pc.local", 5620, 5630)});
    ADD_FAILURE();
  } catch (const ConfigError& e) {
    EXPECT_STREQ(e.what(), "link.yaml: pc.address: expected an IPv4 address, got \"pc.local\"");
    EXPECT_EQ(e.hint(), "fix link.yaml");
  }
}

TEST(ConfigLoader, MockWorksWithoutAReachableRobot) {
  const auto dir = test::tmpDir("cfgmock");
  const Config c =
      ConfigLoader::load(test::writeStation(dir, "  robot_ip: 10.255.255.1\n"), true, false, configFile(),
                         test::contractFiles());
  EXPECT_TRUE(c.mock);
  EXPECT_EQ(c.robot_ip, "10.255.255.1");
}

TEST(ConfigLoader, MaxWaypointsAboveTheCompiledCapacityIsRefused) {
  const auto dir = test::tmpDir("cfglimits");
  test::writeFile(dir / "limits.yaml", "rotation_tol: 1.0e-6\ngripper_width_max_m: 0.08\nmax_waypoints: 100\n");
  const auto p = problemsFor("", dir / "limits.yaml");
  ASSERT_EQ(p.size(), 1u);
  EXPECT_NE(p[0].find("raise kMaxWaypoints and rebuild"), std::string::npos);
}

TEST(ConfigLoader, LibfrankaVersionIsNotAStationSetting) {
  const auto p = problemsFor("  libfranka_version: \"0.10.0\"\n");
  ASSERT_EQ(p.size(), 1u);
  EXPECT_NE(p[0].find("libfranka_version: set it in franka_control/config.yaml only"), std::string::npos);
}

}  // namespace
}  // namespace franka_control
