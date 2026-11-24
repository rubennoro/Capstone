// New (control_deploy/STRUCTURE.md 9.1): the command line, with --local and --keyboard.
#include <gtest/gtest.h>

#include <string>
#include <vector>

#include "franka_control/command_line.h"

namespace franka_control {
namespace {

CommandLine parse(std::vector<std::string> args) {
  args.insert(args.begin(), "franka_control");
  std::vector<char*> argv;
  for (auto& a : args) argv.push_back(a.data());
  return CommandLine::parse(static_cast<int>(argv.size()), argv.data());
}

TEST(CommandLine, LocalAndKeyboardAreParsed) {
  const CommandLine cl = parse({"--station", "s.yaml", "--mock", "--local", "--keyboard", "--ready-file", "r",
                                "--session", "d"});
  EXPECT_EQ(cl.station, "s.yaml");
  EXPECT_TRUE(cl.mock);
  EXPECT_TRUE(cl.local);
  EXPECT_TRUE(cl.keyboard);
  ASSERT_TRUE(cl.ready_file);
  EXPECT_EQ(*cl.ready_file, "r");
  ASSERT_TRUE(cl.session);
  EXPECT_EQ(*cl.session, "d");
  const CommandLine plain = parse({"--station", "s.yaml"});
  EXPECT_FALSE(plain.local);
  EXPECT_FALSE(plain.keyboard);
  EXPECT_FALSE(plain.mock);
}

TEST(CommandLine, CheckConfigTakesLocal) {
  const CommandLine cl = parse({"--check-config", "--station", "s.yaml", "--mock", "--local"});
  EXPECT_TRUE(cl.check_config);
  EXPECT_TRUE(cl.local);
}

TEST(CommandLine, UnknownFlagsAreRefused) {
  EXPECT_THROW(parse({"--station", "s.yaml", "--loca"}), UsageError);
  EXPECT_THROW(parse({"--station", "s.yaml", "--endpoint", "tcp://*:5630"}), UsageError);
  EXPECT_THROW(parse({"--station"}), UsageError);   // a value is needed
  try {
    parse({"--station", "s.yaml", "-k"});
  } catch (const UsageError& e) {
    EXPECT_NE(std::string(e.what()).find("[--local] [--keyboard]"), std::string::npos) << e.what();
  }
}

TEST(CommandLine, VersionAloneStationRequiredOtherwise) {
  EXPECT_TRUE(parse({"--version"}).version);
  EXPECT_THROW(parse({"--mock", "--local"}), UsageError);
  EXPECT_THROW(parse({"--check-config"}), UsageError);
}

}  // namespace
}  // namespace franka_control
