// New (control_deploy/STRUCTURE.md 9.1): build_info.h and the --version line of the built binary.
#include <gtest/gtest.h>

#include <cstdio>
#include <regex>
#include <string>

#include "franka_control/build_info.h"

namespace franka_control {
namespace {

TEST(BuildInfo, TheCommitIsFortyLowercaseHexOrUnknown) {
  EXPECT_TRUE(std::regex_match(std::string(kGitCommit), std::regex("^([0-9a-f]{40}|unknown)$"))) << kGitCommit;
}

TEST(BuildInfo, TheVersionLineOfTheBuiltBinary) {
  const std::string cmd = std::string("'") + FRANKA_CONTROL_BINARY + "' --version";
  FILE* p = popen(cmd.c_str(), "r");
  ASSERT_NE(p, nullptr);
  std::string out;
  char buf[256];
  while (fgets(buf, sizeof(buf), p)) out += buf;
  EXPECT_EQ(pclose(p), 0);
  const std::regex line(
      "^franka_control commit=([0-9a-f]{40}|unknown) dirty=(yes|no) libfranka=([0-9]+\\.[0-9]+\\.[0-9]+|none)\n$");
  std::smatch m;
  ASSERT_TRUE(std::regex_match(out, m, line)) << out;
  EXPECT_EQ(m[1], kGitCommit);
  EXPECT_EQ(m[2], kGitDirty ? "yes" : "no");
  EXPECT_EQ(m[3], FRANKA_CONTROL_LIBFRANKA_VERSION);
}

}  // namespace
}  // namespace franka_control
