// SPDX-License-Identifier: MIT
// Checks that the vendored CLI11 header builds under the project's warning flags and parses as expected.
#include <gtest/gtest.h>

#include <algorithm>
#include <string>
#include <vector>

#include "CLI11.hpp"

TEST(Cli11, ParsesOptions) {
  CLI::App app{"test"};
  std::string session;
  int n = 0;
  app.add_option("--session", session)->required();
  app.add_option("-n", n);
  std::vector<std::string> args{"-n", "3", "--session", "block-raise"};  // CLI11 parses the vector back to front
  std::reverse(args.begin(), args.end());
  ASSERT_NO_THROW(app.parse(args));
  EXPECT_EQ(session, "block-raise");
  EXPECT_EQ(n, 3);
}
