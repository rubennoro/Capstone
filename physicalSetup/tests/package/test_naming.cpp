// SPDX-License-Identifier: MIT
// T3: iPhUMI-identical package naming, including a parity test that runs iPhUMI's parsing rules in Python.
#include <gtest/gtest.h>

#include <unistd.h>

#include <cstdio>
#include <fstream>
#include <map>
#include <set>
#include <sstream>
#include <string>
#include <vector>

#include "rig/package/package.hpp"

namespace rig {
namespace {

TEST(Naming, IsoTimestampMatchesTheIphoneFormat) {
  // 2026-10-01T21:13:57.489Z (a real firstreal package) -> colons replaced.
  const TimeNs t = 1790889237489000000LL + 999999;  // sub-ms part truncated
  EXPECT_EQ(iso_name_timestamp(t), "2026-10-01T21-13-57.489Z");
  EXPECT_EQ(iso_name_timestamp(0), "1970-01-01T00-00-00.000Z");
  EXPECT_EQ(iso_name_timestamp(-1), "1969-12-31T23-59-59.999Z");
}

TEST(Naming, RandomizerDigitsAreOneToNine) {
  std::map<char, int> counts;
  for (uint64_t seed = 0; seed < 2000; ++seed) {
    const std::string d = random_five_digits(seed);
    ASSERT_EQ(d.size(), 5u);
    for (char c : d) {
      ASSERT_GE(c, '1');
      ASSERT_LE(c, '9');
      counts[c]++;
    }
  }
  EXPECT_EQ(counts.size(), 9u);
  for (const auto& kv : counts) EXPECT_GT(kv.second, 800) << kv.first;  // 10000 draws / 9 ~ 1111
  EXPECT_EQ(random_five_digits(42), random_five_digits(42));
}

TEST(Naming, SessionNames) {
  EXPECT_TRUE(validate_session_name("firstreal").ok());
  EXPECT_TRUE(validate_session_name("block-raise-2026-10-30").ok());
  EXPECT_FALSE(validate_session_name("block_raise_2026-10-30").ok());  // PIPELINE 4.1 old example: invalid
  EXPECT_FALSE(validate_session_name("").ok());
  EXPECT_FALSE(validate_session_name("a b").ok());
  EXPECT_FALSE(validate_session_name("a/b").ok());
  EXPECT_FALSE(validate_session_name(std::string(65, 'a')).ok());
  EXPECT_TRUE(validate_session_name(std::string(64, 'a')).ok());
}

TEST(Naming, MakeAndParseRoundTrip) {
  auto n = make_package_name("firstreal", PackageKind::kDemonstration, Side::kLeft, 1790889237489000000LL, 7);
  ASSERT_TRUE(n.ok());
  auto p = parse_package_name(n->str());
  ASSERT_TRUE(p.ok()) << p.status().to_string();
  EXPECT_EQ(p->str(), n->str());
  auto real = parse_package_name("2026-10-01T21-13-57.489Z_75494_firstreal_demonstration_left");
  ASSERT_TRUE(real.ok());
  EXPECT_EQ(real->session, "firstreal");
  EXPECT_EQ(real->side, Side::kLeft);
  EXPECT_EQ(make_package_name("bad_name", PackageKind::kDemonstration, Side::kRight, 0, 1).status().code(),
            StatusCode::kInvalidArgument);
  for (const char* bad : {"2026-10-01T21-13-57.489Z_75494_first_real_demonstration_left",
                          "2026-10-01T21-13-57.489Z_75494_firstreal_demonstration",
                          "2026-10-01T21:13:57.489Z_75494_firstreal_demonstration_left",
                          "2026-10-01T21-13-57.48Z_75494_firstreal_demonstration_left",
                          "2026-10-01T21-13-57.489Z_75094_firstreal_demonstration_left",
                          "2026-10-01T21-13-57.489Z_7549_firstreal_demonstration_left",
                          "2026-10-01T21-13-57.489Z_75494_firstreal_rig_calibration_left",
                          "2026-10-01T21-13-57.489Z_75494_firstreal_demonstration_middle", ""}) {
    EXPECT_FALSE(parse_package_name(bad).ok()) << bad;
  }
}

std::string python3() {
  // /usr/bin/python3 first: a conda python on PATH is fine too (-I, stdlib only), but prefer the system one.
  return access("/usr/bin/python3", X_OK) == 0 ? "/usr/bin/python3" : "python3";
}

TEST(Naming, ParityWithIphumiParser) {
  std::vector<std::string> names;
  uint64_t seed = 1;
  for (PackageKind k : {PackageKind::kDemonstration, PackageKind::kGripperCalibration, PackageKind::kRigCalibration})
    for (Side s : {Side::kLeft, Side::kRight, Side::kHead})
      for (const char* sess : {"firstreal", "block-raise-2026-10-30", "A-1"}) {
        const TimeNs t = 1790889237489000000LL + static_cast<TimeNs>(seed) * 1000003;
        names.push_back(make_package_name(sess, k, s, t, seed).value().str());
        ++seed;
      }
  // Names our parser must reject; iPhUMI must also not accept them as the same package.
  const std::vector<std::string> bad = {"2026-10-01T21-13-57.489Z_75494_block_raise_demonstration_left",
                                        "2026-10-01T21-13-57.489Z_75494_firstreal_rig_calibration_right"};
  const std::string in_path = std::string(RIG_TEST_TMP_ROOT) + "/parity_in_" + std::to_string(getpid()) + ".txt";
  {
    std::ofstream f(in_path);
    for (const auto& n : names) f << n << "\n";
    for (const auto& n : bad) f << n << "\n";
  }
  const std::string cmd = python3() + " -I " + std::string(RIG_SOURCE_DIR) + "/tests/package/iphumi_name_parity.py " +
                          std::string(RIG_REPO_ROOT) + " < " + in_path;
  FILE* p = popen(cmd.c_str(), "r");
  ASSERT_NE(p, nullptr);
  std::string out;
  char buf[4096];
  while (std::fgets(buf, sizeof(buf), p)) out += buf;
  const int rc = pclose(p);
  ASSERT_EQ(rc, 0) << "python parity script failed (exit " << rc << ")\n" << out;
  std::map<std::string, std::vector<std::string>> rows;
  std::istringstream ss(out);
  for (std::string line; std::getline(ss, line);) {
    std::vector<std::string> cols;
    std::stringstream ls(line);
    for (std::string c; std::getline(ls, c, '\t');) cols.push_back(c);
    while (cols.size() < 7) cols.emplace_back();
    rows[cols[0]] = cols;
  }
  ASSERT_EQ(rows.size(), names.size() + bad.size());
  for (const auto& n : names) {
    const auto& r = rows.at(n);
    const PackageName ours = parse_package_name(n).value();
    EXPECT_EQ(r[1], "1") << n;
    EXPECT_EQ(r[2], ours.session) << n;
    EXPECT_EQ(r[3], to_string(ours.kind)) << n;
    EXPECT_EQ(r[4], to_string(ours.side)) << n;
    EXPECT_EQ(r[5], to_string(ours.kind)) << n;  // generic_util get_demonstration_type on the grouped title
    EXPECT_EQ(r[6], ours.session) << n;          // generic_util session filter
  }
  // A '_' in the session or kind is exactly what breaks iPhUMI: it sees != 5 parts (skips the demo).
  for (const auto& n : bad) {
    EXPECT_FALSE(parse_package_name(n).ok()) << n;
    EXPECT_EQ(rows.at(n)[1], "0") << n;
  }
  std::remove(in_path.c_str());
}

}  // namespace
}  // namespace rig
