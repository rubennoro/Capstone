// SPDX-License-Identifier: MIT
// T3: strict scalar parsing of the rig.yaml loader (private helpers in common/config/yaml_reader.hpp).
#include <gtest/gtest.h>

#include <limits>

#include "../../common/config/yaml_reader.hpp"

namespace rig {
namespace cfg {
namespace {

TEST(YamlScalars, Integers) {
  int64_t v = 0;
  EXPECT_TRUE(parse_int_text("0x6A", v));
  EXPECT_EQ(v, 0x6A);
  EXPECT_TRUE(parse_int_text("0X0c", v));
  EXPECT_EQ(v, 12);
  EXPECT_TRUE(parse_int_text("-42", v));
  EXPECT_EQ(v, -42);
  EXPECT_TRUE(parse_int_text("+7", v));
  EXPECT_EQ(v, 7);
  EXPECT_TRUE(parse_int_text("9223372036854775807", v));
  EXPECT_EQ(v, std::numeric_limits<int64_t>::max());
  EXPECT_TRUE(parse_int_text("-9223372036854775808", v));
  EXPECT_EQ(v, std::numeric_limits<int64_t>::min());
  for (const char* bad : {"", "-", "0x", "1.0", "1e3", "9223372036854775808", "0x1g", " 1", "1 ", "true", "0b1"})
    EXPECT_FALSE(parse_int_text(bad, v)) << bad;
}

TEST(YamlScalars, Doubles) {
  double v = 0;
  EXPECT_TRUE(parse_double_text("6.865e-4", v));
  EXPECT_DOUBLE_EQ(v, 6.865e-4);
  EXPECT_TRUE(parse_double_text("60", v));
  EXPECT_DOUBLE_EQ(v, 60.0);
  EXPECT_TRUE(parse_double_text(".5", v));
  EXPECT_TRUE(parse_double_text("-5.", v));
  for (const char* bad : {"", ".", "nan", ".nan", "inf", ".inf", "1e", "0x10", "1e400", "1,5", "e5"})
    EXPECT_FALSE(parse_double_text(bad, v)) << bad;
}

TEST(YamlScalars, Booleans) {
  bool v = false;
  EXPECT_TRUE(parse_bool_text("True", v));
  EXPECT_TRUE(v);
  EXPECT_TRUE(parse_bool_text("false", v));
  EXPECT_FALSE(v);
  for (const char* bad : {"yes", "no", "on", "1", "0", "tRUE", ""}) EXPECT_FALSE(parse_bool_text(bad, v)) << bad;
}

TEST(YamlScalars, Ranges) {
  EXPECT_EQ(open_lo(0, 240).str(), "(0, 240]");
  EXPECT_FALSE(open_lo(0, 240).contains(0));
  EXPECT_TRUE(open_lo(0, 240).contains(240));
  EXPECT_FALSE(closed(0, 1).contains(std::numeric_limits<double>::quiet_NaN()));
  EXPECT_EQ(fmt_double(0.1), "0.1");
  EXPECT_EQ(fmt_double(735.0), "735");
}

}  // namespace
}  // namespace cfg
}  // namespace rig
