#include <gtest/gtest.h>

#include "franka_control/watchdog.h"

namespace franka_control {
namespace {

TEST(Watchdog, NotExpiredAtTheLimitExpiredOneNsAfter) {
  const Watchdog w(500000000);
  EXPECT_FALSE(w.expired(1500000000, 1000000000));
  EXPECT_TRUE(w.expired(1500000001, 1000000000));
  EXPECT_EQ(w.ageNs(1500000001, 1000000000), 500000001);
  EXPECT_FALSE(w.expired(1000000000, 1000000000));
}

}  // namespace
}  // namespace franka_control
