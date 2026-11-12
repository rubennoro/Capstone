#include <gtest/gtest.h>

#include <array>
#include <atomic>
#include <thread>

#include "franka_control/triple_buffer.h"

namespace franka_control {
namespace {

struct Blob {
  std::array<uint64_t, 64> v{};   // all equal in every written copy
};

TEST(TripleBuffer, NewestWinsAndRefreshFalseWhenNothingNew) {
  TripleBuffer<int> b;
  EXPECT_FALSE(b.refresh());
  b.writeSlot() = 1;
  b.publish();
  b.writeSlot() = 2;
  b.publish();
  EXPECT_TRUE(b.refresh());
  EXPECT_EQ(b.readSlot(), 2);
  EXPECT_FALSE(b.refresh());
  EXPECT_EQ(b.readSlot(), 2);
}

TEST(TripleBuffer, ReadersCopyIsStableWhileTheWriterWrites) {
  TripleBuffer<int> b;
  b.writeSlot() = 7;
  b.publish();
  ASSERT_TRUE(b.refresh());
  for (int i = 0; i < 10; ++i) {
    b.writeSlot() = 100 + i;
    b.publish();
    EXPECT_EQ(b.readSlot(), 7);   // the reader's slot is never handed to the writer
  }
  ASSERT_TRUE(b.refresh());
  EXPECT_EQ(b.readSlot(), 109);
}

// Label `threads` (run under TSan): the reader never sees a half-written copy, and values only go forward.
TEST(TripleBufferThreads, ReaderNeverSeesAMixedCopy) {
  TripleBuffer<Blob> b;
  constexpr uint64_t kWrites = 200000;
  std::atomic<bool> done{false};
  std::thread writer([&] {
    for (uint64_t k = 1; k <= kWrites; ++k) {
      Blob& s = b.writeSlot();
      s.v.fill(k);
      b.publish();
    }
    done.store(true);
  });
  uint64_t last = 0, reads = 0;
  bool mixed = false, backwards = false;
  while (!done.load() || b.refresh()) {
    if (!b.refresh()) continue;
    const Blob& s = b.readSlot();
    for (uint64_t x : s.v) mixed = mixed || x != s.v[0];
    backwards = backwards || s.v[0] < last;
    last = s.v[0];
    ++reads;
  }
  writer.join();
  b.refresh();
  EXPECT_FALSE(mixed);
  EXPECT_FALSE(backwards);
  EXPECT_EQ(b.readSlot().v[0], kWrites);
  EXPECT_GT(reads, 0u);
}

}  // namespace
}  // namespace franka_control
