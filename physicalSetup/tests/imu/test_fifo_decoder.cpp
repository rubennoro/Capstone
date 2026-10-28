// SPDX-License-Identifier: MIT
// FifoDecoder on hand-built FIFO byte streams: every tag type, both pairing rules, unwrap, losses.
#include <gtest/gtest.h>

#include <cmath>

#include "imu_test_util.hpp"
#include "rig/drivers/imu/lsm6dsox_fifo.hpp"

namespace rig {
namespace {

using namespace lsm6dsox;  // NOLINT
using test::flat;
using test::ts_word;
using test::word;

std::vector<RawSlot> decode(FifoDecoder& d, const std::vector<std::array<uint8_t, 7>>& ws) {
  std::vector<RawSlot> out;
  const auto b = flat(ws);
  d.push(b.data(), ws.size(), out);
  return out;
}

TEST(FifoDecoder, DecodesEveryTagType) {
  FifoDecoder d({});
  auto out = decode(d, {ts_word(0, 1000), word(kTagGyroNC, 0, 1, -2, 3), word(kTagAccelNC, 0, -100, 200, 4096),
                        word(kTagTemperature, 0, 256 * 6), word(kTagCfgChange, 0), word(0x1F, 0)});
  ASSERT_EQ(out.size(), 1u);
  EXPECT_EQ(out[0].ticks, 1000);
  EXPECT_TRUE(out[0].has_ts && out[0].has_gyro && out[0].has_accel);
  EXPECT_EQ(out[0].gyro, (std::array<int16_t, 3>{1, -2, 3}));
  EXPECT_EQ(out[0].accel, (std::array<int16_t, 3>{-100, 200, 4096}));
  EXPECT_DOUBLE_EQ(d.last_temperature_c(), 31.0);  // 6 * 256 LSB + 25 degC
  EXPECT_EQ(d.stats().words, 6u);
  EXPECT_EQ(d.stats().gyro_words, 1u);
  EXPECT_EQ(d.stats().accel_words, 1u);
  EXPECT_EQ(d.stats().timestamp_words, 1u);
  EXPECT_EQ(d.stats().temperature_words, 1u);
  EXPECT_EQ(d.stats().cfg_change_words, 1u);
  EXPECT_EQ(d.stats().unknown_tags, 1u);
}

TEST(FifoDecoder, ExtremeValuesAndTimestampBytes) {
  FifoDecoder d({});
  auto out = decode(d, {ts_word(1, 0xA1B2C3D4u), word(kTagGyroNC, 1, 32767, -32768, 0),
                        word(kTagAccelNC, 1, -1, 1, -32768)});
  ASSERT_EQ(out.size(), 1u);
  EXPECT_EQ(out[0].ticks, 0xA1B2C3D4LL);
  EXPECT_EQ(out[0].gyro[0], 32767);
  EXPECT_EQ(out[0].gyro[1], -32768);
  EXPECT_EQ(out[0].accel[2], -32768);
}

TEST(FifoDecoder, TagCounterPairingIsOrderIndependent) {
  FifoDecoder d({});
  // Slot order TS,G,XL then G,XL,TS then XL,TS,G: all three are paired correctly by TAG_CNT.
  auto out = decode(d, {ts_word(0, 100), word(kTagGyroNC, 0, 1), word(kTagAccelNC, 0, 2),
                        word(kTagGyroNC, 1, 3), word(kTagAccelNC, 1, 4), ts_word(1, 148),
                        word(kTagAccelNC, 2, 6), ts_word(2, 196), word(kTagGyroNC, 2, 5)});
  ASSERT_EQ(out.size(), 3u);
  EXPECT_EQ(out[0].ticks, 100);
  EXPECT_EQ(out[1].ticks, 148);
  EXPECT_EQ(out[1].gyro[0], 3);
  EXPECT_EQ(out[1].accel[0], 4);
  EXPECT_EQ(out[2].ticks, 196);
  EXPECT_EQ(out[2].gyro[0], 5);
  EXPECT_EQ(d.stats().partial_slots, 0u);
  EXPECT_EQ(d.stats().slots_without_ts, 0u);
}

TEST(FifoDecoder, TimestampFirstPairing) {
  FifoDecoder::Options o;
  o.pair_by_tag_counter = false;
  FifoDecoder d(o);
  auto out = decode(d, {ts_word(0, 100), word(kTagGyroNC, 0, 1), word(kTagAccelNC, 0, 2), ts_word(1, 148),
                        word(kTagGyroNC, 1, 3), word(kTagAccelNC, 1, 4)});
  ASSERT_EQ(out.size(), 2u);
  EXPECT_EQ(out[1].ticks, 148);
  EXPECT_EQ(out[1].accel[0], 4);
}

TEST(FifoDecoder, SlotSpansTwoReads) {
  FifoDecoder d({});
  std::vector<RawSlot> out;
  auto a = flat({ts_word(0, 100), word(kTagGyroNC, 0, 1)});
  d.push(a.data(), 2, out);
  EXPECT_TRUE(out.empty());  // incomplete slot is held
  auto b = flat({word(kTagAccelNC, 0, 2)});
  d.push(b.data(), 1, out);
  ASSERT_EQ(out.size(), 1u);
  EXPECT_EQ(out[0].accel[0], 2);
}

TEST(FifoDecoder, UnwrapsAcross32BitWrap) {
  FifoDecoder d({});
  auto out = decode(d, {ts_word(0, 0xFFFFFFE0u), word(kTagGyroNC, 0), word(kTagAccelNC, 0), ts_word(1, 0x10u),
                        word(kTagGyroNC, 1), word(kTagAccelNC, 1)});
  ASSERT_EQ(out.size(), 2u);
  EXPECT_EQ(out[0].ticks, 0xFFFFFFE0LL);
  EXPECT_EQ(out[1].ticks, 0x100000010LL);
  EXPECT_EQ(d.stats().timestamp_regressions, 0u);
  EXPECT_EQ(FifoDecoder::unwrap(5u, 0x2FFFFFFF0LL), 0x300000005LL);
  EXPECT_EQ(FifoDecoder::unwrap(0xFFFFFFF0u, 0x300000005LL), 0x2FFFFFFF0LL);
}

TEST(FifoDecoder, UnwrapReferenceFromBracket) {
  FifoDecoder d({});
  d.set_unwrap_reference(0x500000000LL + 40);  // 5 wraps already happened
  auto out = decode(d, {ts_word(0, 20), word(kTagGyroNC, 0), word(kTagAccelNC, 0)});
  ASSERT_EQ(out.size(), 1u);
  EXPECT_EQ(out[0].ticks, 0x500000000LL + 20);
}

TEST(FifoDecoder, RegressionCounted) {
  FifoDecoder d({});
  auto out = decode(d, {ts_word(0, 1000), word(kTagGyroNC, 0), word(kTagAccelNC, 0), ts_word(1, 900),
                        word(kTagGyroNC, 1), word(kTagAccelNC, 1)});
  ASSERT_EQ(out.size(), 2u);
  EXPECT_EQ(d.stats().timestamp_regressions, 1u);
  EXPECT_TRUE(out[1].after_gap);
}

TEST(FifoDecoder, TagCounterJumpMeansLostSlots) {
  FifoDecoder d({});
  auto out = decode(d, {ts_word(0, 100), word(kTagGyroNC, 0), word(kTagAccelNC, 0), ts_word(3, 244),
                        word(kTagGyroNC, 3), word(kTagAccelNC, 3)});
  ASSERT_EQ(out.size(), 2u);
  EXPECT_FALSE(out[0].after_gap);
  EXPECT_TRUE(out[1].after_gap);
  EXPECT_EQ(d.stats().lost_slots, 2u);
}

TEST(FifoDecoder, PartialSlotAfterDiscontinuity) {
  FifoDecoder d({});
  std::vector<RawSlot> out;
  auto a = flat({ts_word(0, 100), word(kTagGyroNC, 0, 7), word(kTagAccelNC, 0, 8)});
  d.push(a.data(), 3, out);
  d.mark_discontinuity(out);  // e.g. FIFO overrun: the head of the next read lost its timestamp word
  auto b = flat({word(kTagAccelNC, 1, 9), ts_word(2, 196), word(kTagGyroNC, 2), word(kTagAccelNC, 2)});
  d.push(b.data(), 4, out);
  ASSERT_EQ(out.size(), 2u);  // the head slot (accel only, no timestamp reference) is dropped
  EXPECT_EQ(out[1].ticks, 196);
  EXPECT_TRUE(out[1].after_gap);
  EXPECT_EQ(d.stats().untimed_slots, 1u);
}

TEST(FifoDecoder, PartialSlotRepeatsAreCounted) {
  FifoDecoder d({});
  auto out = decode(d, {ts_word(0, 100), word(kTagGyroNC, 0), word(kTagAccelNC, 0), ts_word(1, 148),
                        word(kTagGyroNC, 1), ts_word(2, 196), word(kTagGyroNC, 2), word(kTagAccelNC, 2)});
  ASSERT_EQ(out.size(), 3u);
  EXPECT_FALSE(out[1].has_accel);
  EXPECT_EQ(d.stats().partial_slots, 1u);
}

TEST(FifoDecoder, DecimatedTimestampsAreExtrapolated) {
  FifoDecoder::Options o;
  o.timestamp_decimation = 8;
  FifoDecoder d(o);
  std::vector<std::array<uint8_t, 7>> ws;
  for (int n = 0; n < 24; ++n) {
    const uint8_t c = static_cast<uint8_t>(n & 3);
    if (n % 8 == 0) ws.push_back(ts_word(c, static_cast<uint32_t>(1000 + 48 * n)));
    ws.push_back(word(kTagGyroNC, c, static_cast<int16_t>(n)));
    ws.push_back(word(kTagAccelNC, c, static_cast<int16_t>(n)));
  }
  ws.push_back(ts_word(0, 1000 + 48 * 24));  // closes slot 23
  auto out = decode(d, ws);
  ASSERT_EQ(out.size(), 24u);
  for (int n = 0; n < 24; ++n) EXPECT_EQ(out[static_cast<std::size_t>(n)].ticks, 1000 + 48 * n) << n;
  EXPECT_EQ(d.stats().slots_without_ts, 21u);
  EXPECT_NEAR(d.period_ticks(), 48.0, 1e-9);
}

}  // namespace
}  // namespace rig
