// SPDX-License-Identifier: MIT
// LSM6DSOX tagged-FIFO decoder. Rules and sources: rig/drivers/imu/lsm6dsox_fifo.hpp.
#include "rig/drivers/imu/lsm6dsox_fifo.hpp"

#include <cmath>

#include "rig/drivers/imu/lsm6dsox_regs.hpp"

namespace rig::lsm6dsox {

namespace {

int16_t le16(const uint8_t* p) { return static_cast<int16_t>(static_cast<uint16_t>(p[0] | (p[1] << 8))); }
uint32_t le32(const uint8_t* p) {
  return static_cast<uint32_t>(p[0]) | (static_cast<uint32_t>(p[1]) << 8) | (static_cast<uint32_t>(p[2]) << 16) |
         (static_cast<uint32_t>(p[3]) << 24);
}
bool odd_parity(uint8_t b) {
  int n = 0;
  for (int i = 0; i < 8; ++i) n += (b >> i) & 1;
  return (n & 1) != 0;
}

// A learned period is accepted only near the nominal one (a wrong slot count would otherwise poison it).
constexpr double kPeriodTolerance = 0.05;
constexpr double kPeriodAlpha = 0.05;

}  // namespace

void FifoDecoder::set_unwrap_reference(int64_t unwrapped_ticks) {
  ref_ticks_ = unwrapped_ticks;
  have_ref_ = true;
}

void FifoDecoder::start_slot(uint8_t cnt, std::vector<RawSlot>& /*out*/) {
  int64_t d = 1;
  if (opt_.pair_by_tag_counter && have_prev_cnt_ && !gap_next_) {
    d = static_cast<int64_t>((cnt - prev_cnt_) & 0x3);
    if (d == 0) d = 4;  // same TAG_CNT again after a complete slot: four slots went missing
  }
  cur_ = RawSlot{};
  cur_.tag_cnt = cnt;
  cur_.after_gap = gap_next_ || d > 1;
  if (d > 1) stats_.lost_slots += static_cast<uint64_t>(d - 1);
  gap_next_ = false;
  slot_index_ += d;
  prev_cnt_ = cnt;
  have_prev_cnt_ = true;
  pending_ = true;
}

void FifoDecoder::finish_slot(std::vector<RawSlot>& out) {
  pending_ = false;
  if (!cur_.has_gyro && !cur_.has_accel) return;  // timestamp-only slot: nothing to emit
  if (!cur_.has_ts) {
    if (!have_last_ts_) {
      ++stats_.untimed_slots;
      if (cur_.after_gap) gap_next_ = true;  // keep the gap flag for the first emitted slot
      return;
    }
    cur_.ticks = last_ts_ + std::llround(static_cast<double>(slot_index_ - last_ts_slot_) * period_ticks_);
    ++stats_.slots_without_ts;
  }
  if (!(cur_.has_gyro && cur_.has_accel)) ++stats_.partial_slots;
  out.push_back(cur_);
}

void FifoDecoder::mark_discontinuity(std::vector<RawSlot>& out) {
  if (pending_) finish_slot(out);
  gap_next_ = true;
  have_prev_cnt_ = false;
  have_last_ts_ = false;
}

void FifoDecoder::push(const uint8_t* words, std::size_t n_words, std::vector<RawSlot>& out) {
  for (std::size_t i = 0; i < n_words; ++i) {
    const uint8_t* w = words + i * kFifoWordBytes;
    const uint8_t tagb = w[0];
    const uint8_t* d = w + 1;
    ++stats_.words;
    if (odd_parity(tagb)) ++stats_.odd_parity_words;
    const uint8_t sensor = tag_sensor(tagb);
    const uint8_t cnt = tag_cnt(tagb);
    if (sensor == kTagTemperature) {
      ++stats_.temperature_words;
      last_temp_c_ = static_cast<double>(le16(d)) / kTempLsbPerC + kTempOffsetC;
      continue;
    }
    if (sensor == kTagCfgChange) {
      ++stats_.cfg_change_words;
      continue;
    }
    if (sensor != kTagGyroNC && sensor != kTagAccelNC && sensor != kTagTimestamp) {
      ++stats_.unknown_tags;
      continue;
    }

    const bool dup = pending_ && ((sensor == kTagGyroNC && cur_.has_gyro) ||
                                  (sensor == kTagAccelNC && cur_.has_accel) ||
                                  (sensor == kTagTimestamp && cur_.has_ts));
    if (opt_.pair_by_tag_counter) {
      if (pending_ && (cnt != cur_.tag_cnt || dup)) finish_slot(out);
    } else {
      if (pending_ && (sensor == kTagTimestamp || dup)) finish_slot(out);
    }
    if (!pending_) start_slot(cnt, out);

    if (sensor == kTagGyroNC) {
      ++stats_.gyro_words;
      cur_.gyro = {le16(d), le16(d + 2), le16(d + 4)};
      cur_.has_gyro = true;
    } else if (sensor == kTagAccelNC) {
      ++stats_.accel_words;
      cur_.accel = {le16(d), le16(d + 2), le16(d + 4)};
      cur_.has_accel = true;
    } else {
      ++stats_.timestamp_words;
      const uint32_t raw = le32(d);
      const int64_t u = have_ref_ ? unwrap(raw, ref_ticks_) : static_cast<int64_t>(raw);
      if (have_prev_ts_word_ && u < prev_ts_word_) {
        ++stats_.timestamp_regressions;
        cur_.after_gap = true;
        have_last_ts_ = false;
      } else if (have_last_ts_ && slot_index_ > last_ts_slot_) {
        const double p = static_cast<double>(u - last_ts_) / static_cast<double>(slot_index_ - last_ts_slot_);
        if (std::abs(p / opt_.nominal_period_ticks - 1.0) < kPeriodTolerance)
          period_ticks_ += kPeriodAlpha * (p - period_ticks_);
      }
      cur_.ticks = u;
      cur_.has_ts = true;
      last_ts_ = u;
      last_ts_slot_ = slot_index_;
      have_last_ts_ = true;
      prev_ts_word_ = u;
      have_prev_ts_word_ = true;
      ref_ticks_ = u;
      have_ref_ = true;
    }
    if (cur_.has_gyro && cur_.has_accel && cur_.has_ts) finish_slot(out);
  }
}

}  // namespace rig::lsm6dsox
