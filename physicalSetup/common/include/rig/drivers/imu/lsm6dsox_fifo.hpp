// SPDX-License-Identifier: MIT
/// @file rig/drivers/imu/lsm6dsox_fifo.hpp
/// @brief Pure decoder of LSM6DSOX tagged FIFO words into time slots (T1). No I/O; unit-tested with byte arrays.
///
/// FIFO word = tag byte (TAG_SENSOR[7:3], TAG_CNT[2:1], TAG_PARITY[0]) + 6 data bytes (DS12814 §6.6.7,
/// Tables 194-196). Data: gyro/accel = X, Y, Z int16 little-endian (DS §9.74-9.76); timestamp = 32-bit counter in
/// data bytes 0..3, little-endian (Linux st_lsm6dsx_buffer.c "B0 = ts[7:0] ... B3 = ts[31:24]"); temperature =
/// int16 in bytes 0..1 [H: assumed by analogy with OUT_TEMP_L/H; only used for display].
///
/// Pairing (which gyro, accel and timestamp words form one sample):
/// - by TAG_CNT (default): TAG_CNT is "a 2-bit counter that identifies the sensor time slot" (DS Table 195), so
///   consecutive gyro/accel/timestamp words with the same TAG_CNT are one slot, in any order. A TAG_CNT step > 1
///   means lost slots.
/// - timestamp-first: a timestamp word opens a new slot and the following data words belong to it (the order the
///   Linux driver assumes: it applies each timestamp word to the data words after it).
/// [H: ec-imu-probe --raw prints the word sequence; both rules must agree on hardware.]
/// Temperature and CFG_CHANGE words never delimit slots. Unknown tags are counted and skipped.
///
/// Timestamps are unwrapped to 64 bits relative to a reference (the last timestamp word, or the last bracketed
/// register read supplied with set_unwrap_reference()): unwrapped = ref + int32(raw - uint32(ref)).
/// Threading: not thread-safe (one owner: the driver poll thread).
#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <vector>

namespace rig::lsm6dsox {

/// One decoded time slot (one IMU sample before scaling and clock mapping).
struct RawSlot {
  int64_t ticks = -1;          ///< Unwrapped chip timestamp of the slot (-1 = unknown, cannot be timed).
  bool has_ts = false;         ///< Timestamp came from a FIFO timestamp word of this slot (else extrapolated).
  bool has_gyro = false;
  bool has_accel = false;
  std::array<int16_t, 3> gyro{};
  std::array<int16_t, 3> accel{};
  uint8_t tag_cnt = 0;
  bool after_gap = false;      ///< Slots were lost before this one (overrun, TAG_CNT jump, discontinuity).
};

struct FifoDecoderStats {
  uint64_t words = 0;
  uint64_t gyro_words = 0;
  uint64_t accel_words = 0;
  uint64_t timestamp_words = 0;
  uint64_t temperature_words = 0;
  uint64_t cfg_change_words = 0;
  uint64_t unknown_tags = 0;
  uint64_t timestamp_regressions = 0;   ///< Timestamp word older than the previous one (not a wrap).
  uint64_t partial_slots = 0;           ///< Emitted slots missing gyro or accel.
  uint64_t slots_without_ts = 0;        ///< Emitted slots timed by extrapolation.
  uint64_t untimed_slots = 0;           ///< Slots dropped: no timestamp reference yet.
  uint64_t lost_slots = 0;              ///< Slots inferred lost from TAG_CNT steps.
  uint64_t odd_parity_words = 0;        ///< Tag bytes with an odd number of set bits (diagnostic for [H]).
};

class FifoDecoder {
 public:
  struct Options {
    bool pair_by_tag_counter = true;
    int timestamp_decimation = 1;        ///< 1 | 8 | 32 (FIFO_CTRL4.DEC_TS_BATCH).
    double nominal_period_ticks = 48.0;  ///< Initial slot period [ticks] (833 Hz: 48 x 25 us); learned online.
  };

  explicit FifoDecoder(Options opt) : opt_(opt), period_ticks_(opt.nominal_period_ticks) {}

  /// Supplies an unwrapped counter value read from TIMESTAMP0..3 (aligns the 32-bit unwrap before the first
  /// timestamp word, and after a discontinuity).
  void set_unwrap_reference(int64_t unwrapped_ticks);

  /// Decodes `n_words` 7-byte words; appends every completed slot to `out` (time order).
  void push(const uint8_t* words, std::size_t n_words, std::vector<RawSlot>& out);

  /// Data were lost before the next word (FIFO overrun, failed bus read): emits the pending slot to `out`,
  /// flags the next slot `after_gap` and stops trusting TAG_CNT steps / extrapolation until the next timestamp.
  void mark_discontinuity(std::vector<RawSlot>& out);

  /// Most recent temperature word [deg C]; NaN until one was decoded.
  double last_temperature_c() const { return last_temp_c_; }
  /// Learned slot period [ticks].
  double period_ticks() const { return period_ticks_; }
  const FifoDecoderStats& stats() const { return stats_; }

  /// Unwraps a 32-bit counter value against a 64-bit reference (nearest value within +-2^31 ticks).
  static int64_t unwrap(uint32_t raw, int64_t reference) {
    const uint32_t delta = raw - static_cast<uint32_t>(static_cast<uint64_t>(reference));
    return reference + static_cast<int32_t>(delta);
  }

 private:
  void start_slot(uint8_t cnt, std::vector<RawSlot>& out);
  void finish_slot(std::vector<RawSlot>& out);

  Options opt_;
  FifoDecoderStats stats_;
  double period_ticks_;
  double last_temp_c_ = std::numeric_limits<double>::quiet_NaN();

  bool have_ref_ = false;
  int64_t ref_ticks_ = 0;
  bool have_last_ts_ = false;   // last_ts_* usable for extrapolation (no discontinuity since)
  int64_t last_ts_ = 0;
  int64_t last_ts_slot_ = 0;
  bool have_prev_ts_word_ = false;  // for regressions (survives discontinuities)
  int64_t prev_ts_word_ = 0;

  bool pending_ = false;
  RawSlot cur_;
  int64_t slot_index_ = 0;
  bool have_prev_cnt_ = false;
  uint8_t prev_cnt_ = 0;
  bool gap_next_ = false;
};

}  // namespace rig::lsm6dsox
