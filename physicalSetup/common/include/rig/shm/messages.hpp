// SPDX-License-Identifier: MIT
/// @file rig/shm/messages.hpp
/// @brief Wire formats of ring payloads (cross-process ABI). FROZEN CONTRACT (T0); encode/decode implemented by T3.
///
/// All structs are trivially copyable PODs with fixed-width fields and explicit padding; host byte order
/// (writer and readers run on the same machine). Bump kShmRingVersion when any of them changes.
#pragma once

#include <cstddef>
#include <cstdint>
#include <type_traits>

#include "rig/span.hpp"
#include "rig/status.hpp"
#include "rig/types.hpp"

namespace rig {

/// Header of a kImageFrame payload, followed immediately by the pixel bytes (planes at their offsets relative
/// to the end of this header).
struct ShmFrameHeader {
  char camera_id[16];        ///< NUL-terminated ImageFrame::camera_id.
  uint64_t seq;              ///< ImageFrame::seq.
  int64_t t_sof_ns;          ///< [ns, MONOTONIC_RAW].
  int64_t t_mid_ns;
  int64_t t_arrival_ns;
  int32_t width;             ///< Of the published image (may be downscaled versus capture).
  int32_t height;
  uint32_t fmt;              ///< PixelFormat.
  uint32_t n_planes;
  uint64_t plane_offset[kMaxPlanes];  ///< Relative to the first pixel byte.
  uint64_t plane_stride[kMaxPlanes];
  int32_t plane_width[kMaxPlanes];
  int32_t plane_height[kMaxPlanes];
  uint64_t data_bytes;       ///< Pixel bytes following the header.
  int32_t src_width;         ///< Capture size before any downscale.
  int32_t src_height;
  // FrameMeta
  int64_t exposure_ns;
  double analog_gain;
  double digital_gain;
  int64_t readout_ns;
  int32_t sensor_mode;
  int32_t focus_position;
  uint32_t focus_status;     ///< FocusStatus.
  uint8_t color_range;       ///< ColorRange (R1 F17; was reserved, so a zeroing older writer reads as kUnknown).
  uint8_t reserved0[3];
};
static_assert(std::is_trivially_copyable_v<ShmFrameHeader> && std::is_standard_layout_v<ShmFrameHeader>);
static_assert(sizeof(ShmFrameHeader) == 200, "ShmFrameHeader layout changed: bump kShmRingVersion");

/// One kImuSample payload.
struct ShmImuSample {
  int64_t t_ns;        ///< [ns, MONOTONIC_RAW].
  int64_t chip_ticks;
  double gyro_rps[3];  ///< [rad/s], IMU body frame.
  double acc_mps2[3];  ///< [m/s^2].
  float temp_c;
  uint32_t flags;      ///< ImuSampleFlags.
};
static_assert(std::is_trivially_copyable_v<ShmImuSample> && std::is_standard_layout_v<ShmImuSample>);
static_assert(sizeof(ShmImuSample) == 72, "ShmImuSample layout changed: bump kShmRingVersion");

/// Bytes needed to publish `frame` (header + pixel data).
inline std::size_t encoded_frame_size(const ImageFrame& frame) noexcept { return sizeof(ShmFrameHeader) + frame.size; }

/// Serializes `frame` (planes copied tightly or as-is; implementation's choice, described by the header) into
/// `out`. Returns bytes written; kOutOfRange if `out` is too small. `src_width/height` = the capture size when
/// the caller downscaled first (pass 0 to use frame.width/height).
Result<std::size_t> encode_frame(const ImageFrame& frame, Span<uint8_t> out, int src_width = 0, int src_height = 0);

/// Parses a kImageFrame payload into a frame that owns a copy of the pixels. kDataLoss if inconsistent.
Result<ImageFrame> decode_frame(Span<const uint8_t> payload);

/// Field-by-field conversions (implemented inline: trivial and needed by T3, T5, T6).
inline ShmImuSample to_shm(const ImuSample& s) noexcept {
  ShmImuSample o{};
  o.t_ns = s.t_ns;
  o.chip_ticks = s.chip_ticks;
  for (int i = 0; i < 3; ++i) {
    o.gyro_rps[i] = s.gyro_rps[i];
    o.acc_mps2[i] = s.acc_mps2[i];
  }
  o.temp_c = s.temp_c;
  o.flags = s.flags;
  return o;
}

inline ImuSample from_shm(const ShmImuSample& s) noexcept {
  ImuSample o;
  o.t_ns = s.t_ns;
  o.chip_ticks = s.chip_ticks;
  o.gyro_rps = Eigen::Vector3d(s.gyro_rps[0], s.gyro_rps[1], s.gyro_rps[2]);
  o.acc_mps2 = Eigen::Vector3d(s.acc_mps2[0], s.acc_mps2[1], s.acc_mps2[2]);
  o.temp_c = s.temp_c;
  o.flags = s.flags;
  return o;
}

}  // namespace rig
