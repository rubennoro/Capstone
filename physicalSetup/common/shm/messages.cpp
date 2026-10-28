// SPDX-License-Identifier: MIT
// kImageFrame payload encode/decode (rig/shm/messages.hpp). Planes are always copied TIGHTLY (stride = row bytes,
// planes back to back), so a ring's slot size depends only on width x height x format (frame_payload_bytes()).
#include <cstring>

#include "rig/shm/messages.hpp"
#include "rig/shm/publisher.hpp"

namespace rig {
namespace {

/// Bytes per sample of plane `i` (see PixelFormat docs in rig/types.hpp). 0 for JPEG/unknown.
int bytes_per_sample(PixelFormat f, int i) {
  switch (f) {
    case PixelFormat::kNV12: return i == 0 ? 1 : 2;
    case PixelFormat::kGRAY8: return 1;
    case PixelFormat::kBGR8:
    case PixelFormat::kRGB8: return 3;
    case PixelFormat::kYUYV: return 2;
    default: return 0;
  }
}

/// Expected plane geometry of a tightly packed image.
bool expected_plane(PixelFormat f, int i, int w, int h, int& pw, int& ph) {
  if (i >= num_planes(f)) return false;
  if (f == PixelFormat::kNV12 && i == 1) {
    pw = w / 2;
    ph = h / 2;
  } else {
    pw = w;
    ph = h;
  }
  return true;
}

void copy_cstr(char* dst, std::size_t n, const std::string& s) {
  std::memset(dst, 0, n);
  std::memcpy(dst, s.data(), std::min(s.size(), n - 1));
}

}  // namespace

std::size_t frame_payload_bytes(int width, int height, PixelFormat fmt) {
  if (width <= 0 || height <= 0) return 0;
  std::size_t n = 0;
  for (int i = 0; i < num_planes(fmt); ++i) {
    int pw = 0, ph = 0;
    expected_plane(fmt, i, width, height, pw, ph);
    n += static_cast<std::size_t>(pw) * static_cast<std::size_t>(ph) * static_cast<std::size_t>(bytes_per_sample(fmt, i));
  }
  return n;
}

Result<std::size_t> encode_frame(const ImageFrame& f, Span<uint8_t> out, int src_width, int src_height) {
  if (!f.data || f.size == 0) return invalid_argument("encode_frame: empty frame");
  if (f.fmt == PixelFormat::kUnknown) return invalid_argument("encode_frame: unknown pixel format");
  ShmFrameHeader h{};
  copy_cstr(h.camera_id, sizeof(h.camera_id), f.camera_id);
  h.seq = f.seq;
  h.t_sof_ns = f.t_sof_ns;
  h.t_mid_ns = f.t_mid_ns;
  h.t_arrival_ns = f.t_arrival_ns;
  h.width = f.width;
  h.height = f.height;
  h.fmt = static_cast<uint32_t>(f.fmt);
  h.n_planes = static_cast<uint32_t>(num_planes(f.fmt));
  h.src_width = src_width > 0 ? src_width : f.width;
  h.src_height = src_height > 0 ? src_height : f.height;
  h.exposure_ns = f.meta.exposure_ns;
  h.analog_gain = f.meta.analog_gain;
  h.digital_gain = f.meta.digital_gain;
  h.readout_ns = f.meta.readout_ns;
  h.sensor_mode = f.meta.sensor_mode;
  h.focus_position = f.meta.focus_position;
  h.focus_status = static_cast<uint32_t>(f.meta.focus_status);
  h.color_range = static_cast<uint8_t>(f.meta.color_range);

  if (f.fmt == PixelFormat::kJPEG) {
    h.plane_offset[0] = 0;
    h.plane_stride[0] = 0;
    h.plane_width[0] = f.width;
    h.plane_height[0] = f.height;
    h.data_bytes = f.size;
    if (out.size() < sizeof(h) + f.size)
      return out_of_range("encode_frame: buffer " + std::to_string(out.size()) + " < " + std::to_string(sizeof(h) + f.size));
    std::memcpy(out.data(), &h, sizeof(h));
    std::memcpy(out.data() + sizeof(h), f.data.get(), f.size);
    return sizeof(h) + f.size;
  }

  // Validate the source planes, then copy tightly.
  std::size_t total = 0;
  for (int i = 0; i < num_planes(f.fmt); ++i) {
    const Plane& p = f.planes[static_cast<std::size_t>(i)];
    int pw = 0, ph = 0;
    expected_plane(f.fmt, i, f.width, f.height, pw, ph);
    if (p.width != pw || p.height != ph)
      return invalid_argument("encode_frame: " + f.camera_id + " plane " + std::to_string(i) + " is " +
                              std::to_string(p.width) + "x" + std::to_string(p.height) + ", expected " +
                              std::to_string(pw) + "x" + std::to_string(ph));
    const std::size_t row = static_cast<std::size_t>(pw) * static_cast<std::size_t>(bytes_per_sample(f.fmt, i));
    if (p.stride < row) return invalid_argument("encode_frame: plane " + std::to_string(i) + " stride < row bytes");
    if (ph > 0 && p.offset + (static_cast<std::size_t>(ph) - 1) * p.stride + row > f.size)
      return invalid_argument("encode_frame: plane " + std::to_string(i) + " extends past the frame buffer");
    h.plane_offset[i] = total;
    h.plane_stride[i] = row;
    h.plane_width[i] = pw;
    h.plane_height[i] = ph;
    total += row * static_cast<std::size_t>(ph);
  }
  h.data_bytes = total;
  if (out.size() < sizeof(h) + total)
    return out_of_range("encode_frame: buffer " + std::to_string(out.size()) + " < " + std::to_string(sizeof(h) + total));
  std::memcpy(out.data(), &h, sizeof(h));
  uint8_t* dst = out.data() + sizeof(h);
  for (int i = 0; i < num_planes(f.fmt); ++i) {
    const Plane& p = f.planes[static_cast<std::size_t>(i)];
    const std::size_t row = h.plane_stride[i];
    const uint8_t* src = f.plane_data(i);
    for (int r = 0; r < p.height; ++r) {
      std::memcpy(dst, src + static_cast<std::size_t>(r) * p.stride, row);
      dst += row;
    }
  }
  return sizeof(h) + total;
}

Result<ImageFrame> decode_frame(Span<const uint8_t> payload) {
  if (payload.size() < sizeof(ShmFrameHeader)) return data_loss("decode_frame: payload shorter than the header");
  ShmFrameHeader h;
  std::memcpy(&h, payload.data(), sizeof(h));
  const PixelFormat fmt = static_cast<PixelFormat>(h.fmt);
  if (h.fmt > static_cast<uint32_t>(PixelFormat::kJPEG) || fmt == PixelFormat::kUnknown)
    return data_loss("decode_frame: bad pixel format " + std::to_string(h.fmt));
  if (h.n_planes != static_cast<uint32_t>(num_planes(fmt))) return data_loss("decode_frame: plane count mismatch");
  if (h.width <= 0 || h.height <= 0) return data_loss("decode_frame: bad size");
  if (h.data_bytes != payload.size() - sizeof(h))
    return data_loss("decode_frame: data_bytes " + std::to_string(h.data_bytes) + " != payload " +
                     std::to_string(payload.size() - sizeof(h)));
  if (h.camera_id[sizeof(h.camera_id) - 1] != '\0') return data_loss("decode_frame: camera_id not terminated");
  ImageFrame f;
  for (uint32_t i = 0; i < h.n_planes; ++i) {
    Plane& p = f.planes[i];
    p.offset = h.plane_offset[i];
    p.stride = h.plane_stride[i];
    p.width = h.plane_width[i];
    p.height = h.plane_height[i];
    if (fmt == PixelFormat::kJPEG) {
      if (p.offset != 0) return data_loss("decode_frame: jpeg offset");
      continue;
    }
    int pw = 0, ph = 0;
    expected_plane(fmt, static_cast<int>(i), h.width, h.height, pw, ph);
    const std::size_t row = static_cast<std::size_t>(pw) * static_cast<std::size_t>(bytes_per_sample(fmt, static_cast<int>(i)));
    if (p.width != pw || p.height != ph || p.stride < row) return data_loss("decode_frame: plane geometry");
    if (p.offset > h.data_bytes || (ph > 0 && (static_cast<std::size_t>(ph) - 1) * p.stride + row > h.data_bytes - p.offset))
      return data_loss("decode_frame: plane " + std::to_string(i) + " outside the data");
  }
  auto buf = std::shared_ptr<uint8_t[]>(new uint8_t[h.data_bytes > 0 ? h.data_bytes : 1]);
  std::memcpy(buf.get(), payload.data() + sizeof(h), h.data_bytes);
  f.data = buf;
  f.size = h.data_bytes;
  f.camera_id = std::string(h.camera_id);
  f.seq = h.seq;
  f.t_sof_ns = h.t_sof_ns;
  f.t_mid_ns = h.t_mid_ns;
  f.t_arrival_ns = h.t_arrival_ns;
  f.width = h.width;
  f.height = h.height;
  f.fmt = fmt;
  f.meta.exposure_ns = h.exposure_ns;
  f.meta.analog_gain = h.analog_gain;
  f.meta.digital_gain = h.digital_gain;
  f.meta.readout_ns = h.readout_ns;
  f.meta.sensor_mode = h.sensor_mode;
  f.meta.focus_position = h.focus_position;
  f.meta.focus_status = static_cast<FocusStatus>(h.focus_status);
  if (h.color_range > static_cast<uint8_t>(ColorRange::kFull))
    return data_loss("decode_frame: bad color_range " + std::to_string(h.color_range));
  f.meta.color_range = static_cast<ColorRange>(h.color_range);
  return f;
}

}  // namespace rig
