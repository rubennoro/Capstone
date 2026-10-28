// SPDX-License-Identifier: MIT
// T3: kImageFrame encode/decode, downscale, and the hub -> shm publishers end to end.
#include <gtest/gtest.h>

#include <unistd.h>

#include <chrono>
#include <cstddef>
#include <cstring>
#include <thread>
#include <vector>

#include "rig/shm/messages.hpp"
#include "rig/shm/publisher.hpp"
#include "rig/time/clock.hpp"

namespace rig {
namespace {

using namespace std::chrono_literals;

/// NV12 frame with padded strides (as Argus pitch-linear buffers have).
ImageFrame nv12(int w, int h, std::size_t pad, uint64_t seq) {
  ImageFrame f;
  f.camera_id = "uw";
  f.seq = seq;
  f.t_sof_ns = 1000 + static_cast<TimeNs>(seq);
  f.t_mid_ns = 2000 + static_cast<TimeNs>(seq);
  f.t_arrival_ns = 3000;
  f.width = w;
  f.height = h;
  f.fmt = PixelFormat::kNV12;
  const std::size_t ys = static_cast<std::size_t>(w) + pad;
  const std::size_t size = ys * h + ys * (h / 2);
  auto buf = std::shared_ptr<uint8_t[]>(new uint8_t[size]);
  for (std::size_t i = 0; i < size; ++i) buf[i] = static_cast<uint8_t>(i * 31 + seq);
  f.planes[0] = Plane{0, ys, w, h};
  f.planes[1] = Plane{ys * h, ys, w / 2, h / 2};
  f.data = buf;
  f.size = size;
  f.meta.exposure_ns = 4000000;
  f.meta.analog_gain = 4.0;
  f.meta.focus_status = FocusStatus::kNotApplicable;
  return f;
}

TEST(FrameCodec, Nv12RoundTripStripsPadding) {
  const ImageFrame f = nv12(64, 48, 16, 7);
  EXPECT_EQ(frame_payload_bytes(64, 48, PixelFormat::kNV12), 64u * 48 * 3 / 2);
  std::vector<uint8_t> buf(encoded_frame_size(f));
  auto n = encode_frame(f, buf, 1640, 1232);
  ASSERT_TRUE(n.ok()) << n.status().to_string();
  EXPECT_EQ(*n, sizeof(ShmFrameHeader) + 64u * 48 * 3 / 2);
  auto d = decode_frame(Span<const uint8_t>(buf.data(), *n));
  ASSERT_TRUE(d.ok()) << d.status().to_string();
  EXPECT_EQ(d->camera_id, "uw");
  EXPECT_EQ(d->seq, 7u);
  EXPECT_EQ(d->t_mid_ns, 2007);
  EXPECT_EQ(d->meta.exposure_ns, 4000000);
  EXPECT_EQ(d->meta.focus_status, FocusStatus::kNotApplicable);
  EXPECT_TRUE(d->valid());
  for (int p = 0; p < 2; ++p) {
    const Plane& a = f.planes[static_cast<std::size_t>(p)];
    const Plane& b = d->planes[static_cast<std::size_t>(p)];
    ASSERT_EQ(a.height, b.height);
    for (int r = 0; r < a.height; ++r)
      ASSERT_EQ(0, std::memcmp(f.plane_data(p) + r * a.stride, d->plane_data(p) + r * b.stride,
                               static_cast<std::size_t>(a.width) * (p == 0 ? 1 : 2)));
  }
  ShmFrameHeader h;
  std::memcpy(&h, buf.data(), sizeof(h));
  EXPECT_EQ(h.src_width, 1640);
  EXPECT_EQ(h.src_height, 1232);
}

TEST(FrameCodec, ColorRangeRoundTrips) {
  // R1 F17: consumers must see full-range NV12 as full, not fall back to limited.
  for (ColorRange cr : {ColorRange::kUnknown, ColorRange::kLimited, ColorRange::kFull}) {
    ImageFrame f = nv12(32, 16, 0, 1);
    f.meta.color_range = cr;
    std::vector<uint8_t> buf(encoded_frame_size(f));
    auto n = encode_frame(f, buf);
    ASSERT_TRUE(n.ok()) << n.status().to_string();
    auto d = decode_frame(Span<const uint8_t>(buf.data(), *n));
    ASSERT_TRUE(d.ok()) << d.status().to_string();
    EXPECT_EQ(d->meta.color_range, cr);
  }
  ImageFrame f = nv12(32, 16, 0, 1);
  std::vector<uint8_t> buf(encoded_frame_size(f));
  auto n = encode_frame(f, buf);
  ASSERT_TRUE(n.ok());
  buf[offsetof(ShmFrameHeader, color_range)] = 7;
  EXPECT_EQ(decode_frame(Span<const uint8_t>(buf.data(), *n)).status().code(), StatusCode::kDataLoss);
}

TEST(FrameCodec, RejectsInconsistentInputAndCorruptPayloads) {
  ImageFrame f = nv12(32, 16, 0, 1);
  std::vector<uint8_t> small(sizeof(ShmFrameHeader) + 10);
  EXPECT_EQ(encode_frame(f, small).status().code(), StatusCode::kOutOfRange);
  ImageFrame bad = f;
  bad.planes[1].height = 16;
  std::vector<uint8_t> buf(4096);
  EXPECT_EQ(encode_frame(bad, buf).status().code(), StatusCode::kInvalidArgument);
  bad = f;
  bad.size = 100;
  EXPECT_EQ(encode_frame(bad, buf).status().code(), StatusCode::kInvalidArgument);
  EXPECT_EQ(encode_frame(ImageFrame{}, buf).status().code(), StatusCode::kInvalidArgument);

  auto n = encode_frame(f, buf).value();
  EXPECT_EQ(decode_frame(Span<const uint8_t>(buf.data(), 10)).status().code(), StatusCode::kDataLoss);
  EXPECT_EQ(decode_frame(Span<const uint8_t>(buf.data(), n - 1)).status().code(), StatusCode::kDataLoss);
  std::vector<uint8_t> c(buf.begin(), buf.begin() + static_cast<std::ptrdiff_t>(n));
  reinterpret_cast<ShmFrameHeader*>(c.data())->fmt = 99;
  EXPECT_EQ(decode_frame(c).status().code(), StatusCode::kDataLoss);
  std::memcpy(c.data(), buf.data(), sizeof(ShmFrameHeader));
  reinterpret_cast<ShmFrameHeader*>(c.data())->plane_offset[1] = 1u << 20;
  EXPECT_EQ(decode_frame(c).status().code(), StatusCode::kDataLoss);
}

TEST(FrameCodec, JpegPassThrough) {
  ImageFrame f;
  f.camera_id = "main";
  f.fmt = PixelFormat::kJPEG;
  f.width = 1280;
  f.height = 720;
  f.t_mid_ns = 5;
  auto buf = std::shared_ptr<uint8_t[]>(new uint8_t[3]{0xFF, 0xD8, 0xFF});
  f.data = buf;
  f.size = 3;
  std::vector<uint8_t> out(sizeof(ShmFrameHeader) + 3);
  ASSERT_EQ(encode_frame(f, out).value(), out.size());
  auto d = decode_frame(out).value();
  EXPECT_EQ(d.size, 3u);
  EXPECT_EQ(d.data[1], 0xD8);
}

TEST(Downscale, SizesAndFormats) {
  int w = 0, h = 0;
  downscaled_size(1640, 1232, 640, w, h);
  EXPECT_EQ(w, 640);
  EXPECT_EQ(h, 480);
  downscaled_size(1920, 1080, 641, w, h);
  EXPECT_EQ(w, 640);
  EXPECT_EQ(h, 360);
  downscaled_size(640, 480, 0, w, h);
  EXPECT_EQ(w, 640);
  const ImageFrame f = nv12(128, 96, 8, 3);
  auto same = downscale_frame(f, 0).value();
  EXPECT_EQ(same.data.get(), f.data.get());  // no copy
  auto s = downscale_frame(f, 64).value();
  EXPECT_EQ(s.width, 64);
  EXPECT_EQ(s.height, 48);
  EXPECT_EQ(s.size, frame_payload_bytes(64, 48, PixelFormat::kNV12));
  EXPECT_EQ(s.t_mid_ns, f.t_mid_ns);
  std::vector<uint8_t> buf(encoded_frame_size(s));
  EXPECT_TRUE(encode_frame(s, buf).ok());
  ImageFrame y = f;
  y.fmt = PixelFormat::kYUYV;
  EXPECT_EQ(downscale_frame(y, 64).status().code(), StatusCode::kInvalidArgument);
}

TEST(ShmPublisher, FramesAndImuReachReaders) {
  const std::string prefix = "/rigtest." + std::to_string(getpid());
  StreamHub<ImageFrame> cam("cam_uw");
  StreamHub<ImuSample> imu("imu0");
  ShmFramePublisherOptions o;
  o.shm_name = prefix + ".cam_uw";
  o.capture_width = 128;
  o.capture_height = 96;
  o.out_width = 64;
  o.n_slots = 4;
  auto fp = ShmFramePublisher::start(cam, o);
  ASSERT_TRUE(fp.ok()) << fp.status().to_string();
  EXPECT_EQ((*fp)->out_height(), 48);
  auto ip = ShmImuPublisher::start(imu, prefix + ".imu0", 64).value();
  auto fr = ShmRingReader::open(o.shm_name, StreamKind::kImageFrame, 0, false).value();
  auto ir = ShmRingReader::open(prefix + ".imu0", StreamKind::kImuSample, 0, false).value();

  cam.publish(nv12(128, 96, 0, 11));
  cam.publish(nv12(100, 96, 0, 12));  // wrong size for this ring: counted as an error, not published
  for (int i = 0; i < 10; ++i) {
    ImuSample s;
    s.t_ns = 100 + i;
    s.acc_mps2 = Eigen::Vector3d(0, 0, kStandardGravity);
    imu.publish(s);
  }
  std::vector<uint8_t> buf(fr->slot_size());
  auto i = fr->wait_next(buf, 2s).value();
  ASSERT_TRUE(i.has_value());
  auto f = decode_frame(Span<const uint8_t>(buf.data(), i->bytes)).value();
  EXPECT_EQ(f.seq, 11u);
  EXPECT_EQ(f.width, 64);
  EXPECT_EQ(i->t_ns, f.t_mid_ns);
  int got = 0;
  ShmImuSample m{};
  const TimeNs deadline = now_raw_ns() + 2000000000LL;
  while (got < 10 && now_raw_ns() < deadline) {
    auto r = ir->wait_next(Span<uint8_t>(reinterpret_cast<uint8_t*>(&m), sizeof(m)), 100ms).value();
    if (!r) continue;
    EXPECT_EQ(m.t_ns, 100 + got);
    EXPECT_DOUBLE_EQ(m.acc_mps2[2], kStandardGravity);
    ++got;
  }
  EXPECT_EQ(got, 10);
  const TimeNs d2 = now_raw_ns() + 2000000000LL;
  while ((*fp)->stats().errors == 0 && now_raw_ns() < d2) std::this_thread::sleep_for(1ms);
  EXPECT_EQ((*fp)->stats().errors, 1u);
  EXPECT_EQ((*fp)->stats().published, 1u);
  cam.close();
  imu.close();
  (*fp)->stop();
  ip->stop();
  EXPECT_EQ(ip->stats().published, 10u);
}

}  // namespace
}  // namespace rig
