// SPDX-License-Identifier: MIT
// StreamHub -> shm ring bridges (rig/shm/publisher.hpp).
#include "rig/shm/publisher.hpp"

#include <atomic>
#include <chrono>
#include <cstring>
#include <mutex>
#include <thread>

#include <opencv2/imgproc.hpp>

namespace rig {

void downscaled_size(int width, int height, int target_width, int& ow, int& oh) {
  if (target_width <= 0 || target_width >= width || width <= 0) {
    ow = width;
    oh = height;
    return;
  }
  ow = target_width & ~1;
  oh = static_cast<int>(static_cast<int64_t>(height) * ow / width);
  oh = std::max(2, oh & ~1);
}

Result<ImageFrame> downscale_frame(const ImageFrame& f, int target_width) {
  int ow = 0, oh = 0;
  downscaled_size(f.width, f.height, target_width, ow, oh);
  if (ow == f.width && oh == f.height) return f;
  if (!f.valid()) return invalid_argument("downscale_frame: invalid frame");
  ImageFrame o = f;  // timestamps, seq, meta
  o.width = ow;
  o.height = oh;
  o.planes = {};
  const std::size_t bytes = frame_payload_bytes(ow, oh, f.fmt);
  if (bytes == 0) return invalid_argument(std::string("downscale_frame: format ") + to_string(f.fmt) + " not supported");
  auto buf = std::shared_ptr<uint8_t[]>(new uint8_t[bytes]);
  try {
    if (f.fmt == PixelFormat::kNV12) {
      const Plane& y = f.planes[0];
      const Plane& uv = f.planes[1];
      cv::Mat ys(y.height, y.width, CV_8UC1, const_cast<uint8_t*>(f.plane_data(0)), y.stride);
      cv::Mat uvs(uv.height, uv.width, CV_8UC2, const_cast<uint8_t*>(f.plane_data(1)), uv.stride);
      cv::Mat yd(oh, ow, CV_8UC1, buf.get(), static_cast<std::size_t>(ow));
      cv::Mat uvd(oh / 2, ow / 2, CV_8UC2, buf.get() + static_cast<std::size_t>(ow) * oh, static_cast<std::size_t>(ow));
      cv::resize(ys, yd, yd.size(), 0, 0, cv::INTER_AREA);
      cv::resize(uvs, uvd, uvd.size(), 0, 0, cv::INTER_AREA);
      o.planes[0] = Plane{0, static_cast<std::size_t>(ow), ow, oh};
      o.planes[1] = Plane{static_cast<std::size_t>(ow) * oh, static_cast<std::size_t>(ow), ow / 2, oh / 2};
    } else {
      const int type = f.fmt == PixelFormat::kGRAY8 ? CV_8UC1 : CV_8UC3;
      if (f.fmt != PixelFormat::kGRAY8 && f.fmt != PixelFormat::kBGR8 && f.fmt != PixelFormat::kRGB8)
        return invalid_argument(std::string("downscale_frame: format ") + to_string(f.fmt) + " not supported");
      const Plane& p = f.planes[0];
      const int bpp = type == CV_8UC1 ? 1 : 3;
      cv::Mat s(p.height, p.width, type, const_cast<uint8_t*>(f.plane_data(0)), p.stride);
      cv::Mat d(oh, ow, type, buf.get(), static_cast<std::size_t>(ow) * bpp);
      cv::resize(s, d, d.size(), 0, 0, cv::INTER_AREA);
      o.planes[0] = Plane{0, static_cast<std::size_t>(ow) * bpp, ow, oh};
    }
  } catch (const cv::Exception& e) {
    return invalid_argument(std::string("downscale_frame: OpenCV: ") + e.what());
  }
  o.data = buf;
  o.size = bytes;
  return o;
}

// ---------------------------------------------------------------------------------------------------------------

struct ShmFramePublisher::Impl {
  ShmFramePublisherOptions opt;
  int ow = 0, oh = 0;
  std::unique_ptr<ShmRingWriter> ring;
  std::shared_ptr<StreamSubscription<ImageFrame>> sub;
  StreamHub<ImageFrame>* hub = nullptr;
  std::atomic<bool> stop{false};
  std::atomic<uint64_t> published{0}, errors{0};
  std::thread th;

  void run() {
    ImageFrame f;
    while (!stop.load(std::memory_order_acquire)) {
      if (!sub->pop(f, std::chrono::milliseconds(50))) {
        if (sub->closed()) break;
        continue;
      }
      auto small = downscale_frame(f, opt.out_width);
      if (!small.ok() || small->width != ow || small->height != oh) {
        errors.fetch_add(1, std::memory_order_relaxed);
        continue;
      }
      Span<uint8_t> slot = ring->begin_write();
      auto n = encode_frame(*small, slot, f.width, f.height);
      if (!n.ok()) {
        // Not committed: the slot stays "being written" (readers treat its old content as overwritten) and the
        // next begin_write() reuses it. No bogus message is ever published.
        errors.fetch_add(1, std::memory_order_relaxed);
        continue;
      }
      if (ring->commit(*n, f.t_mid_ns).ok()) published.fetch_add(1, std::memory_order_relaxed);
    }
  }
};

ShmFramePublisher::ShmFramePublisher() = default;
ShmFramePublisher::~ShmFramePublisher() { stop(); }

Result<std::unique_ptr<ShmFramePublisher>> ShmFramePublisher::start(StreamHub<ImageFrame>& hub,
                                                                    const ShmFramePublisherOptions& o) {
  if (o.fmt == PixelFormat::kJPEG || o.fmt == PixelFormat::kUnknown)
    return invalid_argument(o.shm_name + ": frame publisher needs a raw pixel format");
  auto p = std::unique_ptr<ShmFramePublisher>(new ShmFramePublisher());
  p->impl_ = std::make_unique<Impl>();
  Impl& im = *p->impl_;
  im.opt = o;
  downscaled_size(o.capture_width, o.capture_height, o.out_width, im.ow, im.oh);
  const std::size_t bytes = frame_payload_bytes(im.ow, im.oh, o.fmt);
  if (bytes == 0) return invalid_argument(o.shm_name + ": bad capture size");
  ShmRingOptions ro;
  ro.name = o.shm_name;
  ro.kind = StreamKind::kImageFrame;
  ro.slot_size = sizeof(ShmFrameHeader) + bytes;
  ro.n_slots = o.n_slots;
  ro.stream_name = hub.stream_name();
  RIG_ASSIGN_OR_RETURN(im.ring, ShmRingWriter::create(ro));
  RIG_ASSIGN_OR_RETURN(im.sub, hub.subscribe({o.subscriber_name, DeliveryPolicy::kDropOk, std::max<std::size_t>(1, o.queue)}));
  im.hub = &hub;
  im.th = std::thread([&im] { im.run(); });
  return p;
}

void ShmFramePublisher::stop() {
  if (!impl_) return;
  impl_->stop.store(true, std::memory_order_release);
  if (impl_->th.joinable()) impl_->th.join();
}

ShmPublisherStats ShmFramePublisher::stats() const {
  ShmPublisherStats s;
  s.published = impl_->published.load();
  s.errors = impl_->errors.load();
  s.queue = impl_->sub->stats();
  return s;
}
int ShmFramePublisher::out_width() const { return impl_->ow; }
int ShmFramePublisher::out_height() const { return impl_->oh; }

// ---------------------------------------------------------------------------------------------------------------

struct ShmImuPublisher::Impl {
  std::unique_ptr<ShmRingWriter> ring;
  std::shared_ptr<StreamSubscription<ImuSample>> sub;
  std::atomic<bool> stop{false};
  std::atomic<uint64_t> published{0};
  std::thread th;

  void run() {
    std::vector<ImuSample> batch;
    while (!stop.load(std::memory_order_acquire)) {
      batch.clear();
      if (sub->pop_all(batch, std::chrono::milliseconds(50)) == 0) {
        if (sub->closed()) break;
        continue;
      }
      for (const ImuSample& s : batch) {
        const ShmImuSample m = to_shm(s);
        if (ring->write(Span<const uint8_t>(reinterpret_cast<const uint8_t*>(&m), sizeof(m)), s.t_ns).ok())
          published.fetch_add(1, std::memory_order_relaxed);
      }
    }
  }
};

ShmImuPublisher::ShmImuPublisher() = default;
ShmImuPublisher::~ShmImuPublisher() { stop(); }

Result<std::unique_ptr<ShmImuPublisher>> ShmImuPublisher::start(StreamHub<ImuSample>& hub, const std::string& name,
                                                                std::size_t n_slots, std::size_t queue,
                                                                const std::string& subscriber_name) {
  auto p = std::unique_ptr<ShmImuPublisher>(new ShmImuPublisher());
  p->impl_ = std::make_unique<Impl>();
  Impl& im = *p->impl_;
  ShmRingOptions ro;
  ro.name = name;
  ro.kind = StreamKind::kImuSample;
  ro.slot_size = sizeof(ShmImuSample);
  ro.n_slots = n_slots;
  ro.stream_name = hub.stream_name();
  RIG_ASSIGN_OR_RETURN(im.ring, ShmRingWriter::create(ro));
  RIG_ASSIGN_OR_RETURN(im.sub, hub.subscribe({subscriber_name, DeliveryPolicy::kDropOk, std::max<std::size_t>(1, queue)}));
  im.th = std::thread([&im] { im.run(); });
  return p;
}

void ShmImuPublisher::stop() {
  if (!impl_) return;
  impl_->stop.store(true, std::memory_order_release);
  if (impl_->th.joinable()) impl_->th.join();
}

ShmPublisherStats ShmImuPublisher::stats() const {
  ShmPublisherStats s;
  s.published = impl_->published.load();
  s.queue = impl_->sub->stats();
  return s;
}

}  // namespace rig
