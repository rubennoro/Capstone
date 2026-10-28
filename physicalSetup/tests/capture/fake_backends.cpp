// SPDX-License-Identifier: MIT
#include "fake_backends.hpp"

#include <sched.h>
#include <sys/resource.h>
#include <sys/syscall.h>
#include <unistd.h>

#include <cerrno>
#include <cstring>
#include <thread>

#include "rig/time/clock.hpp"

namespace rig {
namespace testing {
namespace {
void observe(std::vector<int>& cpus, int& nice) {
  cpu_set_t set;
  CPU_ZERO(&set);
  cpus.clear();
  if (sched_getaffinity(0, sizeof(set), &set) == 0)
    for (int i = 0; i < CPU_SETSIZE; ++i)
      if (CPU_ISSET(i, &set)) cpus.push_back(i);
  errno = 0;
  nice = getpriority(PRIO_PROCESS, static_cast<id_t>(::syscall(SYS_gettid)));
}
}  // namespace

FakeCamera::FakeCamera(FakeCameraOptions o) : o_(std::move(o)) {
  cfg_.id = o_.id;
  cfg_.backend = CameraBackendKind::kMock;
  cfg_.fps = o_.fps;
  cfg_.width = o_.width;
  cfg_.height = o_.height;
  cfg_.pixel_format = PixelFormat::kGRAY8;
}

Status FakeCamera::start() {
  start_calls++;
  observe(start_affinity, start_nice);
  std::thread([this] { observe(child_affinity, child_nice); }).join();
  if (!o_.start_error.ok()) return o_.start_error;
  std::lock_guard<std::mutex> lk(mu_);
  if (running_) return failed_precondition("already started");
  running_ = true;
  k_ = 0;
  t0_real_ = now_raw_ns();
  t0_virtual_ = t0_real_;
  return Status::Ok();
}

Status FakeCamera::stop() {
  stop_calls++;
  {
    std::lock_guard<std::mutex> lk(mu_);
    running_ = false;
  }
  cv_.notify_all();
  return Status::Ok();
}

std::optional<ImageFrame> FakeCamera::read(std::chrono::nanoseconds timeout) {
  std::unique_lock<std::mutex> lk(mu_);
  const TimeNs deadline = now_raw_ns() + timeout.count();
  while (true) {
    if (!running_ || !status_.ok()) return std::nullopt;
    if (o_.stall_at >= 0 && k_ >= o_.stall_at) {
      cv_.wait_for(lk, std::chrono::nanoseconds(std::max<TimeNs>(0, deadline - now_raw_ns())));
      return std::nullopt;
    }
    const TimeNs due = t0_real_ + static_cast<TimeNs>(static_cast<double>(k_) * 1e9 / (o_.fps * o_.accel));
    const TimeNs now = now_raw_ns();
    if (now >= due) break;
    if (now >= deadline) return std::nullopt;
    cv_.wait_for(lk, std::chrono::nanoseconds(std::min(due, deadline) - now));
  }
  if (o_.drop_every > 0 && k_ > 0 && k_ % o_.drop_every == 0) {
    ++k_;
    stats_.drops_seq_gap++;
  }
  ImageFrame f;
  f.camera_id = o_.id;
  f.seq = static_cast<uint64_t>(k_);
  f.t_sof_ns = t0_virtual_ + static_cast<TimeNs>(static_cast<double>(k_) * 1e9 / o_.fps);
  f.t_mid_ns = f.t_sof_ns + 1000000;
  f.t_arrival_ns = now_raw_ns();
  f.width = o_.width;
  f.height = o_.height;
  f.fmt = PixelFormat::kGRAY8;
  const std::size_t n = static_cast<std::size_t>(o_.width) * o_.height;
  auto buf = std::shared_ptr<uint8_t[]>(new uint8_t[n]);
  std::memset(buf.get(), static_cast<int>(k_ & 0xFF), n);
  f.data = buf;
  f.size = n;
  f.planes[0] = Plane{0, static_cast<std::size_t>(o_.width), o_.width, o_.height};
  ++k_;
  stats_.frames++;
  stats_.last_t_sof_ns = f.t_sof_ns;
  if (o_.fatal_after >= 0 && static_cast<int64_t>(stats_.frames) >= o_.fatal_after)
    status_ = device_error(o_.id + ": injected fatal error");
  latest_ = f;
  return f;
}

std::optional<ImageFrame> FakeCamera::latest() const {
  std::lock_guard<std::mutex> lk(mu_);
  return latest_;
}
CameraStats FakeCamera::stats() const {
  std::lock_guard<std::mutex> lk(mu_);
  return stats_;
}
Status FakeCamera::status() const {
  std::lock_guard<std::mutex> lk(mu_);
  return status_;
}

FakeImu::FakeImu(FakeImuOptions o) : o_(std::move(o)) {
  cfg_.id = o_.id;
  cfg_.backend = ImuBackendKind::kSim;
  cfg_.odr_hz = o_.odr_hz;
}

Status FakeImu::start() {
  start_calls++;
  if (!o_.start_error.ok()) return o_.start_error;
  std::lock_guard<std::mutex> lk(mu_);
  running_ = true;
  k_ = 0;
  next_bracket_ = 0;
  t0_real_ = now_raw_ns();
  t0_virtual_ = t0_real_;
  return Status::Ok();
}

Status FakeImu::stop() {
  stop_calls++;
  {
    std::lock_guard<std::mutex> lk(mu_);
    running_ = false;
  }
  cv_.notify_all();
  return Status::Ok();
}

std::size_t FakeImu::read(std::vector<ImuSample>& out, std::chrono::nanoseconds timeout) {
  std::unique_lock<std::mutex> lk(mu_);
  const TimeNs deadline = now_raw_ns() + timeout.count();
  const double period_real = 1e9 / (o_.odr_hz * o_.accel);
  while (true) {
    if (!running_ || !status_.ok()) return 0;
    if (o_.max_samples >= 0 && k_ >= o_.max_samples) {
      cv_.wait_for(lk, std::chrono::nanoseconds(std::max<TimeNs>(0, deadline - now_raw_ns())));
      return 0;
    }
    const TimeNs due = t0_real_ + static_cast<TimeNs>(static_cast<double>(k_) * period_real);
    const TimeNs now = now_raw_ns();
    if (now >= due) break;
    if (now >= deadline) return 0;
    cv_.wait_for(lk, std::chrono::nanoseconds(std::min(due, deadline) - now));
  }
  // Every sample whose real due time has passed (a batch, like a FIFO drain).
  const TimeNs now = now_raw_ns();
  std::size_t n = 0;
  while (t0_real_ + static_cast<TimeNs>(static_cast<double>(k_) * period_real) <= now &&
         (o_.max_samples < 0 || k_ < o_.max_samples)) {
    ImuSample s;
    s.t_ns = t0_virtual_ + static_cast<TimeNs>(static_cast<double>(k_) * 1e9 / o_.odr_hz);
    s.chip_ticks = k_ * 48;
    s.acc_mps2 = Eigen::Vector3d(0, 0, kStandardGravity);
    out.push_back(s);
    ++k_;
    ++n;
    stats_.samples++;
    stats_.last_t_ns = s.t_ns;
    if (o_.fatal_after >= 0 && static_cast<int64_t>(stats_.samples) >= o_.fatal_after) {
      status_ = device_error(o_.id + ": injected fatal error");
      break;
    }
  }
  return n;
}

std::size_t FakeImu::read_brackets(std::vector<ClockBracket>& out) {
  std::lock_guard<std::mutex> lk(mu_);
  const TimeNs virt_now = t0_virtual_ + static_cast<TimeNs>(static_cast<double>(k_) * 1e9 / o_.odr_hz);
  std::size_t n = 0;
  const TimeNs period = static_cast<TimeNs>(o_.bracket_period_ms) * 1000000;
  while (t0_virtual_ + next_bracket_ * period < virt_now) {
    const TimeNs t = t0_virtual_ + next_bracket_ * period;
    out.push_back(ClockBracket{t, next_bracket_, t + 1000});
    ++next_bracket_;
    ++n;
  }
  return n;
}

ImuStats FakeImu::stats() const {
  std::lock_guard<std::mutex> lk(mu_);
  return stats_;
}
Status FakeImu::status() const {
  std::lock_guard<std::mutex> lk(mu_);
  return status_;
}

}  // namespace testing
}  // namespace rig
