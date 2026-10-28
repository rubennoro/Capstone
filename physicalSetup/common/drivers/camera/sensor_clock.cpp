// SPDX-License-Identifier: MIT
// Camera timestamp domains (rig/drivers/camera/sensor_clock.hpp; sources there).
#include "rig/drivers/camera/sensor_clock.hpp"

#include <cerrno>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <sstream>

namespace rig {

const char* to_string(SensorClockDomain d) noexcept {
  switch (d) {
    case SensorClockDomain::kUnknown: return "unknown";
    case SensorClockDomain::kMonotonicRaw: return "monotonic_raw";
    case SensorClockDomain::kMonotonic: return "monotonic";
    case SensorClockDomain::kRealtime: return "realtime";
    case SensorClockDomain::kTscOffset: return "tsc-offset_ns";
  }
  return "unknown";
}

namespace {
inline bool plausible(int64_t latency, int64_t max_latency, int64_t max_future) {
  return latency >= -max_future && latency <= max_latency;
}
inline int64_t lat(TimeNs host, uint64_t ts) { return host - static_cast<int64_t>(ts); }
}  // namespace

SensorClockDomain classify_sensor_clock(uint64_t ts, const ClockAnchor& a, int64_t max_latency,
                                        int64_t max_future) noexcept {
  if (ts == 0 || ts > static_cast<uint64_t>(INT64_MAX)) return SensorClockDomain::kUnknown;
  if (plausible(lat(a.raw_ns, ts), max_latency, max_future)) return SensorClockDomain::kMonotonicRaw;
  if (plausible(lat(a.mono_ns, ts), max_latency, max_future)) return SensorClockDomain::kMonotonic;
  if (plausible(lat(a.unix_ns, ts), max_latency, max_future)) return SensorClockDomain::kRealtime;
  return SensorClockDomain::kUnknown;
}

TimeNs sensor_clock_to_raw_ns(uint64_t ts, SensorClockDomain d, const ClockAnchor& a,
                              std::optional<int64_t> tsc_offset_ns) noexcept {
  if (ts > static_cast<uint64_t>(INT64_MAX)) return kInvalidTime;
  const auto t = static_cast<int64_t>(ts);
  switch (d) {
    case SensorClockDomain::kMonotonicRaw: return t;
    case SensorClockDomain::kMonotonic: return t + (a.raw_ns - a.mono_ns);
    case SensorClockDomain::kRealtime: return t + (a.raw_ns - a.unix_ns);
    case SensorClockDomain::kTscOffset:
      // NVIDIA (forum 268916, ShaneCCC): "offset_ns ~= (RTCPU SOF) - (system get raw time)"; raw = tsc - offset_ns.
      if (!tsc_offset_ns) return kInvalidTime;
      return t - *tsc_offset_ns;
    case SensorClockDomain::kUnknown: return kInvalidTime;
  }
  return kInvalidTime;
}

TimeNs gst_clock_time_to_raw_ns(uint64_t base_time, uint64_t running_time, const ClockAnchor& a) noexcept {
  const auto mono = static_cast<int64_t>(base_time + running_time);
  return mono + (a.raw_ns - a.mono_ns);
}

Result<int64_t> read_tsc_offset_ns(const std::string& sysroot) {
  const std::string path = sysroot + "/sys/devices/system/clocksource/clocksource0/offset_ns";
  std::ifstream f(path);
  if (!f) return not_found(path + ": not present (not a Jetson, or no TSC offset node)");
  std::string text;
  std::getline(f, text);
  errno = 0;
  char* end = nullptr;
  const long long v = std::strtoll(text.c_str(), &end, 10);
  if (errno != 0 || end == text.c_str()) return io_error(path + ": cannot parse '" + text + "'");
  return static_cast<int64_t>(v);
}

SensorClockResolver::SensorClockResolver(int frames_needed, int64_t max_latency_ns, int64_t max_future_ns)
    : frames_needed_(frames_needed < 1 ? 1 : frames_needed),
      max_latency_ns_(max_latency_ns),
      max_future_ns_(max_future_ns < 0 ? 0 : max_future_ns) {}

void SensorClockResolver::add(uint64_t sensor_ts, std::optional<uint64_t> tsc, std::optional<int64_t> offset,
                              const ClockAnchor& arrival) {
  if (decision_ || ready()) return;
  samples_.push_back(Sample{sensor_ts, tsc, offset, arrival});
}

Result<SensorClockResolver::Decision> SensorClockResolver::decide() {
  if (decision_) return *decision_;
  if (!ready()) return failed_precondition("sensor clock: need " + std::to_string(frames_needed_) + " frames, have " +
                                           std::to_string(samples_.size()));
  // 1) Classify getSensorTimestamp for every frame; require unanimity.
  SensorClockDomain d0 = classify_sensor_clock(samples_[0].sensor_ts, samples_[0].arrival, max_latency_ns_, max_future_ns_);
  bool unanimous = d0 != SensorClockDomain::kUnknown;
  bool ambiguous = false;
  for (const Sample& s : samples_) {
    if (classify_sensor_clock(s.sensor_ts, s.arrival, max_latency_ns_, max_future_ns_) != d0) unanimous = false;
    if (plausible(lat(s.arrival.raw_ns, s.sensor_ts), max_latency_ns_, max_future_ns_) &&
        plausible(lat(s.arrival.mono_ns, s.sensor_ts), max_latency_ns_, max_future_ns_))
      ambiguous = true;
  }
  // 2) Is TSC - offset_ns plausible for every frame?
  bool tsc_ok = true;
  for (const Sample& s : samples_) {
    if (!s.tsc || !s.offset) {
      tsc_ok = false;
      break;
    }
    const TimeNs raw = sensor_clock_to_raw_ns(*s.tsc, SensorClockDomain::kTscOffset, s.arrival, s.offset);
    if (raw == kInvalidTime || !plausible(s.arrival.raw_ns - raw, max_latency_ns_, max_future_ns_)) tsc_ok = false;
  }

  Decision dec;
  if (unanimous && d0 == SensorClockDomain::kMonotonicRaw) {
    dec.domain = d0;
    dec.raw_mono_ambiguous = ambiguous;
    dec.description = std::string("getSensorTimestamp in monotonic_raw") +
                      (ambiguous ? " (monotonic equally plausible: assumed raw per NVIDIA forum 268916; H verifies)" : "") +
                      (tsc_ok ? "; tsc cross-check available" : "");
  } else if (tsc_ok) {
    dec.domain = SensorClockDomain::kTscOffset;
    dec.use_tsc = true;
    dec.description = std::string("getSensorSofTimestampTsc - offset_ns (getSensorTimestamp classified as ") +
                      (unanimous ? to_string(d0) : "inconsistent") + ")";
  } else if (unanimous) {
    dec.domain = d0;
    dec.description = std::string("getSensorTimestamp in ") + to_string(d0) +
                      ", converted per frame with a clock anchor (no usable TSC); H must verify";
  } else {
    std::ostringstream os;
    os << "sensor clock: no plausible timestamp domain (arrival - ts must be in [" << -max_future_ns_ << ", "
       << max_latency_ns_
       << "] ns); first frame latencies vs raw/mono/realtime = " << lat(samples_[0].arrival.raw_ns, samples_[0].sensor_ts)
       << " / " << lat(samples_[0].arrival.mono_ns, samples_[0].sensor_ts) << " / "
       << lat(samples_[0].arrival.unix_ns, samples_[0].sensor_ts) << " ns, sensor_ts=" << samples_[0].sensor_ts;
    if (samples_[0].tsc) os << ", tsc=" << *samples_[0].tsc;
    if (samples_[0].offset) os << ", offset_ns=" << *samples_[0].offset;
    return device_error(os.str());
  }
  decision_ = dec;
  return dec;
}

TimeNs SensorClockResolver::to_raw_ns(uint64_t sensor_ts, std::optional<uint64_t> tsc, std::optional<int64_t> offset,
                                      const ClockAnchor& arrival) const noexcept {
  if (!decision_) return kInvalidTime;
  if (decision_->use_tsc) {
    if (!tsc) return kInvalidTime;
    return sensor_clock_to_raw_ns(*tsc, SensorClockDomain::kTscOffset, arrival, offset);
  }
  return sensor_clock_to_raw_ns(sensor_ts, decision_->domain, arrival, offset);
}

}  // namespace rig
