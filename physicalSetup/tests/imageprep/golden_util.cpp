// SPDX-License-Identifier: MIT
#include "golden_util.hpp"

#include <cstdio>
#include <cstdlib>
#include <sstream>

#if defined(__SANITIZE_THREAD__)
#include <opencv2/core/utility.hpp>

// TSan builds only (check.sh --sanitize thread):
// - The system OpenCV is not instrumented. Its parallel_for_ pool synchronises with code TSan cannot see, which
//   shows up as false data races on the images it writes; run OpenCV single-threaded so that no suppression of
//   races is needed and our own code is still fully checked.
// - cv::imread loads GDAL (Ubuntu's OpenCV links libgdal for its GDAL codec); GDAL's internal CPL mutexes report a
//   lock-order inversion inside libgdal during codec registration. That is third-party code; suppress only it.
extern "C" const char* __tsan_default_suppressions() { return "deadlock:libgdal.so\n"; }
namespace {
const bool kOpenCvSingleThreaded = [] {
  cv::setNumThreads(0);
  return true;
}();
}  // namespace
#endif

#if defined(__SANITIZE_ADDRESS__)
// ASan/LSan builds only: the system OpenCV's TBB thread pool (libtbb.so.2) intermittently leaves ~12 KB of worker
// allocations at process exit (seen in the F16 ASan run, stack entirely inside libtbb). Third-party; suppress only it.
extern "C" const char* __lsan_default_suppressions() { return "leak:libtbb.so\n"; }
#endif

namespace rig::test {

std::string fnv1a64_hex(const cv::Mat& m) {
  const cv::Mat c = m.isContinuous() ? m : m.clone();
  uint64_t h = 0xCBF29CE484222325ull;
  const auto* p = c.ptr<uint8_t>(0);
  const size_t n = c.total() * c.elemSize();
  for (size_t i = 0; i < n; ++i) {
    h ^= p[i];
    h *= 0x100000001B3ull;
  }
  char buf[17];
  std::snprintf(buf, sizeof(buf), "%016llx", static_cast<unsigned long long>(h));
  return buf;
}

cv::Mat synth_image(int width, int height) {
  cv::Mat m(height, width, CV_8UC3);
  for (int y = 0; y < height; ++y) {
    auto* row = m.ptr<uint8_t>(y);
    for (int x = 0; x < width; ++x) {
      const int64_t xx = x, yy = y;
      for (int c = 0; c < 3; ++c) row[3 * x + c] = static_cast<uint8_t>((xx * 7 + yy * 13 + c * 85 + ((xx * yy) >> 5)) & 255);
    }
  }
  return m;
}

std::string imageprep_data_dir() { return std::string(RIG_SOURCE_DIR) + "/tests/imageprep/data"; }

std::string extended_dir() {
  const char* e = std::getenv("RIG_T4_EXTENDED_DIR");
  return e ? std::string(e) : std::string();
}

CameraModelParams parse_model_line(const std::string& line) {
  std::istringstream is(line);
  CameraModelParams p;
  int kind = 0, n = 0;
  is >> kind >> p.width >> p.height >> p.fx >> p.fy >> p.cx >> p.cy >> p.fov_max_deg >> n;
  p.kind = static_cast<CameraModelKind>(kind);
  p.distortion.resize(static_cast<size_t>(n));
  for (auto& d : p.distortion) is >> d;
  return p;
}

CameraModelParams iphone_uw_virtual() {
  CameraModelParams p;
  p.kind = CameraModelKind::kVirtualPinhole;
  p.width = 640;
  p.height = 480;
  p.fx = p.fy = 246.64919;
  p.cx = 320.16495;
  p.cy = 239.53679;
  return p;
}

std::vector<std::string> split(const std::string& s, const std::string& sep) {
  std::vector<std::string> out;
  size_t pos = 0;
  while (true) {
    const size_t q = s.find(sep, pos);
    out.push_back(s.substr(pos, q == std::string::npos ? std::string::npos : q - pos));
    if (q == std::string::npos) break;
    pos = q + sep.size();
  }
  return out;
}

}  // namespace rig::test
