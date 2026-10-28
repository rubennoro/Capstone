// SPDX-License-Identifier: MIT
// Test-area helper: compiles ec-live's pure model/renderer (an app source, not part of a library) into the
// tests/live helper library, so the tests exercise exactly the code the app runs.
#include "../../episode_collection/apps/ec-live/live_view.cpp"

#if defined(__SANITIZE_THREAD__)
#include <opencv2/core/utility.hpp>
// Same as tests/imageprep/golden_util.cpp: the system OpenCV is not instrumented, so run it single-threaded; GDAL's
// codec registration (cv::imread/imwrite) reports a lock-order inversion inside libgdal (third party).
extern "C" const char* __tsan_default_suppressions() { return "deadlock:libgdal.so\n"; }
namespace {
const bool kOpenCvSingleThreaded = [] {
  cv::setNumThreads(0);
  return true;
}();
}  // namespace
#endif

#if defined(__SANITIZE_ADDRESS__)
extern "C" const char* __lsan_default_suppressions() { return "leak:libtbb.so\n"; }
#endif
