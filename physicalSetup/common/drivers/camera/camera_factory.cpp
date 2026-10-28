// SPDX-License-Identifier: MIT
// T0 STUB of make_camera(). T2 replaces this file with the real factory (argus/gst/mock/replay).
#include "rig/drivers/camera_backend.hpp"

namespace rig {

Result<std::unique_ptr<CameraBackend>> make_camera(const CameraConfig& config) {
  return unimplemented("make_camera(" + config.id + "): camera backends not implemented yet (T2)");
}

}  // namespace rig
