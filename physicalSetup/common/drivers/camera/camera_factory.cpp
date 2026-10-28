// SPDX-License-Identifier: MIT
// make_camera(): backend selection and config sanity checks (rig/drivers/camera_backend.hpp). Owned by T2.
#include <string>

#include "backends.hpp"
#include "rig/drivers/camera/imx519_focus.hpp"
#include "rig/drivers/camera/sensor_modes.hpp"
#include "rig/drivers/camera_backend.hpp"

namespace rig {

Result<std::unique_ptr<CameraBackend>> make_camera(const CameraConfig& config) {
  const std::string ctx = "cameras." + config.id + ".driver";
  if (config.id.empty()) return invalid_argument("make_camera: CameraConfig::id is empty");
  if (!config.sensor.empty() && !is_known_sensor(config.sensor))
    return invalid_argument(ctx + ".sensor: unknown '" + config.sensor + "' (imx519 | imx219)");
  if (config.backend != CameraBackendKind::kReplay) {
    if (config.width <= 0 || config.height <= 0) return invalid_argument(ctx + ": width and height must be > 0");
    if (!(config.fps > 0.0)) return invalid_argument(ctx + ".fps: must be > 0");
  }
  if (!(config.drop_gap_factor > 1.0)) return invalid_argument(ctx + ".drop_gap_factor: must be > 1");
  if (config.buffer_count < 1) return invalid_argument(ctx + ".buffer_count: must be >= 1");
  if (Status s = Imx519Focus::validate(config.focus); !s.ok()) return s.with_context(ctx + ".focus");

  switch (config.backend) {
    case CameraBackendKind::kMock: return cam::make_mock_camera(config);
    case CameraBackendKind::kReplay:
      if (config.replay.path.empty()) return invalid_argument(ctx + ".replay.path: required for backend replay");
      return cam::make_replay_camera(config);
    case CameraBackendKind::kGst:
#if RIG_HAVE_GST
      return cam::make_gst_camera(config);
#else
      return unimplemented(ctx + ".backend: gst not compiled in (configure with -DRIG_WITH_GST=ON)");
#endif
    case CameraBackendKind::kArgus:
#if RIG_HAVE_ARGUS
      return cam::make_argus_camera(config);
#else
      return unimplemented(ctx + ".backend: argus not compiled in (configure with -DRIG_WITH_ARGUS=ON on the Jetson)");
#endif
  }
  return invalid_argument(ctx + ".backend: unknown value");
}

}  // namespace rig
