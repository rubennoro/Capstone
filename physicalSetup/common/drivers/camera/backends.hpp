// SPDX-License-Identifier: MIT
// Private: constructors of the concrete backends, used by camera_factory.cpp.
#pragma once

#include <memory>

#include "rig/drivers/camera_backend.hpp"

namespace rig {
namespace cam {

std::unique_ptr<CameraBackend> make_mock_camera(const CameraConfig& config);
std::unique_ptr<CameraBackend> make_replay_camera(const CameraConfig& config);
#if RIG_HAVE_GST
std::unique_ptr<CameraBackend> make_gst_camera(const CameraConfig& config);  // gst/gst_camera.cpp
#endif
#if RIG_HAVE_ARGUS
std::unique_ptr<CameraBackend> make_argus_camera(const CameraConfig& config);  // argus/argus_camera.cpp
Result<std::vector<SensorMode>> argus_list_modes(int sensor_id, const std::string& sensor);
#endif

}  // namespace cam
}  // namespace rig
