// SPDX-License-Identifier: MIT
// T0 STUB of make_slam_backend(). T6 replaces this file (MockSlam; OrbSlam3Backend behind RIG_WITH_ORBSLAM3).
#include "rig/slam/slam_backend.hpp"

namespace rig {

Result<std::unique_ptr<SlamBackend>> make_slam_backend(SlamBackendKind kind) {
  return unimplemented(std::string("make_slam_backend(") +
                       (kind == SlamBackendKind::kMock ? "mock" : "orbslam3") + "): not implemented yet (T6)");
}

}  // namespace rig
