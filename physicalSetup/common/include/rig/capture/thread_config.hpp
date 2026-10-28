// SPDX-License-Identifier: MIT
/// @file rig/capture/thread_config.hpp
/// @brief Applies a ThreadConfig (CPU affinity, SCHED_FIFO priority or nice value, thread name) to the calling
/// thread. Used by CaptureService's capture threads; also for T5 writer threads and T6's SLAM thread. T3 addition.
///
/// Best effort: every part is attempted; the returned Status lists what failed (e.g. SCHED_FIFO without
/// CAP_SYS_NICE / RLIMIT_RTPRIO on the laptop, or a CPU id that does not exist). Callers log it and continue
/// (PLAN T3: never fatal). Threading: affects only the calling thread.
#pragma once

#include <string>

#include "rig/config/rig_config.hpp"
#include "rig/status.hpp"

namespace rig {

/// @param thread_name set with pthread_setname_np (truncated to 15 chars); empty = unchanged.
Status apply_thread_config(const ThreadConfig& config, const std::string& thread_name);

}  // namespace rig
