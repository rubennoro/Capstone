// SPDX-License-Identifier: MIT
/// @file rig/package/atomic_file.hpp
/// @brief Crash-safe file writes (temp + fsync + rename + fsync(dir)) and fsync helpers. Used for manifest.json and
/// session.json; also for T5's live_checks.json / clock_fit.json and T7's gripper_range.json. T3 addition.
///
/// Guarantee: after a crash at any point, `path` holds either its complete old content (or does not exist, if it
/// never existed) or the complete new content. A crash may leave a hidden temp file `.<name>.tmp.<pid>.<n>` next to
/// it; is_atomic_temp_name() identifies such leftovers (validate_layout ignores them).
/// Threading: thread-safe (distinct temp names per call).
#pragma once

#include <filesystem>
#include <functional>
#include <string>
#include <string_view>

#include "rig/status.hpp"

namespace rig {

/// Steps of write_file_atomic, in order (for crash-injection tests).
enum class AtomicWriteStep {
  kCreateTemp = 0,  ///< before open(O_CREAT|O_EXCL) of the temp file
  kWriteHalf = 1,   ///< after writing the first half of the content
  kFsyncTemp = 2,   ///< after writing all content, before fsync(temp)
  kRename = 3,      ///< before rename(temp, path)
  kFsyncDir = 4,    ///< after rename, before fsync(dir)
};

/// Test hook: called before each step; returning false stops right there without cleanup, exactly like a crash
/// (the function then returns kCancelled). Production code passes nullptr.
using AtomicWriteHook = std::function<bool(AtomicWriteStep)>;

Status write_file_atomic(const std::filesystem::path& path, std::string_view content,
                         const AtomicWriteHook& hook = nullptr);

/// fsync of a file or directory by path (kIoError with errno text).
Status fsync_path(const std::filesystem::path& path);

/// True for names produced as write_file_atomic temp files.
bool is_atomic_temp_name(const std::string& filename);

/// Reads a whole file (kNotFound / kIoError).
Result<std::string> read_file(const std::filesystem::path& path);

}  // namespace rig
