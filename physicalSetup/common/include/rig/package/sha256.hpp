// SPDX-License-Identifier: MIT
/// @file rig/package/sha256.hpp
/// @brief SHA-256 (FIPS 180-4) for content hashes: rig.yaml freeze (config::content_hash), manifest
/// StageRecord::inputs_hash. T3 addition (own module; no external crypto library on the laptop/Jetson build).
/// Threading: Sha256 objects are not shared; the free functions are thread-safe.
#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <string>
#include <string_view>

#include "rig/span.hpp"
#include "rig/status.hpp"

namespace rig {

/// Incremental SHA-256.
class Sha256 {
 public:
  Sha256() noexcept;
  void update(const void* data, std::size_t n) noexcept;
  void update(std::string_view s) noexcept { update(s.data(), s.size()); }
  /// Finishes and returns the 32-byte digest. The object must not be updated afterwards (call reset()).
  std::array<uint8_t, 32> finish() noexcept;
  void reset() noexcept;

 private:
  void block(const uint8_t* p) noexcept;
  std::array<uint32_t, 8> h_{};
  std::array<uint8_t, 64> buf_{};
  std::size_t buf_len_ = 0;
  uint64_t total_bytes_ = 0;
};

/// Lowercase hex of `n` bytes.
std::string to_hex(const uint8_t* data, std::size_t n);
/// Lowercase hex SHA-256 of a byte string.
std::string sha256_hex(std::string_view bytes);
/// Lowercase hex SHA-256 of a file's bytes. kNotFound / kIoError on failure.
Result<std::string> sha256_file_hex(const std::filesystem::path& path);

}  // namespace rig
