// SPDX-License-Identifier: MIT
/// @file rig/package/csv.hpp
/// @brief RFC 4180 CSV: a buffered writer for the high-rate raw files (data.csv, frame_meta.csv, IMU data.csv) and
/// a small reader for validation/tests. T3 addition.
/// Threading: a CsvWriter is used by one thread.
#pragma once

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <memory>
#include <string>
#include <string_view>
#include <vector>

#include "rig/status.hpp"

namespace rig {

/// `s` quoted per RFC 4180 if it contains ',', '"', CR or LF (quotes doubled); otherwise unchanged.
std::string csv_quote(std::string_view s);

class CsvWriter {
 public:
  /// Creates (truncates) `path` and writes `header_line` + '\n'. Data is buffered; written when the buffer exceeds
  /// `buffer_bytes`, on flush(), sync() and close().
  static Result<std::unique_ptr<CsvWriter>> create(const std::filesystem::path& path, std::string_view header_line,
                                                   std::size_t buffer_bytes = 1 << 16);
  ~CsvWriter();  ///< close() (errors are lost: call close() explicitly to see them).
  CsvWriter(const CsvWriter&) = delete;
  CsvWriter& operator=(const CsvWriter&) = delete;

  CsvWriter& field(int64_t v);
  CsvWriter& field(uint64_t v);
  CsvWriter& field(int v) { return field(static_cast<int64_t>(v)); }
  CsvWriter& field(uint32_t v) { return field(static_cast<uint64_t>(v)); }
  /// %.{sig}g; NaN -> "nan" (numpy/pandas read it), +-inf -> "inf"/"-inf".
  CsvWriter& field(double v, int significant_digits = 17);
  CsvWriter& field(std::string_view s);  ///< quoted when needed
  CsvWriter& field(const char* s) { return field(std::string_view(s)); }
  /// Terminates the row. Returns the first error seen so far (latched).
  Status end_row();

  Status flush();
  Status sync();   ///< flush + fsync
  Status close();  ///< flush + close (no fsync); idempotent
  uint64_t rows() const { return rows_; }
  const std::filesystem::path& path() const { return path_; }

 private:
  CsvWriter() = default;
  void sep();
  std::filesystem::path path_;
  int fd_ = -1;
  std::string buf_;
  std::size_t cap_ = 0;
  bool row_open_ = false;
  uint64_t rows_ = 0;
  Status error_;
};

struct CsvTable {
  std::vector<std::string> header;
  std::vector<std::vector<std::string>> rows;
};

/// Parses RFC 4180 text (quoted fields may contain ',' '"' and newlines). kDataLoss on an unterminated quote.
Result<CsvTable> parse_csv(std::string_view text);
Result<CsvTable> read_csv(const std::filesystem::path& path);
/// First line of a file without the line terminator (kNotFound / kIoError; kDataLoss if empty).
Result<std::string> read_first_line(const std::filesystem::path& path);

}  // namespace rig
