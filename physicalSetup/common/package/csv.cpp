// SPDX-License-Identifier: MIT
// RFC 4180 CSV writer/reader (rig/package/csv.hpp).
#include "rig/package/csv.hpp"

#include <fcntl.h>
#include <unistd.h>

#include <cerrno>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <fstream>

#include "rig/package/atomic_file.hpp"

namespace rig {

std::string csv_quote(std::string_view s) {
  if (s.find_first_of(",\"\r\n") == std::string_view::npos) return std::string(s);
  std::string o = "\"";
  for (char c : s) {
    if (c == '"') o += '"';
    o += c;
  }
  return o + "\"";
}

Result<std::unique_ptr<CsvWriter>> CsvWriter::create(const std::filesystem::path& path, std::string_view header,
                                                     std::size_t buffer_bytes) {
  auto w = std::unique_ptr<CsvWriter>(new CsvWriter());
  w->path_ = path;
  w->cap_ = buffer_bytes > 0 ? buffer_bytes : 1;
  w->fd_ = ::open(path.c_str(), O_WRONLY | O_CREAT | O_TRUNC | O_CLOEXEC, 0644);
  if (w->fd_ < 0) return io_error(path.string() + ": create: " + std::strerror(errno));
  w->buf_.reserve(w->cap_ + 256);
  w->buf_.append(header.data(), header.size());
  w->buf_.push_back('\n');
  RIG_RETURN_IF_ERROR(w->flush());
  return w;
}

CsvWriter::~CsvWriter() { (void)close(); }

void CsvWriter::sep() {
  if (row_open_) buf_.push_back(',');
  row_open_ = true;
}

CsvWriter& CsvWriter::field(int64_t v) {
  sep();
  char b[24];
  const int n = std::snprintf(b, sizeof(b), "%lld", static_cast<long long>(v));
  buf_.append(b, static_cast<std::size_t>(n));
  return *this;
}

CsvWriter& CsvWriter::field(uint64_t v) {
  sep();
  char b[24];
  const int n = std::snprintf(b, sizeof(b), "%llu", static_cast<unsigned long long>(v));
  buf_.append(b, static_cast<std::size_t>(n));
  return *this;
}

CsvWriter& CsvWriter::field(double v, int sig) {
  sep();
  if (std::isnan(v)) {
    buf_ += "nan";
  } else if (std::isinf(v)) {
    buf_ += v > 0 ? "inf" : "-inf";
  } else {
    char b[40];
    const int n = std::snprintf(b, sizeof(b), "%.*g", sig < 1 ? 1 : (sig > 17 ? 17 : sig), v);
    buf_.append(b, static_cast<std::size_t>(n));
  }
  return *this;
}

CsvWriter& CsvWriter::field(std::string_view s) {
  sep();
  buf_ += csv_quote(s);
  return *this;
}

Status CsvWriter::end_row() {
  buf_.push_back('\n');
  row_open_ = false;
  ++rows_;
  if (buf_.size() >= cap_) {
    Status s = flush();
    if (!s.ok() && error_.ok()) error_ = s;
  }
  return error_;
}

Status CsvWriter::flush() {
  if (fd_ < 0) return error_.ok() ? failed_precondition(path_.string() + ": closed") : error_;
  std::size_t off = 0;
  while (off < buf_.size()) {
    const ssize_t w = ::write(fd_, buf_.data() + off, buf_.size() - off);
    if (w < 0) {
      if (errno == EINTR) continue;
      Status s = io_error(path_.string() + ": write: " + std::strerror(errno));
      if (error_.ok()) error_ = s;
      return s;
    }
    off += static_cast<std::size_t>(w);
  }
  buf_.clear();
  return error_;
}

Status CsvWriter::sync() {
  RIG_RETURN_IF_ERROR(flush());
  if (::fsync(fd_) != 0) return io_error(path_.string() + ": fsync: " + std::strerror(errno));
  return Status::Ok();
}

Status CsvWriter::close() {
  if (fd_ < 0) return error_;
  Status s = flush();
  if (::close(fd_) != 0 && s.ok()) s = io_error(path_.string() + ": close: " + std::strerror(errno));
  fd_ = -1;
  if (!s.ok() && error_.ok()) error_ = s;
  return s;
}

Result<CsvTable> parse_csv(std::string_view t) {
  std::vector<std::vector<std::string>> lines;
  std::vector<std::string> row;
  std::string field;
  bool in_quotes = false, field_started = false;
  std::size_t i = 0;
  auto end_field = [&] {
    row.push_back(std::move(field));
    field.clear();
    field_started = false;
  };
  auto end_line = [&] {
    end_field();
    lines.push_back(std::move(row));
    row.clear();
  };
  while (i < t.size()) {
    const char c = t[i];
    if (in_quotes) {
      if (c == '"') {
        if (i + 1 < t.size() && t[i + 1] == '"') {
          field += '"';
          i += 2;
          continue;
        }
        in_quotes = false;
      } else {
        field += c;
      }
      ++i;
      continue;
    }
    if (c == '"' && !field_started && field.empty()) {
      in_quotes = true;
      field_started = true;
    } else if (c == ',') {
      end_field();
    } else if (c == '\n' || c == '\r') {
      end_line();
      if (c == '\r' && i + 1 < t.size() && t[i + 1] == '\n') ++i;
    } else {
      field += c;
      field_started = true;
    }
    ++i;
  }
  if (in_quotes) return data_loss("csv: unterminated quoted field");
  if (field_started || !field.empty() || !row.empty()) end_line();
  CsvTable tab;
  if (lines.empty()) return data_loss("csv: empty");
  tab.header = std::move(lines.front());
  tab.rows.assign(std::make_move_iterator(lines.begin() + 1), std::make_move_iterator(lines.end()));
  return tab;
}

Result<CsvTable> read_csv(const std::filesystem::path& path) {
  RIG_ASSIGN_OR_RETURN(std::string text, read_file(path));
  auto t = parse_csv(text);
  if (!t.ok()) return t.status().with_context(path.string());
  return t;
}

Result<std::string> read_first_line(const std::filesystem::path& path) {
  std::ifstream in(path, std::ios::binary);
  if (!in) {
    std::error_code ec;
    if (!std::filesystem::exists(path, ec)) return not_found(path.string() + ": no such file");
    return io_error(path.string() + ": cannot open");
  }
  std::string line;
  if (!std::getline(in, line)) return data_loss(path.string() + ": empty file");
  if (!line.empty() && line.back() == '\r') line.pop_back();
  return line;
}

}  // namespace rig
