// SPDX-License-Identifier: MIT
// Crash-safe writes (rig/package/atomic_file.hpp). Standard POSIX recipe: write a temp file in the same directory,
// fsync it, rename(2) over the target (atomic replacement, POSIX rename()), fsync the directory so the rename itself
// is durable (ext4/xfs require it; fsync(2) NOTES).
#include "rig/package/atomic_file.hpp"

#include <fcntl.h>
#include <unistd.h>

#include <atomic>
#include <cerrno>
#include <cstring>
#include <fstream>
#include <regex>
#include <sstream>

namespace rig {
namespace {

std::atomic<uint64_t> g_counter{0};

Status write_all(int fd, const char* p, std::size_t n, const std::string& what) {
  while (n > 0) {
    const ssize_t w = ::write(fd, p, n);
    if (w < 0) {
      if (errno == EINTR) continue;
      return io_error(what + ": write: " + std::strerror(errno));
    }
    p += w;
    n -= static_cast<std::size_t>(w);
  }
  return Status::Ok();
}

bool step(const AtomicWriteHook& hook, AtomicWriteStep s) { return !hook || hook(s); }

}  // namespace

Status fsync_path(const std::filesystem::path& path) {
  const int fd = ::open(path.c_str(), O_RDONLY | O_CLOEXEC);
  if (fd < 0) return io_error(path.string() + ": open for fsync: " + std::strerror(errno));
  const int r = ::fsync(fd);
  const int e = errno;
  ::close(fd);
  if (r != 0) return io_error(path.string() + ": fsync: " + std::strerror(e));
  return Status::Ok();
}

bool is_atomic_temp_name(const std::string& filename) {
  static const std::regex r("\\..+\\.tmp\\.[0-9]+\\.[0-9]+");
  return std::regex_match(filename, r);
}

Status write_file_atomic(const std::filesystem::path& path, std::string_view content, const AtomicWriteHook& hook) {
  const std::filesystem::path dir = path.has_parent_path() ? path.parent_path() : std::filesystem::path(".");
  const std::filesystem::path tmp =
      dir / ("." + path.filename().string() + ".tmp." + std::to_string(::getpid()) + "." +
             std::to_string(g_counter.fetch_add(1)));
  const std::string what = path.string();
  if (!step(hook, AtomicWriteStep::kCreateTemp)) return cancelled(what + ": simulated crash");
  const int fd = ::open(tmp.c_str(), O_WRONLY | O_CREAT | O_EXCL | O_CLOEXEC, 0644);
  if (fd < 0) return io_error(tmp.string() + ": create: " + std::strerror(errno));
  const std::size_t half = content.size() / 2;
  Status s = write_all(fd, content.data(), half, what);
  if (s.ok() && !step(hook, AtomicWriteStep::kWriteHalf)) {
    ::close(fd);
    return cancelled(what + ": simulated crash");
  }
  if (s.ok()) s = write_all(fd, content.data() + half, content.size() - half, what);
  if (s.ok() && !step(hook, AtomicWriteStep::kFsyncTemp)) {
    ::close(fd);
    return cancelled(what + ": simulated crash");
  }
  if (s.ok() && ::fsync(fd) != 0) s = io_error(tmp.string() + ": fsync: " + std::strerror(errno));
  if (::close(fd) != 0 && s.ok()) s = io_error(tmp.string() + ": close: " + std::strerror(errno));
  if (!s.ok()) {
    ::unlink(tmp.c_str());
    return s;
  }
  if (!step(hook, AtomicWriteStep::kRename)) return cancelled(what + ": simulated crash");
  if (::rename(tmp.c_str(), path.c_str()) != 0) {
    const int e = errno;
    ::unlink(tmp.c_str());
    return io_error(what + ": rename: " + std::strerror(e));
  }
  if (!step(hook, AtomicWriteStep::kFsyncDir)) return cancelled(what + ": simulated crash");
  return fsync_path(dir);
}

Result<std::string> read_file(const std::filesystem::path& path) {
  std::ifstream in(path, std::ios::binary);
  if (!in) {
    std::error_code ec;
    if (!std::filesystem::exists(path, ec)) return not_found(path.string() + ": no such file");
    return io_error(path.string() + ": cannot open");
  }
  std::ostringstream ss;
  ss << in.rdbuf();
  if (in.bad()) return io_error(path.string() + ": read error");
  return ss.str();
}

}  // namespace rig
