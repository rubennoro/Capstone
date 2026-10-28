// SPDX-License-Identifier: MIT
// POSIX shm seqlock ring (rig/shm/ring.hpp).
//
// Seqlock (frozen protocol in ring.hpp, implemented in its fence-correct form; Boehm, "Can seqlocks get along with
// programming language memory models?", MSPC 2012):
//   writer m:  seq.store(2m+1, relaxed); fence(release); payload + slot fields; seq.store(2m+2, release);
//              write_seq.store(m+1, release)
//   reader m:  s1 = seq.load(acquire); [s1 == 2m+2 required]; copy payload + slot fields; fence(acquire);
//              s2 = seq.load(relaxed); s1 == s2 -> consistent.
// The release fence after the odd store keeps payload stores from becoming visible before it, so a reader that
// copied any byte of a newer message necessarily sees s2 != s1. The reader's copy reads memory the writer may be
// changing concurrently (the point of a seqlock): it is done with relaxed atomic word loads in a function that
// ThreadSanitizer does not instrument (the race is benign by construction and detected by the seq re-check).
//
// Writer liveness: the writer holds an open-file-description lock (F_OFD_SETLK, Linux >= 3.15) on byte 0 of the
// segment for its whole life. The kernel drops it when the writer process dies, however it dies, so there is no
// pid-reuse ambiguity. Readers and would-be writers probe it with F_OFD_GETLK (which never acquires). A reader also
// compares the inode behind the name with its own mapping to detect a re-created segment.
#include "rig/shm/ring.hpp"

#include <errno.h>
#include <fcntl.h>
#include <signal.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <thread>

#include "rig/time/clock.hpp"

namespace rig {
namespace {

constexpr std::size_t kSlotAlign = 64;
constexpr std::size_t kHeaderBytes = sizeof(ShmRingHeader);

std::size_t slot_stride_for(std::size_t slot_size) {
  return (sizeof(ShmSlotHeader) + slot_size + kSlotAlign - 1) / kSlotAlign * kSlotAlign;
}

std::string errno_text(int e) { return std::strerror(e); }

Status validate_name(const std::string& name) {
  if (name.size() < 2 || name[0] != '/' || name.find('/', 1) != std::string::npos || name.size() > 200)
    return invalid_argument("shm name '" + name + "': must be '/' + 1..199 characters without another '/'");
  return Status::Ok();
}

/// F_OFD_GETLK probe for a write lock on byte 0: true if another open file description holds a lock.
bool lock_held_by_other(int fd) {
  struct flock fl {};
  fl.l_type = F_WRLCK;
  fl.l_whence = SEEK_SET;
  fl.l_start = 0;
  fl.l_len = 1;
  fl.l_pid = 0;
  if (fcntl(fd, F_OFD_GETLK, &fl) != 0) return true;  // cannot tell: assume alive (never unlink a live segment)
  return fl.l_type != F_UNLCK;
}

bool take_lock(int fd) {
  struct flock fl {};
  fl.l_type = F_WRLCK;
  fl.l_whence = SEEK_SET;
  fl.l_start = 0;
  fl.l_len = 1;
  fl.l_pid = 0;
  return fcntl(fd, F_OFD_SETLK, &fl) == 0;
}

bool same_inode(int fd, const std::string& name) {
  struct stat a {}, b {};
  if (fstat(fd, &a) != 0) return false;
  const int fd2 = shm_open(name.c_str(), O_RDONLY | O_CLOEXEC, 0);
  if (fd2 < 0) return false;
  const bool ok = fstat(fd2, &b) == 0 && a.st_ino == b.st_ino && a.st_dev == b.st_dev;
  close(fd2);
  return ok;
}

/// Copies `n` bytes of a slot payload that may be concurrently overwritten (see file comment). `src` is 64-byte
/// aligned (slot layout); `dst` has no alignment requirement.
#if defined(__SANITIZE_THREAD__)
__attribute__((no_sanitize_thread))
#endif
void racy_copy(uint8_t* dst, const uint8_t* src, std::size_t n) {
  std::size_t i = 0;
  for (; i + 8 <= n; i += 8) {
    const uint64_t w = __atomic_load_n(reinterpret_cast<const uint64_t*>(src + i), __ATOMIC_RELAXED);
    std::memcpy(dst + i, &w, 8);
  }
  for (; i < n; ++i) dst[i] = __atomic_load_n(src + i, __ATOMIC_RELAXED);
}

#if defined(__SANITIZE_THREAD__)
__attribute__((no_sanitize_thread))
#endif
void racy_slot_fields(const ShmSlotHeader* s, uint64_t& bytes, int64_t& t_ns) {
  bytes = __atomic_load_n(&s->payload_bytes, __ATOMIC_RELAXED);
  t_ns = __atomic_load_n(&s->t_ns, __ATOMIC_RELAXED);
}

}  // namespace

// ---------------------------------------------------------------------------------------------------------------
// Writer
// ---------------------------------------------------------------------------------------------------------------

struct ShmRingWriter::Impl {
  std::string name;
  int fd = -1;
  void* map = MAP_FAILED;
  std::size_t map_bytes = 0;
  ShmRingHeader* hdr = nullptr;
  std::size_t slot_size = 0, n_slots = 0, stride = 0;
  uint64_t next = 0;  // next message number
  bool in_write = false;

  ShmSlotHeader* slot(uint64_t m) const {
    return reinterpret_cast<ShmSlotHeader*>(reinterpret_cast<uint8_t*>(hdr) + kHeaderBytes + (m % n_slots) * stride);
  }
  uint8_t* payload(uint64_t m) const { return reinterpret_cast<uint8_t*>(slot(m)) + sizeof(ShmSlotHeader); }

  ~Impl() {
    if (map != MAP_FAILED) munmap(map, map_bytes);
    if (fd >= 0) {
      // Unlink only our own segment (never one that replaced it under the same name).
      if (same_inode(fd, name)) shm_unlink(name.c_str());
      close(fd);  // releases the OFD lock
    }
  }
};

ShmRingWriter::ShmRingWriter() = default;
ShmRingWriter::~ShmRingWriter() = default;

Result<std::unique_ptr<ShmRingWriter>> ShmRingWriter::create(const ShmRingOptions& o) {
  RIG_RETURN_IF_ERROR(validate_name(o.name));
  if (o.slot_size == 0) return invalid_argument(o.name + ": slot_size must be > 0");
  if (o.n_slots < 2) return invalid_argument(o.name + ": n_slots must be >= 2");
  const std::size_t stride = slot_stride_for(o.slot_size);
  if (o.n_slots > (SIZE_MAX - kHeaderBytes) / stride) return invalid_argument(o.name + ": ring too large");
  const std::size_t total = kHeaderBytes + o.n_slots * stride;

  auto w = std::unique_ptr<ShmRingWriter>(new ShmRingWriter());
  w->impl_ = std::make_unique<Impl>();
  Impl& im = *w->impl_;
  im.name = o.name;

  for (int attempt = 0; attempt < 4; ++attempt) {
    const int fd = shm_open(o.name.c_str(), O_CREAT | O_EXCL | O_RDWR | O_CLOEXEC, 0644);
    if (fd >= 0) {
      if (!take_lock(fd)) {  // a recovering writer grabbed the fresh segment first: we lost the race
        close(fd);
        return already_exists(o.name + ": another writer is creating this segment");
      }
      im.fd = fd;
      break;
    }
    if (errno != EEXIST) return io_error(o.name + ": shm_open: " + errno_text(errno));
    // Exists: live writer or stale leftover?
    const int old = shm_open(o.name.c_str(), O_RDWR | O_CLOEXEC, 0);
    if (old < 0) {
      if (errno == ENOENT) continue;  // vanished meanwhile: retry the create
      return io_error(o.name + ": shm_open existing: " + errno_text(errno));
    }
    if (lock_held_by_other(old) || !take_lock(old)) {
      close(old);
      return already_exists(o.name + ": a live writer owns this shm segment");
    }
    // We hold the lock on the stale inode, so no other recovering writer can act on it concurrently.
    if (same_inode(old, o.name)) shm_unlink(o.name.c_str());
    close(old);
  }
  if (im.fd < 0) return unavailable(o.name + ": could not create the segment (concurrent create/unlink)");

  if (ftruncate(im.fd, static_cast<off_t>(total)) != 0)
    return io_error(o.name + ": ftruncate(" + std::to_string(total) + "): " + errno_text(errno));
  im.map = mmap(nullptr, total, PROT_READ | PROT_WRITE, MAP_SHARED, im.fd, 0);
  if (im.map == MAP_FAILED) return io_error(o.name + ": mmap: " + errno_text(errno));
  im.map_bytes = total;
  im.hdr = static_cast<ShmRingHeader*>(im.map);
  im.slot_size = o.slot_size;
  im.n_slots = o.n_slots;
  im.stride = stride;
  // ftruncate zero-fills: every slot seq is 0 ("never written") and write_seq is 0.
  ShmRingHeader* h = im.hdr;
  h->version = kShmRingVersion;
  h->stream_kind = static_cast<uint32_t>(o.kind);
  h->slot_size = o.slot_size;
  h->slot_stride = stride;
  h->n_slots = o.n_slots;
  h->type_tag = o.type_tag;
  h->writer_pid = static_cast<uint32_t>(getpid());
  h->created_raw_ns = now_raw_ns();
  std::strncpy(h->stream_name, o.stream_name.c_str(), sizeof(h->stream_name) - 1);
  h->write_seq.store(0, std::memory_order_relaxed);
  __atomic_store_n(&h->magic, kShmRingMagic, __ATOMIC_RELEASE);  // last (ring.hpp)
  return w;
}

Span<uint8_t> ShmRingWriter::begin_write() {
  Impl& im = *impl_;
  const uint64_t m = im.next;
  if (!im.in_write) {
    im.slot(m)->seq.store(2 * m + 1, std::memory_order_relaxed);
    std::atomic_thread_fence(std::memory_order_release);
    im.in_write = true;
  }
  return Span<uint8_t>(im.payload(m), im.slot_size);
}

Result<uint64_t> ShmRingWriter::commit(std::size_t bytes, TimeNs t_ns) {
  Impl& im = *impl_;
  if (!im.in_write) return failed_precondition(im.name + ": commit() without begin_write()");
  if (bytes > im.slot_size) {
    // Leave the slot odd (torn) until the next successful write: readers treat it as overwritten, never as data.
    return invalid_argument(im.name + ": commit of " + std::to_string(bytes) + " bytes > slot_size " +
                            std::to_string(im.slot_size));
  }
  const uint64_t m = im.next;
  ShmSlotHeader* s = im.slot(m);
  __atomic_store_n(&s->payload_bytes, static_cast<uint64_t>(bytes), __ATOMIC_RELAXED);
  __atomic_store_n(&s->t_ns, t_ns, __ATOMIC_RELAXED);
  s->seq.store(2 * m + 2, std::memory_order_release);
  im.hdr->write_seq.store(m + 1, std::memory_order_release);
  im.next = m + 1;
  im.in_write = false;
  return m;
}

Result<uint64_t> ShmRingWriter::write(Span<const uint8_t> payload, TimeNs t_ns) {
  if (payload.size() > impl_->slot_size)
    return invalid_argument(impl_->name + ": message of " + std::to_string(payload.size()) + " bytes > slot_size " +
                            std::to_string(impl_->slot_size));
  Span<uint8_t> dst = begin_write();
  if (!payload.empty()) std::memcpy(dst.data(), payload.data(), payload.size());
  return commit(payload.size(), t_ns);
}

uint64_t ShmRingWriter::write_seq() const { return impl_->next; }
const std::string& ShmRingWriter::name() const { return impl_->name; }
std::size_t ShmRingWriter::slot_size() const { return impl_->slot_size; }
std::size_t ShmRingWriter::n_slots() const { return impl_->n_slots; }

// ---------------------------------------------------------------------------------------------------------------
// Reader
// ---------------------------------------------------------------------------------------------------------------

struct ShmRingReader::Impl {
  std::string name;
  int fd = -1;
  void* map = MAP_FAILED;
  std::size_t map_bytes = 0;
  const ShmRingHeader* hdr = nullptr;
  std::size_t slot_size = 0, n_slots = 0, stride = 0;
  uint64_t next = 0;
  uint64_t overruns = 0, missed_total = 0, missed_pending = 0;

  const ShmSlotHeader* slot(uint64_t m) const {
    return reinterpret_cast<const ShmSlotHeader*>(reinterpret_cast<const uint8_t*>(hdr) + kHeaderBytes +
                                                  (m % n_slots) * stride);
  }
  const uint8_t* payload(uint64_t m) const { return reinterpret_cast<const uint8_t*>(slot(m)) + sizeof(ShmSlotHeader); }
  uint64_t wseq() const { return hdr->write_seq.load(std::memory_order_acquire); }

  void skip_to(uint64_t m) {
    if (m > next) {
      missed_pending += m - next;
      missed_total += m - next;
      next = m;
    }
  }
  void overrun_resync() {
    ++overruns;
    const uint64_t w = wseq();
    // Oldest message that can still be intact if the writer is busy writing message w.
    const uint64_t oldest = w + 1 > n_slots ? w + 1 - n_slots : 0;
    skip_to(std::max(next + 1, oldest));
  }

  enum class Try { kOk, kNotYet, kTorn, kTooSmall, kCorrupt };
  /// Attempts to copy message m. On kTooSmall, `bytes` holds the message size.
  Try try_read(uint64_t m, Span<uint8_t> out, ShmReadInfo& info) {
    const ShmSlotHeader* s = slot(m);
    const uint64_t want = 2 * m + 2;
    const uint64_t s1 = s->seq.load(std::memory_order_acquire);
    if (s1 < want) return Try::kNotYet;
    if (s1 > want) return Try::kTorn;
    uint64_t bytes = 0;
    int64_t t = kInvalidTime;
    racy_slot_fields(s, bytes, t);
    Try verdict = Try::kOk;
    if (bytes > slot_size) {
      verdict = Try::kCorrupt;
    } else if (bytes > out.size()) {
      verdict = Try::kTooSmall;
    } else {
      racy_copy(out.data(), payload(m), static_cast<std::size_t>(bytes));
    }
    std::atomic_thread_fence(std::memory_order_acquire);
    const uint64_t s2 = s->seq.load(std::memory_order_relaxed);
    if (s2 != s1) return Try::kTorn;
    info.seq = m;
    info.t_ns = t;
    info.bytes = static_cast<std::size_t>(bytes);
    return verdict;
  }

  ~Impl() {
    if (map != MAP_FAILED) munmap(map, map_bytes);
    if (fd >= 0) close(fd);
  }
};

ShmRingReader::ShmRingReader() = default;
ShmRingReader::~ShmRingReader() = default;

Result<std::unique_ptr<ShmRingReader>> ShmRingReader::open(const std::string& name, StreamKind expected_kind,
                                                           uint32_t expected_type_tag, bool start_at_latest) {
  RIG_RETURN_IF_ERROR(validate_name(name));
  auto r = std::unique_ptr<ShmRingReader>(new ShmRingReader());
  r->impl_ = std::make_unique<Impl>();
  Impl& im = *r->impl_;
  im.name = name;
  im.fd = shm_open(name.c_str(), O_RDONLY | O_CLOEXEC, 0);
  if (im.fd < 0) {
    if (errno == ENOENT) return not_found(name + ": no such shm segment (writer not running?)");
    return io_error(name + ": shm_open: " + errno_text(errno));
  }
  struct stat st {};
  if (fstat(im.fd, &st) != 0) return io_error(name + ": fstat: " + errno_text(errno));
  const std::size_t size = static_cast<std::size_t>(st.st_size);
  if (size < kHeaderBytes) return unavailable(name + ": segment is being created (retry)");
  im.map = mmap(nullptr, size, PROT_READ, MAP_SHARED, im.fd, 0);
  if (im.map == MAP_FAILED) return io_error(name + ": mmap: " + errno_text(errno));
  im.map_bytes = size;
  im.hdr = static_cast<const ShmRingHeader*>(im.map);
  const uint64_t magic = __atomic_load_n(&im.hdr->magic, __ATOMIC_ACQUIRE);
  if (magic == 0) return unavailable(name + ": segment is being initialized (retry)");
  if (magic != kShmRingMagic) return failed_precondition(name + ": bad magic (not a rig ring)");
  if (im.hdr->version != kShmRingVersion)
    return failed_precondition(name + ": ring version " + std::to_string(im.hdr->version) + ", this build reads " +
                               std::to_string(kShmRingVersion));
  if (im.hdr->stream_kind != static_cast<uint32_t>(expected_kind))
    return failed_precondition(name + ": stream kind " + std::to_string(im.hdr->stream_kind) + ", expected " +
                               std::to_string(static_cast<uint32_t>(expected_kind)));
  if (expected_type_tag != 0 && im.hdr->type_tag != expected_type_tag)
    return failed_precondition(name + ": type tag mismatch (struct layout differs)");
  im.slot_size = static_cast<std::size_t>(im.hdr->slot_size);
  im.n_slots = static_cast<std::size_t>(im.hdr->n_slots);
  im.stride = static_cast<std::size_t>(im.hdr->slot_stride);
  if (im.n_slots < 2 || im.stride != slot_stride_for(im.slot_size) ||
      im.n_slots > (size - kHeaderBytes) / im.stride)
    return failed_precondition(name + ": inconsistent ring geometry");
  const uint64_t w = im.wseq();
  if (start_at_latest) {
    im.next = w > 0 ? w - 1 : 0;
  } else {
    im.next = w > im.n_slots ? w - im.n_slots : 0;
  }
  return r;
}

Result<std::optional<ShmReadInfo>> ShmRingReader::read_next(Span<uint8_t> out) {
  Impl& im = *impl_;
  for (int attempt = 0; attempt < 64; ++attempt) {
    const uint64_t w = im.wseq();
    if (im.next >= w) return std::optional<ShmReadInfo>();
    if (w - im.next > im.n_slots) {  // lapped: the slots we wanted were reused
      ++im.overruns;
      im.skip_to(w - im.n_slots);
    }
    ShmReadInfo info;
    switch (im.try_read(im.next, out, info)) {
      case Impl::Try::kOk:
        info.missed = im.missed_pending;
        im.missed_pending = 0;
        im.next = info.seq + 1;
        return std::optional<ShmReadInfo>(info);
      case Impl::Try::kNotYet:
        return std::optional<ShmReadInfo>();  // cannot happen for next < write_seq unless re-created
      case Impl::Try::kTorn:
        im.overrun_resync();
        continue;
      case Impl::Try::kTooSmall:
        im.next = info.seq + 1;
        return out_of_range(im.name + ": message " + std::to_string(info.seq) + " has " + std::to_string(info.bytes) +
                            " bytes, buffer " + std::to_string(out.size()) + " (skipped)");
      case Impl::Try::kCorrupt:
        im.next = info.seq + 1;
        return data_loss(im.name + ": slot holds " + std::to_string(info.bytes) + " bytes > slot_size");
    }
  }
  return std::optional<ShmReadInfo>();  // writer laps us faster than we can copy; try again later
}

Result<std::optional<ShmReadInfo>> ShmRingReader::wait_next(Span<uint8_t> out, std::chrono::nanoseconds timeout) {
  const TimeNs deadline = now_raw_ns() + timeout.count();
  auto sleep = std::chrono::microseconds(20);
  while (true) {
    auto r = read_next(out);
    if (!r.ok() || r.value().has_value()) return r;
    const TimeNs now = now_raw_ns();
    if (now >= deadline) return r;
    // Adaptive polling (no futex word in the frozen header): 20 us doubling to 1 ms.
    std::this_thread::sleep_for(std::min<std::chrono::nanoseconds>(sleep, std::chrono::nanoseconds(deadline - now)));
    if (sleep < std::chrono::milliseconds(1)) sleep *= 2;
  }
}

Result<std::optional<ShmReadInfo>> ShmRingReader::read_latest(Span<uint8_t> out) {
  Impl& im = *impl_;
  for (int attempt = 0; attempt < 64; ++attempt) {
    const uint64_t w = im.wseq();
    if (w == 0 || w <= im.next) return std::optional<ShmReadInfo>();
    ShmReadInfo info;
    switch (im.try_read(w - 1, out, info)) {
      case Impl::Try::kOk:
        info.missed = im.missed_pending;  // only overruns seen by read_next; skipped-by-latest are not missed
        im.missed_pending = 0;
        im.next = w;
        return std::optional<ShmReadInfo>(info);
      case Impl::Try::kNotYet:
      case Impl::Try::kTorn:
        continue;  // overwritten meanwhile: retry with the newer one
      case Impl::Try::kTooSmall:
        im.next = w;
        return out_of_range(im.name + ": message " + std::to_string(info.seq) + " has " + std::to_string(info.bytes) +
                            " bytes, buffer " + std::to_string(out.size()) + " (skipped)");
      case Impl::Try::kCorrupt:
        im.next = w;
        return data_loss(im.name + ": slot holds more bytes than slot_size");
    }
  }
  return std::optional<ShmReadInfo>();
}

uint64_t ShmRingReader::overruns() const { return impl_->overruns; }
uint64_t ShmRingReader::missed_total() const { return impl_->missed_total; }
uint64_t ShmRingReader::writer_seq() const { return impl_->wseq(); }
bool ShmRingReader::writer_alive() const {
  const Impl& im = *impl_;
  if (kill(static_cast<pid_t>(im.hdr->writer_pid), 0) != 0 && errno == ESRCH) return false;
  return lock_held_by_other(im.fd) && same_inode(im.fd, im.name);
}
std::size_t ShmRingReader::slot_size() const { return impl_->slot_size; }
std::size_t ShmRingReader::n_slots() const { return impl_->n_slots; }
const std::string& ShmRingReader::name() const { return impl_->name; }

}  // namespace rig
