// SPDX-License-Identifier: MIT
/// @file rig/shm/ring.hpp
/// @brief Cross-process single-writer / many-reader ring of fixed-size slots in POSIX shared memory, each slot
/// guarded by a seqlock (PIPELINE 5 "The shm bus is a deployment interface too"). FROZEN CONTRACT (T0), including
/// the in-memory layout below (it is an ABI between processes); implemented by T3 in common/shm/.
///
/// Segment layout (`shm_open(name)` + `mmap`, size = sizeof(ShmRingHeader) + n_slots * slot_stride):
///   [ShmRingHeader][slot 0: ShmSlotHeader + payload (slot_size bytes) padded to slot_stride] ... [slot n-1]
/// slot_stride = round_up(sizeof(ShmSlotHeader) + slot_size, 64).
///
/// Seqlock protocol for message number m (0-based, monotonically increasing), slot i = m % n_slots:
///   writer: slot.seq = 2m+1 (release)  -> write payload + slot fields -> slot.seq = 2m+2 (release)
///           -> header.write_seq = m+1 (release).
///   reader expecting m: s1 = slot.seq (acquire); s1 < 2m+2 -> not written yet; s1 > 2m+2 -> overwritten (overrun);
///           copy payload; acquire fence; s2 = slot.seq; s1 != s2 -> torn/overwritten (overrun, retry with newer).
/// A reader that detects an overrun resynchronizes to the oldest message still intact and reports the number of
/// messages it missed (ShmReadInfo::missed); overruns are counted, never fatal.
///
/// Names: POSIX names with a leading slash, built as "<shm.prefix>.<stream>" (e.g. "/rig.cam_uw").
/// Lifetime: the writer creates (O_CREAT|O_EXCL; a stale segment whose writer_pid is dead is unlinked and
/// recreated) and unlinks on destruction. Readers keep their mapping valid even after the unlink.
/// Threading: one ShmRingWriter per segment, used by one thread. Each ShmRingReader is used by one thread; any
/// number of readers in any processes.
#pragma once

#include <atomic>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <type_traits>

#include "rig/span.hpp"
#include "rig/status.hpp"
#include "rig/types.hpp"

namespace rig {

/// "RIGRING1" little-endian.
inline constexpr uint64_t kShmRingMagic = 0x31474E4952474952ull;
/// Layout/protocol version; readers refuse other versions with kFailedPrecondition.
inline constexpr uint32_t kShmRingVersion = 1;

/// Payload type of a ring (checked by readers on open).
enum class StreamKind : uint32_t {
  kUnknown = 0,
  kImageFrame = 1,  ///< payload = ShmFrameHeader + pixel bytes (rig/shm/messages.hpp).
  kImuSample = 2,   ///< payload = one ShmImuSample.
  kStatus = 3,      ///< payload = one POD status struct (rig/shm/status_channel.hpp); ring of 2-4 slots.
  kBytes = 4,       ///< opaque bytes.
};

/// Segment header (first bytes of the mapping). Written once by the writer except write_seq.
struct alignas(64) ShmRingHeader {
  uint64_t magic;                   ///< kShmRingMagic; written LAST at creation (release) so readers never see a
                                    ///< half-initialized header.
  uint32_t version;                 ///< kShmRingVersion.
  uint32_t stream_kind;             ///< StreamKind.
  uint64_t slot_size;               ///< Max payload bytes per slot.
  uint64_t slot_stride;             ///< Bytes between slot starts (see file comment).
  uint64_t n_slots;                 ///< Number of slots (>= 2).
  uint32_t type_tag;                ///< For kStatus rings: the struct's kTypeTag; 0 otherwise.
  uint32_t writer_pid;              ///< getpid() of the writer (stale detection, writer_alive()).
  int64_t created_raw_ns;           ///< now_raw_ns() at creation; changes if the writer re-creates the segment.
  char stream_name[64];             ///< NUL-terminated stream name (e.g. "cam_uw").
  alignas(64) std::atomic<uint64_t> write_seq;  ///< Messages committed so far (= next message number).
};

/// Per-slot header, at the start of every slot.
struct alignas(64) ShmSlotHeader {
  std::atomic<uint64_t> seq;  ///< Seqlock word: 0 = never written, 2m+1 = writing m, 2m+2 = holds m.
  uint64_t payload_bytes;     ///< Valid payload bytes (<= slot_size).
  int64_t t_ns;               ///< Message timestamp [ns, MONOTONIC_RAW] (frame t_mid, sample t, status t).
  uint64_t reserved;
};

static_assert(std::atomic<uint64_t>::is_always_lock_free, "shm seqlock needs lock-free 64-bit atomics");
static_assert(std::is_standard_layout_v<ShmRingHeader> && std::is_standard_layout_v<ShmSlotHeader>,
              "shm headers must be standard layout");
static_assert(sizeof(ShmSlotHeader) == 64, "ShmSlotHeader is one cache line");

struct ShmRingOptions {
  std::string name;        ///< POSIX shm name, leading '/', no other '/', <= 200 chars.
  StreamKind kind = StreamKind::kBytes;
  std::size_t slot_size = 0;  ///< Max payload bytes per message (> 0).
  std::size_t n_slots = 8;    ///< >= 2.
  uint32_t type_tag = 0;
  std::string stream_name;    ///< Stored in the header for diagnostics.
};

/// Result of a successful read.
struct ShmReadInfo {
  uint64_t seq = 0;           ///< Message number read.
  TimeNs t_ns = kInvalidTime; ///< Slot timestamp.
  std::size_t bytes = 0;      ///< Payload bytes copied into the caller's buffer.
  uint64_t missed = 0;        ///< Messages skipped since the previous read because of an overrun (0 normally).
};

class ShmRingWriter {
 public:
  /// Creates (or recovers a stale) segment. kAlreadyExists if a live writer owns the name.
  static Result<std::unique_ptr<ShmRingWriter>> create(const ShmRingOptions& options);
  ~ShmRingWriter();  ///< Unlinks the segment.
  ShmRingWriter(const ShmRingWriter&) = delete;
  ShmRingWriter& operator=(const ShmRingWriter&) = delete;

  /// Copies `payload` into the next slot and commits it. Never blocks. kInvalidArgument if larger than slot_size.
  /// Returns the message number written.
  Result<uint64_t> write(Span<const uint8_t> payload, TimeNs t_ns);

  /// Zero-copy variant: begin_write() returns the next slot's payload area (slot_size bytes, valid until commit);
  /// commit() publishes `bytes` of it. Exactly one commit() per begin_write().
  Span<uint8_t> begin_write();
  Result<uint64_t> commit(std::size_t bytes, TimeNs t_ns);

  uint64_t write_seq() const;  ///< Messages committed.
  const std::string& name() const;
  std::size_t slot_size() const;
  std::size_t n_slots() const;

 private:
  ShmRingWriter();
  struct Impl;  // defined by T3
  std::unique_ptr<Impl> impl_;
};

class ShmRingReader {
 public:
  /// Maps an existing segment read-only. kNotFound if absent; kFailedPrecondition on magic/version/kind/type_tag
  /// mismatch (type_tag checked only when expected_type_tag != 0).
  /// @param start_at_latest true: the first read_next() returns the newest message; false: the oldest available.
  static Result<std::unique_ptr<ShmRingReader>> open(const std::string& name, StreamKind expected_kind,
                                                     uint32_t expected_type_tag = 0, bool start_at_latest = true);
  ~ShmRingReader();
  ShmRingReader(const ShmRingReader&) = delete;
  ShmRingReader& operator=(const ShmRingReader&) = delete;

  /// Next message in order into `out`. std::nullopt if no new message. On overrun, skips to the oldest intact
  /// message and reports `missed`. kOutOfRange if `out` is smaller than the message (the message is skipped).
  Result<std::optional<ShmReadInfo>> read_next(Span<uint8_t> out);
  /// Like read_next but waits up to `timeout` (polling or futex; implementation's choice) for a message.
  Result<std::optional<ShmReadInfo>> wait_next(Span<uint8_t> out, std::chrono::nanoseconds timeout);
  /// Newest committed message into `out` (skipping everything older; skipped messages are not counted as missed).
  Result<std::optional<ShmReadInfo>> read_latest(Span<uint8_t> out);

  uint64_t overruns() const;        ///< Overrun events detected.
  uint64_t missed_total() const;    ///< Messages lost to overruns.
  uint64_t writer_seq() const;      ///< header.write_seq now.
  /// False if the writer process is gone (kill(pid, 0) fails with ESRCH) or the segment was re-created.
  bool writer_alive() const;
  std::size_t slot_size() const;
  std::size_t n_slots() const;
  const std::string& name() const;

 private:
  ShmRingReader();
  struct Impl;  // defined by T3
  std::unique_ptr<Impl> impl_;
};

}  // namespace rig
