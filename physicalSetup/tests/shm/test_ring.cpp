// SPDX-License-Identifier: MIT
// T3: shm seqlock ring: protocol, overrun detection, liveness/stale recovery, multi-thread (TSan) and multi-process
// (fork) tests, status channels.
#include <gtest/gtest.h>

#include <signal.h>
#include <sys/mman.h>
#include <sys/wait.h>
#include <unistd.h>

#include <atomic>
#include <chrono>
#include <cstring>
#include <thread>
#include <vector>

#include "rig/shm/ring.hpp"
#include "rig/shm/status_channel.hpp"
#include "rig/time/clock.hpp"

namespace rig {
namespace {

using namespace std::chrono_literals;

std::string uniq(const std::string& n) { return "/rigtest." + std::to_string(getpid()) + "." + n; }

ShmRingOptions opts(const std::string& name, std::size_t slot, std::size_t n, StreamKind k = StreamKind::kBytes) {
  ShmRingOptions o;
  o.name = name;
  o.kind = k;
  o.slot_size = slot;
  o.n_slots = n;
  o.stream_name = "test";
  return o;
}

/// Payload of message m: `words` uint64 all equal to m * 0x9E3779B97F4A7C15 + i (a torn copy mixes messages).
void fill(std::vector<uint8_t>& buf, uint64_t m, std::size_t words) {
  buf.resize(words * 8);
  for (std::size_t i = 0; i < words; ++i) {
    const uint64_t v = m * 0x9E3779B97F4A7C15ull + i;
    std::memcpy(buf.data() + 8 * i, &v, 8);
  }
}
bool consistent(const uint8_t* p, std::size_t bytes, uint64_t m) {
  if (bytes % 8 != 0) return false;
  for (std::size_t i = 0; i < bytes / 8; ++i) {
    uint64_t v;
    std::memcpy(&v, p + 8 * i, 8);
    if (v != m * 0x9E3779B97F4A7C15ull + i) return false;
  }
  return true;
}

TEST(ShmRing, WriteReadInOrder) {
  const std::string name = uniq("basic");
  auto w = ShmRingWriter::create(opts(name, 64, 4)).value();
  EXPECT_EQ(w->slot_size(), 64u);
  EXPECT_EQ(w->n_slots(), 4u);
  auto r = ShmRingReader::open(name, StreamKind::kBytes, 0, /*start_at_latest=*/false).value();
  std::vector<uint8_t> out(64);
  EXPECT_FALSE(r->read_next(out).value().has_value());
  for (uint64_t m = 0; m < 3; ++m) {
    std::vector<uint8_t> p;
    fill(p, m, 4);
    EXPECT_EQ(w->write(p, static_cast<TimeNs>(1000 + m)).value(), m);
  }
  for (uint64_t m = 0; m < 3; ++m) {
    auto i = r->read_next(out).value();
    ASSERT_TRUE(i.has_value());
    EXPECT_EQ(i->seq, m);
    EXPECT_EQ(i->t_ns, static_cast<TimeNs>(1000 + m));
    EXPECT_EQ(i->bytes, 32u);
    EXPECT_EQ(i->missed, 0u);
    EXPECT_TRUE(consistent(out.data(), i->bytes, m));
  }
  EXPECT_FALSE(r->read_next(out).value().has_value());
  EXPECT_EQ(r->writer_seq(), 3u);
  EXPECT_TRUE(r->writer_alive());
  EXPECT_EQ(r->overruns(), 0u);
}

TEST(ShmRing, ZeroCopyAndErrors) {
  const std::string name = uniq("zc");
  EXPECT_EQ(ShmRingWriter::create(opts("noslash", 8, 4)).status().code(), StatusCode::kInvalidArgument);
  EXPECT_EQ(ShmRingWriter::create(opts("/a/b", 8, 4)).status().code(), StatusCode::kInvalidArgument);
  EXPECT_EQ(ShmRingWriter::create(opts(name, 0, 4)).status().code(), StatusCode::kInvalidArgument);
  EXPECT_EQ(ShmRingWriter::create(opts(name, 8, 1)).status().code(), StatusCode::kInvalidArgument);
  EXPECT_EQ(ShmRingReader::open(name, StreamKind::kBytes).status().code(), StatusCode::kNotFound);
  ShmRingOptions o = opts(name, 16, 2, StreamKind::kStatus);
  o.type_tag = 0x1234;
  auto w = ShmRingWriter::create(o).value();
  EXPECT_EQ(ShmRingWriter::create(o).status().code(), StatusCode::kAlreadyExists);  // live writer
  EXPECT_EQ(ShmRingReader::open(name, StreamKind::kBytes).status().code(), StatusCode::kFailedPrecondition);
  EXPECT_EQ(ShmRingReader::open(name, StreamKind::kStatus, 0x9999).status().code(), StatusCode::kFailedPrecondition);
  auto r = ShmRingReader::open(name, StreamKind::kStatus, 0x1234, false).value();
  std::vector<uint8_t> big(17);
  EXPECT_EQ(w->write(big, 1).status().code(), StatusCode::kInvalidArgument);
  EXPECT_EQ(w->commit(1, 1).status().code(), StatusCode::kFailedPrecondition);
  Span<uint8_t> s = w->begin_write();
  ASSERT_EQ(s.size(), 16u);
  std::memset(s.data(), 0xAB, 16);
  EXPECT_EQ(w->commit(16, 42).value(), 0u);
  std::vector<uint8_t> small(8);
  EXPECT_EQ(r->read_next(small).status().code(), StatusCode::kOutOfRange);  // skipped
  EXPECT_FALSE(r->read_next(small).value().has_value());
  w->write(small, 43).value();
  auto i = r->read_next(small).value();
  ASSERT_TRUE(i.has_value());
  EXPECT_EQ(i->seq, 1u);
}

TEST(ShmRing, OverrunIsDetectedAndCounted) {
  const std::string name = uniq("overrun");
  auto w = ShmRingWriter::create(opts(name, 32, 4)).value();
  auto r = ShmRingReader::open(name, StreamKind::kBytes, 0, false).value();
  std::vector<uint8_t> p, out(32);
  for (uint64_t m = 0; m < 2; ++m) {
    fill(p, m, 4);
    w->write(p, 0).value();
  }
  ASSERT_EQ(r->read_next(out).value()->seq, 0u);
  for (uint64_t m = 2; m < 20; ++m) {
    fill(p, m, 4);
    w->write(p, 0).value();
  }
  // Wanted 1; slots now hold 16..19.
  auto i = r->read_next(out).value();
  ASSERT_TRUE(i.has_value());
  EXPECT_EQ(i->seq, 16u);
  EXPECT_EQ(i->missed, 15u);  // 1..15
  EXPECT_TRUE(consistent(out.data(), i->bytes, 16));
  EXPECT_GE(r->overruns(), 1u);
  EXPECT_EQ(r->missed_total(), 15u);
  for (uint64_t m = 17; m < 20; ++m) {
    auto j = r->read_next(out).value();
    ASSERT_TRUE(j.has_value());
    EXPECT_EQ(j->seq, m);
    EXPECT_EQ(j->missed, 0u);
  }
}

TEST(ShmRing, StartPositionsAndReadLatest) {
  const std::string name = uniq("latest");
  auto w = ShmRingWriter::create(opts(name, 32, 4)).value();
  std::vector<uint8_t> p, out(32);
  for (uint64_t m = 0; m < 10; ++m) {
    fill(p, m, 2);
    w->write(p, static_cast<TimeNs>(m)).value();
  }
  auto newest = ShmRingReader::open(name, StreamKind::kBytes, 0, true).value();
  EXPECT_EQ(newest->read_next(out).value()->seq, 9u);
  auto oldest = ShmRingReader::open(name, StreamKind::kBytes, 0, false).value();
  EXPECT_EQ(oldest->read_next(out).value()->seq, 6u);
  auto lat = ShmRingReader::open(name, StreamKind::kBytes, 0, false).value();
  auto l = lat->read_latest(out).value();
  ASSERT_TRUE(l.has_value());
  EXPECT_EQ(l->seq, 9u);
  EXPECT_EQ(l->missed, 0u);  // skipped by read_latest is not "missed"
  EXPECT_FALSE(lat->read_latest(out).value().has_value());  // nothing new
  fill(p, 10, 2);
  w->write(p, 10).value();
  EXPECT_EQ(lat->read_latest(out).value()->seq, 10u);
  EXPECT_EQ(lat->missed_total(), 0u);
}

TEST(ShmRing, WaitNextTimesOutAndWakes) {
  const std::string name = uniq("wait");
  auto w = ShmRingWriter::create(opts(name, 8, 4)).value();
  auto r = ShmRingReader::open(name, StreamKind::kBytes).value();
  std::vector<uint8_t> out(8);
  const auto t0 = std::chrono::steady_clock::now();
  EXPECT_FALSE(r->wait_next(out, 20ms).value().has_value());
  EXPECT_GE(std::chrono::steady_clock::now() - t0, 19ms);
  std::thread t([&] {
    std::this_thread::sleep_for(10ms);
    std::vector<uint8_t> p(8, 1);
    w->write(p, 5).value();
  });
  auto i = r->wait_next(out, 5s).value();
  t.join();
  ASSERT_TRUE(i.has_value());
  EXPECT_EQ(i->t_ns, 5);
}

TEST(ShmRing, WriterDestructionUnlinksAndReaderKeepsMapping) {
  const std::string name = uniq("unlink");
  auto w = ShmRingWriter::create(opts(name, 16, 4)).value();
  auto r = ShmRingReader::open(name, StreamKind::kBytes, 0, false).value();
  std::vector<uint8_t> p, out(16);
  fill(p, 0, 2);
  w->write(p, 1).value();
  EXPECT_TRUE(r->writer_alive());
  w.reset();
  EXPECT_FALSE(r->writer_alive());
  EXPECT_EQ(ShmRingReader::open(name, StreamKind::kBytes).status().code(), StatusCode::kNotFound);
  auto i = r->read_next(out).value();  // mapping still valid after unlink
  ASSERT_TRUE(i.has_value());
  EXPECT_TRUE(consistent(out.data(), i->bytes, 0));
}

// In-process seqlock torture: one writer thread, three readers; tiny ring so overruns and torn reads are frequent.
// Every accepted message must be internally consistent and in increasing order.
TEST(ShmRing, SeqlockUnderContentionNeverReturnsTornData) {
  const std::string name = uniq("torture");
  constexpr std::size_t kWords = 512;  // 4 KiB payload: copies take long enough to be overtaken
  auto w = ShmRingWriter::create(opts(name, kWords * 8, 3)).value();
  std::atomic<bool> done{false};
  std::atomic<uint64_t> bad{0}, reads{0}, missed{0};
  auto reader = [&](bool latest) {
    auto r = ShmRingReader::open(name, StreamKind::kBytes, 0, false).value();
    std::vector<uint8_t> out(kWords * 8);
    int64_t last = -1;
    while (!done.load(std::memory_order_acquire)) {
      auto res = latest ? r->read_latest(out) : r->read_next(out);
      if (!res.ok()) {
        bad++;
        continue;
      }
      if (!res.value()) continue;
      const ShmReadInfo& i = *res.value();
      if (!consistent(out.data(), i.bytes, i.seq) || static_cast<int64_t>(i.seq) <= last) bad++;
      if (static_cast<TimeNs>(i.seq) != i.t_ns) bad++;
      last = static_cast<int64_t>(i.seq);
      reads++;
      missed += i.missed;
    }
  };
  std::vector<std::thread> ts;
  ts.emplace_back(reader, false);
  ts.emplace_back(reader, false);
  ts.emplace_back(reader, true);
  std::vector<uint8_t> p;
  const TimeNs end = now_raw_ns() + 1500000000LL;
  uint64_t m = 0;
  while (now_raw_ns() < end) {
    // Write through begin_write/commit word by word (slow writer: maximizes overlap with readers).
    Span<uint8_t> s = w->begin_write();
    fill(p, m, kWords);
    std::memcpy(s.data(), p.data(), p.size());
    w->commit(p.size(), static_cast<TimeNs>(m)).value();
    ++m;
  }
  done = true;
  for (auto& t : ts) t.join();
  EXPECT_EQ(bad.load(), 0u);
  EXPECT_GT(reads.load(), 100u);
  RecordProperty("messages", static_cast<int>(m));
  RecordProperty("reads", static_cast<int>(reads.load()));
  RecordProperty("missed", static_cast<int>(missed.load()));
}

// ---------------------------------------------------------------------------------------------------------------
// Multi-process
// ---------------------------------------------------------------------------------------------------------------

int wait_child(pid_t pid) {
  int st = 0;
  if (waitpid(pid, &st, 0) != pid) return -1;
  if (WIFEXITED(st)) return WEXITSTATUS(st);
  if (WIFSIGNALED(st)) return 128 + WTERMSIG(st);
  return -1;
}

/// Child: writes `n` messages (payload pattern, t_ns = m) at about `rate_hz`, then holds the segment for `hold_ms`.
[[noreturn]] void child_writer(const std::string& name, uint64_t n, double rate_hz, int hold_ms, int* ready_flag) {
  auto w = ShmRingWriter::create(opts(name, 256 * 8, 16));
  if (!w.ok()) _exit(10);
  __atomic_store_n(ready_flag, 1, __ATOMIC_RELEASE);
  std::vector<uint8_t> p;
  const auto period = std::chrono::nanoseconds(static_cast<int64_t>(1e9 / rate_hz));
  auto next = std::chrono::steady_clock::now();
  for (uint64_t m = 0; m < n; ++m) {
    fill(p, m, 256);
    if (!(*w)->write(p, static_cast<TimeNs>(m)).ok()) _exit(11);
    next += period;
    std::this_thread::sleep_until(next);
  }
  std::this_thread::sleep_for(std::chrono::milliseconds(hold_ms));
  w.value().reset();
  _exit(0);
}

/// Child: reads until the writer has written `n` and nothing is left; exit 0 if consistent and complete
/// (seq + missed accounts for every message), else a code. `slow_us` sleeps after each read (forces overruns).
[[noreturn]] void child_reader(const std::string& name, uint64_t n, int slow_us, bool expect_overrun) {
  std::unique_ptr<ShmRingReader> r;
  for (int i = 0; i < 2000 && !r; ++i) {
    auto o = ShmRingReader::open(name, StreamKind::kBytes, 0, false);
    if (o.ok()) r = std::move(o).value();
    else std::this_thread::sleep_for(1ms);
  }
  if (!r) _exit(20);
  std::vector<uint8_t> out(256 * 8);
  uint64_t accounted = 0;  // messages delivered + missed
  int64_t last = -1;
  const TimeNs deadline = now_raw_ns() + 20000000000LL;
  while (now_raw_ns() < deadline) {
    auto res = r->wait_next(out, 50ms);
    if (!res.ok()) _exit(21);
    if (!res.value()) {
      if (r->writer_seq() >= n && static_cast<uint64_t>(last + 1) >= n) break;
      continue;
    }
    const ShmReadInfo& i = *res.value();
    if (!consistent(out.data(), i.bytes, i.seq)) _exit(22);
    if (last < 0) {
      // Messages written before open() (older than the ring) are not "missed"; start the accounting there.
      last = static_cast<int64_t>(i.seq - i.missed) - 1;
      accounted = static_cast<uint64_t>(last + 1);
    }
    if (static_cast<int64_t>(i.seq) <= last) _exit(23);
    if (i.seq != static_cast<uint64_t>(last + 1) + i.missed) _exit(24);
    last = static_cast<int64_t>(i.seq);
    accounted += 1 + i.missed;
    if (slow_us > 0) std::this_thread::sleep_for(std::chrono::microseconds(slow_us));
  }
  if (accounted != n) _exit(25);
  if (expect_overrun && r->overruns() == 0) _exit(26);
  // (A fast reader normally misses nothing, but a scheduling hiccup > 5 ms may cost it messages; the accounting
  // check above is the invariant.)
  _exit(0);
}

TEST(ShmRingMultiProcess, WriterAndReadersInSeparateProcesses) {
  const std::string name = uniq("mp");
  int* ready = static_cast<int*>(mmap(nullptr, sizeof(int), PROT_READ | PROT_WRITE, MAP_SHARED | MAP_ANONYMOUS, -1, 0));
  ASSERT_NE(ready, MAP_FAILED);
  *ready = 0;
  constexpr uint64_t kN = 3000;
  const pid_t wpid = fork();
  ASSERT_GE(wpid, 0);
  if (wpid == 0) child_writer(name, kN, 3000.0, 300, ready);
  while (!__atomic_load_n(ready, __ATOMIC_ACQUIRE)) std::this_thread::sleep_for(1ms);
  const pid_t fast = fork();
  if (fast == 0) child_reader(name, kN, 0, false);
  const pid_t slow = fork();
  if (slow == 0) child_reader(name, kN, 3000, true);  // 3 ms per read vs 0.33 ms per message, 16 slots
  EXPECT_EQ(wait_child(fast), 0);
  EXPECT_EQ(wait_child(slow), 0);
  EXPECT_EQ(wait_child(wpid), 0);
  EXPECT_EQ(ShmRingReader::open(name, StreamKind::kBytes).status().code(), StatusCode::kNotFound);  // unlinked
  munmap(ready, sizeof(int));
}

TEST(ShmRingMultiProcess, WriterCrashIsDetectedAndSegmentRecovered) {
  const std::string name = uniq("crash");
  int* ready = static_cast<int*>(mmap(nullptr, sizeof(int), PROT_READ | PROT_WRITE, MAP_SHARED | MAP_ANONYMOUS, -1, 0));
  ASSERT_NE(ready, MAP_FAILED);
  *ready = 0;
  const pid_t wpid = fork();
  ASSERT_GE(wpid, 0);
  if (wpid == 0) child_writer(name, 1000000, 2000.0, 0, ready);  // runs until killed
  while (!__atomic_load_n(ready, __ATOMIC_ACQUIRE)) std::this_thread::sleep_for(1ms);
  auto r = ShmRingReader::open(name, StreamKind::kBytes, 0, false).value();
  std::vector<uint8_t> out(256 * 8);
  ASSERT_TRUE(r->wait_next(out, 2s).value().has_value());
  EXPECT_TRUE(r->writer_alive());
  // A second writer must not steal a live segment.
  EXPECT_EQ(ShmRingWriter::create(opts(name, 256 * 8, 16)).status().code(), StatusCode::kAlreadyExists);

  kill(wpid, SIGKILL);  // crash: no destructor, no unlink, possibly mid-write
  EXPECT_EQ(wait_child(wpid), 128 + SIGKILL);
  EXPECT_FALSE(r->writer_alive());
  // The stale segment is still there (nobody unlinked it) ...
  ASSERT_TRUE(ShmRingReader::open(name, StreamKind::kBytes).ok());
  // ... and a restarted writer recovers it.
  auto w2 = ShmRingWriter::create(opts(name, 256 * 8, 16));
  ASSERT_TRUE(w2.ok()) << w2.status().to_string();
  EXPECT_FALSE(r->writer_alive());  // the old reader sees a re-created segment, not its writer
  auto r2 = ShmRingReader::open(name, StreamKind::kBytes, 0, false).value();
  EXPECT_TRUE(r2->writer_alive());
  std::vector<uint8_t> p;
  fill(p, 0, 256);
  (*w2)->write(p, 0).value();
  auto i = r2->read_next(out).value();
  ASSERT_TRUE(i.has_value());
  EXPECT_EQ(i->seq, 0u);
  EXPECT_TRUE(consistent(out.data(), i->bytes, 0));
  munmap(ready, sizeof(int));
}

// ---------------------------------------------------------------------------------------------------------------
// Status channels (inline templates over the ring)
// ---------------------------------------------------------------------------------------------------------------

TEST(StatusChannel, SlamStatusLatestValue) {
  const std::string name = uniq("slam");
  auto w = StatusChannelWriter<SlamStatus>::create(name).value();
  EXPECT_EQ(StatusChannelReader<RecorderStatus>::open(name).status().code(), StatusCode::kFailedPrecondition);
  auto r = StatusChannelReader<SlamStatus>::open(name).value();
  EXPECT_FALSE(r.read_latest().value().has_value());
  SlamStatus s = make_slam_status();
  for (int i = 0; i < 10; ++i) {
    s.heartbeat_seq = static_cast<uint64_t>(i);
    s.p[0] = i * 0.5;
    ASSERT_TRUE(w.publish(s, i).ok());
  }
  auto v = r.read_latest().value();
  ASSERT_TRUE(v.has_value());
  EXPECT_EQ(v->heartbeat_seq, 9u);
  EXPECT_DOUBLE_EQ(v->p[0], 4.5);
  EXPECT_FALSE(r.read_latest().value().has_value());
  EXPECT_TRUE(r.writer_alive());
  EXPECT_EQ(r.writer_seq(), 10u);
  EXPECT_EQ(w.published(), 10u);
}

}  // namespace
}  // namespace rig
