// One writer thread, one reader thread, three copies of T and one atomic index (STRUCTURE.md 3.2).
// Both sides are wait-free (one atomic exchange each); the reader always gets the newest complete copy.
#pragma once

#include <array>
#include <atomic>
#include <cstdint>

namespace franka_control {

template <class T>
class TripleBuffer {
 public:
  TripleBuffer() = default;
  TripleBuffer(const TripleBuffer&) = delete;
  TripleBuffer& operator=(const TripleBuffer&) = delete;

  T& writeSlot() { return slots_[write_]; }                       // writer
  void publish() {                                                // writer: one atomic exchange
    const uint8_t old = middle_.exchange(static_cast<uint8_t>(write_ | kNew), std::memory_order_acq_rel);
    write_ = static_cast<uint8_t>(old & kIndex);
  }
  bool refresh() {                                                // reader: true if something new
    if ((middle_.load(std::memory_order_acquire) & kNew) == 0) return false;
    const uint8_t old = middle_.exchange(read_, std::memory_order_acq_rel);
    read_ = static_cast<uint8_t>(old & kIndex);
    return true;
  }
  const T& readSlot() const { return slots_[read_]; }             // reader

 private:
  static constexpr uint8_t kNew = 0x4;
  static constexpr uint8_t kIndex = 0x3;
  static_assert(std::atomic<uint8_t>::is_always_lock_free, "the triple buffer's index must be a real atomic");

  std::array<T, 3> slots_{};
  uint8_t write_ = 0;                  // writer's own
  uint8_t read_ = 1;                   // reader's own
  std::atomic<uint8_t> middle_{2};     // index of the middle copy, | kNew when the writer published it
};

}  // namespace franka_control
