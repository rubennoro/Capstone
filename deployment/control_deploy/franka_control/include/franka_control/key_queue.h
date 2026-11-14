// New (control_deploy/STRUCTURE.md 3.3, 4.3): the PC keys, and the queue that carries h, g, s from the
// keyboard thread to the messaging thread (the Supervisor belongs to the messaging thread). A mutex is fine: neither
// thread is the 1 kHz loop. Bounded for h and g; a stop is never dropped.
#pragma once

#include <cstddef>
#include <deque>
#include <mutex>
#include <optional>

namespace franka_control {

enum class Key { kHomeArm, kHomeHand, kStop, kQuit };

const char* keyChar(Key k);                         // "h", "g", "s", "q"

class KeyQueue {
 public:
  static constexpr size_t kCapacity = 16;
  void push(Key k);                                 // h, g: dropped with a warning when kCapacity are waiting;
                                                    // s: never dropped (q is never queued)
  std::optional<Key> pop();                         // nullopt when empty

 private:
  std::mutex mutex_;
  std::deque<Key> keys_;
};

}  // namespace franka_control
