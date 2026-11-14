// New (control_deploy/STRUCTURE.md 3.3, 4.3).
#include "franka_control/key_queue.h"

#include <string>

#include "franka_control/log.h"

namespace franka_control {

const char* keyChar(Key k) {
  switch (k) {
    case Key::kHomeArm: return "h";
    case Key::kHomeHand: return "g";
    case Key::kStop: return "s";
    case Key::kQuit: return "q";
  }
  return "?";
}

void KeyQueue::push(Key k) {
  {
    std::lock_guard<std::mutex> lock(mutex_);
    if (keys_.size() < kCapacity) {
      keys_.push_back(k);
      return;
    }
  }
  Log::warning(std::string("keys: ") + std::to_string(kCapacity) + " keys are waiting: key " + keyChar(k) +
               " dropped");
}

std::optional<Key> KeyQueue::pop() {
  std::lock_guard<std::mutex> lock(mutex_);
  if (keys_.empty()) return std::nullopt;
  const Key k = keys_.front();
  keys_.pop_front();
  return k;
}

}  // namespace franka_control
