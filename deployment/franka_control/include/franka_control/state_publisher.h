// C2: the state broadcast (STRUCTURE.md 8.1). PUB socket, ~100 Hz, never blocks, never pauses.
#pragma once

#include <cstdint>
#include <string>

#include <zmq.hpp>

#include "franka_control/clock.h"
#include "franka_control/codec.h"
#include "franka_control/supervisor.h"

namespace franka_control {

class StatePublisher {                             // messaging thread
 public:
  StatePublisher(zmq::context_t& ctx, const std::string& endpoint, int send_hwm, double rate_hz,
                 const Supervisor& supervisor, const Codec& codec, Clock clock);
  std::string endpoint() const;                    // the bound address (tests bind tcp://127.0.0.1:*)
  bool maybePublish();                             // publishes if due; true if it sent
  int64_t nsUntilNext() const;
  uint64_t lastSeq() const { return seq_; }

 private:
  zmq::socket_t socket_;
  const Supervisor& supervisor_;
  const Codec& codec_;
  Clock clock_;
  int64_t period_ns_;
  int64_t next_due_ns_;
  uint64_t seq_ = 0;
};

}  // namespace franka_control
