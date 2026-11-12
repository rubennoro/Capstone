// C3: one reply per request (STRUCTURE.md 8.2). REP socket; a broken request gets ErrorReply and the server keeps
// running (concluding is message_comms' job). The home reply is deferred until homing ends.
#pragma once

#include <cstdint>
#include <string>

#include <zmq.hpp>

#include "franka_control/clock.h"
#include "franka_control/codec.h"
#include "franka_control/shared_state.h"
#include "franka_control/state_publisher.h"
#include "franka_control/supervisor.h"

namespace franka_control {

class CommandServer {                              // messaging thread
 public:
  CommandServer(zmq::context_t& ctx, const std::string& endpoint, Supervisor& supervisor,
                const StatePublisher& publisher, const Codec& codec, SharedState& shared, Clock clock);
  std::string endpoint() const;
  void serviceOnce(int64_t timeout_ns);
  bool homeReplyPending() const { return home_pending_; }

 private:
  std::string handle(const std::string& request);  // the encoded reply ("" when the home reply is deferred)
  void send(const std::string& bytes);

  zmq::socket_t socket_;
  Supervisor& supervisor_;
  const StatePublisher& publisher_;
  const Codec& codec_;
  SharedState& shared_;
  Clock clock_;
  bool home_pending_ = false;
};

}  // namespace franka_control
