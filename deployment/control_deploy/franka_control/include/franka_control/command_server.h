// from rnoro/deployment:deployment/franka_control/include/franka_control/command_server.h @ 91b178b
// changes: the constructor takes the hello reply template (built once by App); handle() answers c3.hello
// (control_deploy/STRUCTURE.md 3.2, 5.5).
// C3: one reply per request. REP socket; a broken request gets ErrorReply and the server keeps running (concluding is
// the laptop manager's job). The home reply is deferred until homing ends.
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
  // hello: every field but t_pc_ns, which is set per request to clock() immediately before encoding.
  CommandServer(zmq::context_t& ctx, const std::string& endpoint, Supervisor& supervisor,
                const StatePublisher& publisher, const Codec& codec, SharedState& shared, Clock clock,
                msg::HelloReply hello);
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
  msg::HelloReply hello_;
  bool home_pending_ = false;
};

}  // namespace franka_control
