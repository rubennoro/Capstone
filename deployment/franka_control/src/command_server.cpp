#include "franka_control/command_server.h"

#include <chrono>
#include <thread>

#include "franka_control/log.h"

namespace franka_control {
namespace {

// Error texts quote what the client sent; keep them plain ASCII so the reply itself always passes the contract.
std::string printable(const std::string& s) {
  std::string out = s;
  for (char& c : out) {
    const auto u = static_cast<unsigned char>(c);
    if (u < 0x20 || u >= 0x7f) c = '?';
  }
  return out;
}

}  // namespace

CommandServer::CommandServer(zmq::context_t& ctx, const std::string& endpoint, Supervisor& supervisor,
                             const StatePublisher& publisher, const Codec& codec, SharedState& shared, Clock clock)
    : socket_(ctx, zmq::socket_type::rep),
      supervisor_(supervisor),
      publisher_(publisher),
      codec_(codec),
      shared_(shared),
      clock_(std::move(clock)) {
  socket_.set(zmq::sockopt::linger, 0);
  socket_.bind(endpoint);
}

std::string CommandServer::endpoint() const { return socket_.get(zmq::sockopt::last_endpoint); }

void CommandServer::send(const std::string& bytes) {
  (void)socket_.send(zmq::const_buffer(bytes.data(), bytes.size()), zmq::send_flags::none);
}

void CommandServer::serviceOnce(int64_t timeout_ns) {
  const auto timeout = std::chrono::milliseconds(std::max<int64_t>(0, (timeout_ns + 999999) / 1000000));
  if (home_pending_) {   // a REP socket must reply before it may receive again
    if (auto reply = supervisor_.pollHome()) {
      home_pending_ = false;
      send(codec_.encode(*reply));
      return;
    }
    std::this_thread::sleep_for(timeout);   // nothing can arrive before this reply; just wait
    return;
  }
  zmq::pollitem_t items[] = {{static_cast<void*>(socket_), 0, ZMQ_POLLIN, 0}};
  zmq::poll(items, 1, timeout);
  if (!(items[0].revents & ZMQ_POLLIN)) return;
  zmq::message_t request;
  if (!socket_.recv(request, zmq::recv_flags::dontwait)) return;
  const std::string reply = handle(std::string(static_cast<const char*>(request.data()), request.size()));
  if (!reply.empty()) send(reply);
}

std::string CommandServer::handle(const std::string& bytes) {
  msg::Any request;
  try {
    request = codec_.decode(bytes);
  } catch (const ContractError& e) {
    Log::warning(std::string("C3 request refused: ") + e.what());
    const std::string schema = e.schema() == "?" ? std::string() : e.schema();
    return codec_.encode(msg::ErrorReply{printable(schema), printable(e.what()), printable(e.field())});
  }
  const std::string schema = msg::schemaOf(request);
  if (!msg::isC3Request(request)) {
    const ContractError e(schema, "", "unexpected_message",
                          "unexpected message: franka_control answers C3 requests only, got " + schema);
    Log::warning(std::string("C3 request refused: ") + e.what());
    return codec_.encode(msg::ErrorReply{schema, e.what(), ""});
  }
  // The watchdog: its own check for this request first, then every decoded request proves the manager is alive.
  supervisor_.watchdogExpiredNow();
  shared_.markAlive(clock_());
  try {
    if (std::holds_alternative<msg::Heartbeat>(request)) return codec_.encode(supervisor_.onHeartbeat());
    if (std::holds_alternative<msg::GetState>(request)) {
      return codec_.encode(supervisor_.stateMessage(publisher_.lastSeq()));
    }
    if (std::holds_alternative<msg::Start>(request)) return codec_.encode(supervisor_.onStart());
    if (std::holds_alternative<msg::Stop>(request)) return codec_.encode(supervisor_.onStop());
    if (std::holds_alternative<msg::Home>(request)) {
      if (auto reply = supervisor_.onHome()) return codec_.encode(*reply);
      home_pending_ = true;
      return "";
    }
    const auto result = supervisor_.onSetWaypoints(std::get<msg::SetWaypoints>(request));
    if (std::holds_alternative<msg::ErrorReply>(result)) return codec_.encode(std::get<msg::ErrorReply>(result));
    const auto& r = std::get<msg::SetWaypointsReply>(result);
    return codec_.encode(r);
  } catch (const std::exception& e) {   // a bug in this process: answer, log it, keep serving
    Log::error(std::string("handling ") + schema + ": " + e.what());
    return codec_.encode(msg::ErrorReply{schema, printable(std::string("franka_control: ") + e.what()), ""});
  }
}

}  // namespace franka_control
