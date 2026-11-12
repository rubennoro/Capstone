#include "franka_control/state_publisher.h"

#include <algorithm>

namespace franka_control {

StatePublisher::StatePublisher(zmq::context_t& ctx, const std::string& endpoint, int send_hwm, double rate_hz,
                               const Supervisor& supervisor, const Codec& codec, Clock clock)
    : socket_(ctx, zmq::socket_type::pub),
      supervisor_(supervisor),
      codec_(codec),
      clock_(std::move(clock)),
      period_ns_(static_cast<int64_t>(1e9 / rate_hz)),
      next_due_ns_(clock_()) {
  socket_.set(zmq::sockopt::linger, 0);
  socket_.set(zmq::sockopt::sndhwm, send_hwm);
  socket_.bind(endpoint);
}

std::string StatePublisher::endpoint() const { return socket_.get(zmq::sockopt::last_endpoint); }

int64_t StatePublisher::nsUntilNext() const { return std::max<int64_t>(0, next_due_ns_ - clock_()); }

bool StatePublisher::maybePublish() {
  const int64_t now = clock_();
  if (now < next_due_ns_) return false;
  const uint64_t seq = seq_ + 1;
  const std::string bytes = codec_.encode(supervisor_.stateMessage(seq));
  seq_ = seq;   // +1 per broadcast, also when a full queue drops it (subscribers then see the gap)
  (void)socket_.send(zmq::const_buffer(bytes.data(), bytes.size()), zmq::send_flags::dontwait);
  next_due_ns_ += period_ns_;
  if (next_due_ns_ <= now) next_due_ns_ = now + period_ns_;   // after a stall: keep the rate, no burst
  return true;
}

}  // namespace franka_control
