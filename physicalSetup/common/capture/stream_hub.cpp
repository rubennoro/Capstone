// SPDX-License-Identifier: MIT
// StreamHub / StreamSubscription (rig/capture/stream_hub.hpp).
//
// Concurrency design:
// - Each subscription owns a Queue: a fixed ring buffer of `capacity` T (allocated once at subscribe), a mutex and
//   a condition variable. Every counter in SubscriberStats is updated under that mutex, so stats() snapshots are
//   self-consistent: offered == delivered + dropped + overflows + depth at any instant.
// - The hub keeps the subscriber list as an immutable shared_ptr<const vector<shared_ptr<Queue>>> (copy-on-write)
//   behind `subs_mu_`. publish() holds subs_mu_ only to copy that pointer, then pushes into each queue under the
//   queue's own mutex (O(1), never waits for a consumer: the consumer holds the queue mutex only to move one item
//   out or for the condvar wait, which releases it). Lock order is always subs_mu_ -> queue mutex; consumers never
//   take subs_mu_. There is no lock inversion.
// - Lossless: a full queue rejects the NEW item (+overflows) and latches the hub flag (atomic, never reset).
//   DropOk: a full queue discards the OLDEST item (+dropped).
// - A subscription destroyed by its consumer marks its queue detached: publish() skips it (a forgotten lossless
//   subscriber must not latch an overflow), and the next list rebuild drops it.
// - close(): marks the hub closed (publish() becomes a no-op), closes every queue and wakes all consumers.
#include "rig/capture/stream_hub.hpp"

#include <algorithm>
#include <atomic>
#include <condition_variable>
#include <mutex>
#include <utility>

namespace rig {

template <class T>
struct StreamSubscription<T>::Queue {
  Queue(const SubscriberOptions& o, std::atomic<bool>* hub_overflow)
      : name(o.name), policy(o.policy), capacity(o.capacity), buf(o.capacity), hub_lossless_overflow(hub_overflow) {}

  const std::string name;
  const DeliveryPolicy policy;
  const std::size_t capacity;

  mutable std::mutex mu;
  std::condition_variable cv;
  std::vector<T> buf;   // ring: items at [head, head + count) mod capacity
  std::size_t head = 0;
  std::size_t count = 0;
  bool closed = false;
  uint64_t offered = 0, delivered = 0, dropped = 0, overflows = 0;
  std::size_t max_depth = 0;

  std::atomic<bool> detached{false};
  // Lives in the hub's Impl; valid while the hub lives. Only touched by push(), i.e. from publish() on the hub.
  std::atomic<bool>* hub_lossless_overflow;

  /// Called by the hub (publisher thread). Returns false if the queue is closed.
  void push(const T& item) {
    bool notify = false;
    {
      std::lock_guard<std::mutex> lk(mu);
      if (closed) return;
      ++offered;
      if (count == capacity) {
        if (policy == DeliveryPolicy::kLossless) {
          ++overflows;
          hub_lossless_overflow->store(true, std::memory_order_release);
          return;
        }
        // kDropOk: discard the oldest.
        buf[head] = T();
        head = (head + 1) % capacity;
        --count;
        ++dropped;
      }
      buf[(head + count) % capacity] = item;
      ++count;
      max_depth = std::max(max_depth, count);
      notify = true;
    }
    if (notify) cv.notify_one();
  }

  /// Under `mu` with count > 0.
  void take_one(T& out) {
    out = std::move(buf[head]);
    buf[head] = T();  // release shared pixel buffers promptly
    head = (head + 1) % capacity;
    --count;
    ++delivered;
  }

  void close() {
    {
      std::lock_guard<std::mutex> lk(mu);
      closed = true;
    }
    cv.notify_all();
  }

  SubscriberStats stats() const {
    std::lock_guard<std::mutex> lk(mu);
    SubscriberStats s;
    s.name = name;
    s.policy = policy;
    s.capacity = capacity;
    s.offered = offered;
    s.delivered = delivered;
    s.dropped = dropped;
    s.overflows = overflows;
    s.depth = count;
    s.max_depth = max_depth;
    return s;
  }
};

// ---------------------------------------------------------------------------------------------------------------
// StreamSubscription
// ---------------------------------------------------------------------------------------------------------------

template <class T>
StreamSubscription<T>::StreamSubscription(std::shared_ptr<Queue> q) : q_(std::move(q)) {}

template <class T>
StreamSubscription<T>::~StreamSubscription() {
  if (q_) q_->detached.store(true, std::memory_order_release);
}

template <class T>
bool StreamSubscription<T>::pop(T& out, std::chrono::nanoseconds timeout) {
  std::unique_lock<std::mutex> lk(q_->mu);
  if (q_->count == 0) {
    if (timeout.count() > 0) q_->cv.wait_for(lk, timeout, [&] { return q_->count > 0 || q_->closed; });
    if (q_->count == 0) return false;
  }
  q_->take_one(out);
  return true;
}

template <class T>
bool StreamSubscription<T>::try_pop(T& out) {
  return pop(out, std::chrono::nanoseconds(0));
}

template <class T>
std::size_t StreamSubscription<T>::pop_all(std::vector<T>& out, std::chrono::nanoseconds timeout) {
  std::unique_lock<std::mutex> lk(q_->mu);
  if (q_->count == 0 && timeout.count() > 0) q_->cv.wait_for(lk, timeout, [&] { return q_->count > 0 || q_->closed; });
  const std::size_t n = q_->count;
  out.reserve(out.size() + n);
  for (std::size_t i = 0; i < n; ++i) {
    out.emplace_back();
    q_->take_one(out.back());
  }
  return n;
}

template <class T>
const std::string& StreamSubscription<T>::name() const {
  return q_->name;
}

template <class T>
SubscriberStats StreamSubscription<T>::stats() const {
  return q_->stats();
}

template <class T>
bool StreamSubscription<T>::closed() const {
  std::lock_guard<std::mutex> lk(q_->mu);
  return q_->closed;
}

// ---------------------------------------------------------------------------------------------------------------
// StreamHub
// ---------------------------------------------------------------------------------------------------------------

template <class T>
struct StreamHub<T>::Impl {
  using Queue = typename StreamSubscription<T>::Queue;
  using List = std::vector<std::shared_ptr<Queue>>;

  explicit Impl(std::string n) : stream_name(std::move(n)), subs(std::make_shared<const List>()) {}

  const std::string stream_name;
  std::atomic<uint64_t> published{0};
  std::atomic<bool> lossless_overflow{false};
  std::atomic<bool> closed{false};

  mutable std::mutex subs_mu;              // guards `subs` (the pointer) and serializes list rebuilds
  std::shared_ptr<const List> subs;        // immutable snapshot

  std::shared_ptr<const List> snapshot() const {
    std::lock_guard<std::mutex> lk(subs_mu);
    return subs;
  }
};

template <class T>
StreamHub<T>::StreamHub(std::string stream_name) : impl_(std::make_unique<Impl>(std::move(stream_name))) {}

template <class T>
StreamHub<T>::~StreamHub() {
  close();
}

template <class T>
Result<std::shared_ptr<StreamSubscription<T>>> StreamHub<T>::subscribe(const SubscriberOptions& options) {
  if (options.capacity == 0)
    return invalid_argument(impl_->stream_name + ": subscriber '" + options.name + "': capacity must be >= 1");
  if (options.name.empty()) return invalid_argument(impl_->stream_name + ": subscriber name must not be empty");
  std::lock_guard<std::mutex> lk(impl_->subs_mu);
  if (impl_->closed.load(std::memory_order_acquire))
    return failed_precondition(impl_->stream_name + ": subscribe('" + options.name + "') after close()");
  auto next = std::make_shared<typename Impl::List>();
  for (const auto& q : *impl_->subs) {
    if (q->detached.load(std::memory_order_acquire)) continue;  // prune abandoned subscriptions
    if (q->name == options.name)
      return already_exists(impl_->stream_name + ": subscriber '" + options.name + "' already exists");
    next->push_back(q);
  }
  auto q = std::make_shared<typename Impl::Queue>(options, &impl_->lossless_overflow);
  next->push_back(q);
  impl_->subs = std::move(next);
  return std::make_shared<StreamSubscription<T>>(q);
}

template <class T>
Status StreamHub<T>::unsubscribe(const std::string& name) {
  std::shared_ptr<typename Impl::Queue> victim;
  {
    std::lock_guard<std::mutex> lk(impl_->subs_mu);
    auto next = std::make_shared<typename Impl::List>();
    for (const auto& q : *impl_->subs) {
      if (q->detached.load(std::memory_order_acquire)) continue;
      if (q->name == name && !victim) {
        victim = q;
        continue;
      }
      next->push_back(q);
    }
    if (!victim) return not_found(impl_->stream_name + ": no subscriber '" + name + "'");
    impl_->subs = std::move(next);
  }
  victim->close();
  return Status::Ok();
}

template <class T>
void StreamHub<T>::publish(const T& item) {
  impl_->published.fetch_add(1, std::memory_order_relaxed);
  if (impl_->closed.load(std::memory_order_acquire)) return;
  const auto list = impl_->snapshot();
  for (const auto& q : *list) {
    if (q->detached.load(std::memory_order_acquire)) continue;
    q->push(item);
  }
}

template <class T>
void StreamHub<T>::close() {
  std::shared_ptr<const typename Impl::List> list;
  {
    std::lock_guard<std::mutex> lk(impl_->subs_mu);
    impl_->closed.store(true, std::memory_order_release);
    list = impl_->subs;
  }
  for (const auto& q : *list) q->close();
}

template <class T>
const std::string& StreamHub<T>::stream_name() const {
  return impl_->stream_name;
}

template <class T>
uint64_t StreamHub<T>::published() const {
  return impl_->published.load(std::memory_order_relaxed);
}

template <class T>
std::vector<SubscriberStats> StreamHub<T>::stats() const {
  const auto list = impl_->snapshot();
  std::vector<SubscriberStats> out;
  for (const auto& q : *list) {
    if (q->detached.load(std::memory_order_acquire)) continue;
    out.push_back(q->stats());
  }
  return out;
}

template <class T>
bool StreamHub<T>::lossless_overflow() const {
  return impl_->lossless_overflow.load(std::memory_order_acquire);
}

template class StreamSubscription<ImageFrame>;
template class StreamSubscription<ImuSample>;
template class StreamSubscription<ClockBracket>;
template class StreamHub<ImageFrame>;
template class StreamHub<ImuSample>;
template class StreamHub<ClockBracket>;

}  // namespace rig
