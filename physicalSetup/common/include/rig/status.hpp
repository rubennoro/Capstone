// SPDX-License-Identifier: MIT
/// @file rig/status.hpp
/// @brief Error handling across module boundaries: `Status` and `Result<T>`. FROZEN CONTRACT (T0).
///
/// Rules (IMPLEMENTATION_PLAN.md section 0, "no exceptions across module boundaries"):
/// - Every public function that can fail returns `Status` or `Result<T>`. Exceptions thrown by third-party code
///   (yaml-cpp, nlohmann_json, std::filesystem, OpenCV) are caught inside the implementing module and converted.
/// - `Status` and `Result` are `[[nodiscard]]`: ignoring one is a compile error under -Werror.
/// - Messages are human-readable, start with the failing context (e.g. a YAML path such as `cameras.uw.driver.fps`
///   or a device such as `/dev/i2c-7 addr 0x6a`), and never end in a newline.
/// - Threading: plain value types; no internal synchronization; safe to copy between threads.
#pragma once

#include <cstdio>
#include <cstdlib>
#include <optional>
#include <string>
#include <string_view>
#include <type_traits>
#include <utility>

namespace rig {

/// Error category. Stable numeric values (they may appear in logs and shm status structs).
enum class StatusCode : int {
  kOk = 0,
  kInvalidArgument = 1,     ///< Caller passed a bad value (config validation errors use this).
  kNotFound = 2,            ///< File, device, camera mode, key ... does not exist.
  kAlreadyExists = 3,       ///< Path or shm segment already exists and must not be overwritten.
  kOutOfRange = 4,          ///< Value outside the allowed range (e.g. read past end, FIFO index).
  kFailedPrecondition = 5,  ///< Call not allowed in the current state (e.g. read() before start()).
  kUnavailable = 6,         ///< Resource temporarily unavailable (device busy, daemon down); may be retried.
  kTimeout = 7,             ///< A blocking operation did not complete within its deadline.
  kIoError = 8,             ///< File-system or syscall failure (errno text in the message).
  kDeviceError = 9,         ///< Hardware/driver reported an error (I2C NACK, Argus error, WHO_AM_I mismatch).
  kDataLoss = 10,           ///< Data was lost or is corrupt (FIFO overrun, lossless queue overflow, torn read).
  kOverrun = 11,            ///< A reader fell behind a ring writer and messages were overwritten.
  kUnimplemented = 12,      ///< Not built in this configuration (e.g. Argus backend with RIG_WITH_ARGUS=OFF).
  kInternal = 13,           ///< Invariant violated; a bug.
  kCancelled = 14,          ///< Operation stopped on request (stop(), Ctrl-C).
};

/// Stable lowercase name of a code, e.g. "invalid_argument".
constexpr const char* to_string(StatusCode c) noexcept {
  switch (c) {
    case StatusCode::kOk: return "ok";
    case StatusCode::kInvalidArgument: return "invalid_argument";
    case StatusCode::kNotFound: return "not_found";
    case StatusCode::kAlreadyExists: return "already_exists";
    case StatusCode::kOutOfRange: return "out_of_range";
    case StatusCode::kFailedPrecondition: return "failed_precondition";
    case StatusCode::kUnavailable: return "unavailable";
    case StatusCode::kTimeout: return "timeout";
    case StatusCode::kIoError: return "io_error";
    case StatusCode::kDeviceError: return "device_error";
    case StatusCode::kDataLoss: return "data_loss";
    case StatusCode::kOverrun: return "overrun";
    case StatusCode::kUnimplemented: return "unimplemented";
    case StatusCode::kInternal: return "internal";
    case StatusCode::kCancelled: return "cancelled";
  }
  return "unknown";
}

/// Outcome of an operation without a value. Default-constructed = OK.
class [[nodiscard]] Status {
 public:
  Status() = default;
  /// @param code    error category; passing kOk yields an OK status and drops the message.
  /// @param message context-first description, no trailing newline.
  Status(StatusCode code, std::string message)
      : code_(code), message_(code == StatusCode::kOk ? std::string() : std::move(message)) {}

  static Status Ok() { return Status(); }

  bool ok() const noexcept { return code_ == StatusCode::kOk; }
  StatusCode code() const noexcept { return code_; }
  const std::string& message() const noexcept { return message_; }

  /// "ok" or "<code>: <message>".
  std::string to_string() const {
    if (ok()) return "ok";
    return std::string(rig::to_string(code_)) + ": " + message_;
  }

  /// Returns a copy with "<context>: " prepended to the message (no-op when OK).
  Status with_context(std::string_view context) const {
    if (ok()) return *this;
    return Status(code_, std::string(context) + ": " + message_);
  }

  friend bool operator==(const Status& a, const Status& b) { return a.code_ == b.code_ && a.message_ == b.message_; }
  friend bool operator!=(const Status& a, const Status& b) { return !(a == b); }

 private:
  StatusCode code_ = StatusCode::kOk;
  std::string message_;
};

/// @name Convenience constructors for error statuses.
/// @{
inline Status invalid_argument(std::string m) { return Status(StatusCode::kInvalidArgument, std::move(m)); }
inline Status not_found(std::string m) { return Status(StatusCode::kNotFound, std::move(m)); }
inline Status already_exists(std::string m) { return Status(StatusCode::kAlreadyExists, std::move(m)); }
inline Status out_of_range(std::string m) { return Status(StatusCode::kOutOfRange, std::move(m)); }
inline Status failed_precondition(std::string m) { return Status(StatusCode::kFailedPrecondition, std::move(m)); }
inline Status unavailable(std::string m) { return Status(StatusCode::kUnavailable, std::move(m)); }
inline Status timeout_error(std::string m) { return Status(StatusCode::kTimeout, std::move(m)); }
inline Status io_error(std::string m) { return Status(StatusCode::kIoError, std::move(m)); }
inline Status device_error(std::string m) { return Status(StatusCode::kDeviceError, std::move(m)); }
inline Status data_loss(std::string m) { return Status(StatusCode::kDataLoss, std::move(m)); }
inline Status overrun_error(std::string m) { return Status(StatusCode::kOverrun, std::move(m)); }
inline Status unimplemented(std::string m) { return Status(StatusCode::kUnimplemented, std::move(m)); }
inline Status internal_error(std::string m) { return Status(StatusCode::kInternal, std::move(m)); }
inline Status cancelled(std::string m) { return Status(StatusCode::kCancelled, std::move(m)); }
/// @}

namespace detail {
[[noreturn]] inline void result_access_failure(const Status& s) {
  std::fprintf(stderr, "rig::Result: value() called on error result: %s\n", s.to_string().c_str());
  std::abort();
}
}  // namespace detail

/// A value of type T or an error `Status`. Never both, never neither.
///
/// - Implicit construction from a T (success) or from a non-OK Status (failure). Constructing from an OK Status is a
///   bug and is turned into `kInternal` so that `ok()` stays truthful.
/// - `value()` / `operator*` / `operator->` on an error result print the status and abort (a bug, not an exception).
/// - T must be move-constructible; T = Status or T = void are not supported (use Status for "no value").
template <class T>
class [[nodiscard]] Result {
  static_assert(!std::is_same_v<std::decay_t<T>, Status>, "Result<Status> is not allowed");
  static_assert(!std::is_void_v<T>, "use Status instead of Result<void>");
  static_assert(!std::is_reference_v<T>, "Result<T&> is not allowed");

 public:
  /// Success.
  Result(const T& v) : value_(v) {}  // NOLINT(google-explicit-constructor)
  /// Success.
  Result(T&& v) : value_(std::move(v)) {}  // NOLINT(google-explicit-constructor)
  /// Success from any type convertible to T (e.g. unique_ptr<Derived> -> unique_ptr<Base>).
  template <class U, class = std::enable_if_t<std::is_convertible_v<U&&, T> &&
                                              !std::is_same_v<std::decay_t<U>, T> &&
                                              !std::is_same_v<std::decay_t<U>, Status> &&
                                              !std::is_same_v<std::decay_t<U>, Result>>>
  Result(U&& v) : value_(T(std::forward<U>(v))) {}  // NOLINT(google-explicit-constructor)
  /// Failure. `s` must not be OK.
  Result(Status s)  // NOLINT(google-explicit-constructor)
      : status_(s.ok() ? internal_error("Result constructed from an OK Status without a value") : std::move(s)) {}

  bool ok() const noexcept { return value_.has_value(); }
  explicit operator bool() const noexcept { return ok(); }

  /// OK status on success, the error otherwise.
  const Status& status() const noexcept { return status_; }

  /// @pre ok(); aborts otherwise.
  T& value() & {
    if (!ok()) detail::result_access_failure(status_);
    return *value_;
  }
  const T& value() const& {
    if (!ok()) detail::result_access_failure(status_);
    return *value_;
  }
  T&& value() && {
    if (!ok()) detail::result_access_failure(status_);
    return std::move(*value_);
  }
  T& operator*() & { return value(); }
  const T& operator*() const& { return value(); }
  T&& operator*() && { return std::move(*this).value(); }
  T* operator->() { return &value(); }
  const T* operator->() const { return &value(); }

  /// The value, or `fallback` on error.
  template <class U>
  T value_or(U&& fallback) const& {
    return ok() ? *value_ : static_cast<T>(std::forward<U>(fallback));
  }

 private:
  std::optional<T> value_;
  Status status_;
};

}  // namespace rig

#define RIG_STATUS_CONCAT_INNER_(a, b) a##b
#define RIG_STATUS_CONCAT_(a, b) RIG_STATUS_CONCAT_INNER_(a, b)

/// Evaluates `expr` (a Status) and returns it from the enclosing function if it is not OK.
#define RIG_RETURN_IF_ERROR(expr)                                          \
  do {                                                                     \
    ::rig::Status RIG_STATUS_CONCAT_(rig_status_, __LINE__) = (expr);      \
    if (!RIG_STATUS_CONCAT_(rig_status_, __LINE__).ok())                   \
      return RIG_STATUS_CONCAT_(rig_status_, __LINE__);                    \
  } while (false)

/// `RIG_ASSIGN_OR_RETURN(auto x, make())`: evaluates `expr` (a Result<T>); on error returns its Status from the
/// enclosing function, otherwise move-assigns the value to `lhs`. Usable once per line.
#define RIG_ASSIGN_OR_RETURN(lhs, expr)                                    \
  auto RIG_STATUS_CONCAT_(rig_result_, __LINE__) = (expr);                 \
  if (!RIG_STATUS_CONCAT_(rig_result_, __LINE__).ok())                     \
    return RIG_STATUS_CONCAT_(rig_result_, __LINE__).status();             \
  lhs = std::move(RIG_STATUS_CONCAT_(rig_result_, __LINE__)).value()
