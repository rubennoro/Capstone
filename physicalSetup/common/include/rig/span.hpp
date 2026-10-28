// SPDX-License-Identifier: MIT
/// @file rig/span.hpp
/// @brief Minimal non-owning contiguous view (C++17 stand-in for std::span). FROZEN CONTRACT (T0).
///
/// Ownership: never owns; the viewed storage must outlive the Span. Threading: same as the viewed storage.
#pragma once

#include <array>
#include <cstddef>
#include <type_traits>
#include <utility>

namespace rig {

/// A pointer + element count. `Span<const T>` is the read-only view; `Span<T>` converts to it implicitly.
template <class T>
class Span {
 public:
  using element_type = T;
  using value_type = std::remove_cv_t<T>;
  using iterator = T*;

  constexpr Span() noexcept = default;
  constexpr Span(T* data, std::size_t size) noexcept : data_(data), size_(size) {}

  /// From any lvalue container with contiguous `data()` / `size()` (std::vector, std::array, std::string, Span<U>).
  template <class C,
            class = std::enable_if_t<!std::is_same_v<std::decay_t<C>, Span> &&
                                     std::is_convertible_v<decltype(std::declval<C&>().data()), T*>>>
  constexpr Span(C& c) noexcept  // NOLINT(google-explicit-constructor)
      : data_(c.data()), size_(static_cast<std::size_t>(c.size())) {}

  /// From a C array.
  template <std::size_t N>
  constexpr Span(T (&arr)[N]) noexcept : data_(arr), size_(N) {}  // NOLINT(google-explicit-constructor)

  constexpr T* data() const noexcept { return data_; }
  constexpr std::size_t size() const noexcept { return size_; }
  constexpr std::size_t size_bytes() const noexcept { return size_ * sizeof(T); }
  constexpr bool empty() const noexcept { return size_ == 0; }
  constexpr T* begin() const noexcept { return data_; }
  constexpr T* end() const noexcept { return data_ + size_; }
  /// @pre i < size() (unchecked).
  constexpr T& operator[](std::size_t i) const noexcept { return data_[i]; }

  /// Elements [offset, offset + count), clamped to the end. `count` = npos means "to the end".
  static constexpr std::size_t npos = static_cast<std::size_t>(-1);
  constexpr Span subspan(std::size_t offset, std::size_t count = npos) const noexcept {
    if (offset > size_) offset = size_;
    const std::size_t rest = size_ - offset;
    return Span(data_ + offset, count > rest ? rest : count);
  }

 private:
  T* data_ = nullptr;
  std::size_t size_ = 0;
};

}  // namespace rig
