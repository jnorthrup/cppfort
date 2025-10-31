#pragma once

#include <optional>
#include <type_traits>
#include <utility>

#include "src/stage0/cpp2_inline.h"

namespace cpp2 {

// Provide light-weight wrappers that mirror the cppfront runtime API.
// These helpers exist purely to make generated code compile for tests.

template <typename T>
constexpr auto move(T&& value) noexcept -> decltype(auto) {
    return std::move(value);
}

template <typename T>
constexpr auto forward(T&& value) noexcept -> decltype(auto) {
    return std::forward<T>(value);
}

template <typename T>
constexpr auto copy(T const& value) -> T {
    return value;
}

template <typename T>
constexpr auto make_optional(T&& value) {
    return std::optional<std::decay_t<T>>(std::forward<T>(value));
}

struct discard_t {
    template <typename T>
    constexpr discard_t& operator=(T&&) noexcept {
        return *this;
    }
};

inline discard_t _{};

} // namespace cpp2

namespace cpp2::impl {

template <typename L, typename R>
constexpr bool cmp_less(L&& lhs, R&& rhs) noexcept {
    return std::forward<L>(lhs) < std::forward<R>(rhs);
}

template <typename L, typename R>
constexpr bool cmp_less_eq(L&& lhs, R&& rhs) noexcept {
    return std::forward<L>(lhs) <= std::forward<R>(rhs);
}

template <typename L, typename R>
constexpr bool cmp_greater(L&& lhs, R&& rhs) noexcept {
    return std::forward<L>(lhs) > std::forward<R>(rhs);
}

template <typename L, typename R>
constexpr bool cmp_greater_eq(L&& lhs, R&& rhs) noexcept {
    return std::forward<L>(lhs) >= std::forward<R>(rhs);
}

template <typename L, typename R>
constexpr bool cmp_equal(L&& lhs, R&& rhs) noexcept {
    return std::forward<L>(lhs) == std::forward<R>(rhs);
}

template <typename L, typename R>
constexpr bool cmp_not_equal(L&& lhs, R&& rhs) noexcept {
    return std::forward<L>(lhs) != std::forward<R>(rhs);
}

} // namespace cpp2::impl

inline auto& _ = cpp2::_;
