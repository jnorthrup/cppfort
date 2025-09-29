// Additional minimal implementation helpers for emitted code
#pragma once

#include <utility>
#include <type_traits>
#include <string>

namespace cpp2::impl {

    template<typename T>
    constexpr T&& move(T& v) noexcept {
        return static_cast<T&&>(v);
    }

    // Basic is<> for std::any and std::string_view wrappers could be added later.
    template<typename T>
    constexpr bool is_any(const std::any&) noexcept { return false; }

    // as_ already defined in cpp2.h; provide an alias name used in tests
    template<typename To, typename From>
    constexpr To as(From&& v) {
        return as_<To>(std::forward<From>(v));
    }

} // namespace cpp2::impl
