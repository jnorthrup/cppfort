// Minimal cpp2 compatibility header (thin stubs for emitted code)
#pragma once

#include <cstdint>
#include <string>
#include <string_view>
#include <type_traits>
#include <utility>
#include <any>

namespace cpp2 {
    using i8 = std::int8_t;
    using i16 = std::int16_t;
    using i32 = std::int32_t;
    using i64 = std::int64_t;
    using u8 = std::uint8_t;
    using u16 = std::uint16_t;
    using u32 = std::uint32_t;
    using u64 = std::uint64_t;

    // Forward declaration for implementation details
    namespace impl {
        // Parameter kind wrappers (pragmatic defaults)
        template<typename T>
        using in = std::add_lvalue_reference_t<std::add_const_t<T>>;

        template<typename T>
        using copy = T;

        template<typename T>
        using move = T;

        template<typename T>
        using out = std::add_lvalue_reference_t<T>;

        // forward helper (perfect-forwarding alias)
        template<typename T>
        using forward = T&&;

        // simple assert-not-null helper
        template<typename P>
        constexpr P assert_not_null(P p) {
            return p;
        }

        // unchecked narrowing and casting helpers
        template<typename To, typename From>
        constexpr To unchecked_narrow(From v) {
            return static_cast<To>(v);
        }

        template<typename To, typename From>
        constexpr To unchecked_cast(From v) {
            return reinterpret_cast<To>(v);
        }

        // simple type inspection / conversion helpers (very small)
        template<typename To, typename From>
        constexpr To as_(From&& v) {
            return static_cast<To>(std::forward<From>(v));
        }

        template<typename T, typename From>
        constexpr bool is(From const&) noexcept {
            // Conservative default: cannot determine dynamic types here.
            return false;
        }
    } // namespace impl

} // namespace cpp2

// UFCS macro helper (captures function-like usage in generated code)
#ifndef CPP2_UFCS
#define CPP2_UFCS(X) (X)
#endif
