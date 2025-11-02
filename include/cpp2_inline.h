// Minimal cpp2 compatibility header for inline use in generated code
// Contains only the definitions actually used by the emitter

#pragma once

#include <cstdint>
#include <type_traits>
#include <utility>
#include <iostream>

namespace cpp2 {
    using i8 = ::std::int8_t;
    using i16 = ::std::int16_t;
    using i32 = ::std::int32_t;
    using i64 = ::std::int64_t;
    using u8 = ::std::uint8_t;
    using u16 = ::std::uint16_t;
    using u32 = ::std::uint32_t;
    using u64 = ::std::uint64_t;

    // Forward declaration for implementation details
    namespace impl {
        // Parameter kind wrappers (pragmatic defaults)
        template<typename T>
        using in = ::std::add_lvalue_reference_t<::std::add_const_t<T>>;

        template<typename T>
        using copy = T;

        template<typename T>
        using move = T;

        template<typename T>
        using out = ::std::add_lvalue_reference_t<T>;

        // forward helper (perfect-forwarding alias)
        template<typename T>
        using forward = T&&;

        // simple assert-not-null helper
        template<typename P>
        inline P assert_not_null(P p) {
            using Raw = ::std::remove_reference_t<P>;
            if constexpr (::std::is_pointer_v<Raw>) {
                if (!p) {
                    ::std::cerr << "cpp2: null pointer access" << ::std::endl;
                    static ::std::remove_pointer_t<Raw> default_value{};
                    return &default_value;
                }
            }
            return p;
        }

        template<typename P>
        inline decltype(auto) deref(P p) {
            return *assert_not_null(p);
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
            return static_cast<To>(::std::forward<From>(v));
        }

        template<typename T, typename From>
        constexpr bool is(From const&) noexcept {
            // Conservative default: cannot determine dynamic types here.
            return false;
        }
    } // namespace impl

    struct contract_system {
        constexpr bool is_active() const noexcept {
            return false;
        }

        template <typename Message>
        void report_violation(Message&&) const noexcept {}
    };

    inline contract_system cpp2_default{};
    inline contract_system type_safety{};
    inline contract_system bounds_safety{};

} // namespace cpp2

// UFCS macro helper (captures function-like usage in generated code)
#ifndef CPP2_UFCS
#define CPP2_UFCS(X) (X)
#endif

inline void set_handler(cpp2::contract_system const&) noexcept {}

#ifndef CPP2_CONTRACT_MSG
#define CPP2_CONTRACT_MSG(message) (message)
#endif

using cpp2::impl::unchecked_narrow;
using cpp2::impl::unchecked_cast;
using cpp2::impl::assert_not_null;
using cpp2::impl::as_;
using cpp2::impl::is;