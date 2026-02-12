//  cpp2_runtime.h - Minimal runtime for cppfort-generated C++ code
//  
//  Copyright 2024-2026 cppfort project
//  SPDX-License-Identifier: Apache-2.0
//
//  This is a standalone header with zero external dependencies.
//  Include this in cppfort-generated code instead of cpp2util.h.

#ifndef CPP2_RUNTIME_H
#define CPP2_RUNTIME_H

#include <string>
#include <sstream>
#include <type_traits>
#include <typeinfo>
#include <stdexcept>
#include <cassert>
#include <any>
#include <optional>
#include <variant>
#include <iomanip>
#include <vector>
#include <memory>
#include <functional>
#include <cstdio>
#include <map>
#include <set>
#include <algorithm>
#if __has_include(<span>)
#include <span>
#endif
#if __has_include(<expected>)
#include <expected>
#endif
#if __has_include(<filesystem>)
#include <filesystem>
#endif
#if __has_include(<source_location>)
#include <source_location>
#endif

namespace cpp2 {

// ============================================================================
//  String conversion for interpolation: "Hello $(name)!"
// ============================================================================

template<typename T>
auto to_string(T const& x) -> std::string {
    if constexpr (std::is_same_v<T, std::string>) {
        return x;
    }
    else if constexpr (std::is_same_v<T, const char*>) {
        return std::string(x);
    }
    else if constexpr (std::is_same_v<T, char>) {
        return std::string(1, x);
    }
    else if constexpr (std::is_same_v<T, bool>) {
        return x ? "true" : "false";
    }
    else if constexpr (std::is_arithmetic_v<T>) {
        return std::to_string(x);
    }
    else if constexpr (requires { std::declval<std::ostream&>() << x; }) {
        std::ostringstream oss;
        oss << x;
        return oss.str();
    }
    else {
        return "(non-printable)";
    }
}

// to_string with format specifier for interpolation: "(expr: fmt)$"
// Uses std::format when available, falls back to ostringstream
template<typename T>
auto to_string(T const& x, std::string_view fmt) -> std::string {
#if __cpp_lib_format >= 202110L
    return std::vformat(fmt, std::make_format_args(x));
#else
    // Fallback: ignore format spec, just convert to string
    (void)fmt;
    return to_string(x);
#endif
}

// ============================================================================
//  Type inspection: x is T
// ============================================================================

template<typename Target, typename Source>
constexpr auto is(Source const& x) -> bool {
    if constexpr (std::is_same_v<std::remove_cvref_t<Target>, std::remove_cvref_t<Source>>) {
        return true;
    }
    else if constexpr (std::is_base_of_v<Target, Source>) {
        return true;
    }
    else if constexpr (std::is_polymorphic_v<Source> && std::is_polymorphic_v<Target>) {
        return dynamic_cast<Target const*>(&x) != nullptr;
    }
    else {
        return false;
    }
}

// Value comparison: x is 42
template<typename T, typename U>
constexpr auto is(T const& x, U const& value) -> bool {
    return x == value;
}

// ============================================================================
//  std::variant specializations for is and as
// ============================================================================

// Helper to check if T is one of Ts...
template<typename T, typename... Ts>
struct is_one_of : std::disjunction<std::is_same<T, Ts>...> {};

// Helper to check if T appears exactly once in Ts...
template<typename T, typename... Ts>
struct appears_once : std::bool_constant<(... + (std::is_same_v<T, Ts> ? 1 : 0)) == 1> {};

// is<T> for variant: check if current alternative is T
// Use type-based check when T appears exactly once
template<typename T, typename... Ts>
requires (is_one_of<T, Ts...>::value && appears_once<T, Ts...>::value)
constexpr auto is(std::variant<Ts...> const& x) -> bool {
    return std::holds_alternative<T>(x);
}

// is<T> for variant with duplicate types: use visit
template<typename T, typename... Ts>
requires (is_one_of<T, Ts...>::value && !appears_once<T, Ts...>::value)
constexpr auto is(std::variant<Ts...> const& x) -> bool {
    return std::visit([](auto const& v) -> bool {
        return std::is_same_v<std::remove_cvref_t<decltype(v)>, T>;
    }, x);
}

// as<T> for variant: extract alternative T (unique case)
template<typename T, typename... Ts>
requires (is_one_of<T, Ts...>::value && appears_once<T, Ts...>::value)
constexpr auto as(std::variant<Ts...> const& x) -> T {
    if (!std::holds_alternative<T>(x)) {
        throw std::bad_variant_access();
    }
    return std::get<T>(x);
}

// as<T> for variant with duplicate types: use visit
template<typename T, typename... Ts>
requires (is_one_of<T, Ts...>::value && !appears_once<T, Ts...>::value)
constexpr auto as(std::variant<Ts...> const& x) -> T {
    return std::visit([](auto const& v) -> T {
        if constexpr (std::is_same_v<std::remove_cvref_t<decltype(v)>, T>) {
            return v;
        } else {
            throw std::bad_variant_access();
        }
    }, x);
}

template<typename T, typename... Ts>
requires (is_one_of<T, Ts...>::value && appears_once<T, Ts...>::value)
constexpr auto as(std::variant<Ts...>& x) -> T& {
    if (!std::holds_alternative<T>(x)) {
        throw std::bad_variant_access();
    }
    return std::get<T>(x);
}

// ============================================================================
//  std::any specializations for is and as
// ============================================================================

// is<T> for any: check type
template<typename T>
constexpr auto is(std::any const& x) -> bool {
    return x.type() == typeid(T);
}

// as<T> for any: extract value
template<typename T>
auto as(std::any const& x) -> T {
    if (x.type() != typeid(T)) {
        throw std::bad_any_cast();
    }
    return std::any_cast<T>(x);
}

template<typename T>
auto as(std::any& x) -> T& {
    if (x.type() != typeid(T)) {
        throw std::bad_any_cast();
    }
    return *std::any_cast<T>(&x);
}

// ============================================================================
//  std::optional specializations for is and as
// ============================================================================

// is<T> for optional: check if has value of type T
template<typename T, typename U>
constexpr auto is(std::optional<U> const& x) -> bool {
    if (!x.has_value()) {
        return false;
    }
    if constexpr (std::is_same_v<T, U>) {
        return true;
    } else {
        return false;
    }
}

// as<T> for optional: extract value - constrained for SFINAE
template<typename T, typename U>
requires (
    std::is_same_v<T, U> ||
    std::is_constructible_v<T, U> ||
    std::is_convertible_v<U, T>
)
auto as(std::optional<U> const& x) -> T {
    if (!x.has_value()) {
        throw std::bad_optional_access();
    }
    if constexpr (std::is_same_v<T, U>) {
        return x.value();
    } else if constexpr (std::is_constructible_v<T, U>) {
        return T(x.value());
    } else {
        return static_cast<T>(x.value());
    }
}

// ============================================================================
//  Type conversion: x as T
// ============================================================================

// General as<> - constrained to only match valid conversions (SFINAE-friendly)
template<typename Target, typename Source>
requires (
    std::is_same_v<Target, std::remove_cvref_t<Source>> ||
    std::is_base_of_v<Target, std::remove_cvref_t<Source>> ||
    std::is_base_of_v<std::remove_cvref_t<Source>, Target> ||
    (std::is_polymorphic_v<std::remove_cvref_t<Source>> && std::is_polymorphic_v<Target>) ||
    std::is_constructible_v<Target, Source> ||
    std::is_convertible_v<Source, Target>
)
constexpr auto as(Source const& x) -> Target {
    if constexpr (std::is_same_v<Target, std::remove_cvref_t<Source>>) {
        return x;
    }
    else if constexpr (std::is_base_of_v<Target, std::remove_cvref_t<Source>>) {
        return static_cast<Target const&>(x);
    }
    else if constexpr (std::is_polymorphic_v<std::remove_cvref_t<Source>> && std::is_polymorphic_v<Target>) {
        if (auto* p = dynamic_cast<Target const*>(&x)) {
            return *p;
        }
        throw std::bad_cast();
    }
    else if constexpr (std::is_constructible_v<Target, Source>) {
        return Target(x);
    }
    else {
        return static_cast<Target>(x);
    }
}

// ============================================================================
//  Bounds checking helpers
// ============================================================================

template<typename T>
constexpr auto bounds_check(T index, T size) -> T {
    if (index < 0 || index >= size) {
        throw std::out_of_range("bounds check failed: index " + 
            std::to_string(index) + " out of range [0, " + 
            std::to_string(size) + ")");
    }
    return index;
}

// ============================================================================
//  Contract assertion handlers
// ============================================================================

// Contract violation handler
class contract_handler {
public:
    using handler_fn = void(*)(const char* msg);

    contract_handler() = default;

    bool is_active() const {
        return handler_ != nullptr;
    }

    void set_handler(handler_fn h = nullptr) {
        handler_ = h;
    }

    void report_violation(const char* msg) const {
        if (handler_) {
            handler_(msg);
        }
    }

private:
    handler_fn handler_ = nullptr;
};

// Default contract handler (throws)
inline void default_contract_handler(const char* msg) {
    throw std::logic_error(msg);
}

// Predefined contract categories
inline contract_handler cpp2_default;
inline contract_handler type_safety;
inline contract_handler bounds_safety;
inline contract_handler null_safety;
inline contract_handler testing;

// Macro for contract messages
#define CPP2_CONTRACT_MSG(msg) msg

// ============================================================================
//  Legacy contract assertions (deprecated)
// ============================================================================

inline void contract_assert(bool condition, const char* message = "contract violation") {
    if (!condition) {
        throw std::logic_error(message);
    }
}

// Pre/post condition macros (can be disabled with CPP2_CONTRACTS_OFF)
#ifndef CPP2_CONTRACTS_OFF
    #define cpp2_pre(cond)  cpp2::contract_assert(cond, "precondition failed: " #cond)
    #define cpp2_post(cond) cpp2::contract_assert(cond, "postcondition failed: " #cond)
#else
    #define cpp2_pre(cond)  ((void)0)
    #define cpp2_post(cond) ((void)0)
#endif

// ============================================================================
//  Discard placeholder
// ============================================================================

struct discard_t {
    template<typename T>
    constexpr discard_t& operator=(T&&) noexcept { return *this; }
};

inline constexpr discard_t _ {};

// ============================================================================
//  impl namespace for internal helpers (backward compat with codegen)
// ============================================================================

namespace impl {
    template<typename T, typename U>
    constexpr auto is_(U const& x) -> bool { return cpp2::is<T>(x); }
    
    template<typename T, typename U>
    constexpr auto as_(U const& x) -> T { return cpp2::as<T>(x); }
}

// ============================================================================
//  UFCS (Unified Function Call Syntax) support
//  Transforms obj.func(args...) to func(obj, args...) for free functions
// ============================================================================

namespace detail {
    // Helper to detect if member function exists
    template<typename T, typename = void>
    struct has_member_impl : std::false_type {};
    
    // SFINAE helper for member function detection
    template<typename... Args>
    struct always_false : std::false_type {};
}

// CPP2_UFCS macro - try member call first, then free function
// Usage: CPP2_UFCS(funcname)(obj, args...)
// This expands to a lambda that tries member then free function
#define CPP2_UFCS(FUNCNAME) \
    [](auto&& obj, auto&&... args) -> decltype(auto) { \
        if constexpr (requires { std::forward<decltype(obj)>(obj).FUNCNAME(std::forward<decltype(args)>(args)...); }) { \
            return std::forward<decltype(obj)>(obj).FUNCNAME(std::forward<decltype(args)>(args)...); \
        } else { \
            return FUNCNAME(std::forward<decltype(obj)>(obj), std::forward<decltype(args)>(args)...); \
        } \
    }

// Nonmember UFCS - always calls free function (for cases where we know it's not a member)
#define CPP2_UFCS_NONMEMBER(FUNCNAME) \
    [](auto&& obj, auto&&... args) -> decltype(auto) { \
        return FUNCNAME(std::forward<decltype(obj)>(obj), std::forward<decltype(args)>(args)...); \
    }

// Nonlocal UFCS - same as CPP2_UFCS but usable in non-local (namespace) scope
// In C++23, generic lambdas work fine in non-local contexts
#define CPP2_UFCS_NONLOCAL(FUNCNAME) CPP2_UFCS(FUNCNAME)

// ============================================================================
//  Forward references and type deduction helpers
// ============================================================================

// CPP2_FORWARD - perfect forwarding macro for forward parameters
#define CPP2_FORWARD(x) std::forward<decltype(x)>(x)

// CPP2_TYPEOF - get the unqualified type of an expression
#define CPP2_TYPEOF(x) std::remove_cvref_t<decltype(x)>

// ============================================================================
//  main() argument handling: main(args)
// ============================================================================

struct args_t {
    int         argc;
    char const* const* argv;
};

inline auto make_args(int argc, char** argv) -> args_t {
    return { argc, const_cast<char const* const*>(argv) };
}

// ============================================================================
//  Narrowing conversions
// ============================================================================

template<typename To, typename From>
constexpr auto unchecked_narrow(From from) noexcept -> To {
    return static_cast<To>(from);
}

template<typename To, typename From>
constexpr auto narrow(From from) -> To {
    auto result = static_cast<To>(from);
    if (static_cast<From>(result) != from) {
        throw std::runtime_error("narrowing conversion failed");
    }
    return result;
}

template<typename To, typename From>
auto unchecked_cast(From from) noexcept -> To {
    return reinterpret_cast<To>(from);
}

// ============================================================================
//  RAII wrappers for C stdio
// ============================================================================

inline auto fopen(const char* filename, const char* mode) -> std::FILE* {
    return std::fopen(filename, mode);
}

// ============================================================================
//  Smart pointer construction: unique.new<T>(args)
//  In generated code, this becomes cpp2::unique_new<T>(args)
// ============================================================================

template<typename T, typename... Args>
auto unique_new(Args&&... args) -> std::unique_ptr<T> {
    return std::make_unique<T>(std::forward<Args>(args)...);
}

template<typename T, typename... Args>
auto shared_new(Args&&... args) -> std::shared_ptr<T> {
    return std::make_shared<T>(std::forward<Args>(args)...);
}

// ============================================================================
//  Cpp2 type aliases (integer, floating point)
// ============================================================================

using i8  = std::int8_t;
using i16 = std::int16_t;
using i32 = std::int32_t;
using i64 = std::int64_t;
using u8  = std::uint8_t;
using u16 = std::uint16_t;
using u32 = std::uint32_t;
using u64 = std::uint64_t;
using f32 = float;
using f64 = double;

} // namespace cpp2

#endif // CPP2_RUNTIME_H
