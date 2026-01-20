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
//  Type conversion: x as T
// ============================================================================

template<typename Target, typename Source>
constexpr auto as(Source const& x) -> Target {
    if constexpr (std::is_same_v<Target, Source>) {
        return x;
    }
    else if constexpr (std::is_base_of_v<Target, Source>) {
        return static_cast<Target const&>(x);
    }
    else if constexpr (std::is_polymorphic_v<Source> && std::is_polymorphic_v<Target>) {
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

} // namespace cpp2

#endif // CPP2_RUNTIME_H
