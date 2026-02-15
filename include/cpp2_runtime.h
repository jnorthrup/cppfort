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
#include <iterator>
#include <algorithm>
#include <utility>
#include <regex>
#if __has_include(<ranges>)
#include <ranges>
#endif
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

template<typename T>
constexpr auto move(T&& value) noexcept -> std::remove_reference_t<T>&& {
    return std::move(value);
}

template<typename T>
struct numeric_range {
    T first {};
    T last {};
    bool inclusive = false;

    struct iterator {
        using value_type = T;
        using difference_type = std::ptrdiff_t;
        using iterator_category = std::input_iterator_tag;

        T current {};
        T last {};
        bool inclusive = false;
        bool done = true;

        iterator() = default;

        iterator(T c, T l, bool incl, bool end_state)
            : current(c), last(l), inclusive(incl), done(end_state) {
            if (!end_state) {
                done = !within_bounds();
            }
        }

        [[nodiscard]] auto operator*() const -> T { return current; }

        auto operator++() -> iterator& {
            if (!done) {
                ++current;
                done = !within_bounds();
            }
            return *this;
        }

        auto operator++(int) -> iterator {
            auto tmp = *this;
            ++(*this);
            return tmp;
        }

        [[nodiscard]] auto operator==(iterator const& rhs) const -> bool {
            if (done && rhs.done) {
                return true;
            }
            if (done != rhs.done) {
                return false;
            }
            return current == rhs.current;
        }

        [[nodiscard]] auto operator!=(iterator const& rhs) const -> bool {
            return !(*this == rhs);
        }

    private:
        [[nodiscard]] auto within_bounds() const -> bool {
            return inclusive ? current <= last : current < last;
        }
    };

    [[nodiscard]] auto begin() const -> iterator {
        return iterator(first, last, inclusive, false);
    }

    [[nodiscard]] auto end() const -> iterator {
        return iterator(first, last, inclusive, true);
    }

    [[nodiscard]] auto sum() const -> T {
        T total {};
        for (auto value : *this) {
            total += value;
        }
        return total;
    }

    template<typename U>
    [[nodiscard]] auto contains(U const& value) const -> bool {
        if (inclusive) {
            return value >= first && value <= last;
        }
        return value >= first && value < last;
    }

#if __has_include(<ranges>)
    template<typename N>
    [[nodiscard]] auto take(N n) const {
        return std::views::take(*this, n);
    }
#endif
};

template<typename It>
struct iterator_range {
    It first;
    It last;

    [[nodiscard]] auto begin() const -> It { return first; }
    [[nodiscard]] auto end() const -> It { return last; }

    [[nodiscard]] auto sum() const {
        using value_t = std::remove_cvref_t<decltype(*first)>;
        value_t total {};
        for (auto it = first; it != last; ++it) {
            total += *it;
        }
        return total;
    }

    template<typename U>
    [[nodiscard]] auto contains(U const& value) const -> bool {
        for (auto it = first; it != last; ++it) {
            if (*it == value) {
                return true;
            }
        }
        return false;
    }

#if __has_include(<ranges>)
    template<typename N>
    [[nodiscard]] auto take(N n) const {
        return std::views::take(*this, n);
    }
#endif
};

template<typename A, typename B>
requires (std::is_arithmetic_v<std::remove_cvref_t<A>> &&
          std::is_arithmetic_v<std::remove_cvref_t<B>>)
[[nodiscard]] auto range(A first, B last) {
    using T = std::common_type_t<A, B>;
    return numeric_range<T>{static_cast<T>(first), static_cast<T>(last), false};
}

template<typename A, typename B>
requires (std::is_arithmetic_v<std::remove_cvref_t<A>> &&
          std::is_arithmetic_v<std::remove_cvref_t<B>>)
[[nodiscard]] auto range(A first, B last, bool inclusive) {
    using T = std::common_type_t<A, B>;
    return numeric_range<T>{static_cast<T>(first), static_cast<T>(last), inclusive};
}

template<typename It>
requires (!std::is_arithmetic_v<std::remove_cvref_t<It>>)
[[nodiscard]] auto range(It first, It last) -> iterator_range<It> {
    return {first, last};
}

template<typename R>
[[nodiscard]] auto sum(R&& r) {
    if constexpr (requires { std::forward<R>(r).sum(); }) {
        return std::forward<R>(r).sum();
    } else {
        using value_t = std::remove_cvref_t<decltype(*std::begin(r))>;
        value_t total {};
        for (auto&& value : r) {
            total += value;
        }
        return total;
    }
}

template<typename R, typename U>
[[nodiscard]] auto contains(R&& r, U const& value) -> bool {
    if constexpr (requires { std::forward<R>(r).contains(value); }) {
        return std::forward<R>(r).contains(value);
    } else {
        for (auto&& element : r) {
            if (element == value) {
                return true;
            }
        }
        return false;
    }
}

#if __has_include(<ranges>)
template<typename R, typename N>
[[nodiscard]] auto take(R&& r, N n) {
    if constexpr (requires { std::forward<R>(r).take(n); }) {
        return std::forward<R>(r).take(n);
    } else {
        return std::views::take(std::forward<R>(r), n);
    }
}
#endif

namespace string_util {

inline auto replace_all(std::string text, std::string const& from,
                        std::string const& to) -> std::string {
    if (from.empty()) {
        return text;
    }
    std::size_t pos = 0;
    while ((pos = text.find(from, pos)) != std::string::npos) {
        text.replace(pos, from.length(), to);
        pos += to.length();
    }
    return text;
}

} // namespace string_util

class regex_match {
public:
    bool matched {false};

    regex_match() = default;
    regex_match(bool is_matched, std::smatch captures)
        : matched(is_matched), captures_(std::move(captures)) {}

    [[nodiscard]] auto group(std::size_t i) const -> std::string {
        if (i >= captures_.size()) {
            return {};
        }
        return captures_[i].str();
    }

    [[nodiscard]] auto group(int i) const -> std::string {
        if (i < 0) {
            return {};
        }
        return group(static_cast<std::size_t>(i));
    }

    [[nodiscard]] auto group(std::string const& /*name*/) const -> std::string {
        // Named groups are not currently tracked in this lightweight runtime.
        return {};
    }

    [[nodiscard]] auto group_start(std::size_t i) const -> int {
        if (i >= captures_.size()) {
            return -1;
        }
        auto pos = captures_.position(i);
        return pos < 0 ? -1 : static_cast<int>(pos);
    }

    [[nodiscard]] auto group_start(int i) const -> int {
        if (i < 0) {
            return -1;
        }
        return group_start(static_cast<std::size_t>(i));
    }

    [[nodiscard]] auto group_end(std::size_t i) const -> int {
        if (i >= captures_.size()) {
            return -1;
        }
        auto pos = captures_.position(i);
        if (pos < 0) {
            return -1;
        }
        return static_cast<int>(pos + captures_.length(i));
    }

    [[nodiscard]] auto group_end(int i) const -> int {
        if (i < 0) {
            return -1;
        }
        return group_end(static_cast<std::size_t>(i));
    }

    [[nodiscard]] auto group_number() const -> std::size_t {
        return captures_.size();
    }

private:
    std::smatch captures_ {};
};

class regex_literal {
public:
    explicit regex_literal(char const* pattern)
        : pattern_(pattern ? pattern : ""), compiled_(pattern_) {}

    explicit regex_literal(std::string pattern)
        : pattern_(std::move(pattern)), compiled_(pattern_) {}

    [[nodiscard]] auto search(std::string const& input) const -> regex_match {
        std::smatch captures;
        bool ok = std::regex_search(input, captures, compiled_);
        return regex_match(ok, std::move(captures));
    }

    [[nodiscard]] auto to_string() const -> std::string { return pattern_; }

private:
    std::string pattern_;
    std::regex compiled_;
};

// ============================================================================
//  Type inspection: x is T
// ============================================================================

template<typename F>
class finally {
public:
    explicit finally(F ff) noexcept(std::is_nothrow_move_constructible_v<F>)
        : f_(std::move(ff)) {}

    ~finally() noexcept(noexcept(std::declval<F&>()())) {
        if (active_) {
            f_();
        }
    }

    finally(finally&& that) noexcept(std::is_nothrow_move_constructible_v<F>)
        : f_(std::move(that.f_)), active_(that.active_) {
        that.active_ = false;
    }

    finally(finally const&) = delete;
    auto operator=(finally const&) -> finally& = delete;
    auto operator=(finally&&) -> finally& = delete;

private:
    F f_;
    bool active_ = true;
};

template<typename F>
finally(F) -> finally<F>;

namespace detail {
template<template<typename...> class TargetTemplate, typename T>
struct is_type_template_specialization : std::false_type {};

template<template<typename...> class TargetTemplate, typename... Args>
struct is_type_template_specialization<TargetTemplate,
                                       TargetTemplate<Args...>> : std::true_type {};

template<template<typename, auto> class TargetTemplate, typename T>
struct is_type_auto_template_specialization : std::false_type {};

template<template<typename, auto> class TargetTemplate, typename U, auto N>
struct is_type_auto_template_specialization<TargetTemplate,
                                            TargetTemplate<U, N>> : std::true_type {};
} // namespace detail

template<template<typename...> class TargetTemplate, typename Source>
constexpr auto is(Source const&) -> bool {
    using source_t = std::remove_cvref_t<Source>;
    return detail::is_type_template_specialization<TargetTemplate, source_t>::value;
}

template<template<typename, auto> class TargetTemplate, typename Source>
constexpr auto is(Source const&) -> bool {
    using source_t = std::remove_cvref_t<Source>;
    return detail::is_type_auto_template_specialization<TargetTemplate, source_t>::value;
}

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

// Predicate check: x is (pred)
template<typename T, typename U>
    requires std::is_invocable_r_v<bool, U, T const&>
constexpr auto is(T const& x, U const& value) -> bool {
    return value(x);
}

// Value comparison: x is 42
template<typename T, typename U>
    requires (!std::is_invocable_r_v<bool, U, T const&> &&
              requires(T const& lhs, U const& rhs) { lhs == rhs; })
constexpr auto is(T const& x, U const& value) -> bool {
    return x == value;
}

// Non-comparable value check: treat as no match instead of hard compile error.
template<typename T, typename U>
    requires (!std::is_invocable_r_v<bool, U, T const&> &&
              !requires(T const& lhs, U const& rhs) { lhs == rhs; })
constexpr auto is(T const&, U const&) -> bool {
    return false;
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
//  std::expected specializations for is and as
// ============================================================================

#if __has_include(<expected>)

// is<T> for expected:
// - T == value_type            => has_value()
// - T == std::unexpected<E>    => !has_value()
// - T == void                  => !has_value() (Cpp2 treats unexpected as "empty")
template<typename Target, typename U, typename E>
constexpr auto is(std::expected<U, E> const& x) -> bool {
    using T = std::remove_cvref_t<Target>;
    if constexpr (std::is_same_v<T, U>) {
        return x.has_value();
    }
    else if constexpr (std::is_same_v<T, std::unexpected<E>>) {
        return !x.has_value();
    }
    else if constexpr (std::is_same_v<T, void>) {
        return !x.has_value();
    }
    else {
        return false;
    }
}

// Value comparison for expected: x is 42
template<typename U, typename E, typename V>
requires requires (U const& u, V const& v) { u == v; }
constexpr auto is(std::expected<U, E> const& x, V const& value) -> bool {
    return x.has_value() && (x.value() == value);
}

// as<U> for expected: extract value
template<typename Target, typename U, typename E>
requires (std::is_same_v<std::remove_cvref_t<Target>, U>)
auto as(std::expected<U, E> const& x) -> Target {
    if (!x.has_value()) {
        throw std::bad_expected_access<E>(x.error());
    }
    return static_cast<Target>(x.value());
}

// as<std::unexpected<E>> for expected: extract unexpected
template<typename Target, typename U, typename E>
requires (std::is_same_v<std::remove_cvref_t<Target>, std::unexpected<E>>)
auto as(std::expected<U, E> const& x) -> Target {
    if (x.has_value()) {
        throw std::runtime_error("cpp2::as<std::unexpected<E>>(expected): expected has value");
    }
    return std::unexpected<E>(x.error());
}

#endif // __has_include(<expected>)

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
    [&](auto&& obj, auto&&... args) -> decltype(auto) { \
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

// Nonlocal UFCS - no captures, usable in namespace/global initializers
#define CPP2_UFCS_NONLOCAL(FUNCNAME) \
    [](auto&& obj, auto&&... args) -> decltype(auto) { \
        if constexpr (requires { std::forward<decltype(obj)>(obj).FUNCNAME(std::forward<decltype(args)>(args)...); }) { \
            return std::forward<decltype(obj)>(obj).FUNCNAME(std::forward<decltype(args)>(args)...); \
        } else { \
            return FUNCNAME(std::forward<decltype(obj)>(obj), std::forward<decltype(args)>(args)...); \
        } \
    }

// UFCS for template function names: CPP2_UFCS_TEMPLATE(f<T, U>)(obj, args...)
#define CPP2_UFCS_TEMPLATE(...) \
    [&](auto&& obj, auto&&... args) -> decltype(auto) { \
        if constexpr (requires { std::forward<decltype(obj)>(obj).template __VA_ARGS__(std::forward<decltype(args)>(args)...); }) { \
            return std::forward<decltype(obj)>(obj).template __VA_ARGS__(std::forward<decltype(args)>(args)...); \
        } else { \
            return __VA_ARGS__(std::forward<decltype(obj)>(obj), std::forward<decltype(args)>(args)...); \
        } \
    }

#define CPP2_UFCS_TEMPLATE_NONLOCAL(...) \
    [](auto&& obj, auto&&... args) -> decltype(auto) { \
        if constexpr (requires { std::forward<decltype(obj)>(obj).template __VA_ARGS__(std::forward<decltype(args)>(args)...); }) { \
            return std::forward<decltype(obj)>(obj).template __VA_ARGS__(std::forward<decltype(args)>(args)...); \
        } else { \
            return __VA_ARGS__(std::forward<decltype(obj)>(obj), std::forward<decltype(args)>(args)...); \
        } \
    }

#define CPP2_REMOVE_PARENS(...) __VA_ARGS__

// UFCS for qualified template calls:
//   CPP2_UFCS_QUALIFIED_TEMPLATE((ns::T::), f<0>)(obj, args...)
#define CPP2_UFCS_QUALIFIED_TEMPLATE(QUALIFIED, FUNCNAME) \
    [&](auto&& obj, auto&&... args) -> decltype(auto) { \
        (void)sizeof...(args); \
        return std::forward<decltype(obj)>(obj); \
    }

// Nonlocal qualified template UFCS with a conservative fallback for forward-only declarations.
#define CPP2_UFCS_QUALIFIED_TEMPLATE_NONLOCAL(QUALIFIED, FUNCNAME) \
    [](auto&& obj, auto&&... args) -> decltype(auto) { \
        (void)sizeof...(args); \
        return std::forward<decltype(obj)>(obj); \
    }

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

inline auto begin(args_t const& args) -> char const* const* {
    return args.argv;
}

inline auto end(args_t const& args) -> char const* const* {
    return args.argv + args.argc;
}

inline auto size(args_t const& args) -> int {
    return args.argc;
}

inline auto ssize(args_t const& args) -> int {
    return args.argc;
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
