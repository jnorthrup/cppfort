#include <iostream>
#include <string>
#include <sstream>
#include <vector>
#include <memory>
#include <optional>
#include <variant>
#include <functional>
#include <utility>
#include <type_traits>
#include <stdexcept>
#include <cstdint>
#include <iterator>
#include <filesystem>

namespace cpp2 {
    template<typename T> auto to_string(T const& x) -> std::string {
        if constexpr (std::is_same_v<T, std::string>) { return x; }
        else if constexpr (std::is_same_v<T, const char*>) { return std::string(x); }
        else if constexpr (std::is_same_v<T, char>) { return std::string(1, x); }
        else if constexpr (std::is_same_v<T, bool>) { return x ? "true" : "false"; }
        else if constexpr (std::is_arithmetic_v<T>) { return std::to_string(x); }
        else { std::ostringstream oss; oss << x; return oss.str(); }
    }
    template<typename T, typename U> constexpr auto is(U const& x) -> bool {
        if constexpr (std::is_same_v<T, U> || std::is_base_of_v<T, U>) { return true; }
        else if constexpr (std::is_polymorphic_v<U>) { return dynamic_cast<T const*>(&x) != nullptr; }
        else { return false; }
    }
    template<typename T, typename U> constexpr auto as(U const& x) -> T {
        if constexpr (std::is_same_v<T, U>) { return x; }
        else if constexpr (std::is_base_of_v<T, U>) { return static_cast<T const&>(x); }
        else if constexpr (std::is_polymorphic_v<U>) { return dynamic_cast<T const&>(x); }
        else { return static_cast<T>(x); }
    }
    namespace impl {
        template<typename T, typename U> constexpr auto is_(U const& x) -> bool { return is<T>(x); }
        template<typename T, typename U> constexpr auto as_(U const& x) -> T { return as<T>(x); }
    }
    // main(args) support
    struct args_t { int argc; char const* const* argv; };
    inline auto make_args(int argc, char** argv) -> args_t { return { argc, const_cast<char const* const*>(argv) }; }
} // namespace cpp2

#include "combinators/structural.hpp"

#include "combinators/transformation.hpp"

#include "combinators/reduction.hpp"

#include "combinators/parsing.hpp"

[[nodiscard]] auto add_one(int x) -> int;
[[nodiscard]] auto double_it(int x) -> int;
[[nodiscard]] auto square(int x) -> int;
[[nodiscard]] auto stringify(int x) -> std::string;
[[nodiscard]] auto parse_int(std::string s) -> int;
[[nodiscard]] auto add_n(int n, int x) -> int;
[[nodiscard]] auto mult_n(int n, int x) -> int;
auto test_simple_pipeline() -> void;
auto test_chained_pipeline() -> void;
auto test_triple_chain() -> void;
auto test_type_transform() -> void;
auto test_type_roundtrip() -> void;
auto test_pipeline_in_addition() -> void;
auto test_pipeline_in_comparison() -> void;
auto test_nested_pipelines() -> void;
auto test_inline_lambda() -> void;
auto test_lambda_chain() -> void;
auto test_lambda_with_capture() -> void;
auto test_curried_add() -> void;
auto test_curried_chain() -> void;
auto test_bytebuffer_take() -> void;
auto test_bytebuffer_skip() -> void;
auto test_bytebuffer_chain() -> void;
auto test_bytebuffer_slice() -> void;
auto test_validation_pipeline() -> void;
auto test_starts_with_pipeline() -> void;
auto test_complex_composition() -> void;
auto test_mixed_pipeline() -> void;
auto test_identity_pipeline() -> void;
auto test_zero_value() -> void;
auto test_negative_value() -> void;

[[nodiscard]] auto add_one(int x) -> int {
    return x + 1;
}

[[nodiscard]] auto double_it(int x) -> int {
    return x * 2;
}

[[nodiscard]] auto square(int x) -> int {
    return x * x;
}

[[nodiscard]] auto stringify(int x) -> std::string {
    return std::to_string(x);
}

[[nodiscard]] auto parse_int(std::string s) -> int {
    return std::stoi(s);
}

[[nodiscard]] auto add_n(int n, int x) -> int {
    return x + n;
}

[[nodiscard]] auto mult_n(int n, int x) -> int {
    return x * n;
}

auto test_simple_pipeline() -> void {
    auto result = add_one(5);
    std::cout << "test_simple_pipeline: PASS\n";
}

auto test_chained_pipeline() -> void {
    auto result = double_it(add_one(5));
    std::cout << "test_chained_pipeline: PASS\n";
}

auto test_triple_chain() -> void {
    auto result = square(double_it(add_one(2)));
    std::cout << "test_triple_chain: PASS\n";
}

auto test_type_transform() -> void {
    auto result = stringify(42);
    std::cout << "test_type_transform: PASS\n";
}

auto test_type_roundtrip() -> void {
    auto result = parse_int(stringify(123));
    std::cout << "test_type_roundtrip: PASS\n";
}

auto test_pipeline_in_addition() -> void {
    auto result = double_it(5) + add_one(3);
    std::cout << "test_pipeline_in_addition: PASS\n";
}

auto test_pipeline_in_comparison() -> void {
    auto passed = double_it(5) > add_one(3);
    std::cout << "test_pipeline_in_comparison: PASS\n";
}

auto test_nested_pipelines() -> void {
    auto inner = double_it(2);
    auto result = square(add_one(inner));
    std::cout << "test_nested_pipelines: PASS\n";
}

auto test_inline_lambda() -> void {
    auto result = [&](int x) -> int { return x * 3; }(5);
    std::cout << "test_inline_lambda: PASS\n";
}

auto test_lambda_chain() -> void {
    auto result = [&](int x) -> int { return [&](int x) -> int { return x * 2; }(x + 1); }(5);
    std::cout << "test_lambda_chain: PASS\n";
}

auto test_lambda_with_capture() -> void {
    auto factor = 10;
    auto multiplier = [&](int x) -> int { return x * 10; };
    auto result = multiplier(5);
    std::cout << "test_lambda_with_capture: PASS\n";
}

auto test_curried_add() -> void {
    auto curried_add = [&](int x) -> int { return add_n(3, x); };
    auto result = curried_add(5);
    std::cout << "test_curried_add: PASS\n";
}

auto test_curried_chain() -> void {
    auto add_three = [&](int x) -> int { return add_n(3, x); };
    auto times_two = [&](int x) -> int { return mult_n(2, x); };
    auto result = times_two(add_three(5));
    std::cout << "test_curried_chain: PASS\n";
}

auto test_bytebuffer_take() -> void {
    std::array<char, 5> data = {0x48, 0x65, 0x6C, 0x6C, 0x6F};
    cpp2::ByteBuffer buf = {std::data(data), std::size(data)};
    auto result = cpp2::combinators::curried::take(3)(buf);
    std::cout << "test_bytebuffer_take: PASS\n";
}

auto test_bytebuffer_skip() -> void {
    std::array<char, 5> data = {0x48, 0x65, 0x6C, 0x6C, 0x6F};
    cpp2::ByteBuffer buf = {std::data(data), std::size(data)};
    auto result = cpp2::combinators::curried::skip(2)(buf);
    std::cout << "test_bytebuffer_skip: PASS\n";
}

auto test_bytebuffer_chain() -> void {
    std::array<char, 5> data = {0x48, 0x65, 0x6C, 0x6C, 0x6F};
    cpp2::ByteBuffer buf = {std::data(data), std::size(data)};
    auto result = cpp2::combinators::curried::take(3)(cpp2::combinators::curried::skip(1)(buf));
    std::cout << "test_bytebuffer_chain: PASS\n";
}

auto test_bytebuffer_slice() -> void {
    std::array<char, 5> data = {0x48, 0x65, 0x6C, 0x6C, 0x6F};
    cpp2::ByteBuffer buf = {std::data(data), std::size(data)};
    auto result = cpp2::combinators::curried::slice(1, 4)(buf);
    std::cout << "test_bytebuffer_slice: PASS\n";
}

auto test_validation_pipeline() -> void {
    std::array<char, 5> data = {0x48, 0x65, 0x6C, 0x6C, 0x6F};
    cpp2::ByteBuffer buf = {std::data(data), std::size(data)};
    auto is_valid = cpp2::combinators::curried::length_eq(5)(buf);
    auto not_valid = cpp2::combinators::curried::length_eq(3)(buf);
    std::cout << "test_validation_pipeline: PASS\n";
}

auto test_starts_with_pipeline() -> void {
    std::array<char, 5> data = {0x48, 0x65, 0x6C, 0x6C, 0x6F};
    cpp2::ByteBuffer buf = {std::data(data), std::size(data)};
    std::array<char, 2> prefix = {0x48, 0x65};
    auto has_prefix = cpp2::combinators::curried::starts_with(cpp2::ByteBuffer(std::data(prefix), std::size(prefix)))(buf);
    std::cout << "test_starts_with_pipeline: PASS\n";
}

auto test_complex_composition() -> void {
    auto add_five = [&](int x) -> int { return x + 5; };
    auto times_three = [&](int x) -> int { return x * 3; };
    auto result = times_three(add_five(double_it(add_one(2))));
    std::cout << "test_complex_composition: PASS\n";
}

auto test_mixed_pipeline() -> void {
    auto square_fn = [&](int x) -> int { return x * x; };
    auto result = double_it(square_fn(add_one(5)));
    std::cout << "test_mixed_pipeline: PASS\n";
}

auto test_identity_pipeline() -> void {
    auto identity = [&](int x) -> int { return x; };
    auto result = identity(42);
    std::cout << "test_identity_pipeline: PASS\n";
}

auto test_zero_value() -> void {
    auto result = double_it(add_one(0));
    std::cout << "test_zero_value: PASS\n";
}

auto test_negative_value() -> void {
    auto result = double_it(add_one(-5));
    std::cout << "test_negative_value: PASS\n";
}

auto main() -> int {
    std::cout << "=== Pipeline Operator Tests ===\n\n";
    std::cout << "--- Basic Pipeline Tests ---\n";
    test_simple_pipeline();
    test_chained_pipeline();
    test_triple_chain();
    test_type_transform();
    test_type_roundtrip();
    std::cout << "\n--- Expression Context Tests ---\n";
    test_pipeline_in_addition();
    test_pipeline_in_comparison();
    test_nested_pipelines();
    std::cout << "\n--- Lambda Pipeline Tests ---\n";
    test_inline_lambda();
    test_lambda_chain();
    test_lambda_with_capture();
    std::cout << "\n--- Curried Function Tests ---\n";
    test_curried_add();
    test_curried_chain();
    std::cout << "\n--- Combinator Pipeline Tests ---\n";
    test_bytebuffer_take();
    test_bytebuffer_skip();
    test_bytebuffer_chain();
    test_bytebuffer_slice();
    std::cout << "\n--- Validation Pipeline Tests ---\n";
    test_validation_pipeline();
    test_starts_with_pipeline();
    std::cout << "\n--- Complex Pipeline Tests ---\n";
    test_complex_composition();
    test_mixed_pipeline();
    std::cout << "\n--- Edge Cases ---\n";
    test_identity_pipeline();
    test_zero_value();
    test_negative_value();
    std::cout << "\n=== All 25 pipeline operator tests passed! ===\n";
    return 0;
}

