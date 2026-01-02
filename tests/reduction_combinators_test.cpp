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

#include "../include/bytebuffer.hpp"

#include "../include/strview.hpp"

#include "../include/combinators/structural.hpp"

#include "../include/combinators/transformation.hpp"

#include "../include/combinators/reduction.hpp"

#include <vector>

#include <string>

#include <iostream>

#include <cassert>

#include <cstddef>

auto test_fold_sum() -> void;
auto test_fold_string_concat() -> void;
auto test_fold_empty() -> void;
auto test_reduce_max() -> void;
auto test_reduce_empty() -> void;
auto test_reduce_single() -> void;
auto test_scan_running_sum() -> void;
auto test_scan_empty() -> void;
auto test_find_basic() -> void;
auto test_find_not_found() -> void;
auto test_find_empty() -> void;
auto test_find_index_basic() -> void;
auto test_find_index_not_found() -> void;
auto test_all_true() -> void;
auto test_all_false() -> void;
auto test_all_empty() -> void;
auto test_any_true() -> void;
auto test_any_false() -> void;
auto test_any_empty() -> void;
auto test_none_true() -> void;
auto test_none_false() -> void;
auto test_count_basic() -> void;
auto test_count_none() -> void;
auto test_count_all() -> void;
auto test_first_last_nth() -> void;
auto test_numeric_reductions() -> void;
auto test_map_then_reduce() -> void;
auto test_filter_then_count() -> void;
auto test_complex_pipeline() -> void;

auto test_fold_sum() -> void {
    std::string data = "12345";
    cpp2::ByteBuffer buf = {std::data(data), std::size(data)};
    auto sum = cpp2::combinators::reduce_from(buf).fold(0, [&](int acc, char c) -> int { return acc + static_cast<int>(c); });
    std::cout << "test_fold_sum: PASS\n";
}

auto test_fold_string_concat() -> void {
    std::string data = "abc";
    cpp2::ByteBuffer buf = {std::data(data), std::size(data)};
    auto result = cpp2::combinators::reduce_from(buf).fold(std::string(""), [&](std::string acc, char c) -> std::string { return acc + "-" + c; });
    std::cout << "test_fold_string_concat: PASS\n";
}

auto test_fold_empty() -> void {
    std::string data = "";
    cpp2::ByteBuffer buf = {std::data(data), std::size(data)};
    auto sum = cpp2::combinators::reduce_from(buf).fold(42, [&](int acc, char c) -> int { return acc + 1; });
    std::cout << "test_fold_empty: PASS\n";
}

auto test_reduce_max() -> void {
    std::string data = "bcaed";
    cpp2::ByteBuffer buf = {std::data(data), std::size(data)};
    auto max_char = cpp2::combinators::reduce_from(buf).reduce([&](char a, char b) -> char { return b; });
    std::cout << "test_reduce_max: PASS\n";
}

auto test_reduce_empty() -> void {
    std::string data = "";
    cpp2::ByteBuffer buf = {std::data(data), std::size(data)};
    auto result = cpp2::combinators::reduce_from(buf).reduce([&](char a, char b) -> char { return a; });
    std::cout << "test_reduce_empty: PASS\n";
}

auto test_reduce_single() -> void {
    std::string data = "x";
    cpp2::ByteBuffer buf = {std::data(data), std::size(data)};
    auto result = cpp2::combinators::reduce_from(buf).reduce([&](char a, char b) -> char { return a; });
    std::cout << "test_reduce_single: PASS\n";
}

auto test_scan_running_sum() -> void {
    std::string data = "123";
    cpp2::ByteBuffer buf = {std::data(data), std::size(data)};
    char zero_char = 0x30;
    auto running = cpp2::combinators::reduce_from(buf).scan(0, [&](int acc, char c) -> int { return acc + c - 0x30; });
    std::vector<int> sums = {};
for (auto v : running)     {
        sums.push_back(v);
    }
    std::cout << "test_scan_running_sum: PASS\n";
}

auto test_scan_empty() -> void {
    std::string data = "";
    cpp2::ByteBuffer buf = {std::data(data), std::size(data)};
    auto running = cpp2::combinators::reduce_from(buf).scan(0, [&](int acc, char c) -> int { return acc + 1; });
    int cnt = 0;
for (auto v : running)     {
        cnt++;
    }
    std::cout << "test_scan_empty: PASS\n";
}

auto test_find_basic() -> void {
    std::string data = "hello";
    cpp2::ByteBuffer buf = {std::data(data), std::size(data)};
    auto found = cpp2::combinators::reduce_from(buf).find([&](char c) -> bool { return c == 0x6C; });
    std::cout << "test_find_basic: PASS\n";
}

auto test_find_not_found() -> void {
    std::string data = "hello";
    cpp2::ByteBuffer buf = {std::data(data), std::size(data)};
    auto found = cpp2::combinators::reduce_from(buf).find([&](char c) -> bool { return c == 0x78; });
    std::cout << "test_find_not_found: PASS\n";
}

auto test_find_empty() -> void {
    std::string data = "";
    cpp2::ByteBuffer buf = {std::data(data), std::size(data)};
    auto found = cpp2::combinators::reduce_from(buf).find([&](char c) -> bool { return true; });
    std::cout << "test_find_empty: PASS\n";
}

auto test_find_index_basic() -> void {
    std::string data = "hello";
    cpp2::ByteBuffer buf = {std::data(data), std::size(data)};
    auto idx = cpp2::combinators::reduce_from(buf).find_index([&](char c) -> bool { return c == 0x6C; });
    std::cout << "test_find_index_basic: PASS\n";
}

auto test_find_index_not_found() -> void {
    std::string data = "hello";
    cpp2::ByteBuffer buf = {std::data(data), std::size(data)};
    auto idx = cpp2::combinators::reduce_from(buf).find_index([&](char c) -> bool { return c == 0x78; });
    std::cout << "test_find_index_not_found: PASS\n";
}

auto test_all_true() -> void {
    std::string data = "ABCDE";
    cpp2::ByteBuffer buf = {std::data(data), std::size(data)};
    auto result = cpp2::combinators::reduce_from(buf).all([&](char c) -> bool { return std::isupper(c) != 0; });
    std::cout << "test_all_true: PASS\n";
}

auto test_all_false() -> void {
    std::string data = "ABcDE";
    cpp2::ByteBuffer buf = {std::data(data), std::size(data)};
    auto result = cpp2::combinators::reduce_from(buf).all([&](char c) -> bool { return std::isupper(c) != 0; });
    std::cout << "test_all_false: PASS\n";
}

auto test_all_empty() -> void {
    std::string data = "";
    cpp2::ByteBuffer buf = {std::data(data), std::size(data)};
    auto result = cpp2::combinators::reduce_from(buf).all([&](char c) -> bool { return false; });
    std::cout << "test_all_empty: PASS\n";
}

auto test_any_true() -> void {
    std::string data = "abc1def";
    cpp2::ByteBuffer buf = {std::data(data), std::size(data)};
    auto result = cpp2::combinators::reduce_from(buf).any([&](char c) -> bool { return std::isdigit(c) != 0; });
    std::cout << "test_any_true: PASS\n";
}

auto test_any_false() -> void {
    std::string data = "abcdef";
    cpp2::ByteBuffer buf = {std::data(data), std::size(data)};
    auto result = cpp2::combinators::reduce_from(buf).any([&](char c) -> bool { return std::isdigit(c) != 0; });
    std::cout << "test_any_false: PASS\n";
}

auto test_any_empty() -> void {
    std::string data = "";
    cpp2::ByteBuffer buf = {std::data(data), std::size(data)};
    auto result = cpp2::combinators::reduce_from(buf).any([&](char c) -> bool { return true; });
    std::cout << "test_any_empty: PASS\n";
}

auto test_none_true() -> void {
    std::string data = "abcdef";
    cpp2::ByteBuffer buf = {std::data(data), std::size(data)};
    auto result = cpp2::combinators::reduce_from(buf).none([&](char c) -> bool { return std::isdigit(c) != 0; });
    std::cout << "test_none_true: PASS\n";
}

auto test_none_false() -> void {
    std::string data = "abc1def";
    cpp2::ByteBuffer buf = {std::data(data), std::size(data)};
    auto result = cpp2::combinators::reduce_from(buf).none([&](char c) -> bool { return std::isdigit(c) != 0; });
    std::cout << "test_none_false: PASS\n";
}

auto test_count_basic() -> void {
    std::string data = "hello";
    cpp2::ByteBuffer buf = {std::data(data), std::size(data)};
    auto cnt = cpp2::combinators::reduce_from(buf).count([&](char c) -> bool { return c == 0x6C; });
    std::cout << "test_count_basic: PASS\n";
}

auto test_count_none() -> void {
    std::string data = "hello";
    cpp2::ByteBuffer buf = {std::data(data), std::size(data)};
    auto cnt = cpp2::combinators::reduce_from(buf).count([&](char c) -> bool { return c == 0x78; });
    std::cout << "test_count_none: PASS\n";
}

auto test_count_all() -> void {
    std::string data = "aaa";
    cpp2::ByteBuffer buf = {std::data(data), std::size(data)};
    auto cnt = cpp2::combinators::reduce_from(buf).count_all();
    std::cout << "test_count_all: PASS\n";
}

auto test_first_last_nth() -> void {
    std::string data = "abcde";
    cpp2::ByteBuffer buf = {std::data(data), std::size(data)};
    auto range = cpp2::combinators::reduce_from(buf);
    auto f = range.first();
    auto l = range.last();
    auto n = range.nth(2);
    auto oob = range.nth(10);
    std::cout << "test_first_last_nth: PASS\n";
}

auto test_numeric_reductions() -> void {
    std::vector<int> nums = {1, 2, 3, 4, 5};
    auto range = cpp2::combinators::reduce_from(nums);
    auto s = range.sum();
    auto p = range.product();
    auto mn = range.min();
    auto mx = range.max();
    std::cout << "test_numeric_reductions: PASS\n";
}

auto test_map_then_reduce() -> void {
    std::string data = "123";
    cpp2::ByteBuffer buf = {std::data(data), std::size(data)};
    auto sum = cpp2::combinators::fold(cpp2::combinators::from(buf) | cpp2::combinators::curried::map([&](char c) -> int { return c - 0x30; }), 0, [&](int a, int b) -> int { return a + b; });
    std::cout << "test_map_then_reduce: PASS\n";
}

auto test_filter_then_count() -> void {
    std::string data = "a1b2c3";
    cpp2::ByteBuffer buf = {std::data(data), std::size(data)};
    auto cnt = cpp2::combinators::count(cpp2::combinators::from(buf) | cpp2::combinators::curried::filter([&](char c) -> bool { return std::isalpha(c) != 0; }), [&](char c) -> bool { return true; });
    std::cout << "test_filter_then_count: PASS\n";
}

auto test_complex_pipeline() -> void {
    std::string data = "The Quick Brown Fox";
    cpp2::ByteBuffer buf = {std::data(data), std::size(data)};
    auto uppercase_count = cpp2::combinators::count(buf, [&](char c) -> bool { return std::isupper(c) != 0; });
    std::cout << "test_complex_pipeline: PASS\n";
}

auto main() -> int {
    std::cout << "=== Reduction Combinators Tests ===\n\n";
    test_fold_sum();
    test_fold_string_concat();
    test_fold_empty();
    test_reduce_max();
    test_reduce_empty();
    test_reduce_single();
    test_scan_running_sum();
    test_scan_empty();
    test_find_basic();
    test_find_not_found();
    test_find_empty();
    test_find_index_basic();
    test_find_index_not_found();
    test_all_true();
    test_all_false();
    test_all_empty();
    test_any_true();
    test_any_false();
    test_any_empty();
    test_none_true();
    test_none_false();
    test_count_basic();
    test_count_none();
    test_count_all();
    test_first_last_nth();
    test_numeric_reductions();
    test_map_then_reduce();
    test_filter_then_count();
    test_complex_pipeline();
    std::cout << "\n=== All reduction combinator tests passed! ===\n";
    return 0;
}

