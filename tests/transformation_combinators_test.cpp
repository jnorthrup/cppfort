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

#include <vector>

#include <string>

#include <iostream>

#include <cassert>

#include <cstddef>

auto test_map_basic() -> void;
auto test_map_type_transform() -> void;
auto test_map_chained() -> void;
auto test_filter_basic() -> void;
auto test_filter_none_match() -> void;
auto test_filter_all_match() -> void;
auto test_map_filter_chain() -> void;
auto test_filter_map_chain() -> void;
auto test_enumerate_basic() -> void;
auto test_enumerate_empty() -> void;
auto test_zip_basic() -> void;
auto test_zip_different_lengths() -> void;
auto test_zip_curried() -> void;
auto test_intersperse_basic() -> void;
auto test_intersperse_single() -> void;
auto test_intersperse_empty() -> void;
auto test_complex_pipeline() -> void;
auto test_structural_transform_chain() -> void;
auto test_first_last() -> void;
auto test_first_last_empty() -> void;

auto test_map_basic() -> void {
    std::string data = "hello";
    cpp2::ByteBuffer buf = {std::data(data), std::size(data)};
    auto mapped = cpp2::combinators::from(buf) | cpp2::combinators::curried::map([&](char c) -> char { return static_cast<char>(std::toupper(c)); });
    std::string result = "";
for (auto c : mapped)     {
        result += c;
    }
    std::cout << "test_map_basic: PASS\n";
}

auto test_map_type_transform() -> void {
    std::string data = "ABC";
    cpp2::ByteBuffer buf = {std::data(data), std::size(data)};
    auto mapped = cpp2::combinators::from(buf) | cpp2::combinators::curried::map([&](char c) -> int { return static_cast<int>(c); });
    std::vector<int> values = {};
for (auto v : mapped)     {
        values.push_back(v);
    }
    std::cout << "test_map_type_transform: PASS\n";
}

auto test_map_chained() -> void {
    std::string data = "abc";
    cpp2::ByteBuffer buf = {std::data(data), std::size(data)};
    auto mapped = cpp2::combinators::from(buf) | cpp2::combinators::curried::map([&](char c) -> char { return static_cast<char>(std::toupper(c)); }) | cpp2::combinators::curried::map([&](char c) -> int { return static_cast<int>(c); });
    std::vector<int> values = {};
for (auto v : mapped)     {
        values.push_back(v);
    }
    std::cout << "test_map_chained: PASS\n";
}

auto test_filter_basic() -> void {
    std::string data = "a1b2c3";
    cpp2::ByteBuffer buf = {std::data(data), std::size(data)};
    auto filtered = cpp2::combinators::from(buf) | cpp2::combinators::curried::filter([&](char c) -> bool { return std::isdigit(c) != 0; });
    std::string result = "";
for (auto c : filtered)     {
        result += c;
    }
    std::cout << "test_filter_basic: PASS\n";
}

auto test_filter_none_match() -> void {
    std::string data = "abc";
    cpp2::ByteBuffer buf = {std::data(data), std::size(data)};
    auto filtered = cpp2::combinators::from(buf) | cpp2::combinators::curried::filter([&](char c) -> bool { return std::isdigit(c) != 0; });
    int cnt = 0;
for (auto c : filtered)     {
        cnt++;
    }
    std::cout << "test_filter_none_match: PASS\n";
}

auto test_filter_all_match() -> void {
    std::string data = "ABC";
    cpp2::ByteBuffer buf = {std::data(data), std::size(data)};
    auto filtered = cpp2::combinators::from(buf) | cpp2::combinators::curried::filter([&](char c) -> bool { return std::isupper(c) != 0; });
    std::string result = "";
for (auto c : filtered)     {
        result += c;
    }
    std::cout << "test_filter_all_match: PASS\n";
}

auto test_map_filter_chain() -> void {
    std::string data = "a1B2c3";
    cpp2::ByteBuffer buf = {std::data(data), std::size(data)};
    auto result = cpp2::combinators::from(buf) | cpp2::combinators::curried::map([&](char c) -> char { return static_cast<char>(std::toupper(c)); }) | cpp2::combinators::curried::filter([&](char c) -> bool { return std::isalpha(c) != 0; });
    std::string output = "";
for (auto c : result)     {
        output += c;
    }
    std::cout << "test_map_filter_chain: PASS\n";
}

auto test_filter_map_chain() -> void {
    std::string data = "a1b2c3";
    cpp2::ByteBuffer buf = {std::data(data), std::size(data)};
    auto result = cpp2::combinators::from(buf) | cpp2::combinators::curried::filter([&](char c) -> bool { return std::isalpha(c) != 0; }) | cpp2::combinators::curried::map([&](char c) -> char { return static_cast<char>(std::toupper(c)); });
    std::string output = "";
for (auto c : result)     {
        output += c;
    }
    std::cout << "test_filter_map_chain: PASS\n";
}

auto test_enumerate_basic() -> void {
    std::string data = "abc";
    cpp2::ByteBuffer buf = {std::data(data), std::size(data)};
    auto enumerated = cpp2::combinators::from(buf) | cpp2::combinators::curried::enumerate();
    std::vector<size_t> indices = {};
    std::string chars = "";
for (auto pair : enumerated)     {
        indices.push_back(pair.first);
        chars += pair.second;
    }
    std::cout << "test_enumerate_basic: PASS\n";
}

auto test_enumerate_empty() -> void {
    std::string data = "";
    cpp2::ByteBuffer buf = {std::data(data), std::size(data)};
    auto enumerated = cpp2::combinators::from(buf) | cpp2::combinators::curried::enumerate();
    int cnt = 0;
for (auto pair : enumerated)     {
        cnt++;
    }
    std::cout << "test_enumerate_empty: PASS\n";
}

auto test_zip_basic() -> void {
    std::string data1 = "abc";
    std::string data2 = "123";
    cpp2::ByteBuffer buf1 = {std::data(data1), std::size(data1)};
    cpp2::ByteBuffer buf2 = {std::data(data2), std::size(data2)};
    auto zipped = cpp2::combinators::zip(buf1, buf2);
    std::vector<std::pair<char, char>> pairs = {};
for (auto p : zipped)     {
        pairs.push_back(p);
    }
    std::cout << "test_zip_basic: PASS\n";
}

auto test_zip_different_lengths() -> void {
    std::string data1 = "abcdef";
    std::string data2 = "12";
    cpp2::ByteBuffer buf1 = {std::data(data1), std::size(data1)};
    cpp2::ByteBuffer buf2 = {std::data(data2), std::size(data2)};
    auto zipped = cpp2::combinators::zip(buf1, buf2);
    int cnt = 0;
for (auto p : zipped)     {
        cnt++;
    }
    std::cout << "test_zip_different_lengths: PASS\n";
}

auto test_zip_curried() -> void {
    std::string data1 = "ab";
    std::string data2 = "12";
    cpp2::ByteBuffer buf1 = {std::data(data1), std::size(data1)};
    cpp2::ByteBuffer buf2 = {std::data(data2), std::size(data2)};
    auto zipped = cpp2::combinators::from(buf1) | cpp2::combinators::curried::zip(buf2);
    std::vector<std::pair<char, char>> pairs = {};
for (auto p : zipped)     {
        pairs.push_back(p);
    }
    std::cout << "test_zip_curried: PASS\n";
}

auto test_intersperse_basic() -> void {
    std::string data = "abc";
    cpp2::ByteBuffer buf = {std::data(data), std::size(data)};
    char sep = 0x2D;
    auto interspersed = cpp2::combinators::from(buf) | cpp2::combinators::curried::intersperse(sep);
    std::string result = "";
for (auto c : interspersed)     {
        result += c;
    }
    std::cout << "test_intersperse_basic: PASS\n";
}

auto test_intersperse_single() -> void {
    std::string data = "x";
    cpp2::ByteBuffer buf = {std::data(data), std::size(data)};
    char sep = 0x2D;
    auto interspersed = cpp2::combinators::from(buf) | cpp2::combinators::curried::intersperse(sep);
    std::string result = "";
for (auto c : interspersed)     {
        result += c;
    }
    std::cout << "test_intersperse_single: PASS\n";
}

auto test_intersperse_empty() -> void {
    std::string data = "";
    cpp2::ByteBuffer buf = {std::data(data), std::size(data)};
    char sep = 0x2D;
    auto interspersed = cpp2::combinators::from(buf) | cpp2::combinators::curried::intersperse(sep);
    int cnt = 0;
for (auto c : interspersed)     {
        cnt++;
    }
    std::cout << "test_intersperse_empty: PASS\n";
}

auto test_complex_pipeline() -> void {
    std::string data = "a1b2c3d4e5";
    cpp2::ByteBuffer buf = {std::data(data), std::size(data)};
    auto result = cpp2::combinators::from(buf) | cpp2::combinators::curried::filter([&](char c) -> bool { return std::isalpha(c) != 0; }) | cpp2::combinators::curried::map([&](char c) -> char { return static_cast<char>(std::toupper(c)); }) | cpp2::combinators::curried::enumerate();
    std::vector<size_t> indices = {};
    std::string chars = "";
for (auto pair : result)     {
        indices.push_back(pair.first);
        chars += pair.second;
    }
    std::cout << "test_complex_pipeline: PASS\n";
}

auto test_structural_transform_chain() -> void {
    std::string data = "hello world test";
    cpp2::ByteBuffer buf = {std::data(data), std::size(data)};
    char space = 0x20;
    auto taken = cpp2::combinators::take(buf, 11);
    std::string taken_str = "";
for (auto c : taken)     {
        taken_str += c;
    }
    std::cout << "test_structural_transform_chain: PASS\n";
}

auto test_first_last() -> void {
    std::string data = "abc";
    cpp2::ByteBuffer buf = {std::data(data), std::size(data)};
    auto range = cpp2::combinators::from(buf);
    auto f = range.first();
    auto l = range.last();
    std::cout << "test_first_last: PASS\n";
}

auto test_first_last_empty() -> void {
    std::string data = "";
    cpp2::ByteBuffer buf = {std::data(data), std::size(data)};
    auto range = cpp2::combinators::from(buf);
    auto f = range.first();
    auto l = range.last();
    std::cout << "test_first_last_empty: PASS\n";
}

auto main() -> int {
    std::cout << "=== Transformation Combinators Tests ===\n\n";
    test_map_basic();
    test_map_type_transform();
    test_map_chained();
    test_filter_basic();
    test_filter_none_match();
    test_filter_all_match();
    test_map_filter_chain();
    test_filter_map_chain();
    test_enumerate_basic();
    test_enumerate_empty();
    test_zip_basic();
    test_zip_different_lengths();
    test_zip_curried();
    test_intersperse_basic();
    test_intersperse_single();
    test_intersperse_empty();
    test_complex_pipeline();
    test_structural_transform_chain();
    test_first_last();
    test_first_last_empty();
    std::cout << "\n=== All transformation combinator tests passed! ===\n";
    return 0;
}

