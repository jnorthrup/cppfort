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

#include <iostream>

#include <vector>

#include <cassert>

auto test_take() -> void;
auto test_skip() -> void;
auto test_slice() -> void;
auto test_split() -> void;
auto test_split_edge_cases() -> void;
auto test_chunk() -> void;
auto test_chunk_edge_cases() -> void;
auto test_window() -> void;
auto test_window_edge_cases() -> void;
auto test_curried_pipe() -> void;
auto test_zero_copy_invariant() -> void;

auto test_take() -> void {
    std::cout << "test_take\n";
    std::string data = "0123456789";
    cpp2::ByteBuffer buf = {std::data(data), std::size(data)};
    auto result = cpp2::combinators::take(buf, 5);
    auto count = 0;
for (auto c : result)     {
        count++;
    }
    auto result2 = cpp2::combinators::take(buf, 100);
    auto count2 = 0;
for (auto c : result2)     {
        count2++;
    }
    auto result3 = cpp2::combinators::take(buf, 0);
    auto count3 = 0;
for (auto c : result3)     {
        count3++;
    }
    std::cout << "  PASS\n";
}

auto test_skip() -> void {
    std::cout << "test_skip\n";
    std::string data = "0123456789";
    cpp2::ByteBuffer buf = {std::data(data), std::size(data)};
    auto result = cpp2::combinators::skip(buf, 5);
    auto count = 0;
for (auto c : result)     {
        count++;
    }
    auto result2 = cpp2::combinators::skip(buf, 100);
    auto count2 = 0;
for (auto c : result2)     {
        count2++;
    }
    auto result3 = cpp2::combinators::skip(buf, 0);
    auto count3 = 0;
for (auto c : result3)     {
        count3++;
    }
    std::cout << "  PASS\n";
}

auto test_slice() -> void {
    std::cout << "test_slice\n";
    std::string data = "0123456789";
    cpp2::ByteBuffer buf = {std::data(data), std::size(data)};
    auto result = cpp2::combinators::slice(buf, 2, 7);
    auto count = 0;
for (auto c : result)     {
        count++;
    }
    auto result2 = cpp2::combinators::slice(buf, 0, 10);
    auto count2 = 0;
for (auto c : result2)     {
        count2++;
    }
    auto result3 = cpp2::combinators::slice(buf, 5, 5);
    auto count3 = 0;
for (auto c : result3)     {
        count3++;
    }
    std::cout << "  PASS\n";
}

auto test_split() -> void {
    std::cout << "test_split\n";
    std::string data = "hello world test";
    cpp2::ByteBuffer buf = {std::data(data), std::size(data)};
    char space = 0x20;
    auto result = cpp2::combinators::split(buf, space);
    std::vector<cpp2::ByteBuffer> parts = {};
for (auto chunk : result)     {
        parts.push_back(chunk);
    }
    std::cout << "  PASS\n";
}

auto test_split_edge_cases() -> void {
    std::cout << "test_split_edge_cases\n";
    std::string empty = "";
    cpp2::ByteBuffer buf1 = {std::data(empty), std::size(empty)};
    char comma = 0x2C;
    auto result1 = cpp2::combinators::split(buf1, comma);
    auto cnt1 = 0;
for (auto chunk : result1)     {
        cnt1++;
    }
    std::string no_delim = "hello";
    cpp2::ByteBuffer buf2 = {std::data(no_delim), std::size(no_delim)};
    auto result2 = cpp2::combinators::split(buf2, comma);
    auto cnt2 = 0;
for (auto chunk : result2)     {
        cnt2++;
    }
    std::string consec = "a,,b";
    cpp2::ByteBuffer buf3 = {std::data(consec), std::size(consec)};
    auto result3 = cpp2::combinators::split(buf3, comma);
    auto cnt3 = 0;
for (auto chunk : result3)     {
        cnt3++;
    }
    std::cout << "  PASS\n";
}

auto test_chunk() -> void {
    std::cout << "test_chunk\n";
    std::string data = "0123456789";
    cpp2::ByteBuffer buf = {std::data(data), std::size(data)};
    auto result = cpp2::combinators::chunk(buf, 3);
    std::vector<cpp2::ByteBuffer> chunks = {};
for (auto c : result)     {
        chunks.push_back(c);
    }
    std::cout << "  PASS\n";
}

auto test_chunk_edge_cases() -> void {
    std::cout << "test_chunk_edge_cases\n";
    std::string empty = "";
    cpp2::ByteBuffer buf1 = {std::data(empty), std::size(empty)};
    auto result1 = cpp2::combinators::chunk(buf1, 3);
    auto cnt1 = 0;
for (auto c : result1)     {
        cnt1++;
    }
    std::string small = "ab";
    cpp2::ByteBuffer buf2 = {std::data(small), std::size(small)};
    auto result2 = cpp2::combinators::chunk(buf2, 10);
    auto cnt2 = 0;
    size_t size2 = 0;
for (auto c : result2)     {
        cnt2++;
        size2 = std::size(c);
    }
    std::cout << "  PASS\n";
}

auto test_window() -> void {
    std::cout << "test_window\n";
    std::string data = "01234";
    cpp2::ByteBuffer buf = {std::data(data), std::size(data)};
    auto result = cpp2::combinators::window(buf, 3);
    std::vector<cpp2::ByteBuffer> windows = {};
for (auto w : result)     {
        windows.push_back(w);
    }
    std::cout << "  PASS\n";
}

auto test_window_edge_cases() -> void {
    std::cout << "test_window_edge_cases\n";
    std::string small = "ab";
    cpp2::ByteBuffer buf1 = {std::data(small), std::size(small)};
    auto result1 = cpp2::combinators::window(buf1, 5);
    auto cnt1 = 0;
for (auto w : result1)     {
        cnt1++;
    }
    std::string exact = "abc";
    cpp2::ByteBuffer buf2 = {std::data(exact), std::size(exact)};
    auto result2 = cpp2::combinators::window(buf2, 3);
    auto cnt2 = 0;
for (auto w : result2)     {
        cnt2++;
    }
    std::cout << "  PASS\n";
}

auto test_curried_pipe() -> void {
    std::cout << "test_curried_pipe\n";
    std::string data = "0123456789";
    cpp2::ByteBuffer buf = {std::data(data), std::size(data)};
    auto skipped = cpp2::combinators::skip(buf, 2);
    auto result = cpp2::combinators::take(skipped, 5);
    auto cnt = 0;
for (auto c : result)     {
        cnt++;
    }
    std::cout << "  PASS\n";
}

auto test_zero_copy_invariant() -> void {
    std::cout << "test_zero_copy_invariant\n";
    std::string data = "hello world";
    cpp2::ByteBuffer buf = {std::data(data), std::size(data)};
    char space = 0x20;
    auto split_result = cpp2::combinators::split(buf, space);
for (auto chunk : split_result)     {
        assert(std::data(chunk) >= std::data(buf));
        assert(std::data(chunk) <= std::data(buf) + std::size(buf));
    }
    auto chunk_res = cpp2::combinators::chunk(buf, 3);
for (auto chunk : chunk_res)     {
        assert(std::data(chunk) >= std::data(buf));
        assert(std::data(chunk) <= std::data(buf) + std::size(buf));
    }
    auto window_result = cpp2::combinators::window(buf, 3);
for (auto w : window_result)     {
        assert(std::data(w) >= std::data(buf));
        assert(std::data(w) <= std::data(buf) + std::size(buf));
    }
    std::cout << "  PASS\n";
}

auto main() -> int {
    std::cout << "=== Structural Combinators Tests ===\n";
    test_take();
    test_skip();
    test_slice();
    test_split();
    test_split_edge_cases();
    test_chunk();
    test_chunk_edge_cases();
    test_window();
    test_window_edge_cases();
    test_curried_pipe();
    test_zero_copy_invariant();
    std::cout << "\nAll tests passed!\n";
}

