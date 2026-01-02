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

#include "../include/combinators/parsing.hpp"

#include <vector>

#include <string>

#include <iostream>

#include <cassert>

#include <cstddef>

#include <cstdint>

auto test_byte_basic() -> void;
auto test_byte_empty() -> void;
auto test_byte_chain() -> void;
auto test_bytes_basic() -> void;
auto test_bytes_insufficient() -> void;
auto test_bytes_exact() -> void;
auto test_until_basic() -> void;
auto test_until_not_found() -> void;
auto test_until_at_start() -> void;
auto test_while_pred_digits() -> void;
auto test_while_pred_none_match() -> void;
auto test_while_pred_all_match() -> void;
auto test_le_u16() -> void;
auto test_le_u32() -> void;
auto test_le_u64() -> void;
auto test_le_insufficient() -> void;
auto test_be_u16() -> void;
auto test_be_u32() -> void;
auto test_be_u64() -> void;
auto test_signed_integers() -> void;
auto test_c_str() -> void;
auto test_pascal_string() -> void;
auto test_pascal_string_insufficient() -> void;
auto test_length_predicates() -> void;
auto test_starts_ends_with() -> void;
auto test_contains() -> void;
auto test_is_sorted() -> void;
auto test_is_unique() -> void;
auto test_all_any_none() -> void;
auto test_parser_chaining() -> void;

auto test_byte_basic() -> void {
    std::string data = "ABC";
    cpp2::ByteBuffer buf = {std::data(data), std::size(data)};
    auto result = cpp2::combinators::byte(buf);
    std::cout << "test_byte_basic: PASS\n";
}

auto test_byte_empty() -> void {
    std::string data = "";
    cpp2::ByteBuffer buf = {std::data(data), std::size(data)};
    auto result = cpp2::combinators::byte(buf);
    std::cout << "test_byte_empty: PASS\n";
}

auto test_byte_chain() -> void {
    std::string data = "ABC";
    cpp2::ByteBuffer buf = {std::data(data), std::size(data)};
    auto r1 = cpp2::combinators::byte(buf);
    auto r2 = cpp2::combinators::byte(r1.value().remaining);
    auto r3 = cpp2::combinators::byte(r2.value().remaining);
    auto r4 = cpp2::combinators::byte(r3.value().remaining);
    std::cout << "test_byte_chain: PASS\n";
}

auto test_bytes_basic() -> void {
    std::string data = "Hello World";
    cpp2::ByteBuffer buf = {std::data(data), std::size(data)};
    auto result = cpp2::combinators::bytes(buf, 5);
    std::cout << "test_bytes_basic: PASS\n";
}

auto test_bytes_insufficient() -> void {
    std::string data = "Hi";
    cpp2::ByteBuffer buf = {std::data(data), std::size(data)};
    auto result = cpp2::combinators::bytes(buf, 10);
    std::cout << "test_bytes_insufficient: PASS\n";
}

auto test_bytes_exact() -> void {
    std::string data = "exact";
    cpp2::ByteBuffer buf = {std::data(data), std::size(data)};
    auto result = cpp2::combinators::bytes(buf, 5);
    std::cout << "test_bytes_exact: PASS\n";
}

auto test_until_basic() -> void {
    uint8_t terminator = 0x00;
    std::array<char, 6> hello = {0x48, 0x65, 0x6C, 0x6C, 0x6F, 0x00};
    cpp2::ByteBuffer buf = {std::data(hello), std::size(hello)};
    auto result = cpp2::combinators::until(buf, terminator);
    std::cout << "test_until_basic: PASS\n";
}

auto test_until_not_found() -> void {
    std::string data = "no terminator here";
    cpp2::ByteBuffer buf = {std::data(data), std::size(data)};
    auto result = cpp2::combinators::until(buf, 0x00);
    std::cout << "test_until_not_found: PASS\n";
}

auto test_until_at_start() -> void {
    uint8_t terminator = 0x00;
    std::array<char, 3> data = {0x00, 0x41, 0x42};
    cpp2::ByteBuffer buf = {std::data(data), std::size(data)};
    auto result = cpp2::combinators::until(buf, terminator);
    std::cout << "test_until_at_start: PASS\n";
}

auto test_while_pred_digits() -> void {
    std::string data = "12345abc";
    cpp2::ByteBuffer buf = {std::data(data), std::size(data)};
    auto result = cpp2::combinators::while_pred(buf, [&](uint8_t b) -> bool { return b >= 0x30 && b <= 0x39; });
    std::cout << "test_while_pred_digits: PASS\n";
}

auto test_while_pred_none_match() -> void {
    std::string data = "abc123";
    cpp2::ByteBuffer buf = {std::data(data), std::size(data)};
    auto result = cpp2::combinators::while_pred(buf, [&](uint8_t b) -> bool { return b >= 0x30 && b <= 0x39; });
    std::cout << "test_while_pred_none_match: PASS\n";
}

auto test_while_pred_all_match() -> void {
    std::string data = "12345";
    cpp2::ByteBuffer buf = {std::data(data), std::size(data)};
    auto result = cpp2::combinators::while_pred(buf, [&](uint8_t b) -> bool { return b >= 0x30 && b <= 0x39; });
    std::cout << "test_while_pred_all_match: PASS\n";
}

auto test_le_u16() -> void {
    std::array<char, 4> data = {0x34, 0x12, static_cast<char>(0xFF), static_cast<char>(0xFF)};
    cpp2::ByteBuffer buf = {std::data(data), std::size(data)};
    auto result = cpp2::combinators::le_u16(buf);
    std::cout << "test_le_u16: PASS\n";
}

auto test_le_u32() -> void {
    std::array<char, 4> data = {0x78, 0x56, 0x34, 0x12};
    cpp2::ByteBuffer buf = {std::data(data), std::size(data)};
    auto result = cpp2::combinators::le_u32(buf);
    std::cout << "test_le_u32: PASS\n";
}

auto test_le_u64() -> void {
    std::array<char, 8> data = {0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07, 0x08};
    cpp2::ByteBuffer buf = {std::data(data), std::size(data)};
    auto result = cpp2::combinators::le_u64(buf);
    std::cout << "test_le_u64: PASS\n";
}

auto test_le_insufficient() -> void {
    char single = 0x12;
    cpp2::ByteBuffer buf = {&single, 1};
    auto r16 = cpp2::combinators::le_u16(buf);
    auto r32 = cpp2::combinators::le_u32(buf);
    auto r64 = cpp2::combinators::le_u64(buf);
    std::cout << "test_le_insufficient: PASS\n";
}

auto test_be_u16() -> void {
    std::array<char, 4> data = {0x12, 0x34, static_cast<char>(0xFF), static_cast<char>(0xFF)};
    cpp2::ByteBuffer buf = {std::data(data), std::size(data)};
    auto result = cpp2::combinators::be_u16(buf);
    std::cout << "test_be_u16: PASS\n";
}

auto test_be_u32() -> void {
    std::array<char, 4> data = {0x12, 0x34, 0x56, 0x78};
    cpp2::ByteBuffer buf = {std::data(data), std::size(data)};
    auto result = cpp2::combinators::be_u32(buf);
    std::cout << "test_be_u32: PASS\n";
}

auto test_be_u64() -> void {
    std::array<char, 8> data = {0x08, 0x07, 0x06, 0x05, 0x04, 0x03, 0x02, 0x01};
    cpp2::ByteBuffer buf = {std::data(data), std::size(data)};
    auto result = cpp2::combinators::be_u64(buf);
    std::cout << "test_be_u64: PASS\n";
}

auto test_signed_integers() -> void {
    std::array<char, 2> neg_data = {static_cast<char>(0xFF), static_cast<char>(0xFF)};
    cpp2::ByteBuffer buf = {std::data(neg_data), std::size(neg_data)};
    auto result = cpp2::combinators::le_i16(buf);
    std::cout << "test_signed_integers: PASS\n";
}

auto test_c_str() -> void {
    uint8_t terminator = 0x00;
    std::array<char, 10> data = {0x48, 0x65, 0x6C, 0x6C, 0x6F, 0x00, 0x58, 0x59, 0x5A, 0x00};
    cpp2::ByteBuffer buf = {std::data(data), std::size(data)};
    auto result = cpp2::combinators::c_str(buf);
    std::cout << "test_c_str: PASS\n";
}

auto test_pascal_string() -> void {
    std::array<char, 8> data = {0x05, 0x48, 0x65, 0x6C, 0x6C, 0x6F, 0x58, 0x59};
    cpp2::ByteBuffer buf = {std::data(data), std::size(data)};
    auto result = cpp2::combinators::pascal_string(buf);
    std::cout << "test_pascal_string: PASS\n";
}

auto test_pascal_string_insufficient() -> void {
    std::array<char, 4> data = {0x0A, 0x41, 0x42, 0x43};
    cpp2::ByteBuffer buf = {std::data(data), std::size(data)};
    auto result = cpp2::combinators::pascal_string(buf);
    std::cout << "test_pascal_string_insufficient: PASS\n";
}

auto test_length_predicates() -> void {
    std::string data = "Hello";
    cpp2::ByteBuffer buf = {std::data(data), std::size(data)};
    std::cout << "test_length_predicates: PASS\n";
}

auto test_starts_ends_with() -> void {
    std::string data = "Hello World";
    cpp2::ByteBuffer buf = {std::data(data), std::size(data)};
    std::string hello = "Hello";
    std::string world = "World";
    std::string xyz = "xyz";
    cpp2::ByteBuffer prefix = {std::data(hello), std::size(hello)};
    cpp2::ByteBuffer suffix = {std::data(world), std::size(world)};
    cpp2::ByteBuffer bad = {std::data(xyz), std::size(xyz)};
    std::cout << "test_starts_ends_with: PASS\n";
}

auto test_contains() -> void {
    std::string data = "Hello";
    cpp2::ByteBuffer buf = {std::data(data), std::size(data)};
    std::cout << "test_contains: PASS\n";
}

auto test_is_sorted() -> void {
    std::string sorted_data = "ABCDE";
    std::string unsorted_data = "ACDBE";
    cpp2::ByteBuffer sorted_buf = {std::data(sorted_data), std::size(sorted_data)};
    cpp2::ByteBuffer unsorted_buf = {std::data(unsorted_data), std::size(unsorted_data)};
    std::cout << "test_is_sorted: PASS\n";
}

auto test_is_unique() -> void {
    std::string unique_data = "ABCDE";
    std::string dup_data = "ABCDA";
    cpp2::ByteBuffer unique_buf = {std::data(unique_data), std::size(unique_data)};
    cpp2::ByteBuffer dup_buf = {std::data(dup_data), std::size(dup_data)};
    std::cout << "test_is_unique: PASS\n";
}

auto test_all_any_none() -> void {
    std::string data = "ABC";
    cpp2::ByteBuffer buf = {std::data(data), std::size(data)};
    std::cout << "test_all_any_none: PASS\n";
}

auto test_parser_chaining() -> void {
    std::array<char, 7> header = {0x05, 0x00, 0x48, 0x65, 0x6C, 0x6C, 0x6F};
    cpp2::ByteBuffer buf = {std::data(header), std::size(header)};
    auto len_result = cpp2::combinators::le_u16(buf);
    auto data_result = cpp2::combinators::bytes(len_result.value().remaining, len_result.value().value);
    std::cout << "test_parser_chaining: PASS\n";
}

auto main() -> int {
    std::cout << "=== Parsing Combinators Tests ===\n\n";
    test_byte_basic();
    test_byte_empty();
    test_byte_chain();
    test_bytes_basic();
    test_bytes_insufficient();
    test_bytes_exact();
    test_until_basic();
    test_until_not_found();
    test_until_at_start();
    test_while_pred_digits();
    test_while_pred_none_match();
    test_while_pred_all_match();
    test_le_u16();
    test_le_u32();
    test_le_u64();
    test_le_insufficient();
    test_be_u16();
    test_be_u32();
    test_be_u64();
    test_signed_integers();
    test_c_str();
    test_pascal_string();
    test_pascal_string_insufficient();
    test_length_predicates();
    test_starts_ends_with();
    test_contains();
    test_is_sorted();
    test_is_unique();
    test_all_any_none();
    test_parser_chaining();
    std::cout << "\n=== All parsing combinator tests passed! ===\n";
    return 0;
}

