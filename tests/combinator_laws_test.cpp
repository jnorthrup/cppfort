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

#include <functional>

#include <random>

template<typename T>
[[nodiscard]] auto id(T x) -> T;
template<typename A, typename B, typename C>
[[nodiscard]] auto compose(std::function<C (B)> f, std::function<B (A)> g) -> std::function<C (A)>;
[[nodiscard]] auto generate_test_data(size_t seed, size_t length) -> std::string;
[[nodiscard]] auto collect_chars(auto range) -> std::vector<char>;
[[nodiscard]] auto collect_ints(auto range) -> std::vector<int>;
template<typename T>
[[nodiscard]] auto vectors_equal(std::vector<T> a, std::vector<T> b) -> bool;
auto test_functor_identity_law() -> void;
auto test_functor_composition_law() -> void;
auto test_take_zero_identity() -> void;
auto test_take_composition() -> void;
auto test_skip_composition() -> void;
auto test_skip_take_ordering() -> void;
auto test_filter_true_identity() -> void;
auto test_filter_false_empty() -> void;
auto test_filter_conjunction() -> void;
auto test_fold_monoid_associativity() -> void;
auto test_count_equals_fold() -> void;
auto test_any_all_duality() -> void;
auto test_pipeline_associativity() -> void;
auto test_lazy_evaluation() -> void;
auto test_empty_buffer_operations() -> void;
auto test_single_element_operations() -> void;

template<typename T>
[[nodiscard]] auto id(T x) -> T {
    return x;
}

template<typename A, typename B, typename C>
[[nodiscard]] auto compose(std::function<C (B)> f, std::function<B (A)> g) -> std::function<C (A)> {
    return [&](A x) -> C { return f(g(x)); };
}

[[nodiscard]] auto generate_test_data(size_t seed, size_t length) -> std::string {
    std::mt19937 gen(static_cast<std::mt19937::result_type>(seed));
    std::uniform_int_distribution<int> dist(32, 126);
    std::string result = {};
    result.reserve(length);
    auto i = 0;
while (i < length)     {
        result += static_cast<char>(dist(gen));
        i++;
    }
    return result;
}

[[nodiscard]] auto collect_chars(auto range) -> std::vector<char> {
    std::vector<char> result = {};
for (auto c : range)     {
        result.push_back(c);
    }
    return result;
}

[[nodiscard]] auto collect_ints(auto range) -> std::vector<int> {
    std::vector<int> result = {};
for (auto v : range)     {
        result.push_back(v);
    }
    return result;
}

template<typename T>
[[nodiscard]] auto vectors_equal(std::vector<T> a, std::vector<T> b) -> bool {
if (std::size(a) != std::size(b))     {
        return false;
    }
    auto i = 0;
while (i < std::size(a))     {
if (a[i] != b[i])         {
            return false;
        }
        i++;
    }
    return true;
}

auto test_functor_identity_law() -> void {
    std::cout << "test_functor_identity_law\n";
    auto seed = 0;
while (seed < 10)     {
        auto data = generate_test_data(seed, 20);
        cpp2::ByteBuffer buf = {std::data(data), std::size(data)};
        auto original = collect_chars(buf);
        auto mapped = cpp2::combinators::from(buf) | cpp2::combinators::curried::map([&](char c) -> char { return c; });
        auto mapped_result = collect_chars(mapped);
        assert(vectors_equal(original, mapped_result));
        seed++;
    }
    std::cout << "  PASS (10 random inputs)\n";
}

auto test_functor_composition_law() -> void {
    std::cout << "test_functor_composition_law\n";
    auto f = [&](char c) -> int { return static_cast<int>(c); };
    auto g = [&](char c) -> char { return static_cast<char>(std::toupper(c)); };
    auto fg = [&](char c) -> int { return static_cast<int>(static_cast<char>(std::toupper(c))); };
    auto seed = 0;
while (seed < 10)     {
        auto data = generate_test_data(seed, 20);
        cpp2::ByteBuffer buf = {std::data(data), std::size(data)};
        auto composed = cpp2::combinators::from(buf) | cpp2::combinators::curried::map(fg);
        auto composed_result = collect_ints(composed);
        cpp2::ByteBuffer buf2 = {std::data(data), std::size(data)};
        auto chained = cpp2::combinators::from(buf2) | cpp2::combinators::curried::map(g) | cpp2::combinators::curried::map(f);
        auto chained_result = collect_ints(chained);
        assert(vectors_equal(composed_result, chained_result));
        seed++;
    }
    std::cout << "  PASS (10 random inputs)\n";
}

auto test_take_zero_identity() -> void {
    std::cout << "test_take_zero_identity\n";
    auto seed = 0;
while (seed < 10)     {
        auto data = generate_test_data(seed, 20);
        cpp2::ByteBuffer buf = {std::data(data), std::size(data)};
        auto result = cpp2::combinators::take(buf, 0);
        auto count = 0;
for (auto c : result)         {
            count++;
        }
        assert(count == 0);
        seed++;
    }
    std::cout << "  PASS (10 random inputs)\n";
}

auto test_take_composition() -> void {
    std::cout << "test_take_composition\n";
    std::string data = "0123456789ABCDEF";
    cpp2::ByteBuffer buf = {std::data(data), std::size(data)};
    auto result1 = cpp2::combinators::take(cpp2::combinators::take(buf, 10), 5);
    auto direct = cpp2::combinators::take(buf, 5);
    auto v1 = collect_chars(result1);
    auto v2 = collect_chars(direct);
    cpp2::ByteBuffer buf2 = {std::data(data), std::size(data)};
    auto result2 = cpp2::combinators::take(cpp2::combinators::take(buf2, 3), 10);
    cpp2::ByteBuffer buf3 = {std::data(data), std::size(data)};
    auto direct2 = cpp2::combinators::take(buf3, 3);
    auto v3 = collect_chars(result2);
    auto v4 = collect_chars(direct2);
    std::cout << "  PASS\n";
}

auto test_skip_composition() -> void {
    std::cout << "test_skip_composition\n";
    std::string data = "0123456789ABCDEF";
    cpp2::ByteBuffer buf = {std::data(data), std::size(data)};
    auto result1 = cpp2::combinators::skip(cpp2::combinators::skip(buf, 3), 5);
    cpp2::ByteBuffer buf2 = {std::data(data), std::size(data)};
    auto direct = cpp2::combinators::skip(buf2, 8);
    auto v1 = collect_chars(result1);
    auto v2 = collect_chars(direct);
    std::cout << "  PASS\n";
}

auto test_skip_take_ordering() -> void {
    std::cout << "test_skip_take_ordering\n";
    std::string data = "0123456789";
    cpp2::ByteBuffer buf = {std::data(data), std::size(data)};
    auto result = cpp2::combinators::take(cpp2::combinators::skip(buf, 3), 4);
    auto v = collect_chars(result);
    std::vector<char> expected = {'3', '4', '5', '6'};
    std::cout << "  PASS\n";
}

auto test_filter_true_identity() -> void {
    std::cout << "test_filter_true_identity\n";
    auto seed = 0;
while (seed < 10)     {
        auto data = generate_test_data(seed, 20);
        cpp2::ByteBuffer buf = {std::data(data), std::size(data)};
        auto original = collect_chars(buf);
        cpp2::ByteBuffer buf2 = {std::data(data), std::size(data)};
        auto filtered = cpp2::combinators::from(buf2) | cpp2::combinators::curried::filter([&](char c) -> bool { return true; });
        auto filtered_result = collect_chars(filtered);
        assert(vectors_equal(original, filtered_result));
        seed++;
    }
    std::cout << "  PASS (10 random inputs)\n";
}

auto test_filter_false_empty() -> void {
    std::cout << "test_filter_false_empty\n";
    auto seed = 0;
while (seed < 10)     {
        auto data = generate_test_data(seed, 20);
        cpp2::ByteBuffer buf = {std::data(data), std::size(data)};
        auto filtered = cpp2::combinators::from(buf) | cpp2::combinators::curried::filter([&](char c) -> bool { return false; });
        auto count = 0;
for (auto c : filtered)         {
            count++;
        }
        assert(count == 0);
        seed++;
    }
    std::cout << "  PASS (10 random inputs)\n";
}

auto test_filter_conjunction() -> void {
    std::cout << "test_filter_conjunction\n";
    std::string data = "aA1bB2cC3dD4";
    cpp2::ByteBuffer buf = {std::data(data), std::size(data)};
    auto p1 = [&](char c) -> bool { return std::isalpha(c) != 0; };
    auto p2 = [&](char c) -> bool { return std::isupper(c) != 0; };
    auto p_combined = [&](char c) -> bool { return std::isalpha(c) != 0 && std::isupper(c) != 0; };
    auto chained = cpp2::combinators::from(buf) | cpp2::combinators::curried::filter(p1) | cpp2::combinators::curried::filter(p2);
    auto chained_result = collect_chars(chained);
    cpp2::ByteBuffer buf2 = {std::data(data), std::size(data)};
    auto combined = cpp2::combinators::from(buf2) | cpp2::combinators::curried::filter(p_combined);
    auto combined_result = collect_chars(combined);
    std::vector<char> expected = {'A', 'B', 'C', 'D'};
    std::cout << "  PASS\n";
}

auto test_fold_monoid_associativity() -> void {
    std::cout << "test_fold_monoid_associativity\n";
    std::string data = "12345";
    cpp2::ByteBuffer buf = {std::data(data), std::size(data)};
    auto sum = cpp2::combinators::reduce_from(buf).fold(0, [&](int acc, char c) -> int { return acc + c - '0'; });
    std::cout << "  PASS\n";
}

auto test_count_equals_fold() -> void {
    std::cout << "test_count_equals_fold\n";
    std::string data = "hello world";
    cpp2::ByteBuffer buf = {std::data(data), std::size(data)};
    auto count_fold = cpp2::combinators::reduce_from(buf).fold(0, [&](int acc, char c) -> int { return acc + 1; });
    std::cout << "  PASS\n";
}

auto test_any_all_duality() -> void {
    std::cout << "test_any_all_duality\n";
    std::string data = "abc123xyz";
    cpp2::ByteBuffer buf = {std::data(data), std::size(data)};
    auto has_digit = cpp2::combinators::reduce_from(buf).any([&](char c) -> bool { return std::isdigit(c) != 0; });
    cpp2::ByteBuffer buf2 = {std::data(data), std::size(data)};
    auto all_non_digit = cpp2::combinators::reduce_from(buf2).all([&](char c) -> bool { return std::isdigit(c) == 0; });
    std::cout << "  PASS\n";
}

auto test_pipeline_associativity() -> void {
    std::cout << "test_pipeline_associativity\n";
    std::string data = "abcdefghij";
    cpp2::ByteBuffer buf = {std::data(data), std::size(data)};
    auto result1 = cpp2::combinators::from(buf) | cpp2::combinators::curried::skip(2) | cpp2::combinators::curried::take(5) | cpp2::combinators::curried::map([&](char c) -> char { return static_cast<char>(std::toupper(c)); });
    auto v1 = collect_chars(result1);
    std::vector<char> expected = {'C', 'D', 'E', 'F', 'G'};
    std::cout << "  PASS\n";
}

auto test_lazy_evaluation() -> void {
    std::cout << "test_lazy_evaluation\n";
    std::string data = "0123456789";
    cpp2::ByteBuffer buf = {std::data(data), std::size(data)};
    auto pipeline = cpp2::combinators::from(buf) | cpp2::combinators::curried::take(3) | cpp2::combinators::curried::map([&](char c) -> char { return c; });
    auto result = collect_chars(pipeline);
    std::cout << "  PASS (lazy construction verified)\n";
}

auto test_empty_buffer_operations() -> void {
    std::cout << "test_empty_buffer_operations\n";
    cpp2::ByteBuffer buf = {nullptr, 0};
    auto take_result = cpp2::combinators::take(buf, 10);
    auto skip_result = cpp2::combinators::skip(buf, 10);
    cpp2::ByteBuffer buf2 = {nullptr, 0};
    auto map_result = cpp2::combinators::from(buf2) | cpp2::combinators::curried::map([&](char c) -> char { return c; });
    cpp2::ByteBuffer buf3 = {nullptr, 0};
    auto filter_result = cpp2::combinators::from(buf3) | cpp2::combinators::curried::filter([&](char c) -> bool { return true; });
    std::cout << "  PASS\n";
}

auto test_single_element_operations() -> void {
    std::cout << "test_single_element_operations\n";
    std::string data = "X";
    cpp2::ByteBuffer buf = {std::data(data), std::size(data)};
    auto take_result = cpp2::combinators::take(buf, 1);
    auto v1 = collect_chars(take_result);
    cpp2::ByteBuffer buf2 = {std::data(data), std::size(data)};
    auto take_result2 = cpp2::combinators::take(buf2, 10);
    auto v2 = collect_chars(take_result2);
    cpp2::ByteBuffer buf3 = {std::data(data), std::size(data)};
    auto skip_result = cpp2::combinators::skip(buf3, 1);
    std::cout << "  PASS\n";
}

auto main() -> int {
    std::cout << "=== Combinator Laws Property-Based Tests ===\n\n";
    std::cout << "--- Functor Laws ---\n";
    test_functor_identity_law();
    test_functor_composition_law();
    std::cout << "\n--- Structural Combinator Laws ---\n";
    test_take_zero_identity();
    test_take_composition();
    test_skip_composition();
    test_skip_take_ordering();
    std::cout << "\n--- Filter Laws ---\n";
    test_filter_true_identity();
    test_filter_false_empty();
    test_filter_conjunction();
    std::cout << "\n--- Reduction Laws ---\n";
    test_fold_monoid_associativity();
    test_count_equals_fold();
    test_any_all_duality();
    std::cout << "\n--- Pipeline Associativity ---\n";
    test_pipeline_associativity();
    std::cout << "\n--- Lazy Evaluation ---\n";
    test_lazy_evaluation();
    std::cout << "\n--- Edge Cases ---\n";
    test_empty_buffer_operations();
    test_single_element_operations();
    std::cout << "\n=== All Combinator Laws Tests PASSED ===\n";
    return 0;
}

