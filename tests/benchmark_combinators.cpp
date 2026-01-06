// Combinator Benchmark Suite
// Compares combinator pipeline performance against hand-written loops
// Target: <5% overhead

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
#include <chrono>
#include <random>
#include <iomanip>

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
} // namespace cpp2

#include "../include/bytebuffer.hpp"
#include "../include/strview.hpp"
#include "../include/combinators/structural.hpp"
#include "../include/combinators/transformation.hpp"
#include "../include/combinators/reduction.hpp"

// ============================================================================
// Benchmark Infrastructure
// ============================================================================

using Clock = std::chrono::high_resolution_clock;
using Duration = std::chrono::duration<double, std::nano>;

// Generate test data
auto generate_data(size_t size, int seed) -> std::string {
    std::mt19937 gen(seed);
    std::uniform_int_distribution<int> dist(0, 255);
    std::string result;
    result.reserve(size);
    for (size_t i = 0; i < size; i++) {
        result += static_cast<char>(dist(gen));
    }
    return result;
}

// Benchmark result
struct BenchmarkResult {
    std::string name;
    double hand_ns;
    double combinator_ns;
    double overhead_pct;
};

auto print_result(const BenchmarkResult& r) -> void {
    std::string status;
    if (r.overhead_pct <= 5.0) { status = "✓ PASS"; }
    else { status = "✗ FAIL"; }
    
    std::cout << std::left << std::setw(35) << r.name
              << std::right << std::setw(12) << std::fixed << std::setprecision(2) << r.hand_ns << " ns  "
              << std::setw(12) << r.combinator_ns << " ns  "
              << std::setw(8) << r.overhead_pct << "%  "
              << status << "\n";
}

constexpr int WARMUP_ITERS = 100;
constexpr int BENCH_ITERS = 10000;

// ============================================================================
// Benchmark: Sum of Elements (fold vs loop)
// ============================================================================

auto bench_sum_hand(const std::string& data) -> int {
    int sum = 0;
    for (size_t i = 0; i < data.size(); i++) {
        sum += static_cast<int>(static_cast<unsigned char>(data[i]));
    }
    return sum;
}

auto bench_sum_combinator(const std::string& data) -> int {
    cpp2::ByteBuffer buf(data.data(), data.size());
    return cpp2::combinators::reduce_from(buf)
        .fold(0, [](int acc, char c) -> int { return acc + static_cast<int>(static_cast<unsigned char>(c)); });
}

auto benchmark_sum(const std::string& data) -> BenchmarkResult {
    // Warmup
    for (int i = 0; i < WARMUP_ITERS; i++) {
        (void)bench_sum_hand(data);
        (void)bench_sum_combinator(data);
    }
    
    // Benchmark hand-written
    auto start = Clock::now();
    for (int j = 0; j < BENCH_ITERS; j++) {
        (void)bench_sum_hand(data);
    }
    double hand_time = static_cast<double>((Clock::now() - start).count()) / BENCH_ITERS;
    
    // Benchmark combinator
    auto start2 = Clock::now();
    for (int k = 0; k < BENCH_ITERS; k++) {
        (void)bench_sum_combinator(data);
    }
    double combinator_time = static_cast<double>((Clock::now() - start2).count()) / BENCH_ITERS;
    
    double overhead = (combinator_time - hand_time) / hand_time * 100.0;
    return BenchmarkResult{"fold (sum)", hand_time, combinator_time, overhead};
}

// ============================================================================
// Benchmark: Filter + Count (filter + loop vs loop)
// ============================================================================

auto bench_filter_count_hand(const std::string& data) -> int {
    int count = 0;
    for (size_t i = 0; i < data.size(); i++) {
        if (static_cast<int>(static_cast<unsigned char>(data[i])) % 2 == 0) {
            count++;
        }
    }
    return count;
}

auto bench_filter_count_combinator(const std::string& data) -> int {
    cpp2::ByteBuffer buf(data.data(), data.size());
    return cpp2::combinators::reduce_from(buf)
        .count([](char c) -> bool { return static_cast<int>(static_cast<unsigned char>(c)) % 2 == 0; });
}

auto benchmark_filter_count(const std::string& data) -> BenchmarkResult {
    // Warmup
    for (int i = 0; i < WARMUP_ITERS; i++) {
        (void)bench_filter_count_hand(data);
        (void)bench_filter_count_combinator(data);
    }
    
    // Benchmark hand-written
    auto start = Clock::now();
    for (int j = 0; j < BENCH_ITERS; j++) {
        (void)bench_filter_count_hand(data);
    }
    double hand_time = static_cast<double>((Clock::now() - start).count()) / BENCH_ITERS;
    
    // Benchmark combinator
    auto start2 = Clock::now();
    for (int k = 0; k < BENCH_ITERS; k++) {
        (void)bench_filter_count_combinator(data);
    }
    double combinator_time = static_cast<double>((Clock::now() - start2).count()) / BENCH_ITERS;
    
    double overhead = (combinator_time - hand_time) / hand_time * 100.0;
    return BenchmarkResult{"filter + count", hand_time, combinator_time, overhead};
}

// ============================================================================
// Benchmark: Take first N elements
// ============================================================================

auto bench_take_hand(const std::string& data, size_t n) -> size_t {
    size_t sum = 0;
    size_t limit = std::min(n, data.size());
    for (size_t i = 0; i < limit; i++) {
        sum += static_cast<size_t>(static_cast<unsigned char>(data[i]));
    }
    return sum;
}

auto bench_take_combinator(const std::string& data, size_t n) -> size_t {
    cpp2::ByteBuffer buf(data.data(), data.size());
    auto result = cpp2::combinators::take(buf, n);
    size_t sum = 0;
    for (char c : result) {
        sum += static_cast<size_t>(static_cast<unsigned char>(c));
    }
    return sum;
}

auto benchmark_take(const std::string& data) -> BenchmarkResult {
    size_t n = 1000;
    
    // Warmup
    for (int i = 0; i < WARMUP_ITERS; i++) {
        (void)bench_take_hand(data, n);
        (void)bench_take_combinator(data, n);
    }
    
    // Benchmark hand-written
    auto start = Clock::now();
    for (int j = 0; j < BENCH_ITERS; j++) {
        (void)bench_take_hand(data, n);
    }
    double hand_time = static_cast<double>((Clock::now() - start).count()) / BENCH_ITERS;
    
    // Benchmark combinator
    auto start2 = Clock::now();
    for (int k = 0; k < BENCH_ITERS; k++) {
        (void)bench_take_combinator(data, n);
    }
    double combinator_time = static_cast<double>((Clock::now() - start2).count()) / BENCH_ITERS;
    
    double overhead = (combinator_time - hand_time) / hand_time * 100.0;
    return BenchmarkResult{"take(1000)", hand_time, combinator_time, overhead};
}

// ============================================================================
// Benchmark: Skip first N elements
// ============================================================================

auto bench_skip_hand(const std::string& data, size_t n) -> size_t {
    size_t sum = 0;
    size_t start_idx = std::min(n, data.size());
    for (size_t i = start_idx; i < data.size(); i++) {
        sum += static_cast<size_t>(static_cast<unsigned char>(data[i]));
    }
    return sum;
}

auto bench_skip_combinator(const std::string& data, size_t n) -> size_t {
    cpp2::ByteBuffer buf(data.data(), data.size());
    auto result = cpp2::combinators::skip(buf, n);
    size_t sum = 0;
    for (char c : result) {
        sum += static_cast<size_t>(static_cast<unsigned char>(c));
    }
    return sum;
}

auto benchmark_skip(const std::string& data) -> BenchmarkResult {
    size_t n = 1000;
    
    // Warmup
    for (int i = 0; i < WARMUP_ITERS; i++) {
        (void)bench_skip_hand(data, n);
        (void)bench_skip_combinator(data, n);
    }
    
    // Benchmark hand-written
    auto start = Clock::now();
    for (int j = 0; j < BENCH_ITERS; j++) {
        (void)bench_skip_hand(data, n);
    }
    double hand_time = static_cast<double>((Clock::now() - start).count()) / BENCH_ITERS;
    
    // Benchmark combinator
    auto start2 = Clock::now();
    for (int k = 0; k < BENCH_ITERS; k++) {
        (void)bench_skip_combinator(data, n);
    }
    double combinator_time = static_cast<double>((Clock::now() - start2).count()) / BENCH_ITERS;
    
    double overhead = (combinator_time - hand_time) / hand_time * 100.0;
    return BenchmarkResult{"skip(1000)", hand_time, combinator_time, overhead};
}

// ============================================================================
// Benchmark: Map transformation
// ============================================================================

auto bench_map_hand(const std::string& data) -> size_t {
    size_t sum = 0;
    for (size_t i = 0; i < data.size(); i++) {
        char c = data[i];
        if (c >= 'a' && c <= 'z') {
            c = static_cast<char>(c - 32);
        }
        sum += static_cast<size_t>(static_cast<unsigned char>(c));
    }
    return sum;
}

auto bench_map_combinator(const std::string& data) -> size_t {
    cpp2::ByteBuffer buf(data.data(), data.size());
    auto mapped = cpp2::combinators::from(buf)
        | cpp2::combinators::curried::map([](char c) -> char {
            if (c >= 'a' && c <= 'z') { return static_cast<char>(c - 32); }
            return c;
        });
    size_t sum = 0;
    for (char c : mapped) {
        sum += static_cast<size_t>(static_cast<unsigned char>(c));
    }
    return sum;
}

auto benchmark_map(const std::string& data) -> BenchmarkResult {
    // Warmup
    for (int i = 0; i < WARMUP_ITERS; i++) {
        (void)bench_map_hand(data);
        (void)bench_map_combinator(data);
    }
    
    // Benchmark hand-written
    auto start = Clock::now();
    for (int j = 0; j < BENCH_ITERS; j++) {
        (void)bench_map_hand(data);
    }
    double hand_time = static_cast<double>((Clock::now() - start).count()) / BENCH_ITERS;
    
    // Benchmark combinator
    auto start2 = Clock::now();
    for (int k = 0; k < BENCH_ITERS; k++) {
        (void)bench_map_combinator(data);
    }
    double combinator_time = static_cast<double>((Clock::now() - start2).count()) / BENCH_ITERS;
    
    double overhead = (combinator_time - hand_time) / hand_time * 100.0;
    return BenchmarkResult{"map (toupper)", hand_time, combinator_time, overhead};
}

// ============================================================================
// Benchmark: Pipeline composition (skip + take + map + fold)
// ============================================================================

auto bench_pipeline_hand(const std::string& data) -> int {
    int sum = 0;
    size_t start_idx = std::min(static_cast<size_t>(100), data.size());
    size_t end_idx = std::min(static_cast<size_t>(600), data.size());
    
    for (size_t i = start_idx; i < end_idx; i++) {
        char c = data[i];
        if (c >= 'a' && c <= 'z') {
            c = static_cast<char>(c - 32);
        }
        sum += static_cast<int>(static_cast<unsigned char>(c));
    }
    return sum;
}

auto bench_pipeline_combinator(const std::string& data) -> int {
    cpp2::ByteBuffer buf(data.data(), data.size());
    auto pipeline = cpp2::combinators::from(buf)
        | cpp2::combinators::curried::skip(100)
        | cpp2::combinators::curried::take(500)
        | cpp2::combinators::curried::map([](char c) -> char {
            if (c >= 'a' && c <= 'z') { return static_cast<char>(c - 32); }
            return c;
        });
    return cpp2::combinators::reduce_from(pipeline)
        .fold(0, [](int acc, char c) -> int { return acc + static_cast<int>(static_cast<unsigned char>(c)); });
}

auto benchmark_pipeline(const std::string& data) -> BenchmarkResult {
    // Warmup
    for (int i = 0; i < WARMUP_ITERS; i++) {
        (void)bench_pipeline_hand(data);
        (void)bench_pipeline_combinator(data);
    }
    
    // Benchmark hand-written
    auto start = Clock::now();
    for (int j = 0; j < BENCH_ITERS; j++) {
        (void)bench_pipeline_hand(data);
    }
    double hand_time = static_cast<double>((Clock::now() - start).count()) / BENCH_ITERS;
    
    // Benchmark combinator
    auto start2 = Clock::now();
    for (int k = 0; k < BENCH_ITERS; k++) {
        (void)bench_pipeline_combinator(data);
    }
    double combinator_time = static_cast<double>((Clock::now() - start2).count()) / BENCH_ITERS;
    
    double overhead = (combinator_time - hand_time) / hand_time * 100.0;
    return BenchmarkResult{"pipeline (skip+take+map+fold)", hand_time, combinator_time, overhead};
}

// ============================================================================
// Benchmark: Find first match
// ============================================================================

auto bench_find_hand(const std::string& data, char target) -> int {
    for (size_t i = 0; i < data.size(); i++) {
        if (data[i] == target) {
            return static_cast<int>(i);
        }
    }
    return -1;
}

auto bench_find_combinator(const std::string& data, char target) -> int {
    cpp2::ByteBuffer buf(data.data(), data.size());
    auto result = cpp2::combinators::reduce_from(buf)
        .find_index([target](char c) -> bool { return c == target; });
    if (result.has_value()) { return static_cast<int>(result.value()); }
    return -1;
}

auto benchmark_find(const std::string& data) -> BenchmarkResult {
    char target = 'X';
    
    // Warmup
    for (int i = 0; i < WARMUP_ITERS; i++) {
        (void)bench_find_hand(data, target);
        (void)bench_find_combinator(data, target);
    }
    
    // Benchmark hand-written
    auto start = Clock::now();
    for (int j = 0; j < BENCH_ITERS; j++) {
        (void)bench_find_hand(data, target);
    }
    double hand_time = static_cast<double>((Clock::now() - start).count()) / BENCH_ITERS;
    
    // Benchmark combinator
    auto start2 = Clock::now();
    for (int k = 0; k < BENCH_ITERS; k++) {
        (void)bench_find_combinator(data, target);
    }
    double combinator_time = static_cast<double>((Clock::now() - start2).count()) / BENCH_ITERS;
    
    double overhead = (combinator_time - hand_time) / hand_time * 100.0;
    return BenchmarkResult{"find", hand_time, combinator_time, overhead};
}

// ============================================================================
// Benchmark: All predicate check
// ============================================================================

auto bench_all_hand(const std::string& data) -> bool {
    for (size_t i = 0; i < data.size(); i++) {
        if (static_cast<int>(static_cast<unsigned char>(data[i])) < 0) {
            return false;
        }
    }
    return true;
}

auto bench_all_combinator(const std::string& data) -> bool {
    cpp2::ByteBuffer buf(data.data(), data.size());
    return cpp2::combinators::reduce_from(buf)
        .all([](char c) -> bool { return static_cast<int>(static_cast<unsigned char>(c)) >= 0; });
}

auto benchmark_all(const std::string& data) -> BenchmarkResult {
    // Warmup
    for (int i = 0; i < WARMUP_ITERS; i++) {
        (void)bench_all_hand(data);
        (void)bench_all_combinator(data);
    }
    
    // Benchmark hand-written
    auto start = Clock::now();
    for (int j = 0; j < BENCH_ITERS; j++) {
        (void)bench_all_hand(data);
    }
    double hand_time = static_cast<double>((Clock::now() - start).count()) / BENCH_ITERS;
    
    // Benchmark combinator
    auto start2 = Clock::now();
    for (int k = 0; k < BENCH_ITERS; k++) {
        (void)bench_all_combinator(data);
    }
    double combinator_time = static_cast<double>((Clock::now() - start2).count()) / BENCH_ITERS;
    
    double overhead = (combinator_time - hand_time) / hand_time * 100.0;
    return BenchmarkResult{"all", hand_time, combinator_time, overhead};
}

// ============================================================================
// MAIN
// ============================================================================

int main() {
    std::cout << "=== Combinator Performance Benchmark ===\n";
    std::cout << "Target: <5% overhead vs hand-written loops\n\n";
    
    std::vector<size_t> sizes = {1000, 10000};
    
    for (size_t size : sizes) {
        std::string data = generate_data(size, 42);
        
        std::cout << "--- Data size: " << size << " bytes ---\n";
        std::cout << std::left << std::setw(35) << "Operation"
                  << std::right << std::setw(15) << "Hand-written"
                  << std::setw(15) << "Combinator"
                  << std::setw(12) << "Overhead"
                  << "  Status\n";
        std::cout << std::string(85, '-') << "\n";
        
        std::vector<BenchmarkResult> results;
        
        results.push_back(benchmark_sum(data));
        results.push_back(benchmark_filter_count(data));
        results.push_back(benchmark_take(data));
        results.push_back(benchmark_skip(data));
        results.push_back(benchmark_map(data));
        results.push_back(benchmark_pipeline(data));
        results.push_back(benchmark_find(data));
        results.push_back(benchmark_all(data));
        
        int passed = 0;
        for (const auto& r : results) {
            print_result(r);
            if (r.overhead_pct <= 5.0) {
                passed++;
            }
        }
        
        std::cout << "\nPassed: " << passed << "/" << results.size() << " benchmarks (<5% overhead)\n\n";
    }
    
    std::cout << "=== Benchmark Complete ===\n";
    return 0;
}
