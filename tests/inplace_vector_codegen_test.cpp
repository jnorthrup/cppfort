// inplace_vector Code Generation Validation Test
// Validates that SBO (Small Buffer Optimization) sizing is correctly applied
// when generating code with cpp2::inplace_vector
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "../include/cpp2/reflection_sbo.hpp"
#include <iostream>
#include <cassert>
#include <string>
#include <cstdint>

using namespace cpp2;

// ============================================================================
// Test: SBO Capacity Computation for inplace_vector Code Generation
// ============================================================================

void test_sbo_capacity_codegen_primitives() {
    std::cout << "Running test_sbo_capacity_codegen_primitives...\n";

    // For primitive types, verify inplace_vector capacity matches SBO calculation
    constexpr auto char_cap = sbo_capacity<char>();
    constexpr auto int_cap = sbo_capacity<int>();
    constexpr auto double_cap = sbo_capacity<double>();

    // inplace_vector<char, N> should have N = sbo_capacity<char>()
    inplace_vector<char, char_cap> char_vec;
    assert(char_vec.capacity() == char_cap);
    assert(char_cap == 64); // 64 bytes / 1 byte = 64

    // inplace_vector<int, N> should have N = sbo_capacity<int>()
    inplace_vector<int, int_cap> int_vec;
    assert(int_vec.capacity() == int_cap);
    assert(int_cap == 9); // 64 bytes / (4 + 3) = 9

    // inplace_vector<double, N> should have N = sbo_capacity<double>()
    inplace_vector<double, double_cap> double_vec;
    assert(double_vec.capacity() == double_cap);
    assert(double_cap == 4); // 64 bytes / (8 + 7) = 4

    std::cout << "  PASS: inplace_vector capacities match SBO calculations\n";
    std::cout << "    char capacity: " << char_cap << "\n";
    std::cout << "    int capacity: " << int_cap << "\n";
    std::cout << "    double capacity: " << double_cap << "\n";
}

void test_sbo_capacity_codegen_aggregates() {
    std::cout << "Running test_sbo_capacity_codegen_aggregates...\n";

    struct Small { int x; };      // 4 bytes
    struct Medium { int x, y; };  // 8 bytes
    struct Large { int data[20]; }; // 80 bytes

    constexpr auto small_cap = sbo_capacity<Small>();
    constexpr auto medium_cap = sbo_capacity<Medium>();
    constexpr auto large_cap = sbo_capacity<Large>();

    // inplace_vector with aggregate types
    inplace_vector<Small, small_cap> small_vec;
    inplace_vector<Medium, medium_cap> medium_vec;
    inplace_vector<Large, large_cap> large_vec;

    assert(small_vec.capacity() == small_cap);
    assert(small_cap == 9);

    assert(medium_vec.capacity() == medium_cap);
    assert(medium_cap == 5);

    assert(large_vec.capacity() == large_cap);
    assert(large_cap == 1); // Clamped to MIN_SBO_CAPACITY

    std::cout << "  PASS: Aggregate inplace_vector capacities correct\n";
    std::cout << "    Small(4b): " << small_cap << "\n";
    std::cout << "    Medium(8b): " << medium_cap << "\n";
    std::cout << "    Large(80b): " << large_cap << "\n";
}

void test_sbo_vector_alias_codegen() {
    std::cout << "Running test_sbo_vector_alias_codegen...\n";

    // sbo_vector<T> is an alias for inplace_vector<T, sbo_capacity<T>()>
    // This test validates the automatic capacity selection

    sbo_vector<int> int_vec;  // Should have capacity = sbo_capacity<int>() = 9
    sbo_vector<double> double_vec;  // Should have capacity = sbo_capacity<double>() = 4

    assert(int_vec.capacity() == 9);
    assert(double_vec.capacity() == 4);

    // Verify we can use the full capacity
    for (int i = 0; i < 9; ++i) {
        int_vec.push_back(i);
    }
    assert(int_vec.size() == 9);

    for (int i = 0; i < 4; ++i) {
        double_vec.push_back(i * 1.5);
    }
    assert(double_vec.size() == 4);

    std::cout << "  PASS: sbo_vector alias automatically selects capacity\n";
    std::cout << "    sbo_vector<int> capacity: " << int_vec.capacity() << "\n";
    std::cout << "    sbo_vector<double> capacity: " << double_vec.capacity() << "\n";
}

void test_sbo_config_codegen_metadata() {
    std::cout << "Running test_sbo_config_codegen_metadata...\n";

    // sbo_config<T> provides compile-time metadata for code generation
    using IntConfig = sbo_config<int>;
    using DoubleConfig = sbo_config<double>;
    using StringConfig = sbo_config<std::string>;

    // Verify config values are constexpr and can be used for code generation
    constexpr auto int_capacity = IntConfig::capacity;
    constexpr auto int_buffer_size = IntConfig::buffer_size;
    constexpr auto int_eligible = IntConfig::eligible;

    static_assert(int_capacity == 9, "int SBO capacity mismatch");
    static_assert(int_buffer_size == 36, "int SBO buffer size mismatch");
    static_assert(int_eligible == true, "int should be SBO eligible");

    // std::string is larger, so capacity should be smaller
    constexpr auto string_capacity = StringConfig::capacity;
    // sizeof(std::string) varies, but should be at least MIN_SBO_CAPACITY
    assert(string_capacity >= 1);

    std::cout << "  PASS: sbo_config provides valid metadata for codegen\n";
    std::cout << "    int: capacity=" << int_capacity << ", buffer=" << int_buffer_size << " bytes\n";
    std::cout << "    string: capacity=" << string_capacity << "\n";
}

void test_reflection_driven_sbo_codegen() {
    std::cout << "Running test_reflection_driven_sbo_codegen...\n";

    // reflection_driven_sbo_size() is the entry point for code generation
    // It uses std::meta when available, otherwise falls back to template metaprogramming

    constexpr auto int_sbo_size = reflection_driven_sbo_size<int>();
    constexpr auto double_sbo_size = reflection_driven_sbo_size<double>();

    // Should match sbo_capacity() result
    assert(int_sbo_size == sbo_capacity<int>());
    assert(double_sbo_size == sbo_capacity<double>());

    // Generate inplace_vector with reflection-driven size
    inplace_vector<int, int_sbo_size> int_vec;
    inplace_vector<double, double_sbo_size> double_vec;

    assert(int_vec.capacity() == int_sbo_size);
    assert(double_vec.capacity() == double_sbo_size);

    std::cout << "  PASS: reflection_driven_sbo_size() produces valid capacities\n";
    std::cout << "    int SBO size: " << int_sbo_size << "\n";
    std::cout << "    double SBO size: " << double_sbo_size << "\n";

#ifdef __cpp_static_reflection
    std::cout << "    Using: C++26 std::meta (native reflection)\n";
#else
    std::cout << "    Using: C++23 template metaprogramming fallback\n";
#endif
}

void test_alignment_aware_sbo_codegen() {
    std::cout << "Running test_alignment_aware_sbo_codegen...\n";

    // Types with stricter alignment should get appropriate capacities
    struct Align8 { alignas(8) int x; };
    struct Align16 { alignas(16) int x; };
    struct Align32 { alignas(32) int x; };

    constexpr auto align8_cap = sbo_capacity<Align8>();
    constexpr auto align16_cap = sbo_capacity<Align16>();
    constexpr auto align32_cap = sbo_capacity<Align32>();

    // Alignment requirements should reduce capacity
    assert(align8_cap >= align16_cap);
    assert(align16_cap >= align32_cap);

    inplace_vector<Align8, align8_cap> align8_vec;
    inplace_vector<Align16, align16_cap> align16_vec;
    inplace_vector<Align32, align32_cap> align32_vec;

    assert(align8_vec.capacity() == align8_cap);
    assert(align16_vec.capacity() == align16_cap);
    assert(align32_vec.capacity() == align32_cap);

    std::cout << "  PASS: Alignment-aware SBO sizing for code generation\n";
    std::cout << "    alignas(8): " << align8_cap << " elements\n";
    std::cout << "    alignas(16): " << align16_cap << " elements\n";
    std::cout << "    alignas(32): " << align32_cap << " elements\n";
}

void test_codegen_buffer_size_validation() {
    std::cout << "Running test_codegen_buffer_size_validation...\n";

    // Validate that inplace_vector buffer size doesn't exceed expected limits
    constexpr auto int_buffer = sbo_buffer_size<int>();
    constexpr auto double_buffer = sbo_buffer_size<double>();

    // Buffer size should be capacity * sizeof(T)
    constexpr auto int_cap = sbo_capacity<int>();
    constexpr auto double_cap = sbo_capacity<double>();

    static_assert(int_buffer == int_cap * sizeof(int), "int buffer size mismatch");
    static_assert(double_buffer == double_cap * sizeof(double), "double buffer size mismatch");

    // Buffer sizes should be reasonable (not exceed 2x target for eligible types)
    static_assert(int_buffer <= DEFAULT_SBO_BUFFER_SIZE * 2, "int buffer too large");
    static_assert(double_buffer <= DEFAULT_SBO_BUFFER_SIZE * 2, "double buffer too large");

    std::cout << "  PASS: inplace_vector buffer sizes validated for code generation\n";
    std::cout << "    int buffer: " << int_buffer << " bytes (target: " << DEFAULT_SBO_BUFFER_SIZE << ")\n";
    std::cout << "    double buffer: " << double_buffer << " bytes (target: " << DEFAULT_SBO_BUFFER_SIZE << ")\n";
}

// ============================================================================
// Main
// ============================================================================

int main() {
    std::cout << "=== inplace_vector Code Generation SBO Validation ===\n";
    std::cout << "Validating SBO sizing for code generation with inplace_vector\n\n";

    test_sbo_capacity_codegen_primitives();
    test_sbo_capacity_codegen_aggregates();
    test_sbo_vector_alias_codegen();
    test_sbo_config_codegen_metadata();
    test_reflection_driven_sbo_codegen();
    test_alignment_aware_sbo_codegen();
    test_codegen_buffer_size_validation();

    std::cout << "\n=== All 7 Tests PASSED ===\n";
    std::cout << "\nValidation Summary:\n";
    std::cout << "- inplace_vector capacity matches SBO calculation for primitives\n";
    std::cout << "- inplace_vector capacity matches SBO calculation for aggregates\n";
    std::cout << "- sbo_vector alias automatically selects correct capacity\n";
    std::cout << "- sbo_config provides valid compile-time metadata\n";
    std::cout << "- reflection_driven_sbo_size() produces correct capacities\n";
    std::cout << "- Alignment requirements handled correctly\n";
    std::cout << "- Buffer size validation for code generation\n";
    std::cout << "\nConclusion: SBO sizing validated for inplace_vector code generation\n";
    std::cout << "Generated code using reflection_driven_sbo_size<T>() will have\n";
    std::cout << "optimal inplace_vector<T, N> capacities for NoEscape aggregates.\n";

    return 0;
}
