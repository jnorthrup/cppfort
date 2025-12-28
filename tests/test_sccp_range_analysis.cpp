//===- test_sccp_range_analysis.cpp - SCCP Range Analysis Tests -----------===//
///
/// Tests for range analysis in SCCP lattice operations.
/// Verifies that integer ranges are properly tracked, merged, and folded.
///
//===----------------------------------------------------------------------===//

#include "../include/LatticeValue.h"
#include "../include/ConstantFolder.h"
#include <cassert>
#include <iostream>
#include <limits>

using namespace cppfort::sccp;

// ============================================================================
// Range Creation Tests
// ============================================================================

void test_create_integer_range() {
    std::cout << "Test: Create integer range\n";

    LatticeValue range = LatticeValue::getIntegerRange(10, 20);

    assert(range.getKind() == LatticeValue::IntegerRange && "Should be IntegerRange kind");
    assert(range.getMin().has_value() && "Range should have min value");
    assert(range.getMax().has_value() && "Range should have max value");
    assert(range.getMin().value() == 10LL && "Min should be 10");
    assert(range.getMax().value() == 20LL && "Max should be 20");

    std::cout << "✓ Create integer range test passed\n\n";
}

void test_range_contains_constant() {
    std::cout << "Test: Check if constant is in range\n";

    LatticeValue range = LatticeValue::getIntegerRange(10, 20);
    LatticeValue constant_in = LatticeValue::getConstant(15LL);
    LatticeValue constant_out = LatticeValue::getConstant(25LL);

    // Meet(range, constant_in_range) = constant
    LatticeValue result1 = LatticeValue::meet(range, constant_in);
    assert(result1.isConstant() && "Constant within range should remain constant");
    assert(result1.getAsInteger().value() == 15LL && "Value should be 15");

    // Meet(range, constant_out_of_range) = Bottom
    LatticeValue result2 = LatticeValue::meet(range, constant_out);
    assert(result2.isBottom() && "Constant outside range should be Bottom");

    std::cout << "✓ Range contains constant test passed\n\n";
}

// ============================================================================
// Range Intersection Tests (Meet Operation)
// ============================================================================

void test_range_meet_overlapping() {
    std::cout << "Test: Meet two overlapping ranges\n";

    LatticeValue range1 = LatticeValue::getIntegerRange(10, 20);
    LatticeValue range2 = LatticeValue::getIntegerRange(15, 25);

    // Intersection: [10, 20] ∩ [15, 25] = [15, 20]
    LatticeValue result = LatticeValue::meet(range1, range2);

    assert(result.getKind() == LatticeValue::IntegerRange && "Result should be range");
    assert(result.getMin().value() == 15LL && "Min should be max(10, 15) = 15");
    assert(result.getMax().value() == 20LL && "Max should be min(20, 25) = 20");

    std::cout << "✓ Range meet overlapping test passed\n\n";
}

void test_range_meet_disjoint() {
    std::cout << "Test: Meet two disjoint ranges\n";

    LatticeValue range1 = LatticeValue::getIntegerRange(10, 20);
    LatticeValue range2 = LatticeValue::getIntegerRange(30, 40);

    // Disjoint ranges: [10, 20] ∩ [30, 40] = ∅ (Bottom)
    LatticeValue result = LatticeValue::meet(range1, range2);

    assert(result.isBottom() && "Disjoint ranges should meet to Bottom");

    std::cout << "✓ Range meet disjoint test passed\n\n";
}

void test_range_meet_subset() {
    std::cout << "Test: Meet range with subset range\n";

    LatticeValue range1 = LatticeValue::getIntegerRange(10, 30);
    LatticeValue range2 = LatticeValue::getIntegerRange(15, 20);

    // [10, 30] ∩ [15, 20] = [15, 20]
    LatticeValue result = LatticeValue::meet(range1, range2);

    assert(result.getKind() == LatticeValue::IntegerRange && "Result should be range");
    assert(result.getMin().value() == 15LL && "Min should be 15");
    assert(result.getMax().value() == 20LL && "Max should be 20");

    std::cout << "✓ Range meet subset test passed\n\n";
}

void test_range_meet_identical() {
    std::cout << "Test: Meet identical ranges\n";

    LatticeValue range1 = LatticeValue::getIntegerRange(10, 20);
    LatticeValue range2 = LatticeValue::getIntegerRange(10, 20);

    // [10, 20] ∩ [10, 20] = [10, 20]
    LatticeValue result = LatticeValue::meet(range1, range2);

    assert(result.getKind() == LatticeValue::IntegerRange && "Result should be range");
    assert(result.getMin().value() == 10LL && "Min should be 10");
    assert(result.getMax().value() == 20LL && "Max should be 20");

    std::cout << "✓ Range meet identical test passed\n\n";
}

// ============================================================================
// Range Arithmetic Tests
// ============================================================================

void test_range_add_constant() {
    std::cout << "Test: Add constant to range\n";

    LatticeValue range = LatticeValue::getIntegerRange(10, 20);
    LatticeValue constant = LatticeValue::getConstant(5LL);

    // [10, 20] + 5 = [15, 25]
    LatticeValue result = ConstantFolder::foldAdd(range, constant);

    assert(result.getKind() == LatticeValue::IntegerRange && "Result should be range");
    assert(result.getMin().value() == 15LL && "Min should be 10 + 5 = 15");
    assert(result.getMax().value() == 25LL && "Max should be 20 + 5 = 25");

    std::cout << "✓ Range add constant test passed\n\n";
}

void test_range_add_negative_constant() {
    std::cout << "Test: Add negative constant to range\n";

    LatticeValue range = LatticeValue::getIntegerRange(10, 20);
    LatticeValue constant = LatticeValue::getConstant(-3LL);

    // [10, 20] + (-3) = [7, 17]
    LatticeValue result = ConstantFolder::foldAdd(range, constant);

    assert(result.getKind() == LatticeValue::IntegerRange && "Result should be range");
    assert(result.getMin().value() == 7LL && "Min should be 10 - 3 = 7");
    assert(result.getMax().value() == 17LL && "Max should be 20 - 3 = 17");

    std::cout << "✓ Range add negative constant test passed\n\n";
}

void test_range_add_range() {
    std::cout << "Test: Add two ranges\n";

    LatticeValue range1 = LatticeValue::getIntegerRange(10, 20);
    LatticeValue range2 = LatticeValue::getIntegerRange(5, 10);

    // [10, 20] + [5, 10] = [15, 30]
    LatticeValue result = ConstantFolder::foldAdd(range1, range2);

    assert(result.getKind() == LatticeValue::IntegerRange && "Result should be range");
    assert(result.getMin().value() == 15LL && "Min should be 10 + 5 = 15");
    assert(result.getMax().value() == 30LL && "Max should be 20 + 10 = 30");

    std::cout << "✓ Range add range test passed\n\n";
}

void test_range_sub_constant() {
    std::cout << "Test: Subtract constant from range\n";

    LatticeValue range = LatticeValue::getIntegerRange(10, 20);
    LatticeValue constant = LatticeValue::getConstant(3LL);

    // [10, 20] - 3 = [7, 17]
    LatticeValue result = ConstantFolder::foldSub(range, constant);

    assert(result.getKind() == LatticeValue::IntegerRange && "Result should be range");
    assert(result.getMin().value() == 7LL && "Min should be 10 - 3 = 7");
    assert(result.getMax().value() == 17LL && "Max should be 20 - 3 = 17");

    std::cout << "✓ Range sub constant test passed\n\n";
}

// ============================================================================
// Boundary Condition Tests
// ============================================================================

void test_range_boundary_min_max() {
    std::cout << "Test: Range with extreme values\n";

    LatticeValue range = LatticeValue::getIntegerRange(INT64_MIN, INT64_MAX);

    assert(range.getKind() == LatticeValue::IntegerRange && "Should be IntegerRange");
    assert(range.getMin().value() == INT64_MIN && "Min should be INT64_MIN");
    assert(range.getMax().value() == INT64_MAX && "Max should be INT64_MAX");

    std::cout << "✓ Range boundary min/max test passed\n\n";
}

void test_range_single_value() {
    std::cout << "Test: Range with single value (point range)\n";

    LatticeValue range = LatticeValue::getIntegerRange(42, 42);

    assert(range.getKind() == LatticeValue::IntegerRange && "Should be IntegerRange");
    assert(range.getMin().value() == 42LL && "Min should be 42");
    assert(range.getMax().value() == 42LL && "Max should be 42");

    // Meeting with constant of same value should produce constant
    LatticeValue constant = LatticeValue::getConstant(42LL);
    LatticeValue result = LatticeValue::meet(range, constant);

    assert(result.isConstant() && "Point range meets constant should be constant");
    assert(result.getAsInteger().value() == 42LL && "Value should be 42");

    std::cout << "✓ Range single value test passed\n\n";
}

void test_range_overflow_detection() {
    std::cout << "Test: Range overflow detection in addition\n";

    LatticeValue range = LatticeValue::getIntegerRange(INT64_MAX - 10, INT64_MAX);
    LatticeValue constant = LatticeValue::getConstant(20LL);

    // Adding 20 to [INT64_MAX-10, INT64_MAX] would overflow
    LatticeValue result = ConstantFolder::foldAdd(range, constant);

    // Implementation should detect overflow and return Top
    assert(result.isTop() && "Overflow should return Top");

    std::cout << "✓ Range overflow detection test passed\n\n";
}

void test_range_underflow_detection() {
    std::cout << "Test: Range underflow detection in subtraction\n";

    LatticeValue range = LatticeValue::getIntegerRange(INT64_MIN, INT64_MIN + 10);
    LatticeValue constant = LatticeValue::getConstant(20LL);

    // Subtracting 20 from [INT64_MIN, INT64_MIN+10] would underflow
    LatticeValue result = ConstantFolder::foldSub(range, constant);

    // Implementation should detect underflow and return Top
    assert(result.isTop() && "Underflow should return Top");

    std::cout << "✓ Range underflow detection test passed\n\n";
}

// ============================================================================
// Main Test Runner
// ============================================================================

int main() {
    std::cout << "========================================\n";
    std::cout << "SCCP Range Analysis Tests\n";
    std::cout << "========================================\n\n";

    // Range creation tests
    test_create_integer_range();
    test_range_contains_constant();

    // Range meet tests
    test_range_meet_overlapping();
    test_range_meet_disjoint();
    test_range_meet_subset();
    test_range_meet_identical();

    // Range arithmetic tests
    test_range_add_constant();
    test_range_add_negative_constant();
    test_range_add_range();
    test_range_sub_constant();

    // Boundary condition tests
    test_range_boundary_min_max();
    test_range_single_value();
    test_range_overflow_detection();
    test_range_underflow_detection();

    std::cout << "========================================\n";
    std::cout << "All Range Analysis Tests Passed!\n";
    std::cout << "========================================\n";

    return 0;
}
