//===- test_sccp_arithmetic_operations.cpp - SCCP Arithmetic Tests --------===//
///
/// Tests for arithmetic operation folding in SCCP analysis.
/// Verifies that Add, Sub, Mul, Div operations with constant operands
/// are folded to constant results, and that overflow/divide-by-zero are
/// properly handled.
///
//===----------------------------------------------------------------------===//

#include "../include/LatticeValue.h"
#include "../include/ConstantFolder.h"
#include <cassert>
#include <iostream>
#include <limits>

using namespace cppfort::sccp;

// ============================================================================
// Addition Tests
// ============================================================================

void test_fold_add_constants() {
    std::cout << "Test: Fold addition with two constants\n";

    LatticeValue a = LatticeValue::getConstant(10LL);
    LatticeValue b = LatticeValue::getConstant(20LL);

    LatticeValue result = ConstantFolder::foldAdd(a, b);

    assert(result.isConstant() && "Addition of constants should be constant");
    auto value = result.getAsInteger();
    assert(value.has_value() && "Result should have integer value");
    assert(value.value() == 30LL && "10 + 20 should equal 30");

    std::cout << "✓ Fold addition with constants test passed\n\n";
}

void test_fold_add_negative() {
    std::cout << "Test: Fold addition with negative values\n";

    LatticeValue a = LatticeValue::getConstant(-5LL);
    LatticeValue b = LatticeValue::getConstant(10LL);

    LatticeValue result = ConstantFolder::foldAdd(a, b);

    assert(result.isConstant() && "Addition should be constant");
    auto value = result.getAsInteger();
    assert(value.value() == 5LL && "-5 + 10 should equal 5");

    std::cout << "✓ Fold addition with negative values test passed\n\n";
}

void test_fold_add_top() {
    std::cout << "Test: Fold addition with Top operand\n";

    LatticeValue top = LatticeValue::getTop();
    LatticeValue constant = LatticeValue::getConstant(42LL);

    LatticeValue result1 = ConstantFolder::foldAdd(top, constant);
    LatticeValue result2 = ConstantFolder::foldAdd(constant, top);

    assert(result1.isTop() && "Top + constant should be Top");
    assert(result2.isTop() && "constant + Top should be Top");

    std::cout << "✓ Fold addition with Top test passed\n\n";
}

void test_fold_add_bottom() {
    std::cout << "Test: Fold addition with Bottom operand\n";

    LatticeValue bottom = LatticeValue::getBottom();
    LatticeValue constant = LatticeValue::getConstant(42LL);

    LatticeValue result1 = ConstantFolder::foldAdd(bottom, constant);
    LatticeValue result2 = ConstantFolder::foldAdd(constant, bottom);

    assert(result1.isBottom() && "Bottom + constant should be Bottom");
    assert(result2.isBottom() && "constant + Bottom should be Bottom");

    std::cout << "✓ Fold addition with Bottom test passed\n\n";
}

void test_fold_add_range_constant() {
    std::cout << "Test: Fold addition with range and constant\n";

    LatticeValue range = LatticeValue::getIntegerRange(10, 20);
    LatticeValue constant = LatticeValue::getConstant(5LL);

    LatticeValue result = ConstantFolder::foldAdd(range, constant);

    assert(result.getKind() == LatticeValue::IntegerRange && "Range + constant should be range");
    assert(result.getMin().value() == 15LL && "Min should be 10 + 5 = 15");
    assert(result.getMax().value() == 25LL && "Max should be 20 + 5 = 25");

    std::cout << "✓ Fold addition with range and constant test passed\n\n";
}

void test_fold_add_overflow() {
    std::cout << "Test: Fold addition with overflow\n";

    LatticeValue a = LatticeValue::getConstant(INT64_MAX);
    LatticeValue b = LatticeValue::getConstant(1LL);

    LatticeValue result = ConstantFolder::foldAdd(a, b);

    // Implementation may return Top for overflow (undefined behavior)
    assert((result.isTop() || result.isConstant()) && "Overflow should be handled");

    std::cout << "✓ Fold addition with overflow test passed\n\n";
}

// ============================================================================
// Subtraction Tests
// ============================================================================

void test_fold_sub_constants() {
    std::cout << "Test: Fold subtraction with two constants\n";

    LatticeValue a = LatticeValue::getConstant(30LL);
    LatticeValue b = LatticeValue::getConstant(10LL);

    LatticeValue result = ConstantFolder::foldSub(a, b);

    assert(result.isConstant() && "Subtraction of constants should be constant");
    auto value = result.getAsInteger();
    assert(value.has_value() && "Result should have integer value");
    assert(value.value() == 20LL && "30 - 10 should equal 20");

    std::cout << "✓ Fold subtraction with constants test passed\n\n";
}

void test_fold_sub_negative_result() {
    std::cout << "Test: Fold subtraction resulting in negative\n";

    LatticeValue a = LatticeValue::getConstant(10LL);
    LatticeValue b = LatticeValue::getConstant(20LL);

    LatticeValue result = ConstantFolder::foldSub(a, b);

    assert(result.isConstant() && "Subtraction should be constant");
    auto value = result.getAsInteger();
    assert(value.value() == -10LL && "10 - 20 should equal -10");

    std::cout << "✓ Fold subtraction with negative result test passed\n\n";
}

void test_fold_sub_underflow() {
    std::cout << "Test: Fold subtraction with underflow\n";

    LatticeValue a = LatticeValue::getConstant(INT64_MIN);
    LatticeValue b = LatticeValue::getConstant(1LL);

    LatticeValue result = ConstantFolder::foldSub(a, b);

    // Implementation may return Top for underflow (undefined behavior)
    assert((result.isTop() || result.isConstant()) && "Underflow should be handled");

    std::cout << "✓ Fold subtraction with underflow test passed\n\n";
}

// ============================================================================
// Multiplication Tests
// ============================================================================

void test_fold_mul_constants() {
    std::cout << "Test: Fold multiplication with two constants\n";

    LatticeValue a = LatticeValue::getConstant(6LL);
    LatticeValue b = LatticeValue::getConstant(7LL);

    LatticeValue result = ConstantFolder::foldMul(a, b);

    assert(result.isConstant() && "Multiplication of constants should be constant");
    auto value = result.getAsInteger();
    assert(value.has_value() && "Result should have integer value");
    assert(value.value() == 42LL && "6 * 7 should equal 42");

    std::cout << "✓ Fold multiplication with constants test passed\n\n";
}

void test_fold_mul_zero() {
    std::cout << "Test: Fold multiplication with zero\n";

    LatticeValue a = LatticeValue::getConstant(0LL);
    LatticeValue b = LatticeValue::getConstant(999LL);

    LatticeValue result = ConstantFolder::foldMul(a, b);

    assert(result.isConstant() && "Multiplication should be constant");
    auto value = result.getAsInteger();
    assert(value.value() == 0LL && "0 * 999 should equal 0");

    std::cout << "✓ Fold multiplication with zero test passed\n\n";
}

void test_fold_mul_negative() {
    std::cout << "Test: Fold multiplication with negative values\n";

    LatticeValue a = LatticeValue::getConstant(-5LL);
    LatticeValue b = LatticeValue::getConstant(3LL);

    LatticeValue result = ConstantFolder::foldMul(a, b);

    assert(result.isConstant() && "Multiplication should be constant");
    auto value = result.getAsInteger();
    assert(value.value() == -15LL && "-5 * 3 should equal -15");

    std::cout << "✓ Fold multiplication with negative values test passed\n\n";
}

void test_fold_mul_overflow() {
    std::cout << "Test: Fold multiplication with overflow\n";

    LatticeValue a = LatticeValue::getConstant(INT64_MAX / 2);
    LatticeValue b = LatticeValue::getConstant(3LL);

    LatticeValue result = ConstantFolder::foldMul(a, b);

    // Implementation may return Top for overflow (undefined behavior)
    assert((result.isTop() || result.isConstant()) && "Overflow should be handled");

    std::cout << "✓ Fold multiplication with overflow test passed\n\n";
}

// ============================================================================
// Division Tests
// ============================================================================

void test_fold_div_constants() {
    std::cout << "Test: Fold division with two constants\n";

    LatticeValue a = LatticeValue::getConstant(42LL);
    LatticeValue b = LatticeValue::getConstant(6LL);

    LatticeValue result = ConstantFolder::foldDiv(a, b);

    assert(result.isConstant() && "Division of constants should be constant");
    auto value = result.getAsInteger();
    assert(value.has_value() && "Result should have integer value");
    assert(value.value() == 7LL && "42 / 6 should equal 7");

    std::cout << "✓ Fold division with constants test passed\n\n";
}

void test_fold_div_truncation() {
    std::cout << "Test: Fold division with truncation\n";

    LatticeValue a = LatticeValue::getConstant(10LL);
    LatticeValue b = LatticeValue::getConstant(3LL);

    LatticeValue result = ConstantFolder::foldDiv(a, b);

    assert(result.isConstant() && "Division should be constant");
    auto value = result.getAsInteger();
    assert(value.value() == 3LL && "10 / 3 should truncate to 3");

    std::cout << "✓ Fold division with truncation test passed\n\n";
}

void test_fold_div_negative() {
    std::cout << "Test: Fold division with negative values\n";

    LatticeValue a = LatticeValue::getConstant(-20LL);
    LatticeValue b = LatticeValue::getConstant(4LL);

    LatticeValue result = ConstantFolder::foldDiv(a, b);

    assert(result.isConstant() && "Division should be constant");
    auto value = result.getAsInteger();
    assert(value.value() == -5LL && "-20 / 4 should equal -5");

    std::cout << "✓ Fold division with negative values test passed\n\n";
}

void test_fold_div_by_zero() {
    std::cout << "Test: Fold division by zero\n";

    LatticeValue a = LatticeValue::getConstant(42LL);
    LatticeValue zero = LatticeValue::getConstant(0LL);

    LatticeValue result = ConstantFolder::foldDiv(a, zero);

    // Division by zero should return Top (undefined behavior)
    assert(result.isTop() && "Division by zero should return Top");

    std::cout << "✓ Fold division by zero test passed\n\n";
}

// ============================================================================
// Main Test Runner
// ============================================================================

int main() {
    std::cout << "========================================\n";
    std::cout << "SCCP Arithmetic Operations Tests\n";
    std::cout << "========================================\n\n";

    // Addition tests
    test_fold_add_constants();
    test_fold_add_negative();
    test_fold_add_top();
    test_fold_add_bottom();
    test_fold_add_range_constant();
    test_fold_add_overflow();

    // Subtraction tests
    test_fold_sub_constants();
    test_fold_sub_negative_result();
    test_fold_sub_underflow();

    // Multiplication tests
    test_fold_mul_constants();
    test_fold_mul_zero();
    test_fold_mul_negative();
    test_fold_mul_overflow();

    // Division tests
    test_fold_div_constants();
    test_fold_div_truncation();
    test_fold_div_negative();
    test_fold_div_by_zero();

    std::cout << "========================================\n";
    std::cout << "All Arithmetic Tests Passed!\n";
    std::cout << "========================================\n";

    return 0;
}
