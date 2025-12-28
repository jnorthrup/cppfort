//===- test_sccp_logical_operations.cpp - SCCP Logical Tests --------------===//
///
/// Tests for logical operation folding in SCCP analysis.
/// Verifies that And, Or, Not operations with constant operands
/// are folded to constant boolean results.
///
//===----------------------------------------------------------------------===//

#include "../include/LatticeValue.h"
#include "../include/ConstantFolder.h"
#include <cassert>
#include <iostream>

using namespace cppfort::sccp;

// ============================================================================
// Logical AND Tests
// ============================================================================

void test_fold_and_true_true() {
    std::cout << "Test: Fold AND with true && true\n";

    LatticeValue a = LatticeValue::getConstant(true);
    LatticeValue b = LatticeValue::getConstant(true);

    LatticeValue result = ConstantFolder::foldAnd(a, b);

    assert(result.isConstant() && "AND of constants should be constant");
    auto value = result.getAsBoolean();
    assert(value.has_value() && "Result should have boolean value");
    assert(value.value() == true && "true && true should equal true");

    std::cout << "✓ Fold AND true && true test passed\n\n";
}

void test_fold_and_true_false() {
    std::cout << "Test: Fold AND with true && false\n";

    LatticeValue a = LatticeValue::getConstant(true);
    LatticeValue b = LatticeValue::getConstant(false);

    LatticeValue result = ConstantFolder::foldAnd(a, b);

    assert(result.isConstant() && "AND should be constant");
    auto value = result.getAsBoolean();
    assert(value.value() == false && "true && false should equal false");

    std::cout << "✓ Fold AND true && false test passed\n\n";
}

void test_fold_and_false_true() {
    std::cout << "Test: Fold AND with false && true\n";

    LatticeValue a = LatticeValue::getConstant(false);
    LatticeValue b = LatticeValue::getConstant(true);

    LatticeValue result = ConstantFolder::foldAnd(a, b);

    assert(result.isConstant() && "AND should be constant");
    auto value = result.getAsBoolean();
    assert(value.value() == false && "false && true should equal false");

    std::cout << "✓ Fold AND false && true test passed\n\n";
}

void test_fold_and_false_false() {
    std::cout << "Test: Fold AND with false && false\n";

    LatticeValue a = LatticeValue::getConstant(false);
    LatticeValue b = LatticeValue::getConstant(false);

    LatticeValue result = ConstantFolder::foldAnd(a, b);

    assert(result.isConstant() && "AND should be constant");
    auto value = result.getAsBoolean();
    assert(value.value() == false && "false && false should equal false");

    std::cout << "✓ Fold AND false && false test passed\n\n";
}

void test_fold_and_top() {
    std::cout << "Test: Fold AND with Top operand\n";

    LatticeValue top = LatticeValue::getTop();
    LatticeValue constant = LatticeValue::getConstant(true);

    LatticeValue result1 = ConstantFolder::foldAnd(top, constant);
    LatticeValue result2 = ConstantFolder::foldAnd(constant, top);

    assert(result1.isTop() && "Top && constant should be Top");
    assert(result2.isTop() && "constant && Top should be Top");

    std::cout << "✓ Fold AND with Top test passed\n\n";
}

void test_fold_and_bottom() {
    std::cout << "Test: Fold AND with Bottom operand\n";

    LatticeValue bottom = LatticeValue::getBottom();
    LatticeValue constant = LatticeValue::getConstant(true);

    LatticeValue result1 = ConstantFolder::foldAnd(bottom, constant);
    LatticeValue result2 = ConstantFolder::foldAnd(constant, bottom);

    assert(result1.isBottom() && "Bottom && constant should be Bottom");
    assert(result2.isBottom() && "constant && Bottom should be Bottom");

    std::cout << "✓ Fold AND with Bottom test passed\n\n";
}

// ============================================================================
// Logical OR Tests
// ============================================================================

void test_fold_or_true_true() {
    std::cout << "Test: Fold OR with true || true\n";

    LatticeValue a = LatticeValue::getConstant(true);
    LatticeValue b = LatticeValue::getConstant(true);

    LatticeValue result = ConstantFolder::foldOr(a, b);

    assert(result.isConstant() && "OR of constants should be constant");
    auto value = result.getAsBoolean();
    assert(value.has_value() && "Result should have boolean value");
    assert(value.value() == true && "true || true should equal true");

    std::cout << "✓ Fold OR true || true test passed\n\n";
}

void test_fold_or_true_false() {
    std::cout << "Test: Fold OR with true || false\n";

    LatticeValue a = LatticeValue::getConstant(true);
    LatticeValue b = LatticeValue::getConstant(false);

    LatticeValue result = ConstantFolder::foldOr(a, b);

    assert(result.isConstant() && "OR should be constant");
    auto value = result.getAsBoolean();
    assert(value.value() == true && "true || false should equal true");

    std::cout << "✓ Fold OR true || false test passed\n\n";
}

void test_fold_or_false_true() {
    std::cout << "Test: Fold OR with false || true\n";

    LatticeValue a = LatticeValue::getConstant(false);
    LatticeValue b = LatticeValue::getConstant(true);

    LatticeValue result = ConstantFolder::foldOr(a, b);

    assert(result.isConstant() && "OR should be constant");
    auto value = result.getAsBoolean();
    assert(value.value() == true && "false || true should equal true");

    std::cout << "✓ Fold OR false || true test passed\n\n";
}

void test_fold_or_false_false() {
    std::cout << "Test: Fold OR with false || false\n";

    LatticeValue a = LatticeValue::getConstant(false);
    LatticeValue b = LatticeValue::getConstant(false);

    LatticeValue result = ConstantFolder::foldOr(a, b);

    assert(result.isConstant() && "OR should be constant");
    auto value = result.getAsBoolean();
    assert(value.value() == false && "false || false should equal false");

    std::cout << "✓ Fold OR false || false test passed\n\n";
}

void test_fold_or_top() {
    std::cout << "Test: Fold OR with Top operand\n";

    LatticeValue top = LatticeValue::getTop();
    LatticeValue constant = LatticeValue::getConstant(true);

    LatticeValue result1 = ConstantFolder::foldOr(top, constant);
    LatticeValue result2 = ConstantFolder::foldOr(constant, top);

    assert(result1.isTop() && "Top || constant should be Top");
    assert(result2.isTop() && "constant || Top should be Top");

    std::cout << "✓ Fold OR with Top test passed\n\n";
}

void test_fold_or_bottom() {
    std::cout << "Test: Fold OR with Bottom operand\n";

    LatticeValue bottom = LatticeValue::getBottom();
    LatticeValue constant = LatticeValue::getConstant(true);

    LatticeValue result1 = ConstantFolder::foldOr(bottom, constant);
    LatticeValue result2 = ConstantFolder::foldOr(constant, bottom);

    assert(result1.isBottom() && "Bottom || constant should be Bottom");
    assert(result2.isBottom() && "constant || Bottom should be Bottom");

    std::cout << "✓ Fold OR with Bottom test passed\n\n";
}

// ============================================================================
// Logical NOT Tests
// ============================================================================

void test_fold_not_true() {
    std::cout << "Test: Fold NOT with true\n";

    LatticeValue a = LatticeValue::getConstant(true);

    LatticeValue result = ConstantFolder::foldNot(a);

    assert(result.isConstant() && "NOT of constant should be constant");
    auto value = result.getAsBoolean();
    assert(value.has_value() && "Result should have boolean value");
    assert(value.value() == false && "!true should equal false");

    std::cout << "✓ Fold NOT true test passed\n\n";
}

void test_fold_not_false() {
    std::cout << "Test: Fold NOT with false\n";

    LatticeValue a = LatticeValue::getConstant(false);

    LatticeValue result = ConstantFolder::foldNot(a);

    assert(result.isConstant() && "NOT of constant should be constant");
    auto value = result.getAsBoolean();
    assert(value.value() == true && "!false should equal true");

    std::cout << "✓ Fold NOT false test passed\n\n";
}

void test_fold_not_top() {
    std::cout << "Test: Fold NOT with Top\n";

    LatticeValue top = LatticeValue::getTop();

    LatticeValue result = ConstantFolder::foldNot(top);

    assert(result.isTop() && "!Top should be Top");

    std::cout << "✓ Fold NOT Top test passed\n\n";
}

void test_fold_not_bottom() {
    std::cout << "Test: Fold NOT with Bottom\n";

    LatticeValue bottom = LatticeValue::getBottom();

    LatticeValue result = ConstantFolder::foldNot(bottom);

    assert(result.isBottom() && "!Bottom should be Bottom");

    std::cout << "✓ Fold NOT Bottom test passed\n\n";
}

void test_fold_not_double_negation() {
    std::cout << "Test: Fold NOT with double negation\n";

    LatticeValue a = LatticeValue::getConstant(true);

    LatticeValue result1 = ConstantFolder::foldNot(a);
    LatticeValue result2 = ConstantFolder::foldNot(result1);

    assert(result2.isConstant() && "!!constant should be constant");
    auto value = result2.getAsBoolean();
    assert(value.value() == true && "!!true should equal true");

    std::cout << "✓ Fold NOT double negation test passed\n\n";
}

// ============================================================================
// Main Test Runner
// ============================================================================

int main() {
    std::cout << "========================================\n";
    std::cout << "SCCP Logical Operations Tests\n";
    std::cout << "========================================\n\n";

    // AND tests
    test_fold_and_true_true();
    test_fold_and_true_false();
    test_fold_and_false_true();
    test_fold_and_false_false();
    test_fold_and_top();
    test_fold_and_bottom();

    // OR tests
    test_fold_or_true_true();
    test_fold_or_true_false();
    test_fold_or_false_true();
    test_fold_or_false_false();
    test_fold_or_top();
    test_fold_or_bottom();

    // NOT tests
    test_fold_not_true();
    test_fold_not_false();
    test_fold_not_top();
    test_fold_not_bottom();
    test_fold_not_double_negation();

    std::cout << "========================================\n";
    std::cout << "All Logical Operations Tests Passed!\n";
    std::cout << "========================================\n";

    return 0;
}
