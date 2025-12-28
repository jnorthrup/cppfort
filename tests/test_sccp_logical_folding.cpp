//===- test_sccp_logical_folding.cpp - SCCP Logical Folding Tests -------------===//
///
/// Tests for logical operation folding in SCCP analysis.
/// Verifies that AND, OR, NOT operations with constant operands are folded
/// to constant results.
///
//===----------------------------------------------------------------------===//

#include "../include/LatticeValue.h"
#include "../include/ConstantFolder.h"
#include <cassert>
#include <iostream>

using namespace cppfort::sccp;

void test_fold_and_true_true() {
    std::cout << "Test: Fold AndOp with true && true\n";

    LatticeValue a = LatticeValue::getConstant(true);
    LatticeValue b = LatticeValue::getConstant(true);

    LatticeValue result = ConstantFolder::foldAnd(a, b);

    assert(result.isConstant() && "And of constants should be constant");
    auto value = result.getAsBoolean();
    assert(value.has_value() && "Result should be boolean");
    assert(value.value() == true && "true && true should equal true");

    std::cout << "✓ Fold AndOp with true && true test passed\n\n";
}

void test_fold_and_true_false() {
    std::cout << "Test: Fold AndOp with true && false\n";

    LatticeValue a = LatticeValue::getConstant(true);
    LatticeValue b = LatticeValue::getConstant(false);

    LatticeValue result = ConstantFolder::foldAnd(a, b);

    assert(result.isConstant() && "And of constants should be constant");
    auto value = result.getAsBoolean();
    assert(value.value() == false && "true && false should equal false");

    std::cout << "✓ Fold AndOp with true && false test passed\n\n";
}

void test_fold_and_false_false() {
    std::cout << "Test: Fold AndOp with false && false\n";

    LatticeValue a = LatticeValue::getConstant(false);
    LatticeValue b = LatticeValue::getConstant(false);

    LatticeValue result = ConstantFolder::foldAnd(a, b);

    assert(result.isConstant() && "And of constants should be constant");
    auto value = result.getAsBoolean();
    assert(value.value() == false && "false && false should equal false");

    std::cout << "✓ Fold AndOp with false && false test passed\n\n";
}

void test_fold_or_true_false() {
    std::cout << "Test: Fold OrOp with true || false\n";

    LatticeValue a = LatticeValue::getConstant(true);
    LatticeValue b = LatticeValue::getConstant(false);

    LatticeValue result = ConstantFolder::foldOr(a, b);

    assert(result.isConstant() && "Or of constants should be constant");
    auto value = result.getAsBoolean();
    assert(value.value() == true && "true || false should equal true");

    std::cout << "✓ Fold OrOp with true || false test passed\n\n";
}

void test_fold_or_false_false() {
    std::cout << "Test: Fold OrOp with false || false\n";

    LatticeValue a = LatticeValue::getConstant(false);
    LatticeValue b = LatticeValue::getConstant(false);

    LatticeValue result = ConstantFolder::foldOr(a, b);

    assert(result.isConstant() && "Or of constants should be constant");
    auto value = result.getAsBoolean();
    assert(value.value() == false && "false || false should equal false");

    std::cout << "✓ Fold OrOp with false || false test passed\n\n";
}

void test_fold_or_true_true() {
    std::cout << "Test: Fold OrOp with true || true\n";

    LatticeValue a = LatticeValue::getConstant(true);
    LatticeValue b = LatticeValue::getConstant(true);

    LatticeValue result = ConstantFolder::foldOr(a, b);

    assert(result.isConstant() && "Or of constants should be constant");
    auto value = result.getAsBoolean();
    assert(value.value() == true && "true || true should equal true");

    std::cout << "✓ Fold OrOp with true || true test passed\n\n";
}

void test_fold_not_true() {
    std::cout << "Test: Fold NotOp with !true\n";

    LatticeValue a = LatticeValue::getConstant(true);

    LatticeValue result = ConstantFolder::foldNot(a);

    assert(result.isConstant() && "Not of constant should be constant");
    auto value = result.getAsBoolean();
    assert(value.value() == false && "!true should equal false");

    std::cout << "✓ Fold NotOp with !true test passed\n\n";
}

void test_fold_not_false() {
    std::cout << "Test: Fold NotOp with !false\n";

    LatticeValue a = LatticeValue::getConstant(false);

    LatticeValue result = ConstantFolder::foldNot(a);

    assert(result.isConstant() && "Not of constant should be constant");
    auto value = result.getAsBoolean();
    assert(value.value() == true && "!false should equal true");

    std::cout << "✓ Fold NotOp with !false test passed\n\n";
}

void test_fold_logical_with_top() {
    std::cout << "Test: Fold logical operations with Top\n";

    LatticeValue top = LatticeValue::getTop();
    LatticeValue constant = LatticeValue::getConstant(true);

    // Top && true = Top
    LatticeValue result1 = ConstantFolder::foldAnd(top, constant);
    assert(result1.isTop() && "Top && constant should be Top");

    // Top || false = Top
    LatticeValue result2 = ConstantFolder::foldOr(top, constant);
    assert(result2.isTop() && "Top || constant should be Top");

    // !Top = Top
    LatticeValue result3 = ConstantFolder::foldNot(top);
    assert(result3.isTop() && "!Top should be Top");

    std::cout << "✓ Fold logical operations with Top test passed\n\n";
}

void test_fold_logical_with_bottom() {
    std::cout << "Test: Fold logical operations with Bottom\n";

    LatticeValue bottom = LatticeValue::getBottom();
    LatticeValue constant = LatticeValue::getConstant(true);

    // Bottom && true = Bottom
    LatticeValue result1 = ConstantFolder::foldAnd(bottom, constant);
    assert(result1.isBottom() && "Bottom && constant should be Bottom");

    // Bottom || false = Bottom
    LatticeValue result2 = ConstantFolder::foldOr(bottom, constant);
    assert(result2.isBottom() && "Bottom || constant should be Bottom");

    // !Bottom = Bottom
    LatticeValue result3 = ConstantFolder::foldNot(bottom);
    assert(result3.isBottom() && "!Bottom should be Bottom");

    std::cout << "✓ Fold logical operations with Bottom test passed\n\n";
}

int main() {
    std::cout << "=== SCCP Logical Folding Tests ===\n\n";

    test_fold_and_true_true();
    test_fold_and_true_false();
    test_fold_and_false_false();
    test_fold_or_true_false();
    test_fold_or_false_false();
    test_fold_or_true_true();
    test_fold_not_true();
    test_fold_not_false();
    test_fold_logical_with_top();
    test_fold_logical_with_bottom();

    std::cout << "=== All Logical Folding tests passed! ===\n";
    return 0;
}
