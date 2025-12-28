//===- test_sccp_comparison_folding.cpp - SCCP Comparison Folding Tests -----===//
///
/// Tests for comparison operation folding in SCCP analysis.
/// Verifies that EQ, NE, LT, GT, LE, GE operations with constant operands
/// are folded to constant boolean results.
///
//===----------------------------------------------------------------------===//

#include "../include/LatticeValue.h"
#include "../include/ConstantFolder.h"
#include <cassert>
#include <iostream>

using namespace cppfort::sccp;

void test_fold_eq_equal() {
    std::cout << "Test: Fold EqOp with equal values\n";

    LatticeValue a = LatticeValue::getConstant(42LL);
    LatticeValue b = LatticeValue::getConstant(42LL);

    LatticeValue result = ConstantFolder::foldCmp(LatticeValue::CmpPredicate::EQ, a, b);

    assert(result.isConstant() && "Eq of constants should be constant");
    auto value = result.getAsBoolean();
    assert(value.has_value() && "Result should be boolean");
    assert(value.value() == true && "42 == 42 should equal true");

    std::cout << "✓ Fold EqOp with equal values test passed\n\n";
}

void test_fold_eq_not_equal() {
    std::cout << "Test: Fold EqOp with not equal values\n";

    LatticeValue a = LatticeValue::getConstant(10LL);
    LatticeValue b = LatticeValue::getConstant(20LL);

    LatticeValue result = ConstantFolder::foldCmp(LatticeValue::CmpPredicate::EQ, a, b);

    assert(result.isConstant() && "Eq of constants should be constant");
    auto value = result.getAsBoolean();
    assert(value.value() == false && "10 == 20 should equal false");

    std::cout << "✓ Fold EqOp with not equal values test passed\n\n";
}

void test_fold_ne_equal() {
    std::cout << "Test: Fold NeOp with equal values\n";

    LatticeValue a = LatticeValue::getConstant(42LL);
    LatticeValue b = LatticeValue::getConstant(42LL);

    LatticeValue result = ConstantFolder::foldCmp(LatticeValue::CmpPredicate::NE, a, b);

    assert(result.isConstant() && "Ne of constants should be constant");
    auto value = result.getAsBoolean();
    assert(value.value() == false && "42 != 42 should equal false");

    std::cout << "✓ Fold NeOp with equal values test passed\n\n";
}

void test_fold_ne_not_equal() {
    std::cout << "Test: Fold NeOp with not equal values\n";

    LatticeValue a = LatticeValue::getConstant(10LL);
    LatticeValue b = LatticeValue::getConstant(20LL);

    LatticeValue result = ConstantFolder::foldCmp(LatticeValue::CmpPredicate::NE, a, b);

    assert(result.isConstant() && "Ne of constants should be constant");
    auto value = result.getAsBoolean();
    assert(value.value() == true && "10 != 20 should equal true");

    std::cout << "✓ Fold NeOp with not equal values test passed\n\n";
}

void test_fold_lt_less() {
    std::cout << "Test: Fold LtOp with less than\n";

    LatticeValue a = LatticeValue::getConstant(10LL);
    LatticeValue b = LatticeValue::getConstant(20LL);

    LatticeValue result = ConstantFolder::foldCmp(LatticeValue::CmpPredicate::LT, a, b);

    assert(result.isConstant() && "Lt of constants should be constant");
    auto value = result.getAsBoolean();
    assert(value.value() == true && "10 < 20 should equal true");

    std::cout << "✓ Fold LtOp with less than test passed\n\n";
}

void test_fold_lt_equal() {
    std::cout << "Test: Fold LtOp with equal values\n";

    LatticeValue a = LatticeValue::getConstant(42LL);
    LatticeValue b = LatticeValue::getConstant(42LL);

    LatticeValue result = ConstantFolder::foldCmp(LatticeValue::CmpPredicate::LT, a, b);

    assert(result.isConstant() && "Lt of constants should be constant");
    auto value = result.getAsBoolean();
    assert(value.value() == false && "42 < 42 should equal false");

    std::cout << "✓ Fold LtOp with equal values test passed\n\n";
}

void test_fold_gt_greater() {
    std::cout << "Test: Fold GtOp with greater than\n";

    LatticeValue a = LatticeValue::getConstant(30LL);
    LatticeValue b = LatticeValue::getConstant(20LL);

    LatticeValue result = ConstantFolder::foldCmp(LatticeValue::CmpPredicate::GT, a, b);

    assert(result.isConstant() && "Gt of constants should be constant");
    auto value = result.getAsBoolean();
    assert(value.value() == true && "30 > 20 should equal true");

    std::cout << "✓ Fold GtOp with greater than test passed\n\n";
}

void test_fold_le_less_or_equal() {
    std::cout << "Test: Fold LeOp with less than\n";

    LatticeValue a = LatticeValue::getConstant(10LL);
    LatticeValue b = LatticeValue::getConstant(20LL);

    LatticeValue result = ConstantFolder::foldCmp(LatticeValue::CmpPredicate::LE, a, b);

    assert(result.isConstant() && "Le of constants should be constant");
    auto value = result.getAsBoolean();
    assert(value.value() == true && "10 <= 20 should equal true");

    std::cout << "✓ Fold LeOp with less than test passed\n\n";
}

void test_fold_le_equal() {
    std::cout << "Test: Fold LeOp with equal values\n";

    LatticeValue a = LatticeValue::getConstant(42LL);
    LatticeValue b = LatticeValue::getConstant(42LL);

    LatticeValue result = ConstantFolder::foldCmp(LatticeValue::CmpPredicate::LE, a, b);

    assert(result.isConstant() && "Le of constants should be constant");
    auto value = result.getAsBoolean();
    assert(value.value() == true && "42 <= 42 should equal true");

    std::cout << "✓ Fold LeOp with equal values test passed\n\n";
}

void test_fold_ge_greater_or_equal() {
    std::cout << "Test: Fold GeOp with greater than\n";

    LatticeValue a = LatticeValue::getConstant(30LL);
    LatticeValue b = LatticeValue::getConstant(20LL);

    LatticeValue result = ConstantFolder::foldCmp(LatticeValue::CmpPredicate::GE, a, b);

    assert(result.isConstant() && "Ge of constants should be constant");
    auto value = result.getAsBoolean();
    assert(value.value() == true && "30 >= 20 should equal true");

    std::cout << "✓ Fold GeOp with greater than test passed\n\n";
}

void test_fold_ge_equal() {
    std::cout << "Test: Fold GeOp with equal values\n";

    LatticeValue a = LatticeValue::getConstant(42LL);
    LatticeValue b = LatticeValue::getConstant(42LL);

    LatticeValue result = ConstantFolder::foldCmp(LatticeValue::CmpPredicate::GE, a, b);

    assert(result.isConstant() && "Ge of constants should be constant");
    auto value = result.getAsBoolean();
    assert(value.value() == true && "42 >= 42 should equal true");

    std::cout << "✓ Fold GeOp with equal values test passed\n\n";
}

void test_fold_cmp_with_top() {
    std::cout << "Test: Fold comparison operations with Top\n";

    LatticeValue top = LatticeValue::getTop();
    LatticeValue constant = LatticeValue::getConstant(10LL);

    // Top == 10 = Top
    LatticeValue result1 = ConstantFolder::foldCmp(LatticeValue::CmpPredicate::EQ, top, constant);
    assert(result1.isTop() && "Top cmp constant should be Top");

    std::cout << "✓ Fold comparison operations with Top test passed\n\n";
}

void test_fold_cmp_with_bottom() {
    std::cout << "Test: Fold comparison operations with Bottom\n";

    LatticeValue bottom = LatticeValue::getBottom();
    LatticeValue constant = LatticeValue::getConstant(10LL);

    // Bottom == 10 = Bottom
    LatticeValue result1 = ConstantFolder::foldCmp(LatticeValue::CmpPredicate::EQ, bottom, constant);
    assert(result1.isBottom() && "Bottom cmp constant should be Bottom");

    std::cout << "✓ Fold comparison operations with Bottom test passed\n\n";
}

int main() {
    std::cout << "=== SCCP Comparison Folding Tests ===\n\n";

    test_fold_eq_equal();
    test_fold_eq_not_equal();
    test_fold_ne_equal();
    test_fold_ne_not_equal();
    test_fold_lt_less();
    test_fold_lt_equal();
    test_fold_gt_greater();
    test_fold_le_less_or_equal();
    test_fold_le_equal();
    test_fold_ge_greater_or_equal();
    test_fold_ge_equal();
    test_fold_cmp_with_top();
    test_fold_cmp_with_bottom();

    std::cout << "=== All Comparison Folding tests passed! ===\n";
    return 0;
}
