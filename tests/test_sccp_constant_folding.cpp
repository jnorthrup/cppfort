//===- test_sccp_constant_folding.cpp - SCCP Constant Folding Tests ---------------===//
///
/// Tests for constant folding in SCCP analysis.
/// Verifies that arithmetic operations with constant operands are folded
/// to constant results.
///
//===----------------------------------------------------------------------===//

#include "../include/LatticeValue.h"
#include "../include/ConstantFolder.h"
#include <cassert>
#include <iostream>

using namespace cppfort::sccp;

void test_fold_add_constants() {
    std::cout << "Test: Fold AddOp with constants\n";

    // 5 + 3 = 8
    LatticeValue a = LatticeValue::getConstant(5LL);
    LatticeValue b = LatticeValue::getConstant(3LL);

    LatticeValue result = ConstantFolder::foldAdd(a, b);

    assert(result.isConstant() && "Add of constants should be constant");
    auto value = result.getAsInteger();
    assert(value.has_value() && "Result should be integer");
    assert(value.value() == 8LL && "5 + 3 should equal 8");

    std::cout << "✓ Fold AddOp with constants test passed\n\n";
}

void test_fold_sub_constants() {
    std::cout << "Test: Fold SubOp with constants\n";

    // 10 - 4 = 6
    LatticeValue a = LatticeValue::getConstant(10LL);
    LatticeValue b = LatticeValue::getConstant(4LL);

    LatticeValue result = ConstantFolder::foldSub(a, b);

    assert(result.isConstant() && "Sub of constants should be constant");
    auto value = result.getAsInteger();
    assert(value.value() == 6LL && "10 - 4 should equal 6");

    std::cout << "✓ Fold SubOp with constants test passed\n\n";
}

void test_fold_mul_constants() {
    std::cout << "Test: Fold MulOp with constants\n";

    // 7 * 6 = 42
    LatticeValue a = LatticeValue::getConstant(7LL);
    LatticeValue b = LatticeValue::getConstant(6LL);

    LatticeValue result = ConstantFolder::foldMul(a, b);

    assert(result.isConstant() && "Mul of constants should be constant");
    auto value = result.getAsInteger();
    assert(value.value() == 42LL && "7 * 6 should equal 42");

    std::cout << "✓ Fold MulOp with constants test passed\n\n";
}

void test_fold_div_constants() {
    std::cout << "Test: Fold DivOp with constants\n";

    // 20 / 4 = 5
    LatticeValue a = LatticeValue::getConstant(20LL);
    LatticeValue b = LatticeValue::getConstant(4LL);

    LatticeValue result = ConstantFolder::foldDiv(a, b);

    assert(result.isConstant() && "Div of constants should be constant");
    auto value = result.getAsInteger();
    assert(value.value() == 5LL && "20 / 4 should equal 5");

    std::cout << "✓ Fold DivOp with constants test passed\n\n";
}

void test_fold_add_with_range() {
    std::cout << "Test: Fold AddOp with range and constant\n";

    // [0, 100] + 50 = [50, 150] (range addition)
    LatticeValue range = LatticeValue::getIntegerRange(0LL, 100LL);
    LatticeValue constant = LatticeValue::getConstant(50LL);

    LatticeValue result = ConstantFolder::foldAdd(range, constant);

    assert(result.getKind() == LatticeValue::IntegerRange && "Result should be range");
    auto min = result.getMin();
    auto max = result.getMax();
    assert(min.value() == 50LL && "Min should be 50");
    assert(max.value() == 150LL && "Max should be 150");

    std::cout << "✓ Fold AddOp with range and constant test passed\n\n";
}

void test_fold_add_two_ranges() {
    std::cout << "Test: Fold AddOp with two ranges\n";

    // [0, 50] + [0, 50] = [0, 100]
    LatticeValue range1 = LatticeValue::getIntegerRange(0LL, 50LL);
    LatticeValue range2 = LatticeValue::getIntegerRange(0LL, 50LL);

    LatticeValue result = ConstantFolder::foldAdd(range1, range2);

    assert(result.getKind() == LatticeValue::IntegerRange && "Result should be range");
    auto min = result.getMin();
    auto max = result.getMax();
    assert(min.value() == 0LL && "Min should be 0");
    assert(max.value() == 100LL && "Max should be 100");

    std::cout << "✓ Fold AddOp with two ranges test passed\n\n";
}

void test_fold_with_top() {
    std::cout << "Test: Fold operations with Top\n";

    LatticeValue top = LatticeValue::getTop();
    LatticeValue constant = LatticeValue::getConstant(10LL);

    // Top + 10 = Top (unknown without knowing left operand)
    LatticeValue result1 = ConstantFolder::foldAdd(top, constant);
    assert(result1.isTop() && "Top op constant should be Top");

    // Top - 5 = Top
    LatticeValue result2 = ConstantFolder::foldSub(top, LatticeValue::getConstant(5LL));
    assert(result2.isTop() && "Top op constant should be Top");

    std::cout << "✓ Fold operations with Top test passed\n\n";
}

void test_fold_div_by_zero_returns_top() {
    std::cout << "Test: Fold DivOp by zero returns Top\n";

    LatticeValue a = LatticeValue::getConstant(10LL);
    LatticeValue zero = LatticeValue::getConstant(0LL);

    // Division by zero should return Top (undefined behavior)
    LatticeValue result = ConstantFolder::foldDiv(a, zero);

    assert(result.isTop() && "Div by zero should return Top (undefined)");

    std::cout << "✓ Fold DivOp by zero test passed\n\n";
}

int main() {
    std::cout << "=== SCCP Constant Folding Tests ===\n\n";

    test_fold_add_constants();
    test_fold_sub_constants();
    test_fold_mul_constants();
    test_fold_div_constants();
    test_fold_add_with_range();
    test_fold_add_two_ranges();
    test_fold_with_top();
    test_fold_div_by_zero_returns_top();

    std::cout << "=== All Constant Folding tests passed! ===\n";
    return 0;
}
