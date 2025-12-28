//===- test_sccp_float_special.cpp - SCCP Float Special Values Tests ---------===//
///
/// Tests for special float value handling in SCCP analysis.
/// Verifies that NaN and Infinity values are properly tracked and propagated
/// through lattice operations.
///
//===----------------------------------------------------------------------===//

#include "../include/LatticeValue.h"
#include "../include/ConstantFolder.h"
#include <cassert>
#include <iostream>
#include <cmath>
#include <limits>

using namespace cppfort::sccp;

void test_float_special_kind() {
    std::cout << "Test: FloatSpecial kind exists\n";

    LatticeValue nan = LatticeValue::getFloatSpecial(LatticeValue::FloatSpecialValue::NaN);
    assert(nan.getKind() == LatticeValue::FloatSpecial && "NaN should be FloatSpecial kind");

    LatticeValue inf = LatticeValue::getFloatSpecial(LatticeValue::FloatSpecialValue::PosInfinity);
    assert(inf.getKind() == LatticeValue::FloatSpecial && "Infinity should be FloatSpecial kind");

    std::cout << "✓ FloatSpecial kind test passed\n\n";
}

void test_nan_creation() {
    std::cout << "Test: Create NaN value\n";

    LatticeValue nan = LatticeValue::getFloatSpecial(LatticeValue::FloatSpecialValue::NaN);

    assert(nan.isFloatSpecial() && "NaN should be FloatSpecial");
    auto value = nan.getAsFloatSpecial();
    assert(value.has_value() && "NaN should have FloatSpecial value");
    assert(value.value() == LatticeValue::FloatSpecialValue::NaN && "Value should be NaN");

    std::cout << "✓ Create NaN value test passed\n\n";
}

void test_infinity_creation() {
    std::cout << "Test: Create Infinity values\n";

    LatticeValue posInf = LatticeValue::getFloatSpecial(LatticeValue::FloatSpecialValue::PosInfinity);
    assert(posInf.isFloatSpecial() && "PosInfinity should be FloatSpecial");
    assert(posInf.getAsFloatSpecial().value() == LatticeValue::FloatSpecialValue::PosInfinity);

    LatticeValue negInf = LatticeValue::getFloatSpecial(LatticeValue::FloatSpecialValue::NegInfinity);
    assert(negInf.isFloatSpecial() && "NegInfinity should be FloatSpecial");
    assert(negInf.getAsFloatSpecial().value() == LatticeValue::FloatSpecialValue::NegInfinity);

    std::cout << "✓ Create Infinity values test passed\n\n";
}

void test_nan_propagation_through_add() {
    std::cout << "Test: NaN propagation through addition\n";

    LatticeValue nan = LatticeValue::getFloatSpecial(LatticeValue::FloatSpecialValue::NaN);
    LatticeValue constant = LatticeValue::getConstant(42LL);

    // NaN + anything = NaN
    LatticeValue result1 = ConstantFolder::foldAdd(nan, constant);
    assert(result1.isFloatSpecial() && "NaN + constant should be FloatSpecial");
    assert(result1.getAsFloatSpecial().value() == LatticeValue::FloatSpecialValue::NaN);

    // constant + NaN = NaN
    LatticeValue result2 = ConstantFolder::foldAdd(constant, nan);
    assert(result2.isFloatSpecial() && "constant + NaN should be FloatSpecial");

    std::cout << "✓ NaN propagation through addition test passed\n\n";
}

void test_nan_propagation_through_sub() {
    std::cout << "Test: NaN propagation through subtraction\n";

    LatticeValue nan = LatticeValue::getFloatSpecial(LatticeValue::FloatSpecialValue::NaN);
    LatticeValue constant = LatticeValue::getConstant(42LL);

    // NaN - anything = NaN
    LatticeValue result1 = ConstantFolder::foldSub(nan, constant);
    assert(result1.isFloatSpecial() && "NaN - constant should be FloatSpecial");

    std::cout << "✓ NaN propagation through subtraction test passed\n\n";
}

void test_infinity_arithmetic() {
    std::cout << "Test: Infinity arithmetic\n";

    LatticeValue inf = LatticeValue::getFloatSpecial(LatticeValue::FloatSpecialValue::PosInfinity);
    LatticeValue constant = LatticeValue::getConstant(42LL);

    // Infinity + constant = Infinity
    LatticeValue result1 = ConstantFolder::foldAdd(inf, constant);
    assert(result1.isFloatSpecial() && "Infinity + constant should be FloatSpecial");

    // Infinity - constant = Infinity
    LatticeValue result2 = ConstantFolder::foldSub(inf, constant);
    assert(result2.isFloatSpecial() && "Infinity - constant should be FloatSpecial");

    std::cout << "✓ Infinity arithmetic test passed\n\n";
}

void test_nan_meet_operations() {
    std::cout << "Test: NaN meet operations\n";

    LatticeValue nan = LatticeValue::getFloatSpecial(LatticeValue::FloatSpecialValue::NaN);
    LatticeValue constant = LatticeValue::getConstant(42LL);

    // meet(NaN, constant) = Bottom (different types)
    LatticeValue result = LatticeValue::meet(nan, constant);
    assert(result.isBottom() && "NaN meet constant should be Bottom");

    std::cout << "✓ NaN meet operations test passed\n\n";
}

void test_float_special_with_top() {
    std::cout << "Test: FloatSpecial with Top\n";

    LatticeValue nan = LatticeValue::getFloatSpecial(LatticeValue::FloatSpecialValue::NaN);
    LatticeValue top = LatticeValue::getTop();

    // NaN meet Top = NaN (Top is identity)
    LatticeValue result = LatticeValue::meet(nan, top);
    assert(result.isFloatSpecial() && "NaN meet Top should be NaN");

    std::cout << "✓ FloatSpecial with Top test passed\n\n";
}

int main() {
    std::cout << "=== SCCP Float Special Values Tests ===\n\n";

    test_float_special_kind();
    test_nan_creation();
    test_infinity_creation();
    test_nan_propagation_through_add();
    test_nan_propagation_through_sub();
    test_infinity_arithmetic();
    test_nan_meet_operations();
    test_float_special_with_top();

    std::cout << "=== All Float Special Values tests passed! ===\n";
    return 0;
}
