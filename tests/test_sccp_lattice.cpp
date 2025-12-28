#include "../include/LatticeValue.h"
#include <cassert>
#include <iostream>

using namespace cppfort::sccp;

void test_lattice_top() {
    std::cout << "Test: LatticeValue Top kind\n";

    LatticeValue top = LatticeValue::getTop();

    assert(top.getKind() == LatticeValue::Kind::Top && "Value should be Top");
    assert(!top.isConstant() && "Top should not be constant");
    assert(!top.isBottom() && "Top should not be bottom");

    std::cout << "✓ LatticeValue Top test passed\n\n";
}

void test_lattice_bottom() {
    std::cout << "Test: LatticeValue Bottom kind\n";

    LatticeValue bottom = LatticeValue::getBottom();

    assert(bottom.getKind() == LatticeValue::Kind::Bottom && "Value should be Bottom");
    assert(!bottom.isConstant() && "Bottom should not be constant");
    assert(bottom.isBottom() && "Value should be bottom");
    assert(!bottom.isTop() && "Bottom should not be top");

    std::cout << "✓ LatticeValue Bottom test passed\n\n";
}

void test_lattice_constant_integer() {
    std::cout << "Test: LatticeValue constant integer\n";

    LatticeValue constant = LatticeValue::getConstant(42LL);

    assert(constant.getKind() == LatticeValue::Kind::Constant && "Value should be Constant");
    assert(constant.isConstant() && "Value should be constant");
    assert(!constant.isTop() && "Constant should not be top");
    assert(!constant.isBottom() && "Constant should not be bottom");

    // Get constant value
    std::optional<int64_t> value = constant.getAsInteger();
    assert(value.has_value() && "Should have integer value");
    assert(value.value() == 42 && "Value should be 42");

    std::cout << "✓ LatticeValue constant integer test passed\n\n";
}

void test_lattice_constant_boolean() {
    std::cout << "Test: LatticeValue constant boolean\n";

    LatticeValue const_true = LatticeValue::getConstant(true);
    LatticeValue const_false = LatticeValue::getConstant(false);

    assert(const_true.isConstant() && "True should be constant");
    assert(const_false.isConstant() && "False should be constant");

    std::optional<bool> true_val = const_true.getAsBoolean();
    std::optional<bool> false_val = const_false.getAsBoolean();

    assert(true_val.has_value() && "Should have boolean value");
    assert(false_val.has_value() && "Should have boolean value");
    assert(true_val.value() == true && "True value should be true");
    assert(false_val.value() == false && "False value should be false");

    std::cout << "✓ LatticeValue constant boolean test passed\n\n";
}

void test_lattice_meet_top_with_constant() {
    std::cout << "Test: Lattice meet operation - Top with Constant\n";

    LatticeValue top = LatticeValue::getTop();
    LatticeValue constant = LatticeValue::getConstant(42LL);

    // meet(Top, Constant) = Constant
    LatticeValue result = LatticeValue::meet(top, constant);

    assert(result.isConstant() && "Result should be constant");
    std::optional<int64_t> value = result.getAsInteger();
    assert(value.has_value() && "Result should have integer value");
    assert(value.value() == 42LL && "Result value should be 42");

    std::cout << "✓ Lattice meet Top with Constant test passed\n\n";
}

void test_lattice_meet_constant_with_constant() {
    std::cout << "Test: Lattice meet operation - Constant with Constant (same)\n";

    LatticeValue c1 = LatticeValue::getConstant(42LL);
    LatticeValue c2 = LatticeValue::getConstant(42LL);

    // meet(Constant, Constant) with same value = Constant
    LatticeValue result = LatticeValue::meet(c1, c2);

    assert(result.isConstant() && "Result should be constant");
    std::optional<int64_t> value = result.getAsInteger();
    assert(value.value() == 42LL && "Result value should be 42");

    std::cout << "✓ Lattice meet constant with constant (same) test passed\n\n";
}

void test_lattice_meet_constant_with_constant_different() {
    std::cout << "Test: Lattice meet operation - Constant with Constant (different)\n";

    LatticeValue c1 = LatticeValue::getConstant(42LL);
    LatticeValue c2 = LatticeValue::getConstant(99LL);

    // meet(Constant, Constant) with different values = Bottom (conflict)
    LatticeValue result = LatticeValue::meet(c1, c2);

    assert(result.isBottom() && "Conflicting constants should result in Bottom");

    std::cout << "✓ Lattice meet constant with constant (different) test passed\n\n";
}

void test_lattice_meet_with_bottom() {
    std::cout << "Test: Lattice meet operation - anything with Bottom = Bottom\n";

    LatticeValue constant = LatticeValue::getConstant(42LL);
    LatticeValue bottom = LatticeValue::getBottom();
    LatticeValue top = LatticeValue::getTop();

    // meet(anything, Bottom) = Bottom
    LatticeValue result1 = LatticeValue::meet(constant, bottom);
    LatticeValue result2 = LatticeValue::meet(top, bottom);

    assert(result1.isBottom() && "meet(constant, bottom) should be bottom");
    assert(result2.isBottom() && "meet(top, bottom) should be bottom");

    std::cout << "✓ Lattice meet with Bottom test passed\n\n";
}

void test_lattice_range_creation() {
    std::cout << "Test: LatticeValue with integer range\n";

    LatticeValue range = LatticeValue::getIntegerRange(0LL, 100LL);

    assert(range.getKind() == LatticeValue::IntegerRange && "Value should be IntegerRange");
    assert(!range.isConstant() && "Range should not be constant");
    assert(!range.isTop() && "Range should not be top");
    assert(!range.isBottom() && "Range should not be bottom");

    auto min = range.getMin();
    auto max = range.getMax();

    assert(min.has_value() && "Should have min value");
    assert(max.has_value() && "Should have max value");
    assert(min.value() == 0LL && "Min should be 0");
    assert(max.value() == 100LL && "Max should be 100");

    std::cout << "✓ LatticeValue range creation test passed\n\n";
}

void test_lattice_range_meet() {
    std::cout << "Test: Lattice meet operation - ranges\n";

    LatticeValue range1 = LatticeValue::getIntegerRange(0LL, 50LL);
    LatticeValue range2 = LatticeValue::getIntegerRange(25LL, 100LL);

    // meet([0,50], [25,100]) = [25,50] (intersection)
    LatticeValue result = LatticeValue::meet(range1, range2);

    assert(result.getKind() == LatticeValue::IntegerRange && "Result should be IntegerRange");
    auto min = result.getMin();
    auto max = result.getMax();

    assert(min.has_value() && "Should have min value");
    assert(max.has_value() && "Should have max value");
    assert(min.value() == 25LL && "Min should be 25 (intersection)");
    assert(max.value() == 50LL && "Max should be 50 (intersection)");

    std::cout << "✓ LatticeValue range meet test passed\n\n";
}

void test_lattice_range_meet_disjoint() {
    std::cout << "Test: Lattice meet operation - disjoint ranges\n";

    LatticeValue range1 = LatticeValue::getIntegerRange(0LL, 10LL);
    LatticeValue range2 = LatticeValue::getIntegerRange(20LL, 30LL);

    // meet([0,10], [20,30]) = Bottom (no intersection)
    LatticeValue result = LatticeValue::meet(range1, range2);

    assert(result.isBottom() && "Disjoint ranges should result in Bottom");

    std::cout << "✓ LatticeValue range meet disjoint test passed\n\n";
}

void test_lattice_range_with_constant() {
    std::cout << "Test: Lattice meet operation - range with constant\n";

    LatticeValue range = LatticeValue::getIntegerRange(0LL, 100LL);
    LatticeValue constant = LatticeValue::getConstant(50LL);

    // meet([0,100], 50) = 50 (constant within range)
    LatticeValue result = LatticeValue::meet(range, constant);

    assert(result.isConstant() && "Result should be constant");
    auto value = result.getAsInteger();
    assert(value.value() == 50LL && "Value should be 50");

    // meet([0,100], 200) = Bottom (constant outside range)
    LatticeValue constant2 = LatticeValue::getConstant(200LL);
    LatticeValue result2 = LatticeValue::meet(range, constant2);

    assert(result2.isBottom() && "Constant outside range should result in Bottom");

    std::cout << "✓ LatticeValue range with constant test passed\n\n";
}

int main() {
    std::cout << "=== SCCP LatticeValue Tests ===\n\n";

    test_lattice_top();
    test_lattice_bottom();
    test_lattice_constant_integer();
    test_lattice_constant_boolean();
    test_lattice_meet_top_with_constant();
    test_lattice_meet_constant_with_constant();
    test_lattice_meet_constant_with_constant_different();
    test_lattice_meet_with_bottom();
    test_lattice_range_creation();
    test_lattice_range_meet();
    test_lattice_range_meet_disjoint();
    test_lattice_range_with_constant();

    std::cout << "=== All LatticeValue tests passed! ===\n";
    return 0;
}
