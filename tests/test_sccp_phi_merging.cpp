//===- test_sccp_phi_merging.cpp - SCCP Phi Node Merging Tests ---------------===//
///
/// Tests for phi node constant merging in SCCP analysis.
/// Verifies that phi nodes correctly merge values from multiple predecessors
/// using the meet operation.
///
//===----------------------------------------------------------------------===//

#include "../include/LatticeValue.h"
#include "../include/DataflowAnalysis.h"
#include <cassert>
#include <iostream>
#include <vector>

using namespace cppfort::sccp;

void test_phi_merge_all_same_constants() {
    std::cout << "Test: Phi merge with all same constants\n";

    // All inputs are constant 42
    std::vector<LatticeValue> inputs = {
        LatticeValue::getConstant(42LL),
        LatticeValue::getConstant(42LL),
        LatticeValue::getConstant(42LL)
    };

    LatticeValue result = DataflowAnalysis::mergePhiInputs(inputs);

    assert(result.isConstant() && "Phi with same constants should be constant");
    assert(result.getAsInteger().value() == 42LL && "Result should be 42");

    std::cout << "✓ Phi merge with all same constants test passed\n\n";
}

void test_phi_merge_different_constants() {
    std::cout << "Test: Phi merge with different constants\n";

    // Inputs are different constants
    std::vector<LatticeValue> inputs = {
        LatticeValue::getConstant(10LL),
        LatticeValue::getConstant(20LL),
        LatticeValue::getConstant(30LL)
    };

    LatticeValue result = DataflowAnalysis::mergePhiInputs(inputs);

    assert(result.isBottom() && "Phi with different constants should be Bottom");

    std::cout << "✓ Phi merge with different constants test passed\n\n";
}

void test_phi_merge_with_top() {
    std::cout << "Test: Phi merge with Top\n";

    // One input is Top
    std::vector<LatticeValue> inputs = {
        LatticeValue::getConstant(42LL),
        LatticeValue::getTop(),
        LatticeValue::getConstant(42LL)
    };

    LatticeValue result = DataflowAnalysis::mergePhiInputs(inputs);

    assert(result.isTop() && "Phi with Top input should be Top");

    std::cout << "✓ Phi merge with Top test passed\n\n";
}

void test_phi_merge_all_top() {
    std::cout << "Test: Phi merge with all Top\n";

    std::vector<LatticeValue> inputs = {
        LatticeValue::getTop(),
        LatticeValue::getTop()
    };

    LatticeValue result = DataflowAnalysis::mergePhiInputs(inputs);

    assert(result.isTop() && "Phi with all Top should be Top");

    std::cout << "✓ Phi merge with all Top test passed\n\n";
}

void test_phi_merge_with_bottom() {
    std::cout << "Test: Phi merge with Bottom\n";

    // One input is Bottom (unreachable)
    std::vector<LatticeValue> inputs = {
        LatticeValue::getConstant(42LL),
        LatticeValue::getBottom(),
        LatticeValue::getConstant(42LL)
    };

    LatticeValue result = DataflowAnalysis::mergePhiInputs(inputs);

    assert(result.isBottom() && "Phi with Bottom input should be Bottom");

    std::cout << "✓ Phi merge with Bottom test passed\n\n";
}

void test_phi_merge_empty_inputs() {
    std::cout << "Test: Phi merge with empty inputs\n";

    std::vector<LatticeValue> inputs;

    LatticeValue result = DataflowAnalysis::mergePhiInputs(inputs);

    assert(result.isTop() && "Phi with no inputs should be Top (undefined)");

    std::cout << "✓ Phi merge with empty inputs test passed\n\n";
}

void test_phi_merge_single_input() {
    std::cout << "Test: Phi merge with single input\n";

    std::vector<LatticeValue> inputs = {
        LatticeValue::getConstant(99LL)
    };

    LatticeValue result = DataflowAnalysis::mergePhiInputs(inputs);

    assert(result.isConstant() && "Phi with single input should be that constant");
    assert(result.getAsInteger().value() == 99LL && "Result should be 99");

    std::cout << "✓ Phi merge with single input test passed\n\n";
}

void test_phi_merge_with_ranges() {
    std::cout << "Test: Phi merge with integer ranges\n";

    // Two ranges that overlap
    std::vector<LatticeValue> inputs = {
        LatticeValue::getIntegerRange(0LL, 50LL),
        LatticeValue::getIntegerRange(25LL, 75LL)
    };

    LatticeValue result = DataflowAnalysis::mergePhiInputs(inputs);

    assert(result.getKind() == LatticeValue::IntegerRange && "Merged ranges should be range");
    assert(result.getMin().value() == 25LL && "Min should be intersection start");
    assert(result.getMax().value() == 50LL && "Max should be intersection end");

    std::cout << "✓ Phi merge with integer ranges test passed\n\n";
}

void test_phi_merge_disjoint_ranges() {
    std::cout << "Test: Phi merge with disjoint ranges\n";

    // Ranges that don't overlap
    std::vector<LatticeValue> inputs = {
        LatticeValue::getIntegerRange(0LL, 10LL),
        LatticeValue::getIntegerRange(50LL, 100LL)
    };

    LatticeValue result = DataflowAnalysis::mergePhiInputs(inputs);

    assert(result.isBottom() && "Disjoint ranges should merge to Bottom");

    std::cout << "✓ Phi merge with disjoint ranges test passed\n\n";
}

void test_phi_merge_range_and_constant() {
    std::cout << "Test: Phi merge with range and constant\n";

    std::vector<LatticeValue> inputs = {
        LatticeValue::getIntegerRange(0LL, 100LL),
        LatticeValue::getConstant(50LL)
    };

    LatticeValue result = DataflowAnalysis::mergePhiInputs(inputs);

    // Constant 50 is within range [0, 100], so result is the constant
    assert(result.isConstant() && "Range and in-range constant should be constant");
    assert(result.getAsInteger().value() == 50LL && "Result should be 50");

    std::cout << "✓ Phi merge with range and constant test passed\n\n";
}

int main() {
    std::cout << "=== SCCP Phi Node Merging Tests ===\n\n";

    test_phi_merge_all_same_constants();
    test_phi_merge_different_constants();
    test_phi_merge_with_top();
    test_phi_merge_all_top();
    test_phi_merge_with_bottom();
    test_phi_merge_empty_inputs();
    test_phi_merge_single_input();
    test_phi_merge_with_ranges();
    test_phi_merge_disjoint_ranges();
    test_phi_merge_range_and_constant();

    std::cout << "=== All Phi Node Merging tests passed! ===\n";
    return 0;
}
