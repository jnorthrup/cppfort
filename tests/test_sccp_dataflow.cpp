//===- test_sccp_dataflow.cpp - SCCP Dataflow Analysis Tests ----------------===//
///
/// Tests for dataflow analysis engine in SCCP.
/// Verifies initialization, worklist processing, and lattice value propagation.
///
//===----------------------------------------------------------------------===//

#include "../include/LatticeValue.h"
#include "../include/SCCPWorklist.h"
#include "../include/DataflowAnalysis.h"
#include <cassert>
#include <iostream>
#include <unordered_map>

using namespace cppfort::sccp;

void test_dataflow_initialization() {
    std::cout << "Test: Dataflow analysis initialization\n";

    DataflowAnalysis analysis;

    // Initially, all values should be uninitialized (no lattice values stored)
    assert(analysis.getValueCount() == 0 && "Initial value count should be 0");

    std::cout << "✓ Dataflow initialization test passed\n\n";
}

void test_set_and_get_lattice_value() {
    std::cout << "Test: Set and get lattice value\n";

    DataflowAnalysis analysis;
    void* key = (void*)0x1000;

    // Set a constant value
    LatticeValue constant = LatticeValue::getConstant(42LL);
    analysis.setLatticeValue(key, constant);

    // Get the value back
    LatticeValue retrieved = analysis.getLatticeValue(key);
    assert(retrieved.isConstant() && "Retrieved value should be constant");
    assert(retrieved.getAsInteger().value() == 42LL && "Value should be 42");

    std::cout << "✓ Set and get lattice value test passed\n\n";
}

void test_get_missing_value_returns_top() {
    std::cout << "Test: Get missing value returns Top\n";

    DataflowAnalysis analysis;
    void* key = (void*)0x2000;

    // Getting a value that doesn't exist should return Top
    LatticeValue value = analysis.getLatticeValue(key);
    assert(value.isTop() && "Missing value should return Top");

    std::cout << "✓ Get missing value returns Top test passed\n\n";
}

void test_update_value_detects_change() {
    std::cout << "Test: Update value detects change\n";

    DataflowAnalysis analysis;
    void* key = (void*)0x3000;

    // Set initial value
    LatticeValue initial = LatticeValue::getTop();
    analysis.setLatticeValue(key, initial);

    // Update with a different value
    LatticeValue updated = LatticeValue::getConstant(10LL);
    bool changed = analysis.updateLatticeValue(key, updated);

    assert(changed && "Update from Top to Constant should report change");
    assert(analysis.getLatticeValue(key).isConstant() && "Value should be updated");

    // Update again with same value
    bool changedAgain = analysis.updateLatticeValue(key, updated);
    assert(!changedAgain && "Update with same value should not report change");

    std::cout << "✓ Update value detects change test passed\n\n";
}

void test_worklist_integration() {
    std::cout << "Test: Worklist integration\n";

    DataflowAnalysis analysis;
    SCCPWorklist& worklist = analysis.getWorklist();

    assert(worklist.empty() && "Initial worklist should be empty");

    // Add items to worklist
    void* item1 = (void*)0x4000;
    void* item2 = (void*)0x5000;
    worklist.enqueue(item1);
    worklist.enqueue(item2);

    assert(worklist.size() == 2 && "Worklist should have 2 items");

    std::cout << "✓ Worklist integration test passed\n\n";
}

void test_value_count() {
    std::cout << "Test: Value count tracking\n";

    DataflowAnalysis analysis;

    assert(analysis.getValueCount() == 0 && "Initial count should be 0");

    void* key1 = (void*)0x6000;
    void* key2 = (void*)0x7000;

    analysis.setLatticeValue(key1, LatticeValue::getConstant(1LL));
    assert(analysis.getValueCount() == 1 && "Count should be 1");

    analysis.setLatticeValue(key2, LatticeValue::getConstant(2LL));
    assert(analysis.getValueCount() == 2 && "Count should be 2");

    std::cout << "✓ Value count tracking test passed\n\n";
}

void test_edge_case_null_key() {
    std::cout << "Test: Edge case - null key handling\n";

    DataflowAnalysis analysis;

    // Setting value with null key should work (though unusual)
    LatticeValue value = LatticeValue::getConstant(99LL);
    analysis.setLatticeValue(nullptr, value);

    LatticeValue retrieved = analysis.getLatticeValue(nullptr);
    assert(retrieved.isConstant() && "Null key should retrieve value");

    std::cout << "✓ Null key handling test passed\n\n";
}

int main() {
    std::cout << "=== SCCP Dataflow Analysis Tests ===\n\n";

    test_dataflow_initialization();
    test_set_and_get_lattice_value();
    test_get_missing_value_returns_top();
    test_update_value_detects_change();
    test_worklist_integration();
    test_value_count();
    test_edge_case_null_key();

    std::cout << "=== All Dataflow Analysis tests passed! ===\n";
    return 0;
}
