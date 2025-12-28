//===- test_sccp_worklist.cpp - SCCP Worklist Tests -------------------------===//
///
/// Tests for the worklist algorithm used in SCCP analysis.
/// The worklist tracks which operations need to be reprocessed when their
/// lattice values change during dataflow analysis.
///
//===----------------------------------------------------------------------===//

#include "../include/SCCPWorklist.h"
#include <cassert>
#include <iostream>

using namespace cppfort::sccp;

void test_worklist_empty() {
    std::cout << "Test: Worklist is empty on creation\n";

    SCCPWorklist worklist;

    assert(worklist.empty() && "New worklist should be empty");
    assert(worklist.size() == 0 && "Size should be 0");

    std::cout << "✓ Worklist empty test passed\n\n";
}

void test_worklist_enqueue_dequeue() {
    std::cout << "Test: Worklist enqueue and dequeue\n";

    SCCPWorklist worklist;

    // Enqueue some items (using void* as opaque identifiers)
    void* item1 = (void*)0x1;
    void* item2 = (void*)0x2;
    void* item3 = (void*)0x3;

    worklist.enqueue(item1);
    worklist.enqueue(item2);
    worklist.enqueue(item3);

    assert(!worklist.empty() && "Worklist should not be empty");
    assert(worklist.size() == 3 && "Size should be 3");

    // Dequeue items (FIFO order)
    void* result1 = worklist.dequeue();
    void* result2 = worklist.dequeue();
    void* result3 = worklist.dequeue();

    assert(result1 == item1 && "First dequeue should return item1");
    assert(result2 == item2 && "Second dequeue should return item2");
    assert(result3 == item3 && "Third dequeue should return item3");
    assert(worklist.empty() && "Worklist should be empty after dequeue all");

    std::cout << "✓ Worklist enqueue/dequeue test passed\n\n";
}

void test_worklist_no_duplicates() {
    std::cout << "Test: Worklist prevents duplicate entries\n";

    SCCPWorklist worklist;

    void* item1 = (void*)0x1;

    // Enqueue same item multiple times
    worklist.enqueue(item1);
    worklist.enqueue(item1);
    worklist.enqueue(item1);

    // Should only have one entry
    assert(worklist.size() == 1 && "Worklist should prevent duplicates");

    void* result = worklist.dequeue();
    assert(result == item1 && "Should return the item");
    assert(worklist.empty() && "Worklist should be empty");

    std::cout << "✓ Worklist no duplicates test passed\n\n";
}

void test_worklist_dequeue_empty() {
    std::cout << "Test: Worklist dequeue empty returns nullptr\n";

    SCCPWorklist worklist;

    void* result = worklist.dequeue();

    assert(result == nullptr && "Dequeue empty should return nullptr");

    std::cout << "✓ Worklist dequeue empty test passed\n\n";
}

void test_worklist_clear() {
    std::cout << "Test: Worklist clear\n";

    SCCPWorklist worklist;

    worklist.enqueue((void*)0x1);
    worklist.enqueue((void*)0x2);
    worklist.enqueue((void*)0x3);

    worklist.clear();

    assert(worklist.empty() && "Worklist should be empty after clear");
    assert(worklist.size() == 0 && "Size should be 0");

    std::cout << "✓ Worklist clear test passed\n\n";
}

void test_worklist_contains() {
    std::cout << "Test: Worklist contains check\n";

    SCCPWorklist worklist;

    void* item1 = (void*)0x1;
    void* item2 = (void*)0x2;

    worklist.enqueue(item1);

    assert(worklist.contains(item1) && "Worklist should contain item1");
    assert(!worklist.contains(item2) && "Worklist should not contain item2");

    std::cout << "✓ Worklist contains test passed\n\n";
}

void test_worklist_reenqueue() {
    std::cout << "Test: Worklist allows re-enqueue after dequeue\n";

    SCCPWorklist worklist;

    void* item1 = (void*)0x1;

    worklist.enqueue(item1);
    worklist.dequeue(); // Remove from worklist

    // Should be able to re-enqueue
    worklist.enqueue(item1);

    assert(worklist.size() == 1 && "Worklist should have 1 item after re-enqueue");

    std::cout << "✓ Worklist re-enqueue test passed\n\n";
}

int main() {
    std::cout << "=== SCCP Worklist Tests ===\n\n";

    test_worklist_empty();
    test_worklist_enqueue_dequeue();
    test_worklist_no_duplicates();
    test_worklist_dequeue_empty();
    test_worklist_clear();
    test_worklist_contains();
    test_worklist_reenqueue();

    std::cout << "=== All Worklist tests passed! ===\n";
    return 0;
}
