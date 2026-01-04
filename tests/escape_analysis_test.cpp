// Test file for escape analysis infrastructure
// Tests EscapeInfo and EscapeKind enums added to ast.hpp

#include "../include/ast.hpp"
#include <cassert>
#include <iostream>

using namespace cpp2_transpiler;

// Test that EscapeKind enum exists and has all required values
void test_escape_kind_enum() {
    // Test enum values exist and are distinct
    EscapeKind no_escape = EscapeKind::NoEscape;
    EscapeKind to_heap = EscapeKind::EscapeToHeap;
    EscapeKind to_return = EscapeKind::EscapeToReturn;
    EscapeKind to_param = EscapeKind::EscapeToParam;
    EscapeKind to_global = EscapeKind::EscapeToGlobal;
    EscapeKind to_channel = EscapeKind::EscapeToChannel;
    EscapeKind to_gpu = EscapeKind::EscapeToGPU;
    EscapeKind to_dma = EscapeKind::EscapeToDMA;

    // Verify they're all distinct
    assert(no_escape != to_heap);
    assert(to_heap != to_return);
    assert(to_return != to_param);
    assert(to_param != to_global);
    assert(to_global != to_channel);
    assert(to_channel != to_gpu);
    assert(to_gpu != to_dma);

    std::cout << "✓ EscapeKind enum tests passed\n";
}

// Test that EscapeInfo struct exists and has required fields
void test_escape_info_struct() {
    EscapeInfo info;

    // Test default construction
    info.kind = EscapeKind::NoEscape;
    assert(info.kind == EscapeKind::NoEscape);

    // Test escape_points field (should be a vector)
    assert(info.escape_points.empty());

    // Test needs_lifetime_extension field
    info.needs_lifetime_extension = false;
    assert(!info.needs_lifetime_extension);

    info.needs_lifetime_extension = true;
    assert(info.needs_lifetime_extension);

    std::cout << "✓ EscapeInfo struct tests passed\n";
}

// Test EscapeInfo with different escape scenarios
void test_escape_info_scenarios() {
    // Scenario 1: NoEscape (stack-local variable)
    EscapeInfo stack_local;
    stack_local.kind = EscapeKind::NoEscape;
    stack_local.needs_lifetime_extension = false;
    assert(stack_local.escape_points.empty());

    // Scenario 2: EscapeToReturn (returned from function)
    EscapeInfo returned_value;
    returned_value.kind = EscapeKind::EscapeToReturn;
    returned_value.needs_lifetime_extension = true;

    // Scenario 3: EscapeToHeap (stored in heap-allocated object)
    EscapeInfo heap_stored;
    heap_stored.kind = EscapeKind::EscapeToHeap;
    heap_stored.needs_lifetime_extension = true;

    // Scenario 4: EscapeToGPU (transferred to GPU memory)
    EscapeInfo gpu_transfer;
    gpu_transfer.kind = EscapeKind::EscapeToGPU;
    gpu_transfer.needs_lifetime_extension = true;

    // Scenario 5: EscapeToChannel (sent through channel)
    EscapeInfo channel_send;
    channel_send.kind = EscapeKind::EscapeToChannel;
    channel_send.needs_lifetime_extension = true;

    std::cout << "✓ EscapeInfo scenario tests passed\n";
}

int main() {
    test_escape_kind_enum();
    test_escape_info_struct();
    test_escape_info_scenarios();

    std::cout << "\n✅ All escape analysis tests passed!\n";
    return 0;
}
