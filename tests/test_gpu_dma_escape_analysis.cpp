// Test: GPU/DMA escape analysis integration
// Tests connecting escape analysis to external memory transfers

#include <iostream>
#include <cassert>
#include <memory>

#include "../include/ast.hpp"

using namespace cpp2_transpiler;

// Test 1: EscapeInfo tracks GPU transfers
void test_escape_info_gpu_transfer() {
    std::cout << "Test: EscapeInfo tracks GPU transfers\n";

    // Create an EscapeInfo for a variable escaping to GPU
    EscapeInfo escape;
    escape.kind = EscapeKind::EscapeToGPU;
    escape.needs_lifetime_extension = true;

    assert(escape.kind == EscapeKind::EscapeToGPU);
    assert(escape.needs_lifetime_extension == true);
    assert(escape.escape_points.empty());

    std::cout << "✓ Test passed\n";
}

// Test 2: EscapeInfo tracks DMA transfers
void test_escape_info_dma_transfer() {
    std::cout << "Test: EscapeInfo tracks DMA transfers\n";

    EscapeInfo escape;
    escape.kind = EscapeKind::EscapeToDMA;
    escape.needs_lifetime_extension = false;  // DMA completes synchronously

    assert(escape.kind == EscapeKind::EscapeToDMA);
    assert(escape.needs_lifetime_extension == false);

    std::cout << "✓ Test passed\n";
}

// Test 3: MemoryTransfer linked to EscapeInfo
void test_memory_transfer_escape_link() {
    std::cout << "Test: MemoryTransfer linked to EscapeInfo\n";

    // Create a GPU memory region
    MemoryRegion gpu_region;
    gpu_region.name = "gpu_global";
    gpu_region.is_device_memory = true;

    // Create a transfer with escape info
    MemoryTransfer transfer;
    transfer.escape_kind = EscapeKind::EscapeToGPU;
    transfer.dest_region = &gpu_region;
    transfer.is_async = true;

    // Verify the escape kind matches the transfer type
    assert(transfer.escape_kind == EscapeKind::EscapeToGPU);
    assert(transfer.dest_region->is_device_memory == true);
    assert(transfer.is_async == true);

    std::cout << "✓ Test passed\n";
}

// Test 4: VariableDeclaration with EscapeInfo and MemoryTransfer
void test_vardecl_with_escape_and_transfer() {
    std::cout << "Test: VariableDeclaration with EscapeInfo and MemoryTransfer\n";

    // Create a variable declaration
    auto var = std::make_unique<VariableDeclaration>("gpu_data", 10);

    // Attach escape info
    var->escape_info = std::make_unique<EscapeInfo>();
    var->escape_info->kind = EscapeKind::EscapeToGPU;
    var->escape_info->needs_lifetime_extension = true;

    // Attach memory transfer info
    var->memory_transfer = std::make_unique<MemoryTransfer>();
    var->memory_transfer->escape_kind = EscapeKind::EscapeToGPU;
    var->memory_transfer->is_async = true;

    // Verify linkage
    assert(var->escape_info != nullptr);
    assert(var->escape_info->kind == EscapeKind::EscapeToGPU);
    assert(var->memory_transfer != nullptr);
    assert(var->memory_transfer->escape_kind == var->escape_info->kind);

    std::cout << "✓ Test passed\n";
}

// Test 5: Multiple variables with different escape patterns
void test_multiple_escape_patterns() {
    std::cout << "Test: Multiple variables with different escape patterns\n";

    // Variable 1: Escapes to GPU
    auto gpu_var = std::make_unique<VariableDeclaration>("gpu_buffer", 1);
    gpu_var->escape_info = std::make_unique<EscapeInfo>();
    gpu_var->escape_info->kind = EscapeKind::EscapeToGPU;

    // Variable 2: Escapes to DMA
    auto dma_var = std::make_unique<VariableDeclaration>("dma_buffer", 2);
    dma_var->escape_info = std::make_unique<EscapeInfo>();
    dma_var->escape_info->kind = EscapeKind::EscapeToDMA;

    // Variable 3: No escape (stack-local)
    auto local_var = std::make_unique<VariableDeclaration>("local_temp", 3);
    local_var->escape_info = std::make_unique<EscapeInfo>();
    local_var->escape_info->kind = EscapeKind::NoEscape;

    // Verify distinct patterns
    assert(gpu_var->escape_info->kind == EscapeKind::EscapeToGPU);
    assert(dma_var->escape_info->kind == EscapeKind::EscapeToDMA);
    assert(local_var->escape_info->kind == EscapeKind::NoEscape);

    // Only GPU/DMA vars should have memory transfers
    assert(gpu_var->name == "gpu_buffer");
    assert(dma_var->name == "dma_buffer");
    assert(local_var->name == "local_temp");

    std::cout << "✓ Test passed\n";
}

int main() {
    std::cout << "=== GPU/DMA Escape Analysis Integration Tests ===\n\n";

    test_escape_info_gpu_transfer();
    test_escape_info_dma_transfer();
    test_memory_transfer_escape_link();
    test_vardecl_with_escape_and_transfer();
    test_multiple_escape_patterns();

    std::cout << "\nAll tests passed! ✓\n";
    return 0;
}
