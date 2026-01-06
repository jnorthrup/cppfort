// Test file for lifecycle-based optimization
// Tests optimization pass that eliminates transfers based on escape analysis

#include "../include/ast.hpp"
#include <cassert>
#include <iostream>
#include <memory>
#include <vector>

using namespace cpp2_transpiler;

// Helper to simulate optimization decision
struct TransferOptimizationResult {
    bool transfer_eliminated = false;
    std::string reason;
    std::vector<std::string> kept_transfers;
    std::vector<std::string> eliminated_transfers;
};

// Mock optimization pass that makes decisions based on escape analysis
TransferOptimizationResult optimize_transfers(
    const std::vector<std::pair<std::string, EscapeKind>>& var_escape_pairs) {

    TransferOptimizationResult result;

    for (const auto& [var_name, escape_kind] : var_escape_pairs) {
        if (escape_kind == EscapeKind::NoEscape) {
            // Optimization: No transfer needed for NoEscape variables
            result.eliminated_transfers.push_back(var_name);
            result.transfer_eliminated = true;
        } else if (escape_kind == EscapeKind::EscapeToGPU ||
                   escape_kind == EscapeKind::EscapeToDMA) {
            // Transfer required for GPU/DMA escapes
            result.kept_transfers.push_back(var_name);
        } else {
            // Default: keep transfer for other escape types
            result.kept_transfers.push_back(var_name);
        }
    }

    if (!result.eliminated_transfers.empty()) {
        result.reason = "Eliminated transfers for " + std::to_string(
            result.eliminated_transfers.size()) + " variables with NoEscape kind";
    }

    return result;
}

// Test: Basic optimization - eliminate NoEscape transfers
void test_optimize_noescape_transfers() {
    std::vector<std::pair<std::string, EscapeKind>> inputs = {
        {"local_x", EscapeKind::NoEscape},
        {"local_y", EscapeKind::NoEscape},
        {"result", EscapeKind::EscapeToGPU}
    };

    auto result = optimize_transfers(inputs);

    assert(result.transfer_eliminated == true);
    assert(result.eliminated_transfers.size() == 2);
    assert(result.kept_transfers.size() == 1);
    assert(result.kept_transfers[0] == "result");
    assert(result.reason.find("NoEscape") != std::string::npos);

    std::cout << "✓ Optimize NoEscape transfers test passed\n";
    std::cout << "  - Eliminated: " << result.eliminated_transfers.size() << " transfers\n";
    std::cout << "  - Kept: " << result.kept_transfers.size() << " transfers\n";
}

// Test: All NoEscape - eliminate all transfers
void test_optimize_all_noescape() {
    std::vector<std::pair<std::string, EscapeKind>> inputs = {
        {"x", EscapeKind::NoEscape},
        {"y", EscapeKind::NoEscape},
        {"z", EscapeKind::NoEscape}
    };

    auto result = optimize_transfers(inputs);

    assert(result.transfer_eliminated == true);
    assert(result.eliminated_transfers.size() == 3);
    assert(result.kept_transfers.empty());

    std::cout << "✓ Optimize all NoEscape transfers test passed\n";
}

// Test: No optimization needed - all escapes required
void test_no_optimization_needed() {
    std::vector<std::pair<std::string, EscapeKind>> inputs = {
        {"data", EscapeKind::EscapeToGPU},
        {"buffer", EscapeKind::EscapeToDMA},
        {"result", EscapeKind::EscapeToReturn}
    };

    auto result = optimize_transfers(inputs);

    assert(result.transfer_eliminated == false);
    assert(result.eliminated_transfers.empty());
    assert(result.kept_transfers.size() == 3);

    std::cout << "✓ No optimization needed test passed\n";
}

// Test: Mixed escape kinds
void test_mixed_escape_kinds() {
    std::vector<std::pair<std::string, EscapeKind>> inputs = {
        {"local_temp", EscapeKind::NoEscape},
        {"gpu_data", EscapeKind::EscapeToGPU},
        {"dma_buffer", EscapeKind::EscapeToDMA},
        {"heap_alloc", EscapeKind::EscapeToHeap},
        {"loop_var", EscapeKind::NoEscape}
    };

    auto result = optimize_transfers(inputs);

    assert(result.transfer_eliminated == true);
    assert(result.eliminated_transfers.size() == 2);
    assert(result.kept_transfers.size() == 3);

    // Verify eliminated are NoEscape
    for (const auto& var : result.eliminated_transfers) {
        assert(var == "local_temp" || var == "loop_var");
    }

    std::cout << "✓ Mixed escape kinds optimization test passed\n";
}

// Test: Kernel function optimization
void test_kernel_optimization() {
    // Simulate kernel function with variables
    std::vector<std::pair<std::string, EscapeKind>> kernel_vars = {
        {"block_id", EscapeKind::NoEscape},     // Local register
        {"thread_id", EscapeKind::NoEscape},    // Local register
        {"shared_data", EscapeKind::NoEscape},  // Shared memory (still NoEscape)
        {"global_result", EscapeKind::EscapeToGPU}  // Global memory
    };

    auto result = optimize_transfers(kernel_vars);

    assert(result.transfer_eliminated == true);
    assert(result.eliminated_transfers.size() == 3);
    assert(result.kept_transfers.size() == 1);
    assert(result.kept_transfers[0] == "global_result");

    std::cout << "✓ Kernel function optimization test passed\n";
}

// Test: Optimization with DMA async considerations
void test_dma_optimization_with_async() {
    // DMA transfers have additional considerations:
    // - Async transfers don't block
    // - But they still consume DMA engine resources
    // - Optimizing them is still valuable

    std::vector<std::pair<std::string, EscapeKind>> dma_vars = {
        {"staging_buffer", EscapeKind::NoEscape},    // Can be eliminated
        {"dma_desc", EscapeKind::EscapeToDMA},       // Required for DMA
        {"async_data", EscapeKind::EscapeToDMA},     // Required for transfer
        {"local_cache", EscapeKind::NoEscape}        // Can be eliminated
    };

    auto result = optimize_transfers(dma_vars);

    assert(result.transfer_eliminated == true);
    assert(result.eliminated_transfers.size() == 2);
    assert(result.kept_transfers.size() == 2);

    std::cout << "✓ DMA optimization with async test passed\n";
}

// Test: Performance impact simulation
void test_optimization_performance_impact() {
    // Simulate performance impact of optimization
    struct PerformanceMetrics {
        size_t transfers_before = 0;
        size_t transfers_after = 0;
        double reduction_percent = 0.0;
    };

    std::vector<std::pair<std::string, EscapeKind>> kernel_vars;

    // Simulate a kernel with 100 variables
    for (int i = 0; i < 90; ++i) {
        kernel_vars.push_back({"local_var_" + std::to_string(i), EscapeKind::NoEscape});
    }
    for (int i = 0; i < 10; ++i) {
        kernel_vars.push_back({"global_var_" + std::to_string(i), EscapeKind::EscapeToGPU});
    }

    auto result = optimize_transfers(kernel_vars);

    PerformanceMetrics metrics;
    metrics.transfers_before = kernel_vars.size();
    metrics.transfers_after = result.kept_transfers.size();
    metrics.reduction_percent =
        100.0 * (1.0 - (double)metrics.transfers_after / metrics.transfers_before);

    assert(metrics.transfers_before == 100);
    assert(metrics.transfers_after == 10);
    assert(metrics.reduction_percent == 90.0);

    std::cout << "✓ Optimization performance impact test passed\n";
    std::cout << "  - Transfers reduced by " << metrics.reduction_percent << "%\n";
    std::cout << "  - From " << metrics.transfers_before << " to "
              << metrics.transfers_after << " transfers\n";
}

// Test: Optimization with nested scopes
void test_optimization_with_nested_scopes() {
    // Inner scope variables that are NoEscape relative to kernel
    std::vector<std::pair<std::string, EscapeKind>> outer_vars = {
        {"outer_loop", EscapeKind::NoEscape}
    };

    std::vector<std::pair<std::string, EscapeKind>> inner_vars = {
        {"inner_temp", EscapeKind::NoEscape},
        {"inner_result", EscapeKind::EscapeToGPU}
    };

    auto outer_result = optimize_transfers(outer_vars);
    auto inner_result = optimize_transfers(inner_vars);

    assert(outer_result.eliminated_transfers.size() == 1);
    assert(inner_result.eliminated_transfers.size() == 1);
    assert(inner_result.kept_transfers.size() == 1);

    std::cout << "✓ Optimization with nested scopes test passed\n";
}

int main() {
    std::cout << "=== Lifecycle-Based Optimization Test Suite ===\n\n";

    std::cout << "--- Basic Optimization Tests ---\n";
    test_optimize_noescape_transfers();
    test_optimize_all_noescape();
    test_no_optimization_needed();

    std::cout << "\n--- Advanced Optimization Tests ---\n";
    test_mixed_escape_kinds();
    test_kernel_optimization();
    test_dma_optimization_with_async();

    std::cout << "\n--- Performance Tests ---\n";
    test_optimization_performance_impact();
    test_optimization_with_nested_scopes();

    std::cout << "\n✅ All lifecycle-based optimization tests passed!\n";
    std::cout << "\nKey Optimizations:\n";
    std::cout << "  ✓ NoEscape variables eliminate transfers\n";
    std::cout << "  ✓ GPU/DMA variables require transfers\n";
    std::cout << "  ✓ Up to 90% reduction in unnecessary transfers\n";

    return 0;
}
