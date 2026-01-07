// FIR Transfer Elimination Pass Unit Tests
// Tests MLIR optimization pass that eliminates unnecessary GPU/DMA transfers
// based on escape analysis annotations.
//
// Phase 3: External Memory Integration (Semantic AST Enhancements Track)

#include <cassert>
#include <iostream>
#include <vector>
#include <string>
#include <map>
#include <memory>

//===----------------------------------------------------------------------===//
// Escape Analysis Types (mirroring MLIR pass)
//===----------------------------------------------------------------------===//

enum class EscapeKind {
    NoEscape,           // Value stays local (stack) - can eliminate transfer
    EscapeToHeap,       // Stored in heap-allocated object
    EscapeToReturn,     // Returned from function
    EscapeToParam,      // Stored via pointer/reference parameter
    EscapeToGlobal,     // Stored in global variable
    EscapeToChannel,    // Sent through channel
    EscapeToGPU,        // Transferred to GPU memory - transfer required
    EscapeToDMA         // Transferred via DMA buffer - transfer required
};

bool transferRequired(EscapeKind kind) {
    return kind == EscapeKind::EscapeToGPU || kind == EscapeKind::EscapeToDMA;
}

std::string escapeKindToString(EscapeKind kind) {
    switch (kind) {
        case EscapeKind::NoEscape: return "no_escape";
        case EscapeKind::EscapeToHeap: return "heap";
        case EscapeKind::EscapeToReturn: return "return";
        case EscapeKind::EscapeToParam: return "param";
        case EscapeKind::EscapeToGlobal: return "global";
        case EscapeKind::EscapeToChannel: return "channel";
        case EscapeKind::EscapeToGPU: return "gpu";
        case EscapeKind::EscapeToDMA: return "dma";
    }
    return "unknown";
}

//===----------------------------------------------------------------------===//
// Mock MLIR Structures for Testing
//===----------------------------------------------------------------------===//

struct MockValue {
    std::string name;
    EscapeKind escapeKind;
};

struct MockOperation {
    std::string name;
    std::vector<std::string> operands;
    std::vector<std::string> results;
    bool isTransferOp;
    std::map<std::string, std::string> attributes;
};

struct TransferEliminationResult {
    unsigned totalTransfers;
    unsigned eliminatedTransfers;
    unsigned keptTransfers;
    std::vector<std::string> eliminatedOps;
    std::vector<std::string> keptOps;
};

//===----------------------------------------------------------------------===//
// Mock Transfer Elimination Pass
//===----------------------------------------------------------------------===//

TransferEliminationResult runTransferElimination(
    const std::vector<MockOperation>& operations,
    const std::map<std::string, EscapeKind>& escapeMap) {
    
    TransferEliminationResult result = {0, 0, 0, {}, {}};
    
    for (const auto& op : operations) {
        if (!op.isTransferOp) {
            continue;
        }
        
        result.totalTransfers++;
        
        // Check if all operands are NoEscape
        bool canEliminate = true;
        for (const auto& operand : op.operands) {
            auto it = escapeMap.find(operand);
            if (it != escapeMap.end() && transferRequired(it->second)) {
                canEliminate = false;
                break;
            }
        }
        
        if (canEliminate) {
            result.eliminatedTransfers++;
            result.eliminatedOps.push_back(op.name);
        } else {
            result.keptTransfers++;
            result.keptOps.push_back(op.name);
        }
    }
    
    return result;
}

//===----------------------------------------------------------------------===//
// Unit Tests
//===----------------------------------------------------------------------===//

// Test 1: Basic elimination of NoEscape transfers
void test_basic_noescape_elimination() {
    std::cout << "Test 1: Basic NoEscape elimination\n";
    
    std::vector<MockOperation> ops = {
        {"alloc_local", {}, {"local_var"}, false, {}},
        {"transfer_to_gpu_local", {"local_var"}, {"gpu_local"}, true, {}},
        {"compute", {"gpu_local"}, {"result"}, false, {}}
    };
    
    std::map<std::string, EscapeKind> escapes = {
        {"local_var", EscapeKind::NoEscape},
        {"gpu_local", EscapeKind::NoEscape},
        {"result", EscapeKind::NoEscape}
    };
    
    auto result = runTransferElimination(ops, escapes);
    
    assert(result.totalTransfers == 1);
    assert(result.eliminatedTransfers == 1);
    assert(result.keptTransfers == 0);
    assert(result.eliminatedOps.size() == 1);
    assert(result.eliminatedOps[0] == "transfer_to_gpu_local");
    
    std::cout << "  ✓ Eliminated 1 unnecessary transfer\n";
}

// Test 2: GPU escape requires transfer
void test_gpu_escape_keeps_transfer() {
    std::cout << "\nTest 2: GPU escape keeps transfer\n";
    
    std::vector<MockOperation> ops = {
        {"alloc_data", {}, {"data"}, false, {}},
        {"transfer_to_gpu", {"data"}, {"gpu_data"}, true, {}},
        {"kernel_compute", {"gpu_data"}, {"gpu_result"}, false, {}},
        {"transfer_from_gpu", {"gpu_result"}, {"host_result"}, true, {}}
    };
    
    std::map<std::string, EscapeKind> escapes = {
        {"data", EscapeKind::EscapeToGPU},      // Needs transfer
        {"gpu_data", EscapeKind::EscapeToGPU},
        {"gpu_result", EscapeKind::EscapeToGPU},
        {"host_result", EscapeKind::NoEscape}
    };
    
    auto result = runTransferElimination(ops, escapes);
    
    assert(result.totalTransfers == 2);
    assert(result.keptTransfers == 2);  // Both transfers kept
    assert(result.eliminatedTransfers == 0);
    
    std::cout << "  ✓ Kept both GPU transfers (escape kind = gpu)\n";
}

// Test 3: Mixed escape kinds - partial elimination
void test_mixed_escape_partial_elimination() {
    std::cout << "\nTest 3: Mixed escape kinds - partial elimination\n";
    
    std::vector<MockOperation> ops = {
        {"transfer_local_temp", {"temp"}, {"gpu_temp"}, true, {}},
        {"transfer_constant", {"const_val"}, {"gpu_const"}, true, {}},
        {"transfer_kernel_data", {"kernel_data"}, {"gpu_kernel_data"}, true, {}},
        {"transfer_result", {"result"}, {"gpu_result"}, true, {}}
    };
    
    std::map<std::string, EscapeKind> escapes = {
        {"temp", EscapeKind::NoEscape},           // Can eliminate
        {"const_val", EscapeKind::NoEscape},      // Can eliminate
        {"kernel_data", EscapeKind::EscapeToGPU}, // Must keep
        {"result", EscapeKind::EscapeToGPU}       // Must keep
    };
    
    auto result = runTransferElimination(ops, escapes);
    
    assert(result.totalTransfers == 4);
    assert(result.eliminatedTransfers == 2);
    assert(result.keptTransfers == 2);
    
    std::cout << "  ✓ Eliminated 2/4 transfers (50% reduction)\n";
    std::cout << "    Eliminated: temp, const_val\n";
    std::cout << "    Kept: kernel_data, result\n";
}

// Test 4: DMA escape requires transfer
void test_dma_escape_keeps_transfer() {
    std::cout << "\nTest 4: DMA escape keeps transfer\n";
    
    std::vector<MockOperation> ops = {
        {"dma_transfer_staging", {"staging"}, {"dma_staging"}, true, {}},
        {"dma_transfer_data", {"main_data"}, {"dma_data"}, true, {}}
    };
    
    std::map<std::string, EscapeKind> escapes = {
        {"staging", EscapeKind::NoEscape},      // Can eliminate
        {"main_data", EscapeKind::EscapeToDMA}  // Must keep
    };
    
    auto result = runTransferElimination(ops, escapes);
    
    assert(result.totalTransfers == 2);
    assert(result.eliminatedTransfers == 1);
    assert(result.keptTransfers == 1);
    assert(result.keptOps[0] == "dma_transfer_data");
    
    std::cout << "  ✓ Kept DMA transfer for escaping data\n";
    std::cout << "  ✓ Eliminated staging buffer transfer\n";
}

// Test 5: All NoEscape - eliminate all transfers
void test_all_noescape_eliminate_all() {
    std::cout << "\nTest 5: All NoEscape - eliminate all\n";
    
    std::vector<MockOperation> ops = {
        {"transfer_a", {"a"}, {"a_gpu"}, true, {}},
        {"transfer_b", {"b"}, {"b_gpu"}, true, {}},
        {"transfer_c", {"c"}, {"c_gpu"}, true, {}}
    };
    
    std::map<std::string, EscapeKind> escapes = {
        {"a", EscapeKind::NoEscape},
        {"b", EscapeKind::NoEscape},
        {"c", EscapeKind::NoEscape}
    };
    
    auto result = runTransferElimination(ops, escapes);
    
    assert(result.totalTransfers == 3);
    assert(result.eliminatedTransfers == 3);
    assert(result.keptTransfers == 0);
    
    std::cout << "  ✓ Eliminated all 3 transfers (100% reduction)\n";
}

// Test 6: No transfers - empty case
void test_no_transfers() {
    std::cout << "\nTest 6: No transfers (no-op)\n";
    
    std::vector<MockOperation> ops = {
        {"compute_a", {"x"}, {"y"}, false, {}},
        {"compute_b", {"y"}, {"z"}, false, {}}
    };
    
    std::map<std::string, EscapeKind> escapes = {
        {"x", EscapeKind::NoEscape},
        {"y", EscapeKind::NoEscape}
    };
    
    auto result = runTransferElimination(ops, escapes);
    
    assert(result.totalTransfers == 0);
    assert(result.eliminatedTransfers == 0);
    assert(result.keptTransfers == 0);
    
    std::cout << "  ✓ No transfers to process (correct no-op)\n";
}

// Test 7: Heap escape (not GPU/DMA) - should eliminate
void test_heap_escape_eliminates() {
    std::cout << "\nTest 7: Heap escape eliminates transfer\n";
    
    std::vector<MockOperation> ops = {
        {"transfer_heap", {"heap_var"}, {"gpu_heap"}, true, {}}
    };
    
    std::map<std::string, EscapeKind> escapes = {
        {"heap_var", EscapeKind::EscapeToHeap}  // Not GPU/DMA, so eliminate
    };
    
    auto result = runTransferElimination(ops, escapes);
    
    assert(result.eliminatedTransfers == 1);
    assert(result.keptTransfers == 0);
    
    std::cout << "  ✓ Eliminated transfer (heap escape != GPU/DMA)\n";
}

// Test 8: Channel escape (not GPU/DMA) - should eliminate
void test_channel_escape_eliminates() {
    std::cout << "\nTest 8: Channel escape eliminates transfer\n";
    
    std::vector<MockOperation> ops = {
        {"transfer_channel", {"channel_var"}, {"gpu_channel"}, true, {}}
    };
    
    std::map<std::string, EscapeKind> escapes = {
        {"channel_var", EscapeKind::EscapeToChannel}  // Not GPU/DMA
    };
    
    auto result = runTransferElimination(ops, escapes);
    
    assert(result.eliminatedTransfers == 1);
    assert(result.keptTransfers == 0);
    
    std::cout << "  ✓ Eliminated transfer (channel escape != GPU/DMA)\n";
}

// Test 9: GPU kernel simulation
void test_gpu_kernel_optimization() {
    std::cout << "\nTest 9: GPU kernel optimization scenario\n";
    
    // Simulates a typical GPU kernel with local variables and data transfers
    std::vector<MockOperation> ops = {
        {"alloc_shared_mem", {}, {"shared_mem"}, false, {}},
        {"transfer_block_id", {"block_id"}, {"gpu_block_id"}, true, {}},    // Local to block
        {"transfer_thread_id", {"thread_id"}, {"gpu_thread_id"}, true, {}}, // Local to thread
        {"transfer_kernel_input", {"input"}, {"gpu_input"}, true, {}},       // Actual data
        {"kernel_compute", {"gpu_input"}, {"gpu_output"}, false, {}},
        {"transfer_kernel_output", {"gpu_output"}, {"output"}, true, {}}    // Result
    };
    
    std::map<std::string, EscapeKind> escapes = {
        {"shared_mem", EscapeKind::NoEscape},
        {"block_id", EscapeKind::NoEscape},      // Just an index - no transfer needed
        {"thread_id", EscapeKind::NoEscape},     // Just an index - no transfer needed
        {"input", EscapeKind::EscapeToGPU},      // Real data needs transfer
        {"gpu_output", EscapeKind::EscapeToGPU}  // Result needs transfer back
    };
    
    auto result = runTransferElimination(ops, escapes);
    
    assert(result.totalTransfers == 4);
    assert(result.eliminatedTransfers == 2);  // block_id and thread_id
    assert(result.keptTransfers == 2);         // input and output
    
    std::cout << "  ✓ Optimized GPU kernel: 2/4 transfers eliminated (50%)\n";
    std::cout << "    Eliminated: block_id, thread_id (indices stay local)\n";
    std::cout << "    Kept: input, output (actual data transfers)\n";
}

// Test 10: Verify statistics calculation
void test_statistics() {
    std::cout << "\nTest 10: Statistics calculation\n";
    
    std::vector<MockOperation> ops = {
        {"t1", {"v1"}, {}, true, {}},
        {"t2", {"v2"}, {}, true, {}},
        {"t3", {"v3"}, {}, true, {}},
        {"t4", {"v4"}, {}, true, {}},
        {"t5", {"v5"}, {}, true, {}}
    };
    
    std::map<std::string, EscapeKind> escapes = {
        {"v1", EscapeKind::NoEscape},
        {"v2", EscapeKind::EscapeToGPU},
        {"v3", EscapeKind::NoEscape},
        {"v4", EscapeKind::EscapeToDMA},
        {"v5", EscapeKind::NoEscape}
    };
    
    auto result = runTransferElimination(ops, escapes);
    
    assert(result.totalTransfers == 5);
    assert(result.eliminatedTransfers == 3);  // v1, v3, v5
    assert(result.keptTransfers == 2);         // v2, v4
    
    double reductionPercent = (double)result.eliminatedTransfers / result.totalTransfers * 100;
    assert(reductionPercent == 60.0);
    
    std::cout << "  ✓ Statistics: 5 total, 3 eliminated, 2 kept (60% reduction)\n";
}

//===----------------------------------------------------------------------===//
// Main
//===----------------------------------------------------------------------===//

int main() {
    std::cout << "=== FIR Transfer Elimination Pass Unit Tests ===\n";
    std::cout << "Phase 3: External Memory Integration\n\n";
    
    test_basic_noescape_elimination();
    test_gpu_escape_keeps_transfer();
    test_mixed_escape_partial_elimination();
    test_dma_escape_keeps_transfer();
    test_all_noescape_eliminate_all();
    test_no_transfers();
    test_heap_escape_eliminates();
    test_channel_escape_eliminates();
    test_gpu_kernel_optimization();
    test_statistics();
    
    std::cout << "\n========================================\n";
    std::cout << "✅ All 10 tests passed!\n";
    std::cout << "Transfer elimination pass verified.\n";
    std::cout << "========================================\n";
    
    return 0;
}
