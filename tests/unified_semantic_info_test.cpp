// Unified Semantic Info Tests
// Phase 5: Semantic AST Enhancements Track
//
// Tests SemanticInfo struct, query methods, and semantic dump.
//
// TDD Red Phase: These tests define expected behavior before implementation.

#include <cassert>
#include <iostream>
#include <vector>
#include <string>
#include <memory>
#include <optional>
#include <sstream>

namespace cpp2_transpiler {

//===----------------------------------------------------------------------===//
// Existing Types (from ast.hpp)
//===----------------------------------------------------------------------===//

enum class EscapeKind {
    NoEscape,
    EscapeToHeap,
    EscapeToReturn,
    EscapeToParam,
    EscapeToGlobal,
    EscapeToChannel,
    EscapeToGPU,
    EscapeToDMA
};

enum class OwnershipKind {
    Owned,
    Borrowed,
    MutBorrowed,
    Moved
};

struct LifetimeRegion {
    void* scope_start = nullptr;
    void* scope_end = nullptr;
    std::vector<LifetimeRegion*> nested_regions;
    std::string name;  // For debugging
    
    bool outlives(const LifetimeRegion& other) const {
        for (const auto* nested : nested_regions) {
            if (nested == &other) return true;
            if (nested->outlives(other)) return true;
        }
        return false;
    }
};

struct EscapeInfo {
    EscapeKind kind = EscapeKind::NoEscape;
    std::vector<void*> escape_points;
    bool needs_lifetime_extension = false;
};

struct BorrowInfo {
    OwnershipKind kind = OwnershipKind::Owned;
    void* owner = nullptr;
    std::vector<void*> active_borrows;
    LifetimeRegion* lifetime = nullptr;
};

struct MemoryRegion {
    std::string name;
    std::size_t size_bytes = 0;
    bool is_device_memory = false;
};

struct MemoryTransfer {
    EscapeKind escape_kind = EscapeKind::NoEscape;
    MemoryRegion* source_region = nullptr;
    MemoryRegion* dest_region = nullptr;
    bool is_async = false;
};

struct ChannelTransfer {
    EscapeKind escape_kind = EscapeKind::EscapeToChannel;
    void* send_point = nullptr;
    void* recv_point = nullptr;
    OwnershipKind ownership_transfer = OwnershipKind::Moved;
    std::string channel_name;
};

//===----------------------------------------------------------------------===//
// NEW: Safety Contract (for contract annotations)
//===----------------------------------------------------------------------===//

struct SafetyContract {
    enum class Kind {
        Precondition,   // [[expects: condition]]
        Postcondition,  // [[ensures: condition]]
        Assertion,      // [[assert: condition]]
        Invariant       // Type invariant
    };
    
    Kind kind;
    std::string condition;      // The condition expression as string
    std::string message;        // Optional message
    bool is_audit = false;      // Audit-level contract (may be expensive)
    
    SafetyContract(Kind k, std::string cond, std::string msg = "")
        : kind(k), condition(std::move(cond)), message(std::move(msg)) {}
};

//===----------------------------------------------------------------------===//
// NEW: Kernel Launch Context (for GPU operations)
//===----------------------------------------------------------------------===//

struct KernelLaunch {
    std::string kernel_name;
    std::string grid_dims;      // e.g., "256,256"
    std::string block_dims;     // e.g., "32"
    std::string memory_policy;  // "coherent", "streaming", "private"
    bool is_active = false;
};

//===----------------------------------------------------------------------===//
// NEW: Unified Semantic Info (Phase 5)
//===----------------------------------------------------------------------===//

struct SemanticInfo {
    // Ownership and borrowing
    BorrowInfo borrow;
    
    // Escape analysis
    EscapeInfo escape;
    
    // Memory location
    std::optional<MemoryRegion> memory_region;
    std::optional<MemoryTransfer> active_transfer;
    
    // Concurrency
    std::optional<ChannelTransfer> channel_transfer;
    std::optional<KernelLaunch> kernel_context;
    
    // Lifetime bounds
    LifetimeRegion lifetime;
    std::vector<LifetimeRegion*> must_outlive;
    
    // Safety contracts
    std::vector<SafetyContract> contracts;
    
    // Constructors
    SemanticInfo() = default;
    
    //===------------------------------------------------------------------===//
    // Query Methods
    //===------------------------------------------------------------------===//
    
    /// Returns true if this semantic info indicates safe operation
    bool is_safe() const {
        // Not safe if:
        // 1. Value has escaped and is being used (use-after-move)
        if (borrow.kind == OwnershipKind::Moved) {
            return false;
        }
        
        // 2. Mutable borrow with existing borrows (aliasing)
        if (borrow.kind == OwnershipKind::MutBorrowed && 
            !borrow.active_borrows.empty()) {
            return false;
        }
        
        // 3. DMA transfer with async and potential aliasing
        if (active_transfer && active_transfer->is_async) {
            // Async transfers need special handling
            return escape.kind == EscapeKind::NoEscape;
        }
        
        return true;
    }
    
    /// Returns true if this value can be optimized away
    bool can_optimize_away() const {
        // Cannot optimize away if escapes to external memory (GPU/DMA)
        if (escape.kind == EscapeKind::EscapeToGPU ||
            escape.kind == EscapeKind::EscapeToDMA ||
            escape.kind == EscapeKind::EscapeToChannel) {
            return false;
        }
        
        // Can optimize away if:
        // 1. NoEscape - value stays local
        if (escape.kind == EscapeKind::NoEscape) {
            return true;
        }
        
        // 2. No active transfers and no external escapes
        if (!active_transfer && !channel_transfer) {
            // If also owned and no borrows, can potentially inline
            if (borrow.kind == OwnershipKind::Owned && 
                borrow.active_borrows.empty()) {
                return true;
            }
        }
        
        return false;
    }
    
    /// Returns human-readable explanation of semantics
    std::string explain_semantics() const {
        std::ostringstream ss;
        
        // Ownership
        ss << "Ownership: ";
        switch (borrow.kind) {
            case OwnershipKind::Owned: ss << "owned"; break;
            case OwnershipKind::Borrowed: ss << "borrowed (immutable)"; break;
            case OwnershipKind::MutBorrowed: ss << "borrowed (mutable)"; break;
            case OwnershipKind::Moved: ss << "MOVED (invalid)"; break;
        }
        ss << "\n";
        
        // Escape
        ss << "Escape: ";
        switch (escape.kind) {
            case EscapeKind::NoEscape: ss << "local (no escape)"; break;
            case EscapeKind::EscapeToHeap: ss << "escapes to heap"; break;
            case EscapeKind::EscapeToReturn: ss << "escapes via return"; break;
            case EscapeKind::EscapeToParam: ss << "escapes via parameter"; break;
            case EscapeKind::EscapeToGlobal: ss << "escapes to global"; break;
            case EscapeKind::EscapeToChannel: ss << "escapes to channel"; break;
            case EscapeKind::EscapeToGPU: ss << "escapes to GPU"; break;
            case EscapeKind::EscapeToDMA: ss << "escapes to DMA"; break;
        }
        ss << "\n";
        
        // Memory region
        if (memory_region) {
            ss << "Memory: " << memory_region->name;
            if (memory_region->is_device_memory) {
                ss << " (device)";
            }
            ss << "\n";
        }
        
        // Active transfer
        if (active_transfer) {
            ss << "Transfer: ";
            if (active_transfer->is_async) {
                ss << "async ";
            }
            ss << "active\n";
        }
        
        // Channel
        if (channel_transfer) {
            ss << "Channel: " << channel_transfer->channel_name << "\n";
        }
        
        // Kernel
        if (kernel_context && kernel_context->is_active) {
            ss << "Kernel: " << kernel_context->kernel_name << "\n";
        }
        
        // Contracts
        if (!contracts.empty()) {
            ss << "Contracts: " << contracts.size() << " active\n";
        }
        
        // Safety
        ss << "Safe: " << (is_safe() ? "yes" : "NO") << "\n";
        ss << "Optimizable: " << (can_optimize_away() ? "yes" : "no") << "\n";
        
        return ss.str();
    }
    
    /// Generate MLIR attributes for this semantic info
    std::string to_mlir_attributes() const {
        std::ostringstream ss;
        
        // Ownership attribute
        switch (borrow.kind) {
            case OwnershipKind::Owned:
                ss << "#cpp2.owned";
                break;
            case OwnershipKind::Borrowed:
                ss << "#cpp2.borrowed";
                break;
            case OwnershipKind::MutBorrowed:
                ss << "#cpp2.mut_borrowed";
                break;
            case OwnershipKind::Moved:
                ss << "#cpp2.moved";
                break;
        }
        
        // Escape attribute
        if (escape.kind != EscapeKind::NoEscape) {
            ss << ", #cpp2.escape<\"";
            switch (escape.kind) {
                case EscapeKind::EscapeToHeap: ss << "heap"; break;
                case EscapeKind::EscapeToReturn: ss << "return"; break;
                case EscapeKind::EscapeToGPU: ss << "gpu"; break;
                case EscapeKind::EscapeToDMA: ss << "dma"; break;
                case EscapeKind::EscapeToChannel: ss << "channel"; break;
                default: ss << "other"; break;
            }
            ss << "\">";
        }
        
        return ss.str();
    }
};

} // namespace cpp2_transpiler

using namespace cpp2_transpiler;

//===----------------------------------------------------------------------===//
// Unit Tests
//===----------------------------------------------------------------------===//

// Test 1: SemanticInfo default construction
void test_semantic_info_default() {
    std::cout << "Test 1: SemanticInfo default construction\n";
    
    SemanticInfo info;
    
    assert(info.borrow.kind == OwnershipKind::Owned);
    assert(info.escape.kind == EscapeKind::NoEscape);
    assert(!info.memory_region.has_value());
    assert(!info.active_transfer.has_value());
    assert(!info.channel_transfer.has_value());
    assert(!info.kernel_context.has_value());
    assert(info.contracts.empty());
    
    std::cout << "  ✓ Default SemanticInfo has correct initial values\n";
}

// Test 2: is_safe() - owned value is safe
void test_is_safe_owned() {
    std::cout << "\nTest 2: is_safe() - owned value\n";
    
    SemanticInfo info;
    info.borrow.kind = OwnershipKind::Owned;
    
    assert(info.is_safe() == true);
    
    std::cout << "  ✓ Owned value is safe\n";
}

// Test 3: is_safe() - moved value is NOT safe
void test_is_safe_moved() {
    std::cout << "\nTest 3: is_safe() - moved value\n";
    
    SemanticInfo info;
    info.borrow.kind = OwnershipKind::Moved;
    
    assert(info.is_safe() == false);
    
    std::cout << "  ✓ Moved value is NOT safe (use-after-move)\n";
}

// Test 4: is_safe() - mutable borrow with existing borrows
void test_is_safe_aliasing() {
    std::cout << "\nTest 4: is_safe() - aliasing detection\n";
    
    SemanticInfo info;
    info.borrow.kind = OwnershipKind::MutBorrowed;
    info.borrow.active_borrows.push_back(reinterpret_cast<void*>(0x1234));  // Fake borrow
    
    assert(info.is_safe() == false);
    
    std::cout << "  ✓ Mutable borrow with existing borrows is NOT safe\n";
}

// Test 5: can_optimize_away() - NoEscape
void test_can_optimize_noescape() {
    std::cout << "\nTest 5: can_optimize_away() - NoEscape\n";
    
    SemanticInfo info;
    info.escape.kind = EscapeKind::NoEscape;
    
    assert(info.can_optimize_away() == true);
    
    std::cout << "  ✓ NoEscape value can be optimized away\n";
}

// Test 6: can_optimize_away() - escapes to GPU
void test_can_optimize_gpu_escape() {
    std::cout << "\nTest 6: can_optimize_away() - GPU escape\n";
    
    SemanticInfo info;
    info.escape.kind = EscapeKind::EscapeToGPU;
    
    assert(info.can_optimize_away() == false);
    
    std::cout << "  ✓ GPU-escaping value cannot be optimized away\n";
}

// Test 7: explain_semantics() - basic output
void test_explain_semantics_basic() {
    std::cout << "\nTest 7: explain_semantics() - basic output\n";
    
    SemanticInfo info;
    info.borrow.kind = OwnershipKind::Owned;
    info.escape.kind = EscapeKind::NoEscape;
    
    std::string explanation = info.explain_semantics();
    
    assert(explanation.find("Ownership: owned") != std::string::npos);
    assert(explanation.find("Escape: local") != std::string::npos);
    assert(explanation.find("Safe: yes") != std::string::npos);
    
    std::cout << "  ✓ explain_semantics() generates readable output\n";
}

// Test 8: explain_semantics() - with memory region
void test_explain_semantics_memory() {
    std::cout << "\nTest 8: explain_semantics() - with memory region\n";
    
    SemanticInfo info;
    info.memory_region = MemoryRegion{"gpu_global", 1024, true};
    
    std::string explanation = info.explain_semantics();
    
    assert(explanation.find("Memory: gpu_global") != std::string::npos);
    assert(explanation.find("(device)") != std::string::npos);
    
    std::cout << "  ✓ Memory region included in explanation\n";
}

// Test 9: to_mlir_attributes() - owned
void test_mlir_attributes_owned() {
    std::cout << "\nTest 9: to_mlir_attributes() - owned\n";
    
    SemanticInfo info;
    info.borrow.kind = OwnershipKind::Owned;
    
    std::string attrs = info.to_mlir_attributes();
    
    assert(attrs.find("#cpp2.owned") != std::string::npos);
    
    std::cout << "  ✓ Generates #cpp2.owned attribute\n";
}

// Test 10: to_mlir_attributes() - borrowed
void test_mlir_attributes_borrowed() {
    std::cout << "\nTest 10: to_mlir_attributes() - borrowed\n";
    
    SemanticInfo info;
    info.borrow.kind = OwnershipKind::Borrowed;
    
    std::string attrs = info.to_mlir_attributes();
    
    assert(attrs.find("#cpp2.borrowed") != std::string::npos);
    
    std::cout << "  ✓ Generates #cpp2.borrowed attribute\n";
}

// Test 11: to_mlir_attributes() - with escape
void test_mlir_attributes_escape() {
    std::cout << "\nTest 11: to_mlir_attributes() - with escape\n";
    
    SemanticInfo info;
    info.borrow.kind = OwnershipKind::Owned;
    info.escape.kind = EscapeKind::EscapeToGPU;
    
    std::string attrs = info.to_mlir_attributes();
    
    assert(attrs.find("#cpp2.owned") != std::string::npos);
    assert(attrs.find("#cpp2.escape<\"gpu\">") != std::string::npos);
    
    std::cout << "  ✓ Generates #cpp2.escape attribute\n";
}

// Test 12: SafetyContract construction
void test_safety_contract() {
    std::cout << "\nTest 12: SafetyContract construction\n";
    
    SafetyContract pre(SafetyContract::Kind::Precondition, "x > 0", "x must be positive");
    SafetyContract post(SafetyContract::Kind::Postcondition, "result >= 0");
    
    assert(pre.kind == SafetyContract::Kind::Precondition);
    assert(pre.condition == "x > 0");
    assert(pre.message == "x must be positive");
    assert(post.kind == SafetyContract::Kind::Postcondition);
    
    std::cout << "  ✓ SafetyContract stores condition and message\n";
}

// Test 13: SemanticInfo with contracts
void test_semantic_info_contracts() {
    std::cout << "\nTest 13: SemanticInfo with contracts\n";
    
    SemanticInfo info;
    info.contracts.emplace_back(SafetyContract::Kind::Precondition, "n >= 0");
    info.contracts.emplace_back(SafetyContract::Kind::Postcondition, "result > 0");
    
    assert(info.contracts.size() == 2);
    
    std::string explanation = info.explain_semantics();
    assert(explanation.find("Contracts: 2") != std::string::npos);
    
    std::cout << "  ✓ Contracts tracked in SemanticInfo\n";
}

// Test 14: KernelLaunch context
void test_kernel_context() {
    std::cout << "\nTest 14: KernelLaunch context\n";
    
    SemanticInfo info;
    info.kernel_context = KernelLaunch{
        "compute_kernel",
        "256,256",
        "32",
        "streaming",
        true
    };
    
    assert(info.kernel_context->is_active == true);
    assert(info.kernel_context->kernel_name == "compute_kernel");
    
    std::string explanation = info.explain_semantics();
    assert(explanation.find("Kernel: compute_kernel") != std::string::npos);
    
    std::cout << "  ✓ Kernel context tracked correctly\n";
}

// Test 15: Channel transfer in SemanticInfo
void test_channel_transfer_semantic() {
    std::cout << "\nTest 15: Channel transfer in SemanticInfo\n";
    
    SemanticInfo info;
    info.channel_transfer = ChannelTransfer{};
    info.channel_transfer->channel_name = "data_chan";
    info.channel_transfer->ownership_transfer = OwnershipKind::Moved;
    
    std::string explanation = info.explain_semantics();
    assert(explanation.find("Channel: data_chan") != std::string::npos);
    
    std::cout << "  ✓ Channel transfer in semantic info\n";
}

// Test 16: Async DMA safety check
void test_async_dma_safety() {
    std::cout << "\nTest 16: Async DMA safety check\n";
    
    SemanticInfo info;
    info.active_transfer = MemoryTransfer{};
    info.active_transfer->is_async = true;
    info.escape.kind = EscapeKind::EscapeToGPU;  // Escapes during async
    
    assert(info.is_safe() == false);
    
    // But if NoEscape, it's safe
    info.escape.kind = EscapeKind::NoEscape;
    assert(info.is_safe() == true);
    
    std::cout << "  ✓ Async DMA safety correctly evaluated\n";
}

// Test 17: Lifetime tracking
void test_lifetime_tracking() {
    std::cout << "\nTest 17: Lifetime tracking\n";
    
    LifetimeRegion outer;
    outer.name = "function_scope";
    
    LifetimeRegion inner;
    inner.name = "loop_scope";
    
    outer.nested_regions.push_back(&inner);
    
    SemanticInfo info;
    info.lifetime = outer;
    info.must_outlive.push_back(&inner);
    
    assert(info.lifetime.outlives(inner) == true);
    assert(info.must_outlive.size() == 1);
    
    std::cout << "  ✓ Lifetime relationships tracked\n";
}

// Test 18: Full semantic dump
void test_full_semantic_dump() {
    std::cout << "\nTest 18: Full semantic dump\n";
    
    SemanticInfo info;
    info.borrow.kind = OwnershipKind::MutBorrowed;
    info.escape.kind = EscapeKind::EscapeToChannel;
    info.memory_region = MemoryRegion{"host_memory", 4096, false};
    info.channel_transfer = ChannelTransfer{};
    info.channel_transfer->channel_name = "output_chan";
    info.contracts.emplace_back(SafetyContract::Kind::Invariant, "size > 0");
    
    std::string dump = info.explain_semantics();
    
    // Verify all sections present
    assert(dump.find("Ownership:") != std::string::npos);
    assert(dump.find("Escape:") != std::string::npos);
    assert(dump.find("Memory:") != std::string::npos);
    assert(dump.find("Channel:") != std::string::npos);
    assert(dump.find("Contracts:") != std::string::npos);
    assert(dump.find("Safe:") != std::string::npos);
    assert(dump.find("Optimizable:") != std::string::npos);
    
    std::cout << "  ✓ Full semantic dump contains all sections\n";
    std::cout << "\n  Sample dump:\n";
    std::cout << "  ----------------------------------------\n";
    // Print each line indented
    std::istringstream iss(dump);
    std::string line;
    while (std::getline(iss, line)) {
        std::cout << "  | " << line << "\n";
    }
    std::cout << "  ----------------------------------------\n";
}

//===----------------------------------------------------------------------===//
// Main
//===----------------------------------------------------------------------===//

int main() {
    std::cout << "=== Unified Semantic Info Tests ===\n";
    std::cout << "Phase 5: Semantic AST Enhancements Track\n\n";
    
    test_semantic_info_default();
    test_is_safe_owned();
    test_is_safe_moved();
    test_is_safe_aliasing();
    test_can_optimize_noescape();
    test_can_optimize_gpu_escape();
    test_explain_semantics_basic();
    test_explain_semantics_memory();
    test_mlir_attributes_owned();
    test_mlir_attributes_borrowed();
    test_mlir_attributes_escape();
    test_safety_contract();
    test_semantic_info_contracts();
    test_kernel_context();
    test_channel_transfer_semantic();
    test_async_dma_safety();
    test_lifetime_tracking();
    test_full_semantic_dump();
    
    std::cout << "\n========================================\n";
    std::cout << "✅ All 18 tests passed!\n";
    std::cout << "Unified semantic info implementation verified.\n";
    std::cout << "========================================\n";
    
    return 0;
}
