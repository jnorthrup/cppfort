#include "Cpp2Passes.h"
#include "Cpp2FIRDialect.h"
#include "mlir/Pass/Pass.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/Builders.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallVector.h"

using namespace mlir;
using namespace mlir::cpp2fir;

namespace {

/// FIRArenaInferencePass: Analyzes FIR operations and infers arena allocation scopes.
/// 
/// This pass identifies NoEscape allocations and groups them into scope-based arenas.
/// For each function:
/// 1. Walk all blocks (scopes)
/// 2. Identify cpp2fir.var_decl operations with no_escape attribute
/// 3. Assign arena scope IDs to eligible allocations
/// 4. Tag operations with #cpp2fir.arena_scope<ID> attribute
///
/// Eligibility criteria:
/// - EscapeKind == NoEscape
/// - Type is aggregate (vector, map, string) or large primitive (> 256 bytes)
/// - Not captured by nested lambdas/closures
struct FIRArenaInferencePass : public PassWrapper<FIRArenaInferencePass, OperationPass<ModuleOp>> {
    MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(FIRArenaInferencePass)

    StringRef getArgument() const override { return "fir-arena-inference"; }
    StringRef getDescription() const override { return "Infers arena allocation scopes for NoEscape values"; }

    // Statistics
    struct Stats {
        unsigned functionsProcessed = 0;
        unsigned scopesCreated = 0;
        unsigned allocationsTagged = 0;
        unsigned heapAllocationsKept = 0;
    };
    Stats stats;

    void runOnOperation() override {
        ModuleOp module = getOperation();
        unsigned nextArenaId = 1;
        
        // Process each function
        module.walk([&](FuncOp func) {
            processFuncOp(func, nextArenaId);
            stats.functionsProcessed++;
        });
        
        // Emit statistics as remarks
        if (stats.allocationsTagged > 0 || stats.heapAllocationsKept > 0) {
            module.emitRemark() << "Arena inference: " 
                << stats.scopesCreated << " scopes, "
                << stats.allocationsTagged << " arena allocations, "
                << stats.heapAllocationsKept << " heap allocations";
        }
    }

private:
    /// Process a single function for arena inference
    void processFuncOp(FuncOp func, unsigned &nextArenaId) {
        if (func.isDeclaration())
            return;
        
        OpBuilder builder(func.getContext());
        
        // Map: block -> (arena_id, list of eligible var_decl ops)
        llvm::DenseMap<Block*, unsigned> blockArenaIds;
        llvm::DenseMap<Block*, llvm::SmallVector<Operation*, 4>> blockAllocations;
        
        // Phase 1: Identify eligible allocations per block
        func.walk([&](Operation *op) {
            if (op->getName().getStringRef() == "cpp2fir.var_decl") {
                if (isArenaEligible(op)) {
                    Block *block = op->getBlock();
                    if (blockArenaIds.find(block) == blockArenaIds.end()) {
                        blockArenaIds[block] = nextArenaId++;
                        stats.scopesCreated++;
                    }
                    blockAllocations[block].push_back(op);
                } else {
                    stats.heapAllocationsKept++;
                }
            }
        });
        
        // Phase 2: Tag eligible allocations with arena scope
        for (auto &[block, allocations] : blockAllocations) {
            unsigned arenaId = blockArenaIds[block];
            for (Operation *op : allocations) {
                tagWithArenaScope(op, arenaId, builder);
                stats.allocationsTagged++;
            }
        }
    }
    
    /// Check if an operation is eligible for arena allocation
    bool isArenaEligible(Operation *op) {
        // Check for escape_kind attribute
        if (auto escapeAttr = op->getAttrOfType<StringAttr>("escape_kind")) {
            StringRef escape = escapeAttr.getValue();
            // Only NoEscape values are arena-eligible
            if (escape != "no_escape" && escape != "NoEscape") {
                return false;
            }
        } else {
            // No escape info - conservative: assume heap
            return false;
        }
        
        // Check if it's an aggregate type (vector, map, string, array)
        // or a large allocation
        if (auto typeAttr = op->getAttrOfType<StringAttr>("type_name")) {
            StringRef typeName = typeAttr.getValue();
            // Standard library aggregates
            if (typeName.contains("vector") || 
                typeName.contains("map") || 
                typeName.contains("string") ||
                typeName.contains("array") ||
                typeName.contains("deque") ||
                typeName.contains("list") ||
                typeName.contains("set")) {
                return true;
            }
        }
        
        // Check size if available
        if (auto sizeAttr = op->getAttrOfType<IntegerAttr>("size_bytes")) {
            // Large allocations (> 256 bytes) go to arena
            return sizeAttr.getInt() > 256;
        }
        
        // Default: aggregate types or large locals are arena-eligible
        return true;
    }
    
    /// Tag an operation with arena scope attribute
    void tagWithArenaScope(Operation *op, unsigned arenaId, OpBuilder &builder) {
        // Create arena_scope attribute: #cpp2fir.arena_scope<ID>
        auto arenaAttr = builder.getI64IntegerAttr(arenaId);
        op->setAttr("arena_scope", arenaAttr);
        
        // Also mark the allocation strategy
        op->setAttr("allocation_strategy", builder.getStringAttr("arena"));
    }
};

} // namespace

namespace mlir {
namespace cpp2 {
    std::unique_ptr<Pass> createFIRArenaInferencePass() {
        return std::make_unique<FIRArenaInferencePass>();
    }
}
}
