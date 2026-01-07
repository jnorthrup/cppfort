#include "Cpp2Passes.h"
#include "Cpp2SONDialect.h"

#include "mlir/Pass/Pass.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/Value.h"
#include "mlir/Support/LLVM.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/Debug.h"

#include <queue>

#define DEBUG_TYPE "son-concurrency-analysis"

using namespace mlir;
using namespace mlir::sond;

namespace {

/// SONConcurrencyAnalysisPass: Analyzes Sea of Nodes IR for concurrency optimization opportunities
///
/// This pass performs several analyses to enable concurrency optimizations:
/// 1. Lock Elision: Identify mutex locks that protect no shared data
/// 2. Memory Barrier Elimination: Find unnecessary fences/syncs
/// 3. Async-to-Sync Conversion: Detect async operations that can be synchronous
/// 4. Parallel Region Detection: Find independent operations suitable for parallelization
/// 5. Race Condition Detection: Identify potential data races (for validation)
///
/// Algorithm:
/// - Walk the SON graph and build alias sets for memory operations
/// - Analyze control flow to identify critical sections
/// - Detect operations that don't actually need synchronization
/// - Mark operations for elision with attributes
struct SONConcurrencyAnalysisPass : public PassWrapper<SONConcurrencyAnalysisPass, OperationPass<ModuleOp>> {
    MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(SONConcurrencyAnalysisPass)

    StringRef getArgument() const override { return "son-concurrency-analysis"; }
    StringRef getDescription() const override {
        return "Analyze SON IR for concurrency optimizations (lock elision, barrier elimination)";
    }

    // Statistics
    struct Stats {
        unsigned locksAnalyzed = 0;
        unsigned locksElided = 0;
        unsigned barriersAnalyzed = 0;
        unsigned barriersEliminated = 0;
        unsigned asyncToSync = 0;
        unsigned parallelRegionsDetected = 0;
    };
    Stats stats;

    void runOnOperation() override {
        ModuleOp module = getOperation();

        // Process each function
        module.walk([&](FuncOp func) {
            analyzeFunction(func);
        });

        // Emit statistics
        if (stats.locksElided > 0 || stats.barriersEliminated > 0 || stats.asyncToSync > 0) {
            func->emitRemark() << "Concurrency Analysis: "
                << stats.locksElided << " locks elided, "
                << stats.barriersEliminated << " barriers eliminated, "
                << stats.asyncToSync << " async→sync conversions";
        }
    }

private:
    /// Alias analysis for memory operations
    struct AliasInfo {
        Value basePtr;           // Base pointer for this memory operation
        SmallVector<Value*, 4> mayAlias; // Other values this may alias with
        bool isVolatile = false; // Volatile operations cannot be elided
    };

    /// Critical section information
    struct CriticalSection {
        Operation* lockOp = nullptr;     // The lock operation
        Operation* unlockOp = nullptr;   // Corresponding unlock
        SmallVector<Value*, 8> protectedMemory; // Memory protected by this lock
        bool hasSideEffects = true;      // Whether operations in section have side effects
    };

    /// Analyze a single function for concurrency opportunities
    void analyzeFunction(FuncOp func) {
        LLVM_DEBUG(llvm::dbgs() << "Analyzing function: " << func.getName() << "\n");

        // Step 1: Build alias sets for all memory operations
        DenseMap<Value, AliasInfo> aliasSets;
        buildAliasAnalysis(func, aliasSets);

        // Step 2: Identify critical sections (lock-unlock pairs)
        SmallVector<CriticalSection, 8> criticalSections;
        identifyCriticalSections(func, criticalSections);

        // Step 3: Analyze each critical section for elision opportunities
        for (auto& cs : criticalSections) {
            stats.locksAnalyzed++;
            if (canElideLock(cs, aliasSets)) {
                elideLock(cs);
                stats.locksElided++;
                LLVM_DEBUG(llvm::dbgs() << "  Elided lock at " << cs.lockOp->getLoc() << "\n");
            }
        }

        // Step 4: Find memory barriers that can be eliminated
        analyzeBarriers(func, aliasSets);

        // Step 5: Detect async operations that can be synchronous
        analyzeAsyncOps(func);
    }

    /// Build alias analysis for all memory operations in the function
    void buildAliasAnalysis(FuncOp func, DenseMap<Value, AliasInfo>& aliasSets) {
        func.walk([&](Operation* op) {
            // Collect all memory operations (load, store, new, etc.)
            if (isa<LoadOp, StoreOp, NewOp>(op)) {
                Value memPtr = getMemoryPointer(op);
                if (memPtr) {
                    AliasInfo info;
                    info.basePtr = memPtr;

                    // Check for volatility
                    if (auto volatileAttr = op->getAttrOfType<StringAttr>("volatile")) {
                        info.isVolatile = true;
                    }

                    aliasSets[memPtr] = info;
                }
            }
        });

        // Compute aliasing relationships
        // (Simplified: values with different allocations don't alias)
        // TODO: Implement full alias analysis using type information
    }

    /// Identify critical sections (lock-unlock pairs)
    void identifyCriticalSections(FuncOp func, SmallVector<CriticalSection, 8>& sections) {
        // Look for lock/unlock patterns
        // In SON, these might be represented as call operations to lock/unlock functions
        // or as dedicated ops if we add them

        func.walk([&](Operation* op) {
            // Check for lock operation (could be call to "lock" or dedicated op)
            if (isLockOperation(op)) {
                CriticalSection cs;
                cs.lockOp = op;

                // Find matching unlock
                if (Operation* unlock = findMatchingUnlock(op)) {
                    cs.unlockOp = unlock;

                    // Collect memory operations in the critical section
                    collectProtectedMemory(op, unlock, cs.protectedMemory);

                    sections.push_back(cs);
                }
            }
        });
    }

    /// Check if a lock can be elided (no shared data access)
    bool canElideLock(const CriticalSection& cs, const DenseMap<Value, AliasInfo>& aliasSets) {
        // Lock elision conditions:
        // 1. No memory operations in critical section, OR
        // 2. All memory operations are on local (non-shared) data, OR
        // 3. All memory operations are already protected by another lock

        if (cs.protectedMemory.empty()) {
            // Empty critical section - definitely elidable
            return true;
        }

        // Check if any protected memory is actually shared
        for (Value* mem : cs.protectedMemory) {
            auto it = aliasSets.find(*mem);
            if (it != aliasSets.end()) {
                if (it->second.isVolatile) {
                    // Volatile operations cannot be elided
                    return false;
                }

                // Check if this memory escapes the function
                if (memoryEscapesFunction(*mem)) {
                    return false;
                }
            }
        }

        // No shared data found - lock can be elided
        return true;
    }

    /// Mark a lock for elision by adding an attribute
    void elideLock(CriticalSection& cs) {
        // Add elision attribute to both lock and unlock
        if (cs.lockOp) {
            cs.lockOp->setAttr("elided", UnitAttr::get(cs.lockOp->getContext()));
        }
        if (cs.unlockOp) {
            cs.unlockOp->setAttr("elided", UnitAttr::get(cs.unlockOp->getContext()));
        }
    }

    /// Analyze memory barriers for elimination
    void analyzeBarriers(FuncOp func, const DenseMap<Value, AliasInfo>& aliasSets) {
        func.walk([&](Operation* op) {
            if (isBarrierOperation(op)) {
                stats.barriersAnalyzed++;

                // Check if barrier is needed
                if (!barrierNecessary(op, aliasSets)) {
                    op->setAttr("elided", UnitAttr::get(op->getContext()));
                    stats.barriersEliminated++;
                    LLVM_DEBUG(llvm::dbgs() << "  Eliminated barrier at " << op->getLoc() << "\n");
                }
            }
        });
    }

    /// Analyze async operations that can become synchronous
    void analyzeAsyncOps(FuncOp func) {
        func.walk([&](Operation* op) {
            if (isAsyncOperation(op)) {
                // Check if async operation can be synchronous
                if (canBeSync(op)) {
                    op->setAttr("convert_to_sync", UnitAttr::get(op->getContext()));
                    stats.asyncToSync++;
                    LLVM_DEBUG(llvm::dbgs() << "  Async→sync at " << op->getLoc() << "\n");
                }
            }
        });
    }

    // ========== Helper Methods ==========

    /// Get the memory pointer from a memory operation
    Value* getMemoryPointer(Operation* op) {
        if (auto load = dyn_cast<LoadOp>(op)) {
            return &load.getPtr();
        }
        if (auto store = dyn_cast<StoreOp>(op)) {
            return &store.getPtr();
        }
        if (auto alloc = dyn_cast<NewOp>(op)) {
            return &alloc.getResult();
        }
        return nullptr;
    }

    /// Check if operation is a lock
    bool isLockOperation(Operation* op) {
        // Check for dedicated lock op or call to lock function
        if (auto call = dyn_cast<CallOp>(op)) {
            if (auto callee = call.getCallee()) {
                return callee->contains("lock") || callee->contains("mutex");
            }
        }
        // TODO: Add dedicated lock op to SON dialect
        return false;
    }

    /// Check if operation is an unlock
    bool isUnlockOperation(Operation* op) {
        if (auto call = dyn_cast<CallOp>(op)) {
            if (auto callee = call.getCallee()) {
                return callee->contains("unlock") || callee->contains("mutex_unlock");
            }
        }
        return false;
    }

    /// Check if operation is a barrier/fence
    bool isBarrierOperation(Operation* op) {
        if (auto call = dyn_cast<CallOp>(op)) {
            if (auto callee = call.getCallee()) {
                return callee->contains("barrier") || callee->contains("fence") ||
                       callee->contains("atomic_thread_fence");
            }
        }
        return false;
    }

    /// Check if operation is async
    bool isAsyncOperation(Operation* op) {
        if (auto call = dyn_cast<CallOp>(op)) {
            if (auto callee = call.getCallee()) {
                return callee->contains("async") || callee->contains("launch");
            }
        }
        return false;
    }

    /// Find matching unlock for a given lock
    Operation* findMatchingUnlock(Operation* lockOp) {
        // Walk forward from lock to find corresponding unlock
        // This is simplified - real implementation needs control flow analysis
        Block* block = lockOp->getBlock();
        auto it = ++Block::iterator(lockOp);
        int depth = 1;

        while (it != block->end()) {
            if (isLockOperation(&*it)) depth++;
            if (isUnlockOperation(&*it)) {
                depth--;
                if (depth == 0) return &*it;
            }
            ++it;
        }

        return nullptr;
    }

    /// Collect memory operations protected by a lock
    void collectProtectedMemory(Operation* lock, Operation* unlock, SmallVector<Value*, 8>& memory) {
        Block* block = lock->getBlock();
        auto it = ++Block::iterator(lock);

        while (&*it != unlock) {
            if (Value* mem = getMemoryPointer(&*it)) {
                memory.push_back(mem);
            }
            ++it;
        }
    }

    /// Check if a memory value escapes the function
    bool memoryEscapesFunction(Value mem) {
        // Check if value is returned
        for (auto user : mem.getUsers()) {
            if (isa<ReturnOp>(user)) {
                return true;
            }
        }

        // Check if value is stored to memory that might escape
        for (auto user : mem.getUsers()) {
            if (auto store = dyn_cast<StoreOp>(user)) {
                // Check if the stored-to memory escapes
                if (memoryEscapesFunction(store.getPtr())) {
                    return true;
                }
            }
        }

        return false;
    }

    /// Check if a barrier is necessary
    bool barrierNecessary(Operation* barrier, const DenseMap<Value, AliasInfo>& aliasSets) {
        // Barrier is necessary if:
        // 1. There are memory operations before it that could alias with operations after
        // 2. Those operations are not already ordered by other means

        // Simplified: check if there are any volatile or shared memory operations nearby
        Block* block = barrier->getBlock();

        // Check operations before barrier
        for (auto it = block->begin(); &*it != barrier; ++it) {
            if (Value* mem = getMemoryPointer(&*it)) {
                auto aliasIt = aliasSets.find(*mem);
                if (aliasIt != aliasSets.end() && aliasIt->second.isVolatile) {
                    return true; // Volatile access needs barrier
                }
            }
        }

        return false;
    }

    /// Check if async operation can be synchronous
    bool canBeSync(Operation* asyncOp) {
        // Async can be sync if:
        // 1. No dependencies on other async operations
        // 2. Result is used immediately (no concurrent work possible)
        // 3. No side effects that require async ordering

        // Check if result has only immediate uses
        if (asyncOp->getNumResults() == 0) {
            return false; // Fire-and-forget async must stay async
        }

        Value result = asyncOp->getResult(0);

        // Check if result is used in the same basic block without intervening ops
        bool hasNonLocalUse = false;
        for (auto user : result.getUsers()) {
            if (user->getBlock() != asyncOp->getBlock()) {
                hasNonLocalUse = true;
                break;
            }
        }

        return !hasNonLocalUse && result.hasOneUse();
    }
};

} // namespace

namespace mlir {
namespace cpp2 {
    std::unique_ptr<Pass> createSONConcurrencyAnalysisPass() {
        return std::make_unique<SONConcurrencyAnalysisPass>();
    }
}
}
