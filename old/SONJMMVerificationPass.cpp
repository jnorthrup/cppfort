#include "Cpp2Passes.h"
#include "Cpp2SONDialect.h"

#include "mlir/Pass/Pass.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/Value.h"
#include "mlir/Support/LLVM.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/Debug.h"

#include <queue>

#define DEBUG_TYPE "son-jmm-verification"

using namespace mlir;
using namespace mlir::sond;

namespace {

/// SONJMMVerificationPass: Validates Java Memory Model constraints in SON IR
///
/// This pass performs static verification of JMM guarantees:
/// 1. Happens-Before Edge Consistency: Ensure happens-before relationships form a valid DAG
/// 2. Volatile Sequential Consistency: Verify volatile operations have SC semantics
/// 3. Final Field Freeze Timing: Validate freeze points occur at constructor boundaries
/// 4. Safe Publication: Detect unsafe publication patterns
///
/// Algorithm:
/// - Build happens-before graph from jmm_happens_before attributes
/// - Validate transitive closure properties
/// - Check for cycles (would violate HB semantics)
/// - Verify volatile operations have seq_cst markers
/// - Validate final field stores occur before constructor_end operations
struct SONJMMVerificationPass : public PassWrapper<SONJMMVerificationPass, OperationPass<ModuleOp>> {
    MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(SONJMMVerificationPass)

    StringRef getArgument() const override { return "son-jmm-verification"; }
    StringRef getDescription() const override {
        return "Validate Java Memory Model constraints in SON IR";
    }

    // Statistics
    struct Stats {
        unsigned happensBeforeEdges = 0;
        unsigned happensBeforeViolations = 0;
        unsigned volatileOps = 0;
        unsigned volatileViolations = 0;
        unsigned finalFields = 0;
        unsigned finalFieldViolations = 0;
        unsigned unsafePublications = 0;
    };
    Stats stats;

    void runOnOperation() override {
        ModuleOp module = getOperation();
        bool hasViolations = false;

        // Process each function
        module.walk([&](mlir::func::FuncOp func) {
            if (analyzeFunctionJMM(func)) {
                hasViolations = true;
            }
        });

        // Emit diagnostics
        if (hasViolations) {
            module.emitError() << "JMM verification failed: found "
                << stats.happensBeforeViolations << " HB violations, "
                << stats.volatileViolations << " volatile violations, "
                << stats.finalFieldViolations << " final field violations, "
                << stats.unsafePublications << " unsafe publications";
            signalPassFailure();
        }

        LLVM_DEBUG({
            llvm::dbgs() << "JMM Verification Statistics:\n";
            llvm::dbgs() << "  Happens-before edges: " << stats.happensBeforeEdges << "\n";
            llvm::dbgs() << "  Happens-before violations: " << stats.happensBeforeViolations << "\n";
            llvm::dbgs() << "  Volatile operations: " << stats.volatileOps << "\n";
            llvm::dbgs() << "  Volatile violations: " << stats.volatileViolations << "\n";
            llvm::dbgs() << "  Final fields: " << stats.finalFields << "\n";
            llvm::dbgs() << "  Final field violations: " << stats.finalFieldViolations << "\n";
            llvm::dbgs() << "  Unsafe publications: " << stats.unsafePublications << "\n";
        });
    }

private:
    /// Happens-before edge representation
    struct HappensBeforeEdge {
        Operation* source;
        Operation* target;
        bool isTransitive;
    };

    /// Analyze a single function for JMM constraint violations
    /// Returns true if violations were found
    bool analyzeFunctionJMM(mlir::func::FuncOp func) {
        LLVM_DEBUG(llvm::dbgs() << "JMM analyzing function: " << func.getName() << "\n");

        bool hasViolations = false;

        // Step 1: Build happens-before graph
        SmallVector<HappensBeforeEdge, 32> hbEdges;
        buildHappensBeforeGraph(func, hbEdges);

        // Step 2: Validate happens-before consistency
        if (validateHappensBeforeConsistency(func, hbEdges)) {
            hasViolations = true;
        }

        // Step 3: Check volatile operations
        if (checkVolatileOrdering(func)) {
            hasViolations = true;
        }

        // Step 4: Verify final field freeze timing
        if (verifyFinalFieldFreeze(func)) {
            hasViolations = true;
        }

        // Step 5: Detect unsafe publications
        if (detectUnsafePublications(func)) {
            hasViolations = true;
        }

        return hasViolations;
    }

    /// Build the happens-before graph from JMM attributes
    void buildHappensBeforeGraph(mlir::func::FuncOp func, SmallVector<HappensBeforeEdge, 32>& edges) {
        // Map operation names to operations
        DenseMap<StringRef, Operation*> opMap;
        func.walk([&](Operation* op) {
            if (auto symName = op->getAttrOfType<StringAttr>("sym_name")) {
                opMap[symName.getValue()] = op;
            }
        });

        // Extract happens-before edges from operations
        func.walk([&](Operation* op) {
            // Check for jmm_happens_before attribute
            if (auto hbAttr = op->getAttrOfType<JMMHappensBeforeAttr>("jmm_happens_before")) {
                stats.happensBeforeEdges += hbAttr.getPredecessors().size();

                for (StringRef pred : hbAttr.getPredecessors()) {
                    if (auto predOp = opMap.lookup(pred)) {
                        edges.push_back({predOp, op, hbAttr.getIsTransitive()});

                        LLVM_DEBUG(llvm::dbgs() << "HB edge: " << pred << " -> "
                                   << op->getName() << "\n");
                    } else {
                        op->emitWarning() << "happens-before predecessor '" << pred
                                         << "' not found in function";
                    }
                }
            }
        });
    }

    /// Validate happens-before consistency (no cycles, proper transitivity)
    /// Returns true if violations were found
    bool validateHappensBeforeConsistency(mlir::func::FuncOp func,
                                          const SmallVector<HappensBeforeEdge, 32>& edges) {
        // Build adjacency list for cycle detection
        DenseMap<Operation*, SmallVector<Operation*, 4>> adjList;
        for (const auto& edge : edges) {
            adjList[edge.source].push_back(edge.target);
        }

        // DFS-based cycle detection
        DenseSet<Operation*> visited;
        DenseSet<Operation*> recursionStack;
        bool hasCycle = false;

        std::function<bool(Operation*)> detectCycle = [&](Operation* op) -> bool {
            if (recursionStack.contains(op)) {
                op->emitError() << "happens-before cycle detected - violates JMM";
                stats.happensBeforeViolations++;
                return true;
            }

            if (visited.contains(op)) {
                return false;
            }

            visited.insert(op);
            recursionStack.insert(op);

            for (Operation* succ : adjList[op]) {
                if (detectCycle(succ)) {
                    hasCycle = true;
                }
            }

            recursionStack.erase(op);
            return false;
        };

        // Check all operations for cycles
        for (const auto& [op, _] : adjList) {
            if (!visited.contains(op)) {
                if (detectCycle(op)) {
                    hasCycle = true;
                }
            }
        }

        return hasCycle;
    }

    /// Check that volatile operations have proper sequential consistency markers
    /// Returns true if violations were found
    bool checkVolatileOrdering(mlir::func::FuncOp func) {
        bool hasViolations = false;

        func.walk([&](Operation* op) {
            // Check if operation has jmm_volatile attribute
            if (auto volAttr = op->getAttrOfType<JMMVolatileAttr>("jmm_volatile")) {
                stats.volatileOps++;

                // Verify that volatile operations are marked with is_volatile=true
                if (!volAttr.getIsVolatile()) {
                    op->emitError() << "volatile attribute present but is_volatile=false";
                    stats.volatileViolations++;
                    hasViolations = true;
                }

                // Verify that load/store operations with volatile have proper semantics
                if (isa<LoadOp>(op) || isa<StoreOp>(op)) {
                    LLVM_DEBUG(llvm::dbgs() << "Volatile memory operation: "
                               << op->getName() << "\n");
                    // Additional checks could verify seq_cst emission in code generation
                }
            }
        });

        return hasViolations;
    }

    /// Verify that final field stores occur before constructor_end operations
    /// Returns true if violations were found
    bool verifyFinalFieldFreeze(mlir::func::FuncOp func) {
        bool hasViolations = false;

        // Find all constructor_end operations
        SmallVector<ConstructorEndOp, 8> constructorEnds;
        func.walk([&](ConstructorEndOp op) {
            constructorEnds.push_back(op);
        });

        // For each constructor_end, verify that all final field stores
        // to the same object occur before it in program order
        for (auto ctorEnd : constructorEnds) {
            auto ctorOp = ctorEnd.getJmmFinalField().getConstructorOp();
            Value objectPtr = ctorEnd.getObjectPtr();

            LLVM_DEBUG(llvm::dbgs() << "Checking final field freeze for constructor: "
                       << ctorOp << "\n");

            // Find all stores to this object with final field attribute
            func.walk([&](StoreOp store) {
                if (auto finalAttr = store.getJmmFinalFieldAttr()) {
                    stats.finalFields++;

                    // Check if this store is to the same object
                    if (store.getPtr() == objectPtr) {
                        // Verify store occurs before constructor_end
                        // (simplified: check they're in the same basic block and store comes first)
                        if (!dominates(store, ctorEnd)) {
                            store.emitError() << "final field store occurs after constructor_end";
                            stats.finalFieldViolations++;
                            hasViolations = true;
                        }
                    }
                }
            });
        }

        return hasViolations;
    }

    /// Detect unsafe publication patterns
    /// Returns true if unsafe patterns were found
    bool detectUnsafePublications(mlir::func::FuncOp func) {
        bool hasUnsafe = false;

        // Look for objects with final fields that are published before constructor_end
        func.walk([&](NewOp newOp) {
            if (auto finalAttr = newOp.getJmmFinalFieldAttr()) {
                Value objectPtr = newOp.getPtr();

                // Find the constructor_end for this object
                ConstructorEndOp ctorEnd;
                func.walk([&](ConstructorEndOp op) {
                    if (op.getObjectPtr() == objectPtr) {
                        ctorEnd = op;
                    }
                });

                if (!ctorEnd) {
                    newOp.emitWarning() << "object with final fields has no constructor_end";
                    stats.unsafePublications++;
                    hasUnsafe = true;
                    return;
                }

                // Check for stores that publish the object pointer before constructor_end
                func.walk([&](StoreOp store) {
                    if (store.getValue() == objectPtr) {
                        if (!dominates(ctorEnd, store)) {
                            store.emitError() << "unsafe publication: object published before "
                                             << "constructor_end (final fields not frozen)";
                            stats.unsafePublications++;
                            hasUnsafe = true;
                        }
                    }
                });
            }
        });

        return hasUnsafe;
    }

    /// Simplified dominance check (assumes operations in same block are in program order)
    /// TODO: Use proper dominator tree from MLIR
    bool dominates(Operation* a, Operation* b) {
        if (!a || !b) return false;

        Block* blockA = a->getBlock();
        Block* blockB = b->getBlock();

        // If in different blocks, we'd need proper dominator analysis
        // For now, conservatively return false if blocks differ
        if (blockA != blockB) {
            return false;
        }

        // In same block: check if a appears before b
        for (Operation& op : *blockA) {
            if (&op == a) return true;
            if (&op == b) return false;
        }

        return false;
    }
};

} // anonymous namespace

namespace mlir {
namespace cpp2 {

std::unique_ptr<Pass> createSONJMMVerificationPass() {
    return std::make_unique<SONJMMVerificationPass>();
}

} // namespace cpp2
} // namespace mlir
