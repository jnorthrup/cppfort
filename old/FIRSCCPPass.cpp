//===- FIRSCCPPass.cpp - FIR Dialect SCCP Pass ---------------------------===//
///
/// Sparse Conditional Constant Propagation pass for FIR dialect.
/// Uses the standalone SCCP library (LatticeValue, ConstantFolder, DataflowAnalysis).
///
//===----------------------------------------------------------------------===//

#include "Cpp2Passes.h"
#include "Cpp2FIRDialect.h"

// Include our standalone SCCP library
#include "LatticeValue.h"
#include "ConstantFolder.h"
#include "DataflowAnalysis.h"

#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "llvm/Support/Debug.h"

#define DEBUG_TYPE "fir-sccp"

using namespace mlir;
using namespace mlir::cpp2fir;
using namespace cppfort::sccp;

namespace {

/// SCCP Pass for FIR Dialect
/// Uses our standalone SCCP library for analysis and constant propagation
struct FIRSCCPPass : public PassWrapper<FIRSCCPPass, OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(FIRSCCPPass)

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<Cpp2FIRDialect>();
  }

  void runOnOperation() override {
    ModuleOp module = getOperation();
    DataflowAnalysis analysis;

    LLVM_DEBUG(llvm::dbgs() << "=== FIR SCCP Pass Start ===\n");

    // Phase 1: Initialize - all values start at Top
    // Mark entry blocks as reachable
    module.walk([&](FuncOp func) {
      if (func.getBody().empty()) return;

      LLVM_DEBUG(llvm::dbgs() << "Initializing function: " << func.getSymName() << "\n");

      // Mark entry block as reachable
      analysis.markBlockReachable(&func.getBody().front());

      // Add all operations in function to worklist
      for (Operation &op : func.getBody().front()) {
        analysis.getWorklist().enqueue(&op);
        LLVM_DEBUG(llvm::dbgs() << "  Enqueued op: " << op.getName() << "\n");
      }
    });

    // Phase 2: Process worklist to fixed point
    LLVM_DEBUG(llvm::dbgs() << "\n=== Phase 2: Dataflow Analysis ===\n");
    [[maybe_unused]] int iterations = 0;
    while (!analysis.getWorklist().empty()) {
      Operation* op = static_cast<Operation*>(analysis.getWorklist().dequeue());
      if (!op) continue;

      LLVM_DEBUG(llvm::dbgs() << "Processing op [" << iterations++ << "]: " << op->getName() << "\n");

      // Skip operations in unreachable blocks
      Block* block = op->getBlock();
      if (block && !analysis.isBlockReachable(block)) {
        LLVM_DEBUG(llvm::dbgs() << "  Skipping unreachable op\n");
        continue;
      }

      // Compute lattice value for this operation
      LatticeValue result = computeLatticeValue(op, analysis);
      LLVM_DEBUG(llvm::dbgs() << "  Computed lattice: " << result.toString() << "\n");

      // Update results
      for (Value v : op->getResults()) {
        if (analysis.updateLatticeValue(v.getAsOpaquePointer(), result)) {
          LLVM_DEBUG(llvm::dbgs() << "  Lattice changed - adding users to worklist\n");
          // Value changed - add users to worklist
          for (Operation* user : v.getUsers()) {
            analysis.getWorklist().enqueue(user);
          }
        }
      }
    }
    LLVM_DEBUG(llvm::dbgs() << "Dataflow converged after " << iterations << " iterations\n");

    // Phase 3: Rewrite IR with discovered constants
    LLVM_DEBUG(llvm::dbgs() << "\n=== Phase 3: IR Rewriting ===\n");
    PatternRewriter rewriter(module.getContext());
    int replacements = 0;
    module.walk([&](Operation* op) {
      // Skip constants
      if (isa<ConstantOp>(op)) return;

      // Skip operations in unreachable blocks
      Block* block = op->getBlock();
      if (block && !analysis.isBlockReachable(block)) {
        return;
      }

      rewriter.setInsertionPointAfter(op);

      // Replace constant results
      for (Value result : op->getResults()) {
        LatticeValue val = analysis.getLatticeValue(result.getAsOpaquePointer());

        if (val.isConstant()) {
          if (auto intVal = val.getAsInteger()) {
            LLVM_DEBUG(llvm::dbgs() << "Replacing " << op->getName()
                       << " with constant: " << intVal.value() << "\n");
            // Replace with constant
            auto constOp = rewriter.create<ConstantOp>(
              op->getLoc(),
              result.getType(),
              rewriter.getIntegerAttr(result.getType(), intVal.value()));
            rewriter.replaceOp(op, {constOp.getResult()});
            replacements++;
            break; // Op replaced, stop processing results
          } else if (auto boolVal = val.getAsBoolean()) {
            LLVM_DEBUG(llvm::dbgs() << "Replacing " << op->getName()
                       << " with constant: " << (boolVal.value() ? "true" : "false") << "\n");
            auto constOp = rewriter.create<ConstantOp>(
              op->getLoc(),
              result.getType(),
              rewriter.getIntegerAttr(result.getType(), boolVal.value() ? 1 : 0));
            rewriter.replaceOp(op, {constOp.getResult()});
            replacements++;
            break;
          }
        }
      }
    });
    LLVM_DEBUG(llvm::dbgs() << "Replaced " << replacements << " operations with constants\n");
    LLVM_DEBUG(llvm::dbgs() << "=== FIR SCCP Pass End ===\n\n");
  }

  StringRef getArgument() const final { return "fir-sccp"; }
  StringRef getDescription() const final {
    return "Sparse Conditional Constant Propagation for FIR dialect";
  }

private:
  /// Compute lattice value for an operation using our ConstantFolder
  LatticeValue computeLatticeValue(Operation* op, DataflowAnalysis& analysis) {
    // Constants are known
    if (auto constOp = dyn_cast<ConstantOp>(op)) {
      if (auto intAttr = llvm::dyn_cast<mlir::IntegerAttr>(constOp.getValue())) {
        return LatticeValue::getConstant(intAttr.getInt());
      }
      // Could handle other attribute types here
    }

    // Get operand lattice values
    SmallVector<LatticeValue, 4> operands;
    for (Value v : op->getOperands()) {
      operands.push_back(analysis.getLatticeValue(v.getAsOpaquePointer()));
    }

    // Binary operations
    if (operands.size() == 2) {
      LatticeValue lhs = operands[0];
      LatticeValue rhs = operands[1];

      if (isa<AddOp>(op)) {
        return ConstantFolder::foldAdd(lhs, rhs);
      } else if (isa<SubOp>(op)) {
        return ConstantFolder::foldSub(lhs, rhs);
      } else if (isa<MulOp>(op)) {
        return ConstantFolder::foldMul(lhs, rhs);
      } else if (isa<DivOp>(op)) {
        return ConstantFolder::foldDiv(lhs, rhs);
      } else if (isa<AndOp>(op)) {
        return ConstantFolder::foldAnd(lhs, rhs);
      } else if (isa<OrOp>(op)) {
        return ConstantFolder::foldOr(lhs, rhs);
      } else if (auto cmpOp = dyn_cast<CmpOp>(op)) {
        StringRef pred = cmpOp.getPredicate();
        LatticeValue::CmpPredicate predicate;

        if (pred == "lt" || pred == "<") predicate = LatticeValue::CmpPredicate::LT;
        else if (pred == "le" || pred == "<=") predicate = LatticeValue::CmpPredicate::LE;
        else if (pred == "gt" || pred == ">") predicate = LatticeValue::CmpPredicate::GT;
        else if (pred == "ge" || pred == ">=") predicate = LatticeValue::CmpPredicate::GE;
        else if (pred == "eq" || pred == "==") predicate = LatticeValue::CmpPredicate::EQ;
        else if (pred == "ne" || pred == "!=") predicate = LatticeValue::CmpPredicate::NE;
        else return LatticeValue::getTop();

        return ConstantFolder::foldCmp(predicate, lhs, rhs);
      }
    }

    // Unary operations
    if (operands.size() == 1) {
      if (isa<NotOp>(op)) {
        return ConstantFolder::foldNot(operands[0]);
      }
    }

    // Phi operation
    if (auto phiOp = dyn_cast<PhiOp>(op)) {
      std::vector<LatticeValue> phiInputs(operands.begin(), operands.end());
      return DataflowAnalysis::mergePhiInputs(phiInputs);
    }

    // Default: Top (unknown)
    return LatticeValue::getTop();
  }
};

} // namespace

//===----------------------------------------------------------------------===//
// Pass Creation and Registration
//===----------------------------------------------------------------------===//

namespace mlir {
namespace cpp2 {

std::unique_ptr<Pass> createFIRSCCPPass() {
  return std::make_unique<FIRSCCPPass>();
}

} // namespace cpp2
} // namespace mlir

// Static registration
static mlir::PassRegistration<FIRSCCPPass> firSccpPass;
