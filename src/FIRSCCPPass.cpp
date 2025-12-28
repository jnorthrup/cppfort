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
#include "mlir/Pass/Pass.h"

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

    // Phase 1: Initialize - all values start at Top
    // Mark entry blocks as reachable
    module.walk([&](FuncOp func) {
      if (func.getBody().empty()) return;

      // Mark entry block as reachable
      analysis.markBlockReachable(&func.getBody().front());

      // Add all operations in function to worklist
      for (Operation &op : func.getBody().front()) {
        analysis.getWorklist().enqueue(&op);
      }
    });

    // Phase 2: Process worklist to fixed point
    while (!analysis.getWorklist().empty()) {
      Operation* op = static_cast<Operation*>(analysis.getWorklist().dequeue());
      if (!op) continue;

      // Skip operations in unreachable blocks
      Block* block = op->getBlock();
      if (block && !analysis.isBlockReachable(block)) {
        continue;
      }

      // Compute lattice value for this operation
      LatticeValue result = computeLatticeValue(op, analysis);

      // Update results
      for (Value v : op->getResults()) {
        if (analysis.updateLatticeValue(v.getAsOpaquePointer(), result)) {
          // Value changed - add users to worklist
          for (Operation* user : v.getUsers()) {
            analysis.getWorklist().enqueue(user);
          }
        }
      }
    }

    // Phase 3: Rewrite IR with discovered constants
    IRRewriter rewriter(module.getContext());
    module.walk([&](Operation* op) {
      // Skip constants
      if (isa<Cpp2FIR_ConstantOp>(op)) return;

      // Skip operations in unreachable blocks
      Block* block = op->getBlock();
      if (block && !analysis.isBlockReachable(block)) {
        return;
      }

      rewriter.setInsertionPointAfter(op);

      // Replace constant results
      bool changed = false;
      for (Value result : op->getResults()) {
        LatticeValue val = analysis.getLatticeValue(result.getAsOpaquePointer());

        if (val.isConstant()) {
          if (auto intVal = val.getAsInteger()) {
            // Replace with constant
            auto constOp = rewriter.create<Cpp2FIR_ConstantOp>(
              op->getLoc(),
              result.getType(),
              rewriter.getIntegerAttr(result.getType(), intVal.value()));
            Result replacements[] = {constOp.getResult()};
            rewriter.replaceOp(op, replacements);
            changed = true;
            break; // Op replaced, stop processing results
          } else if (auto boolVal = val.getAsBoolean()) {
            auto constOp = rewriter.create<Cpp2FIR_ConstantOp>(
              op->getLoc(),
              result.getType(),
              rewriter.getIntegerAttr(result.getType(), boolVal.value() ? 1 : 0));
            Result replacements[] = {constOp.getResult()};
            rewriter.replaceOp(op, replacements);
            changed = true;
            break;
          }
        }
      }
    });
  }

  StringRef getArgument() const final { return "fir-sccp"; }
  StringRef getDescription() const final {
    return "Sparse Conditional Constant Propagation for FIR dialect";
  }

private:
  /// Compute lattice value for an operation using our ConstantFolder
  LatticeValue computeLatticeValue(Operation* op, DataflowAnalysis& analysis) {
    // Constants are known
    if (auto constOp = dyn_cast<Cpp2FIR_ConstantOp>(op)) {
      if (auto intAttr = constOp.getValue().dyn_cast<IntegerAttr>()) {
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

      if (isa<Cpp2FIR_AddOp>(op)) {
        return ConstantFolder::foldAdd(lhs, rhs);
      } else if (isa<Cpp2FIR_SubOp>(op)) {
        return ConstantFolder::foldSub(lhs, rhs);
      } else if (isa<Cpp2FIR_MulOp>(op)) {
        return ConstantFolder::foldMul(lhs, rhs);
      } else if (isa<Cpp2FIR_DivOp>(op)) {
        return ConstantFolder::foldDiv(lhs, rhs);
      } else if (isa<Cpp2FIR_AndOp>(op)) {
        return ConstantFolder::foldAnd(lhs, rhs);
      } else if (isa<Cpp2FIR_OrOp>(op)) {
        return ConstantFolder::foldOr(lhs, rhs);
      } else if (auto cmpOp = dyn_cast<Cpp2FIR_CmpOp>(op)) {
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
      if (isa<Cpp2FIR_NotOp>(op)) {
        return ConstantFolder::foldNot(operands[0]);
      }
    }

    // Phi operation
    if (auto phiOp = dyn_cast<Cpp2FIR_PhiOp>(op)) {
      return DataflowAnalysis::mergePhiInputs(operands);
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
