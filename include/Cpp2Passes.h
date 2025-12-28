#ifndef CPP2_PASSES_H
#define CPP2_PASSES_H

#include "mlir/Pass/Pass.h"

namespace mlir {
namespace cpp2 {

//===----------------------------------------------------------------------===//
// Dialect Conversion Passes
//===----------------------------------------------------------------------===//

/// Create a pass to convert FIR dialect to SON dialect
std::unique_ptr<Pass> createConvertFIRToSONPass();

/// Initialize FIR to SON pass registration
void initConvertFIRToSONPass();

/// Create SCCP pass for FIR dialect
/// Uses standalone SCCP library (LatticeValue, DataflowAnalysis, ConstantFolder)
std::unique_ptr<Pass> createFIRSCCPPass();

//===----------------------------------------------------------------------===//
// Sea of Nodes Optimization Passes (from Chapter 24)
//===----------------------------------------------------------------------===//

/// Create SCCP (Sparse Conditional Constant Propagation) pass
/// Implements the optimistic dataflow algorithm from Chapter 24:
/// - Types start at TOP and fall to BOTTOM
/// - Conditional analysis skips unreachable code
/// - Interprocedural extension for whole-program analysis
std::unique_ptr<Pass> createSCCPPass();

/// Create iterative peephole pass
/// Implements the worklist-based peephole optimization from IterPeeps.java:
/// - Random worklist pull order for coverage
/// - Iterates until fixed point (no more progress)
/// - Applies fold() and idealize() transformations
std::unique_ptr<Pass> createIterPeepsPass();

/// Create dead code elimination pass for SON
/// Removes operations with no users and unreachable control flow
std::unique_ptr<Pass> createSONDCEPass();

/// Create common subexpression elimination pass for SON
/// Uses hash-consing to identify and eliminate redundant operations
std::unique_ptr<Pass> createSONCSEPass();

/// Create loop optimization pass for SON
/// Handles loop invariant code motion, strength reduction, etc.
std::unique_ptr<Pass> createSONLoopOptPass();

//===----------------------------------------------------------------------===//
// Pass Registration
//===----------------------------------------------------------------------===//

/// Register all Cpp2 passes
inline void registerCpp2Passes() {
  initConvertFIRToSONPass();
  // SCCP and IterPeeps are registered via static initializers
}

} // namespace cpp2
} // namespace mlir

#endif // CPP2_PASSES_H
