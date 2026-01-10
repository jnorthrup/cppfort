// Cpp2SON Dialect JMM Constraint Verification
// Validates happens-before consistency, volatile ordering, final field timing, unsafe publication
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef CPP2SON_JMM_VERIFICATION_H
#define CPP2SON_JMM_VERIFICATION_H

#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"

namespace mlir {
namespace sond {

/// Verify JMM constraints in a module
/// Returns true if all constraints are satisfied, false otherwise
bool verifyJMMConstraints(ModuleOp module);

/// Create a pass that verifies JMM constraints
std::unique_ptr<Pass> createJMMVerificationPass();

} // namespace sond
} // namespace mlir

#endif // CPP2SON_JMM_VERIFICATION_H
