#include "Cpp2Dialect.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/DialectImplementation.h"
#include "mlir/IR/OpImplementation.h"
#include "llvm/ADT/TypeSwitch.h"

using namespace mlir;
using namespace mlir::cpp2;

#include "Cpp2OpsDialect.cpp.inc"

#define GET_TYPEDEF_CLASSES
#include "Cpp2OpsTypes.cpp.inc"

#define GET_OP_CLASSES
#include "Cpp2Ops.cpp.inc"

void Cpp2Dialect::initialize() {
  addOperations<
#define GET_OP_LIST
#include "Cpp2Ops.cpp.inc"
  >();
  addTypes<
#define GET_TYPEDEF_LIST
#include "Cpp2OpsTypes.cpp.inc"
  >();
}

// Constant folder for AddOp
OpFoldResult AddOp::fold(FoldAdaptor adaptor) {
  auto lhs = adaptor.getLhs();
  auto rhs = adaptor.getRhs();

  if (!lhs || !rhs) return {};

  if (auto lhsInt = lhs.dyn_cast<IntegerAttr>()) {
    if (auto rhsInt = rhs.dyn_cast<IntegerAttr>()) {
      auto lhsVal = lhsInt.getInt();
      auto rhsVal = rhsInt.getInt();
      return IntegerAttr::get(lhsInt.getType(), lhsVal + rhsVal);
    }
  }
  return {};
}

// Constant folder for SubOp
OpFoldResult SubOp::fold(FoldAdaptor adaptor) {
  auto lhs = adaptor.getLhs();
  auto rhs = adaptor.getRhs();

  if (!lhs || !rhs) return {};

  if (auto lhsInt = lhs.dyn_cast<IntegerAttr>()) {
    if (auto rhsInt = rhs.dyn_cast<IntegerAttr>()) {
      auto lhsVal = lhsInt.getInt();
      auto rhsVal = rhsInt.getInt();
      return IntegerAttr::get(lhsInt.getType(), lhsVal - rhsVal);
    }
  }
  return {};
}

// Constant folder for MulOp
OpFoldResult MulOp::fold(FoldAdaptor adaptor) {
  auto lhs = adaptor.getLhs();
  auto rhs = adaptor.getRhs();

  if (!lhs || !rhs) return {};

  if (auto lhsInt = lhs.dyn_cast<IntegerAttr>()) {
    if (auto rhsInt = rhs.dyn_cast<IntegerAttr>()) {
      auto lhsVal = lhsInt.getInt();
      auto rhsVal = rhsInt.getInt();
      return IntegerAttr::get(lhsInt.getType(), lhsVal * rhsVal);
    }
  }
  return {};
}

// Constant folder for DivOp
OpFoldResult DivOp::fold(FoldAdaptor adaptor) {
  auto lhs = adaptor.getLhs();
  auto rhs = adaptor.getRhs();

  if (!lhs || !rhs) return {};

  if (auto lhsInt = lhs.dyn_cast<IntegerAttr>()) {
    if (auto rhsInt = rhs.dyn_cast<IntegerAttr>()) {
      auto rhsVal = rhsInt.getInt();
      if (rhsVal == 0) return {};
      auto lhsVal = lhsInt.getInt();
      return IntegerAttr::get(lhsInt.getType(), lhsVal.sdiv(rhsVal));
    }
  }
  return {};
}

// Constant folder for ConstantOp
OpFoldResult ConstantOp::fold(FoldAdaptor adaptor) {
  return getValue();
}

// Canonicalizer for AddOp
void AddOp::getCanonicalizationPatterns(RewritePatternSet &results, MLIRContext *context) {
  // Add(0, x) -> x
  // Add(x, 0) -> x
  // Add(C1, C2) -> C3 (handled by folder)
}

// Canonicalizer for PhiOp
void PhiOp::getCanonicalizationPatterns(RewritePatternSet &results, MLIRContext *context) {
  // Phi(x, x, ..., x) -> x (all same value)
  // Phi with single input -> value
}