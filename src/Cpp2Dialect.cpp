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

  if (auto lhsInt = dyn_cast<IntegerAttr>(lhs)) {
    if (auto rhsInt = dyn_cast<IntegerAttr>(rhs)) {
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

  if (auto lhsInt = dyn_cast<IntegerAttr>(lhs)) {
    if (auto rhsInt = dyn_cast<IntegerAttr>(rhs)) {
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

  if (auto lhsInt = dyn_cast<IntegerAttr>(lhs)) {
    if (auto rhsInt = dyn_cast<IntegerAttr>(rhs)) {
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

  if (auto lhsInt = dyn_cast<IntegerAttr>(lhs)) {
    if (auto rhsInt = dyn_cast<IntegerAttr>(rhs)) {
      auto rhsVal = rhsInt.getInt();
      if (rhsVal == 0) return {};
      auto lhsVal = lhsInt.getInt();
      return IntegerAttr::get(lhsInt.getType(), lhsVal / rhsVal);
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

// Implementations for RegionBranchOpInterface

void IfOp::getSuccessorRegions(RegionBranchPoint point,
                               SmallVectorImpl<RegionSuccessor> &regions) {
  // If the predecessor is the IfOp itself, then we can branch into the then
  // or else region.
  if (point.isParent()) {
    regions.push_back(RegionSuccessor(&getThenRegion()));
    if (!getElseRegion().empty())
      regions.push_back(RegionSuccessor(&getElseRegion()));
    return;
  }

  // Otherwise, the predecessor is the then or else region, which branches
  // back to the parent op.
  regions.push_back(RegionSuccessor(getResults()));
}

void LoopOp::getSuccessorRegions(RegionBranchPoint point,
                                 SmallVectorImpl<RegionSuccessor> &regions) {
  // If the predecessor is the LoopOp itself, then we can branch into the body.
  if (point.isParent()) {
    regions.push_back(RegionSuccessor(&getBody(), getBody().getArguments()));
    return;
  }

  // If the predecessor is the body, we can branch back to the body or to the
  // parent op.
  regions.push_back(RegionSuccessor(&getBody(), getBody().getArguments()));
  regions.push_back(RegionSuccessor(getResults()));
}

// Implementations for Cpp2Dialect attribute parsing/printing
Attribute Cpp2Dialect::parseAttribute(DialectAsmParser &parser, Type type) const {
  StringRef attrTag;
  if (failed(parser.parseKeyword(&attrTag)))
    return Attribute();
  // We don't have any custom attributes yet.
  parser.emitError(parser.getNameLoc(), "unknown attribute");
  return Attribute();
}

void Cpp2Dialect::printAttribute(Attribute attr, DialectAsmPrinter &printer) const {
  printer << "unknown_attribute";
}
