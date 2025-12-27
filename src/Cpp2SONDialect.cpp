#include "Cpp2SONDialect.h"

#include "mlir/IR/Builders.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/DialectImplementation.h"
#include "mlir/IR/PatternMatch.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/TypeSwitch.h"
#include <cmath>

using namespace mlir;
using namespace mlir::sond;

#include "Cpp2SONOpsDialect.cpp.inc"

// Include generated type definitions
#define GET_TYPEDEF_CLASSES
#include "Cpp2SONOpsTypes.cpp.inc"

// Include generated attribute definitions  
#define GET_ATTRDEF_CLASSES
#include "Cpp2SONOpsAttrDefs.cpp.inc"

#define GET_OP_CLASSES
#include "Cpp2SONOps.cpp.inc"

void Cpp2SONDialect::initialize() {
  addOperations<
#define GET_OP_LIST
#include "Cpp2SONOps.cpp.inc"
  >();
  
  // Register types from the Sea of Nodes type system
  addTypes<
#define GET_TYPEDEF_LIST
#include "Cpp2SONOpsTypes.cpp.inc"
  >();
  
  // Register attributes
  addAttributes<
#define GET_ATTRDEF_LIST
#include "Cpp2SONOpsAttrDefs.cpp.inc"
  >();
}

//===----------------------------------------------------------------------===//
// Helper Functions for Folding (Chapter 24 Transfer Functions)
//===----------------------------------------------------------------------===//

// Helper for comparison folding - matches Java BoolNode predicates
static std::optional<bool> evalICmp(llvm::StringRef pred, int64_t lhs, int64_t rhs) {
  if (pred == "lt" || pred == "<") return lhs < rhs;
  if (pred == "le" || pred == "<=") return lhs <= rhs;
  if (pred == "gt" || pred == ">") return lhs > rhs;
  if (pred == "ge" || pred == ">=") return lhs >= rhs;
  if (pred == "eq" || pred == "==") return lhs == rhs;
  if (pred == "ne" || pred == "!=") return lhs != rhs;
  return std::nullopt;
}

// Helper for float comparison folding
static std::optional<bool> evalFCmp(llvm::StringRef pred, double lhs, double rhs) {
  if (pred == "lt" || pred == "<") return lhs < rhs;
  if (pred == "le" || pred == "<=") return lhs <= rhs;
  if (pred == "gt" || pred == ">") return lhs > rhs;
  if (pred == "ge" || pred == ">=") return lhs >= rhs;
  if (pred == "eq" || pred == "==") return lhs == rhs;
  if (pred == "ne" || pred == "!=") return lhs != rhs;
  return std::nullopt;
}

// Negate comparison predicate (from IfNode.java negate())
static llvm::StringRef negatePredicate(llvm::StringRef pred) {
  if (pred == "lt" || pred == "<") return ">=";
  if (pred == "le" || pred == "<=") return ">";
  if (pred == "gt" || pred == ">") return "<=";
  if (pred == "ge" || pred == ">=") return "<";
  if (pred == "eq" || pred == "==") return "!=";
  if (pred == "ne" || pred == "!=") return "==";
  return pred;
}

// Swap comparison operands (from IfNode.java swap())
static llvm::StringRef swapPredicate(llvm::StringRef pred) {
  if (pred == "lt" || pred == "<") return ">";
  if (pred == "le" || pred == "<=") return ">=";
  if (pred == "gt" || pred == ">") return "<";
  if (pred == "ge" || pred == ">=") return "<=";
  return pred;  // eq, ne are symmetric
}

//===----------------------------------------------------------------------===//
// Arithmetic Fold Implementations
// Corresponds to compute() methods in ArithNode.java subclasses
//===----------------------------------------------------------------------===//

::mlir::OpFoldResult AddOp::fold(AddOp::GenericAdaptor<llvm::ArrayRef<mlir::Attribute>> adaptor) {
  auto lhs = adaptor.getLhs();
  auto rhs = adaptor.getRhs();
  if (!lhs || !rhs) return {};

  // Integer constant folding
  if (auto lhsInt = dyn_cast<IntegerAttr>(lhs)) {
    if (auto rhsInt = dyn_cast<IntegerAttr>(rhs)) {
      return IntegerAttr::get(lhsInt.getType(), lhsInt.getInt() + rhsInt.getInt());
    }
  }
  
  // Identity: x + 0 = x
  if (auto rhsInt = dyn_cast<IntegerAttr>(rhs)) {
    if (rhsInt.getInt() == 0) return getLhs();
  }
  if (auto lhsInt = dyn_cast<IntegerAttr>(lhs)) {
    if (lhsInt.getInt() == 0) return getRhs();
  }
  
  return {};
}

::mlir::OpFoldResult SubOp::fold(SubOp::GenericAdaptor<llvm::ArrayRef<mlir::Attribute>> adaptor) {
  auto lhs = adaptor.getLhs();
  auto rhs = adaptor.getRhs();
  
  // x - x = 0
  if (getLhs() == getRhs()) {
    return IntegerAttr::get(getResult().getType(), 0);
  }
  
  if (!lhs || !rhs) return {};

  if (auto lhsInt = dyn_cast<IntegerAttr>(lhs)) {
    if (auto rhsInt = dyn_cast<IntegerAttr>(rhs)) {
      return IntegerAttr::get(lhsInt.getType(), lhsInt.getInt() - rhsInt.getInt());
    }
  }
  
  // x - 0 = x
  if (auto rhsInt = dyn_cast<IntegerAttr>(rhs)) {
    if (rhsInt.getInt() == 0) return getLhs();
  }
  
  return {};
}

::mlir::OpFoldResult MulOp::fold(MulOp::GenericAdaptor<llvm::ArrayRef<mlir::Attribute>> adaptor) {
  auto lhs = adaptor.getLhs();
  auto rhs = adaptor.getRhs();
  if (!lhs || !rhs) return {};

  if (auto lhsInt = dyn_cast<IntegerAttr>(lhs)) {
    if (auto rhsInt = dyn_cast<IntegerAttr>(rhs)) {
      return IntegerAttr::get(lhsInt.getType(), lhsInt.getInt() * rhsInt.getInt());
    }
  }
  
  // x * 0 = 0
  if (auto rhsInt = dyn_cast<IntegerAttr>(rhs)) {
    if (rhsInt.getInt() == 0) return rhs;
  }
  if (auto lhsInt = dyn_cast<IntegerAttr>(lhs)) {
    if (lhsInt.getInt() == 0) return lhs;
  }
  
  // x * 1 = x
  if (auto rhsInt = dyn_cast<IntegerAttr>(rhs)) {
    if (rhsInt.getInt() == 1) return getLhs();
  }
  if (auto lhsInt = dyn_cast<IntegerAttr>(lhs)) {
    if (lhsInt.getInt() == 1) return getRhs();
  }
  
  return {};
}

::mlir::OpFoldResult DivOp::fold(DivOp::GenericAdaptor<llvm::ArrayRef<mlir::Attribute>> adaptor) {
  auto lhs = adaptor.getLhs();
  auto rhs = adaptor.getRhs();
  
  // x / x = 1 (if x != 0)
  if (getLhs() == getRhs()) {
    if (auto rhsInt = dyn_cast_if_present<IntegerAttr>(rhs)) {
      if (rhsInt.getInt() != 0)
        return IntegerAttr::get(getResult().getType(), 1);
    }
  }
  
  if (!lhs || !rhs) return {};

  if (auto lhsInt = dyn_cast<IntegerAttr>(lhs)) {
    if (auto rhsInt = dyn_cast<IntegerAttr>(rhs)) {
      auto rhsVal = rhsInt.getInt();
      if (rhsVal == 0) return {};  // Division by zero
      return IntegerAttr::get(lhsInt.getType(), lhsInt.getInt() / rhsVal);
    }
  }
  
  // x / 1 = x
  if (auto rhsInt = dyn_cast<IntegerAttr>(rhs)) {
    if (rhsInt.getInt() == 1) return getLhs();
  }
  
  // 0 / x = 0 (if x != 0)
  if (auto lhsInt = dyn_cast<IntegerAttr>(lhs)) {
    if (lhsInt.getInt() == 0) return lhs;
  }
  
  return {};
}

::mlir::OpFoldResult MinusOp::fold(MinusOp::GenericAdaptor<llvm::ArrayRef<mlir::Attribute>> adaptor) {
  auto in = adaptor.getInput();
  if (!in) return {};
  
  if (auto intAttr = dyn_cast<IntegerAttr>(in)) {
    return IntegerAttr::get(intAttr.getType(), -intAttr.getInt());
  }
  return {};
}

//===----------------------------------------------------------------------===//
// Floating Point Fold Implementations
//===----------------------------------------------------------------------===//

::mlir::OpFoldResult AddFOp::fold(AddFOp::GenericAdaptor<llvm::ArrayRef<mlir::Attribute>> adaptor) {
  auto lhs = adaptor.getLhs();
  auto rhs = adaptor.getRhs();
  if (!lhs || !rhs) return {};

  if (auto lhsFloat = dyn_cast<FloatAttr>(lhs)) {
    if (auto rhsFloat = dyn_cast<FloatAttr>(rhs)) {
      return FloatAttr::get(lhsFloat.getType(), 
                            lhsFloat.getValueAsDouble() + rhsFloat.getValueAsDouble());
    }
  }
  return {};
}

::mlir::OpFoldResult SubFOp::fold(SubFOp::GenericAdaptor<llvm::ArrayRef<mlir::Attribute>> adaptor) {
  auto lhs = adaptor.getLhs();
  auto rhs = adaptor.getRhs();
  if (!lhs || !rhs) return {};

  if (auto lhsFloat = dyn_cast<FloatAttr>(lhs)) {
    if (auto rhsFloat = dyn_cast<FloatAttr>(rhs)) {
      return FloatAttr::get(lhsFloat.getType(),
                            lhsFloat.getValueAsDouble() - rhsFloat.getValueAsDouble());
    }
  }
  return {};
}

::mlir::OpFoldResult MulFOp::fold(MulFOp::GenericAdaptor<llvm::ArrayRef<mlir::Attribute>> adaptor) {
  auto lhs = adaptor.getLhs();
  auto rhs = adaptor.getRhs();
  if (!lhs || !rhs) return {};

  if (auto lhsFloat = dyn_cast<FloatAttr>(lhs)) {
    if (auto rhsFloat = dyn_cast<FloatAttr>(rhs)) {
      return FloatAttr::get(lhsFloat.getType(),
                            lhsFloat.getValueAsDouble() * rhsFloat.getValueAsDouble());
    }
  }
  return {};
}

::mlir::OpFoldResult DivFOp::fold(DivFOp::GenericAdaptor<llvm::ArrayRef<mlir::Attribute>> adaptor) {
  auto lhs = adaptor.getLhs();
  auto rhs = adaptor.getRhs();
  if (!lhs || !rhs) return {};

  if (auto lhsFloat = dyn_cast<FloatAttr>(lhs)) {
    if (auto rhsFloat = dyn_cast<FloatAttr>(rhs)) {
      double rhsVal = rhsFloat.getValueAsDouble();
      if (rhsVal == 0.0) return {};
      return FloatAttr::get(lhsFloat.getType(),
                            lhsFloat.getValueAsDouble() / rhsVal);
    }
  }
  return {};
}

::mlir::OpFoldResult MinusFOp::fold(MinusFOp::GenericAdaptor<llvm::ArrayRef<mlir::Attribute>> adaptor) {
  auto in = adaptor.getInput();
  if (!in) return {};
  
  if (auto floatAttr = dyn_cast<FloatAttr>(in)) {
    return FloatAttr::get(floatAttr.getType(), -floatAttr.getValueAsDouble());
  }
  return {};
}

//===----------------------------------------------------------------------===//
// Bitwise/Logical Fold Implementations
//===----------------------------------------------------------------------===//

::mlir::OpFoldResult AndOp::fold(AndOp::GenericAdaptor<llvm::ArrayRef<mlir::Attribute>> adaptor) {
  auto lhs = adaptor.getLhs();
  auto rhs = adaptor.getRhs();
  
  // x & x = x
  if (getLhs() == getRhs()) return getLhs();
  
  if (!lhs || !rhs) return {};

  if (auto lhsBool = dyn_cast<BoolAttr>(lhs)) {
    if (auto rhsBool = dyn_cast<BoolAttr>(rhs)) {
      return BoolAttr::get(lhsBool.getContext(), lhsBool.getValue() && rhsBool.getValue());
    }
  }
  if (auto lhsInt = dyn_cast<IntegerAttr>(lhs)) {
    if (auto rhsInt = dyn_cast<IntegerAttr>(rhs)) {
      return IntegerAttr::get(lhsInt.getType(), lhsInt.getInt() & rhsInt.getInt());
    }
  }
  
  // x & 0 = 0
  if (auto rhsInt = dyn_cast<IntegerAttr>(rhs)) {
    if (rhsInt.getInt() == 0) return rhs;
  }
  if (auto lhsInt = dyn_cast<IntegerAttr>(lhs)) {
    if (lhsInt.getInt() == 0) return lhs;
  }
  
  // x & -1 = x (all ones)
  if (auto rhsInt = dyn_cast<IntegerAttr>(rhs)) {
    if (rhsInt.getInt() == -1) return getLhs();
  }
  
  return {};
}

::mlir::OpFoldResult OrOp::fold(OrOp::GenericAdaptor<llvm::ArrayRef<mlir::Attribute>> adaptor) {
  auto lhs = adaptor.getLhs();
  auto rhs = adaptor.getRhs();
  
  // x | x = x
  if (getLhs() == getRhs()) return getLhs();
  
  if (!lhs || !rhs) return {};

  if (auto lhsBool = dyn_cast<BoolAttr>(lhs)) {
    if (auto rhsBool = dyn_cast<BoolAttr>(rhs)) {
      return BoolAttr::get(lhsBool.getContext(), lhsBool.getValue() || rhsBool.getValue());
    }
  }
  if (auto lhsInt = dyn_cast<IntegerAttr>(lhs)) {
    if (auto rhsInt = dyn_cast<IntegerAttr>(rhs)) {
      return IntegerAttr::get(lhsInt.getType(), lhsInt.getInt() | rhsInt.getInt());
    }
  }
  
  // x | 0 = x
  if (auto rhsInt = dyn_cast<IntegerAttr>(rhs)) {
    if (rhsInt.getInt() == 0) return getLhs();
  }
  if (auto lhsInt = dyn_cast<IntegerAttr>(lhs)) {
    if (lhsInt.getInt() == 0) return getRhs();
  }
  
  return {};
}

::mlir::OpFoldResult XorOp::fold(XorOp::GenericAdaptor<llvm::ArrayRef<mlir::Attribute>> adaptor) {
  auto lhs = adaptor.getLhs();
  auto rhs = adaptor.getRhs();
  
  // x ^ x = 0
  if (getLhs() == getRhs()) {
    return IntegerAttr::get(getResult().getType(), 0);
  }
  
  if (!lhs || !rhs) return {};

  if (auto lhsInt = dyn_cast<IntegerAttr>(lhs)) {
    if (auto rhsInt = dyn_cast<IntegerAttr>(rhs)) {
      return IntegerAttr::get(lhsInt.getType(), lhsInt.getInt() ^ rhsInt.getInt());
    }
  }
  
  // x ^ 0 = x
  if (auto rhsInt = dyn_cast<IntegerAttr>(rhs)) {
    if (rhsInt.getInt() == 0) return getLhs();
  }
  if (auto lhsInt = dyn_cast<IntegerAttr>(lhs)) {
    if (lhsInt.getInt() == 0) return getRhs();
  }
  
  return {};
}

::mlir::OpFoldResult NotOp::fold(NotOp::GenericAdaptor<llvm::ArrayRef<mlir::Attribute>> adaptor) {
  auto in = adaptor.getInput();
  if (!in) return {};

  if (auto b = dyn_cast<BoolAttr>(in)) {
    return BoolAttr::get(b.getContext(), !b.getValue());
  }
  if (auto i = dyn_cast<IntegerAttr>(in)) {
    // Bitwise NOT
    return IntegerAttr::get(i.getType(), ~i.getInt());
  }
  return {};
}

::mlir::OpFoldResult ShlOp::fold(ShlOp::GenericAdaptor<llvm::ArrayRef<mlir::Attribute>> adaptor) {
  auto lhs = adaptor.getLhs();
  auto rhs = adaptor.getRhs();
  if (!lhs || !rhs) return {};

  if (auto lhsInt = dyn_cast<IntegerAttr>(lhs)) {
    if (auto rhsInt = dyn_cast<IntegerAttr>(rhs)) {
      auto shift = rhsInt.getInt();
      if (shift < 0 || shift >= 64) return {};
      return IntegerAttr::get(lhsInt.getType(), lhsInt.getInt() << shift);
    }
  }
  
  // x << 0 = x
  if (auto rhsInt = dyn_cast<IntegerAttr>(rhs)) {
    if (rhsInt.getInt() == 0) return getLhs();
  }
  
  return {};
}

::mlir::OpFoldResult ShrOp::fold(ShrOp::GenericAdaptor<llvm::ArrayRef<mlir::Attribute>> adaptor) {
  auto lhs = adaptor.getLhs();
  auto rhs = adaptor.getRhs();
  if (!lhs || !rhs) return {};

  if (auto lhsInt = dyn_cast<IntegerAttr>(lhs)) {
    if (auto rhsInt = dyn_cast<IntegerAttr>(rhs)) {
      auto shift = rhsInt.getInt();
      if (shift < 0 || shift >= 64) return {};
      // Logical shift right (unsigned)
      uint64_t val = static_cast<uint64_t>(lhsInt.getInt());
      return IntegerAttr::get(lhsInt.getType(), static_cast<int64_t>(val >> shift));
    }
  }
  
  // x >> 0 = x
  if (auto rhsInt = dyn_cast<IntegerAttr>(rhs)) {
    if (rhsInt.getInt() == 0) return getLhs();
  }
  
  return {};
}

::mlir::OpFoldResult SarOp::fold(SarOp::GenericAdaptor<llvm::ArrayRef<mlir::Attribute>> adaptor) {
  auto lhs = adaptor.getLhs();
  auto rhs = adaptor.getRhs();
  if (!lhs || !rhs) return {};

  if (auto lhsInt = dyn_cast<IntegerAttr>(lhs)) {
    if (auto rhsInt = dyn_cast<IntegerAttr>(rhs)) {
      auto shift = rhsInt.getInt();
      if (shift < 0 || shift >= 64) return {};
      // Arithmetic shift right (signed)
      return IntegerAttr::get(lhsInt.getType(), lhsInt.getInt() >> shift);
    }
  }
  
  // x >> 0 = x
  if (auto rhsInt = dyn_cast<IntegerAttr>(rhs)) {
    if (rhsInt.getInt() == 0) return getLhs();
  }
  
  return {};
}

//===----------------------------------------------------------------------===//
// Comparison Fold Implementation
//===----------------------------------------------------------------------===//

::mlir::OpFoldResult CmpOp::fold(CmpOp::GenericAdaptor<llvm::ArrayRef<mlir::Attribute>> adaptor) {
  auto lhs = adaptor.getLhs();
  auto rhs = adaptor.getRhs();
  
  // Same operand comparisons
  if (getLhs() == getRhs()) {
    auto pred = getPredicate();
    // x == x, x <= x, x >= x are true
    if (pred == "eq" || pred == "==" || pred == "le" || pred == "<=" ||
        pred == "ge" || pred == ">=") {
      auto resultTy = getResult().getType();
      if (auto intTy = dyn_cast<IntegerType>(resultTy)) {
        return IntegerAttr::get(intTy, 1);
      }
      return BoolAttr::get(getContext(), true);
    }
    // x != x, x < x, x > x are false
    if (pred == "ne" || pred == "!=" || pred == "lt" || pred == "<" ||
        pred == "gt" || pred == ">") {
      auto resultTy = getResult().getType();
      if (auto intTy = dyn_cast<IntegerType>(resultTy)) {
        return IntegerAttr::get(intTy, 0);
      }
      return BoolAttr::get(getContext(), false);
    }
  }
  
  if (!lhs || !rhs) return {};

  auto predAttr = (*this)->getAttrOfType<StringAttr>("predicate");
  if (!predAttr) return {};

  // Integer comparison
  if (auto lhsInt = dyn_cast<IntegerAttr>(lhs)) {
    if (auto rhsInt = dyn_cast<IntegerAttr>(rhs)) {
      auto res = evalICmp(predAttr.getValue(), lhsInt.getInt(), rhsInt.getInt());
      if (!res.has_value()) return {};

      auto resultTy = getResult().getType();
      if (auto intTy = dyn_cast<IntegerType>(resultTy)) {
        return IntegerAttr::get(intTy, *res ? 1 : 0);
      }
      return BoolAttr::get(getContext(), *res);
    }
  }
  
  // Float comparison
  if (auto lhsFloat = dyn_cast<FloatAttr>(lhs)) {
    if (auto rhsFloat = dyn_cast<FloatAttr>(rhs)) {
      auto res = evalFCmp(predAttr.getValue(), 
                          lhsFloat.getValueAsDouble(), 
                          rhsFloat.getValueAsDouble());
      if (!res.has_value()) return {};

      auto resultTy = getResult().getType();
      if (auto intTy = dyn_cast<IntegerType>(resultTy)) {
        return IntegerAttr::get(intTy, *res ? 1 : 0);
      }
      return BoolAttr::get(getContext(), *res);
    }
  }
  
  return {};
}

//===----------------------------------------------------------------------===//
// Custom Assembly Format for ConstantOp
//===----------------------------------------------------------------------===//

void ConstantOp::print(OpAsmPrinter &p) {
  p << " ";
  p << getValue();
  p.printOptionalAttrDict((*this)->getAttrs(), /*elidedAttrs=*/{"value"});

  // Only print explicit type if attribute doesn't already include it
  if (!llvm::isa<TypedAttr>(getValue())) {
    p << " : ";
    p << getType();
  }
}

ParseResult ConstantOp::parse(OpAsmParser &parser, OperationState &result) {
  Attribute valueAttr;

  // Parse the constant value attribute
  if (parser.parseAttribute(valueAttr))
    return failure();

  // Parse optional attributes
  if (parser.parseOptionalAttrDict(result.attributes))
    return failure();

  // For typed attributes, extract the type
  Type resultType;
  if (auto typedAttr = llvm::dyn_cast<TypedAttr>(valueAttr)) {
    resultType = typedAttr.getType();
  } else {
    // For untyped attributes, parse explicit type
    if (parser.parseColon() || parser.parseType(resultType))
      return failure();
  }

  result.addAttribute("value", valueAttr);
  result.addTypes(resultType);
  return success();
}

//===----------------------------------------------------------------------===//
// Canonicalization Patterns (from idealize() methods)
//===----------------------------------------------------------------------===//

// AddOp canonicalizations (from AddNode.java idealize())
void AddOp::getCanonicalizationPatterns(RewritePatternSet &patterns, MLIRContext *context) {
  // Patterns would be added here for:
  // - (a + b) + c -> a + (b + c) if b,c are constants
  // - a + (-b) -> a - b
  // - (-a) + b -> b - a
  // These are registered via patterns.add<...>()
}

void SubOp::getCanonicalizationPatterns(RewritePatternSet &patterns, MLIRContext *context) {
  // Patterns for:
  // - a - (-b) -> a + b
  // - 0 - a -> -a
}

void MulOp::getCanonicalizationPatterns(RewritePatternSet &patterns, MLIRContext *context) {
  // Patterns for:
  // - a * 2^n -> a << n (strength reduction)
  // - (-a) * (-b) -> a * b
}
