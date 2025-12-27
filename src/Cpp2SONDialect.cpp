#include "Cpp2SONDialect.h"

#include "mlir/IR/Builders.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "llvm/ADT/StringRef.h"

using namespace mlir;
using namespace mlir::sond;

#include "Cpp2SONOpsDialect.cpp.inc"

#define GET_OP_CLASSES
#include "Cpp2SONOps.cpp.inc"

void Cpp2SONDialect::initialize() {
  addOperations<
#define GET_OP_LIST
#include "Cpp2SONOps.cpp.inc"
  >();
}

// Helper for comparison folding
static std::optional<bool> evalICmp(llvm::StringRef pred, int64_t lhs, int64_t rhs) {
  if (pred == "lt") return lhs < rhs;
  if (pred == "le") return lhs <= rhs;
  if (pred == "gt") return lhs > rhs;
  if (pred == "ge") return lhs >= rhs;
  if (pred == "eq") return lhs == rhs;
  if (pred == "ne") return lhs != rhs;
  return std::nullopt;
}

// Fold implementations using GenericAdaptor<llvm::ArrayRef<Attribute>>
::mlir::OpFoldResult AddOp::fold(AddOp::GenericAdaptor<llvm::ArrayRef<mlir::Attribute>> adaptor) {
  auto lhs = adaptor.getLhs();
  auto rhs = adaptor.getRhs();
  if (!lhs || !rhs) return {};

  if (auto lhsInt = dyn_cast<IntegerAttr>(lhs)) {
    if (auto rhsInt = dyn_cast<IntegerAttr>(rhs)) {
      return IntegerAttr::get(lhsInt.getType(), lhsInt.getInt() + rhsInt.getInt());
    }
  }
  return {};
}

::mlir::OpFoldResult SubOp::fold(SubOp::GenericAdaptor<llvm::ArrayRef<mlir::Attribute>> adaptor) {
  auto lhs = adaptor.getLhs();
  auto rhs = adaptor.getRhs();
  if (!lhs || !rhs) return {};

  if (auto lhsInt = dyn_cast<IntegerAttr>(lhs)) {
    if (auto rhsInt = dyn_cast<IntegerAttr>(rhs)) {
      return IntegerAttr::get(lhsInt.getType(), lhsInt.getInt() - rhsInt.getInt());
    }
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
  return {};
}

::mlir::OpFoldResult DivOp::fold(DivOp::GenericAdaptor<llvm::ArrayRef<mlir::Attribute>> adaptor) {
  auto lhs = adaptor.getLhs();
  auto rhs = adaptor.getRhs();
  if (!lhs || !rhs) return {};

  if (auto lhsInt = dyn_cast<IntegerAttr>(lhs)) {
    if (auto rhsInt = dyn_cast<IntegerAttr>(rhs)) {
      auto rhsVal = rhsInt.getInt();
      if (rhsVal == 0) return {};
      return IntegerAttr::get(lhsInt.getType(), lhsInt.getInt() / rhsVal);
    }
  }
  return {};
}

::mlir::OpFoldResult AndOp::fold(AndOp::GenericAdaptor<llvm::ArrayRef<mlir::Attribute>> adaptor) {
  auto lhs = adaptor.getLhs();
  auto rhs = adaptor.getRhs();
  if (!lhs || !rhs) return {};

  if (auto lhsBool = dyn_cast<BoolAttr>(lhs)) {
    if (auto rhsBool = dyn_cast<BoolAttr>(rhs)) {
      return BoolAttr::get(lhsBool.getContext(), lhsBool.getValue() && rhsBool.getValue());
    }
  }
  if (auto lhsInt = dyn_cast<IntegerAttr>(lhs)) {
    if (auto rhsInt = dyn_cast<IntegerAttr>(rhs)) {
      // Common representation for i1 values
      bool l = lhsInt.getInt() != 0;
      bool r = rhsInt.getInt() != 0;
      return IntegerAttr::get(lhsInt.getType(), (l && r) ? 1 : 0);
    }
  }
  return {};
}

::mlir::OpFoldResult OrOp::fold(OrOp::GenericAdaptor<llvm::ArrayRef<mlir::Attribute>> adaptor) {
  auto lhs = adaptor.getLhs();
  auto rhs = adaptor.getRhs();
  if (!lhs || !rhs) return {};

  if (auto lhsBool = dyn_cast<BoolAttr>(lhs)) {
    if (auto rhsBool = dyn_cast<BoolAttr>(rhs)) {
      return BoolAttr::get(lhsBool.getContext(), lhsBool.getValue() || rhsBool.getValue());
    }
  }
  if (auto lhsInt = dyn_cast<IntegerAttr>(lhs)) {
    if (auto rhsInt = dyn_cast<IntegerAttr>(rhs)) {
      bool l = lhsInt.getInt() != 0;
      bool r = rhsInt.getInt() != 0;
      return IntegerAttr::get(lhsInt.getType(), (l || r) ? 1 : 0);
    }
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
    return IntegerAttr::get(i.getType(), (i.getInt() == 0) ? 1 : 0);
  }
  return {};
}

::mlir::OpFoldResult CmpOp::fold(CmpOp::GenericAdaptor<llvm::ArrayRef<mlir::Attribute>> adaptor) {
  auto lhs = adaptor.getLhs();
  auto rhs = adaptor.getRhs();
  if (!lhs || !rhs) return {};

  auto predAttr = (*this)->getAttrOfType<StringAttr>("predicate");
  if (!predAttr) return {};

  if (auto lhsInt = dyn_cast<IntegerAttr>(lhs)) {
    if (auto rhsInt = dyn_cast<IntegerAttr>(rhs)) {
      auto res = evalICmp(predAttr.getValue(), lhsInt.getInt(), rhsInt.getInt());
      if (!res.has_value()) return {};

      // Prefer i1 IntegerAttr if result type is integer, otherwise BoolAttr.
      auto resultTy = getResult().getType();
      if (auto intTy = dyn_cast<IntegerType>(resultTy)) {
        return IntegerAttr::get(intTy, *res ? 1 : 0);
      }
      return BoolAttr::get(getContext(), *res);
    }
  }
  return {};
}

// Custom assembly format for ConstantOp
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
