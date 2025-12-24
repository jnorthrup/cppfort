#include "Cpp2SONDialect.h"

#include "mlir/IR/Builders.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/IR/BuiltinAttributes.h"

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
