//===----------------------------------------------------------------------===//
// Cpp2 Sea-of-Nodes Dialect Implementation
// TrikeShed Math-Based SoN Compiler
//===----------------------------------------------------------------------===//

#include "mlir/IR/Builders.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/DialectImplementation.h"
#include "mlir/IR/MLIRContext.h"

#include "Cpp2SONDialect.h"
#include "Cpp2SONOpsOps.h.inc"

using namespace mlir;
using namespace cpp2;

//===----------------------------------------------------------------------===//
// Cpp2SON Dialect
//===----------------------------------------------------------------------===//

Cpp2SONDialect::Cpp2SONDialect(mlir::MLIRContext *context)
    : mlir::Dialect(getDialectNamespace(), context, 
                    mlir::TypeID::get<Cpp2SONDialect>()) {
  initialize();
}

void Cpp2SONDialect::initialize() {
  addOperations<
#define GET_OP_LIST
#include "Cpp2SONOpsOps.cpp.inc"
#undef GET_OP_LIST
  >();
}

//===----------------------------------------------------------------------===//
// Op Builders
//===----------------------------------------------------------------------===//

void IndexedOp::build(mlir::OpBuilder &builder, mlir::OperationState &state,
                      mlir::Type resultType, mlir::Value domain, mlir::Value accessor) {
  state.addTypes(resultType);
  state.addOperands({domain, accessor});
}
