//===----------------------------------------------------------------------===//
// Cpp2 Sea-of-Nodes Dialect Header
// TrikeShed Math-Based SoN Compiler
//===----------------------------------------------------------------------===//

#ifndef CPP2_SON_DIALECT_H
#define CPP2_SON_DIALECT_H

#include "mlir/IR/Dialect.h"
#include "mlir/IR/Builders.h"

#include "Cpp2SONOpsOps.h.inc"

namespace cpp2 {

class Cpp2SONDialect : public mlir::Dialect {
 public:
  explicit Cpp2SONDialect(mlir::MLIRContext *context);
  static llvm::StringRef getDialectNamespace() { return "cpp2"; }

 private:
  void initialize();
};

}  // namespace cpp2

#define GET_OP_CLASSES
#include "Cpp2SONOpsOps.h.inc"

#endif  // CPP2_SON_DIALECT_H
