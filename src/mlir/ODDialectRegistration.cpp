#include "mlir/IR/Dialect.h"
#include "mlir/IR/MLIRContext.h"

#if __has_include("ODDialect.cpp.inc")
#include "ODDialect.cpp.inc"
#endif

void registerODDialect(mlir::DialectRegistry &registry) {
#if __has_include("ODDialect.cpp.inc")
    registry.insert<cppfort::mlir_son::ODDialect>();
#endif
}
