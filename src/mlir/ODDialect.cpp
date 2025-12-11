// Minimal ODDialect.cpp: include generated TableGen output if available
#include "mlir/IR/Dialect.h"
#include "mlir/IR/OpImplementation.h"

// Generated file may not exist until TableGen runs; include guarded
#if __has_include("ODDialect.cpp.inc")
#include "ODDialect.cpp.inc"
#endif

namespace cppfort::mlir_son {

// Nothing here yet. Generated files handle dialect class definitions.

} // namespace cppfort::mlir_son
