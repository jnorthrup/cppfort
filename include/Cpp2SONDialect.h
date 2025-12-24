#ifndef CPP2_SON_DIALECT_H
#define CPP2_SON_DIALECT_H

#include "mlir/IR/Dialect.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/Interfaces/InferTypeOpInterface.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "mlir/Bytecode/BytecodeOpInterface.h"

#include "Cpp2SONOpsDialect.h.inc"

#define GET_OP_CLASSES
#include "Cpp2SONOps.h.inc"

#endif // CPP2_SON_DIALECT_H
