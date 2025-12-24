#ifndef CPP2_FIR_DIALECT_H
#define CPP2_FIR_DIALECT_H

#include "mlir/IR/Dialect.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/Interfaces/FunctionInterfaces.h"
#include "mlir/Interfaces/CallInterfaces.h"
#include "mlir/Interfaces/ControlFlowInterfaces.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "mlir/Bytecode/BytecodeOpInterface.h"

#include "Cpp2FIROpsDialect.h.inc"

#define GET_OP_CLASSES
#include "Cpp2FIROps.h.inc"

#endif // CPP2_FIR_DIALECT_H
