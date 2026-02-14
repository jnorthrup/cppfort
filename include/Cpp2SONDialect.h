#ifndef CPP2_SON_DIALECT_H
#define CPP2_SON_DIALECT_H

#include "mlir/IR/Dialect.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/Interfaces/InferTypeOpInterface.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "mlir/Interfaces/ControlFlowInterfaces.h"
#include "mlir/Interfaces/CallInterfaces.h"
#include "mlir/Interfaces/FunctionInterfaces.h"
#include "mlir/Bytecode/BytecodeOpInterface.h"

// Include dialect definition
#include "Cpp2SONOpsDialect.h.inc"

// Include generated types
#define GET_TYPEDEF_CLASSES
#include "Cpp2SONOpsTypes.h.inc"

// Include generated attribute declarations
#define GET_ATTRDEF_CLASSES
#include "Cpp2SONOpsAttrDefs.h.inc"

// Include generated operations
#define GET_OP_CLASSES
#include "Cpp2SONOps.h.inc"

#endif // CPP2_SON_DIALECT_H
