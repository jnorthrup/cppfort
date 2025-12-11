#pragma once

#include "mlir/IR/Dialect.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/Types.h"
#include "mlir/IR/Builders.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "mlir/Interfaces/ControlFlowInterfaces.h"

#include "Cpp2OpsDialect.h.inc"

#define GET_TYPEDEF_CLASSES
#include "Cpp2OpsTypes.h.inc"

#define GET_OP_CLASSES
#include "Cpp2Ops.h.inc"