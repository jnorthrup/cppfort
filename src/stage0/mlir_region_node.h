// Lightweight shim header to alias our RegionNode and related stubs into the
// `mlir` namespace for builds that don't link against system MLIR headers.
//
// This file mirrors the intentions in `src/stage0/region_node.h` where we
// provide a compatible surface for components that reference `mlir::RegionNode`.
// If the full MLIR headers are available on the system, prefer those instead
// by setting `USE_MLIR` in the build system.

#pragma once

#include "region_node.h"

// If the real MLIR headers are present, the source code would normally use
// them. For this project we provide a lightweight alias that places our
// `cppfort::ir::mlir::RegionNode` type into the `mlir` namespace so code that
// expects `mlir::RegionNode` can compile without system MLIR being installed.

namespace mlir {
    using RegionNode = cppfort::ir::mlir::RegionNode;
    using OpStub = cppfort::ir::mlir::OpStub;
    using ValueStub = cppfort::ir::mlir::ValueStub;
}
