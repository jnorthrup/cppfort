#ifndef CPP2_PASSES_H
#define CPP2_PASSES_H

#include "mlir/Pass/Pass.h"

namespace mlir {
namespace cpp2 {

/// Create a pass to convert FIR dialect to SON dialect
std::unique_ptr<Pass> createConvertFIRToSONPass();

/// Initialize FIR to SON pass registration
void initConvertFIRToSONPass();

/// Register all Cpp2 passes
inline void registerCpp2Passes() {
  initConvertFIRToSONPass();
}

} // namespace cpp2
} // namespace mlir

#endif // CPP2_PASSES_H
