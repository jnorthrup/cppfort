#pragma once

#include <string>
#include <string_view>
#include <vector>

#include <mlir/IR/BuiltinOps.h>
#include <mlir/IR/MLIRContext.h>
#include <mlir/IR/OwningOpRef.h>

#include "cpp2_mlir_loader.h"
#include "pijul_orbit_builder.h"

namespace cppfort::stage0 {

/**
 * Lightweight assembler that converts simple Cpp2 source fragments into
 * cppfort-specific MLIR operations carrying orbit-style metadata.
 */
class Cpp2MlirAssembler {
public:
    /**
     * Build an MLIR module from the given Cpp2 source snippet. The assembler
     * currently recognises the minimal function form:
     *
     *   name: () -> return_type = { body }
     *
     * and emits a `cppfort.func` with nested `cppfort.cutout` operations that
     * mirror the evidence spans produced by the orbit tooling.
     */
    static mlir::OwningOpRef<mlir::ModuleOp> buildMinimalFunctionModule(
        std::string_view source,
        mlir::MLIRContext& context,
        const std::string& moduleName = "cpp2_minimal");

    /**
     * Assemble an MLIR module from orbit matches that already capture evidence
     * spans. This is the preferred entry point for real code, as it preserves
     * the named cutouts and hierarchy metadata computed by the orbit pipeline.
     */
    static mlir::OwningOpRef<mlir::ModuleOp> buildModuleFromMatches(
        std::string_view source,
        const std::vector<cppfort::pijul::OrbitMatchInfo>& matches,
        mlir::MLIRContext& context,
        const std::string& moduleName = "cpp2_capture");
};

} // namespace cppfort::stage0
