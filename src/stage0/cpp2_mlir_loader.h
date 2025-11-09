#pragma once

#include <string>
#include <string_view>
#include <vector>

#include "evidence.h"
#include "pijul_types.h"

#include <mlir/IR/BuiltinOps.h>
#include <mlir/IR/MLIRContext.h>
#include <mlir/IR/OwningOpRef.h>

namespace cppfort::stage0 {

struct NamedCutout {
    std::string name;
    EvidenceSpan span;
    cppfort::pijul::NodeContext context;
};

class Cpp2MlirLoader {
public:
    struct Options {
        bool attach_orbit_metadata = true;
    };

    explicit Cpp2MlirLoader(mlir::MLIRContext& context);

    mlir::OwningOpRef<mlir::ModuleOp> buildModule(const std::string& moduleName,
                                                  const std::vector<NamedCutout>& cutouts,
                                                  const Options& options);

    std::string moduleToString(mlir::ModuleOp module) const;

private:
    mlir::MLIRContext& context_;
};

std::string sanitize_symbol(std::string_view input);

} // namespace cppfort::stage0
