#pragma once

#include <string>
#include <vector>

#include <mlir/IR/BuiltinOps.h>

namespace cppfort::stage0 {

// Applies the initial cpp2 → cpp1 rewrite pattern. Returns true when a
// function rewrite was applied and populates the generated cpp1 text and
// the name of the synthesized function.
bool rewrite_first_function_to_cpp1(mlir::ModuleOp module,
                                    std::string& generated_cpp1,
                                    std::string& rewritten_function_name);

struct CutoutRewrite {
    std::string cutoutName;
    std::vector<int64_t> hierarchyPath;
    std::string hierarchyLabel;
    std::string text;
};

std::vector<CutoutRewrite> collect_cutout_rewrites(mlir::ModuleOp module);

} // namespace cppfort::stage0
