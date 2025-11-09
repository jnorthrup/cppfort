#include <iostream>
#include <sstream>
#include <string>
#include <unordered_map>
#include <vector>

#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/IR/Builders.h>
#include <mlir/IR/MLIRContext.h>

#include "cpp2_mlir_loader.h"
#include "cpp2_mlir_rewriter.h"
#include "heap_limiter.h"

using namespace cppfort::stage0;

namespace {

std::string signatureKey(const std::vector<int64_t>& path,
                         const std::string& label) {
    std::ostringstream oss;
    oss << "[";
    for (std::size_t i = 0; i < path.size(); ++i) {
        if (i != 0) {
            oss << '.';
        }
        oss << path[i];
    }
    oss << "]";
    if (!label.empty()) {
        oss << label;
    }
    return oss.str();
}

std::unordered_map<std::string, std::string> makeRewriteMap(
    const std::vector<CutoutRewrite>& rewrites) {
    std::unordered_map<std::string, std::string> map;
    map.reserve(rewrites.size());
    for (const auto& rewrite : rewrites) {
        if (rewrite.text.empty()) {
            continue;
        }
        map.emplace(signatureKey(rewrite.hierarchyPath, rewrite.hierarchyLabel),
                    rewrite.text);
    }
    return map;
}

NamedCutout makeCutout(const std::string& snippet,
                       const std::string& patternName,
                       std::vector<int64_t> path,
                       std::size_t ordinal,
                       const std::string& label) {
    NamedCutout cutout;
    cutout.name = patternName;
    cutout.span.start_pos = 0;
    cutout.span.end_pos = snippet.size();
    cutout.span.content = snippet;
    cutout.span.pattern_name = patternName;
    cutout.span.match_length = snippet.size();
    cutout.span.confidence = 1.0;

    cutout.context.start_pos = 0;
    cutout.context.end_pos = snippet.size();
    cutout.context.hierarchy_index = std::move(path);
    cutout.context.hierarchy_depth = cutout.context.hierarchy_index.size();
    cutout.context.hierarchy_ordinal = ordinal;
    cutout.context.hierarchy_label = label;
    return cutout;
}

} // namespace

int main() {
    if (!cppfort::stage0::ensure_heap_limit(std::cerr)) {
        return 1;
    }

    std::cout << "=== cpp2_mlir_rewriter smoke test ===\n";

    const std::string cpp2Snippet =
        "template<T>\n"
        "[[nodiscard]] accumulate: (range: in T, init: copy T) -> T requires Summable<T> = {\n"
        "    total := init;\n"
        "    return total;\n"
        "}";

    mlir::MLIRContext context;
    context.allowUnregisteredDialects(true);
    context.loadDialect<mlir::func::FuncDialect>();

    std::vector<NamedCutout> cutouts;
    cutouts.push_back(makeCutout(
        cpp2Snippet,
        "cpp2_template_function_definition",
        {0},
        0,
        "a"));
    cutouts.push_back(makeCutout(
        "total := init;",
        "cpp2_var_auto",
        {0, 0},
        0,
        "a"));
    cutouts.push_back(makeCutout(
        "_pre: (count > 0) = \"count positive\"",
        "cpp2_contract_pre",
        {1},
        1,
        "b"));
    cutouts.push_back(makeCutout(
        "_post: (sum >= 0) = \"stable\" requires ok",
        "cpp2_contract_post",
        {2},
        2,
        "c"));
    cutouts.push_back(makeCutout(
        "values.push_back(42)",
        "cpp2_ufcs_call",
        {3},
        3,
        "d"));
    cutouts.push_back(makeCutout(
        "for item: items { total += item; }",
        "cpp2_for_with_body",
        {4},
        4,
        "e"));
    cutouts.push_back(makeCutout(
        "while running { tick(); }",
        "cpp2_while_loop",
        {5},
        5,
        "f"));
    cutouts.push_back(makeCutout(
        "loop { work(); }",
        "cpp2_loop_forever",
        {6},
        6,
        "g"));
    cutouts.push_back(makeCutout(
        "inspect value { handle_case(); }",
        "cpp2_inspect_expression",
        {7},
        7,
        "h"));
    cutouts.push_back(makeCutout(
        "is Value => handle_case();",
        "cpp2_inspect_case_is",
        {7, 0},
        0,
        "a"));

    Cpp2MlirLoader loader(context);
    auto module = loader.buildModule("rewrite_test", cutouts, Cpp2MlirLoader::Options{});
    if (!module) {
        std::cerr << "Failed to build MLIR module for cutouts\n";
        return 1;
    }

    mlir::OpBuilder builder(&context);
    if (auto funcOp = module->lookupSymbol<mlir::func::FuncOp>("rewrite_test")) {
        funcOp->setAttr("cppfort.cutout_count",
                        builder.getI64IntegerAttr(static_cast<int64_t>(cutouts.size())));
    } else {
        std::cerr << "Unable to locate synthesized function symbol\n";
        return 1;
    }

    std::string generated;
    std::string rewrittenName;
    const bool applied = rewrite_first_function_to_cpp1(*module, generated, rewrittenName);
    if (!applied) {
        std::cerr << "Rewrite pattern did not apply\n";
        return 1;
    }

    const auto rewrites = collect_cutout_rewrites(*module);
    const auto rewriteMap = makeRewriteMap(rewrites);

    const std::string functionKey = signatureKey({0}, "a");
    const std::string variableKey = signatureKey({0, 0}, "a");
    const std::string contractPreKey = signatureKey({1}, "b");
    const std::string contractPostKey = signatureKey({2}, "c");
    const std::string ufcsKey = signatureKey({3}, "d");
    const std::string forKey = signatureKey({4}, "e");
    const std::string whileKey = signatureKey({5}, "f");
    const std::string loopKey = signatureKey({6}, "g");
    const std::string inspectKey = signatureKey({7}, "h");
    const std::string inspectCaseKey = signatureKey({7, 0}, "a");

    const std::string expected =
        "template<T>\n"
        "[[nodiscard]] T accumulate(const T& range, T init) requires Summable<T> {\n"
        "    total := init;\n"
        "    return total;\n"
        "}";

    std::cout << "Generated C++:\n" << generated << "\n";
    std::cout << "Expected C++:\n" << expected << "\n";
    std::cout << "Rewritten name: " << rewrittenName << "\n";

    if (generated != expected) {
        std::cerr << "Mismatch in generated C++ code\n";
        return 1;
    }
    if (rewrittenName != "cpp1_accumulate") {
        std::cerr << "Unexpected rewritten function name: " << rewrittenName << "\n";
        return 1;
    }
    auto funcIt = rewriteMap.find(functionKey);
    if (funcIt == rewriteMap.end() || funcIt->second != expected) {
        std::cerr << "Missing function rewrite entry\n";
        return 1;
    }
    auto varIt = rewriteMap.find(variableKey);
    if (varIt == rewriteMap.end() || varIt->second != "auto total = init;") {
        std::cerr << "Missing variable rewrite entry\n";
        return 1;
    }
    auto preIt = rewriteMap.find(contractPreKey);
    if (preIt == rewriteMap.end() || preIt->second != "assert((count > 0) && \"count positive\");") {
        std::cerr << "Missing contract pre rewrite entry\n";
        return 1;
    }
    auto postIt = rewriteMap.find(contractPostKey);
    if (postIt == rewriteMap.end() ||
        postIt->second != "/* post: sum >= 0 -> \"stable\" requires ok */") {
        std::cerr << "Missing contract post rewrite entry\n";
        return 1;
    }
    auto ufcsIt = rewriteMap.find(ufcsKey);
    if (ufcsIt == rewriteMap.end() || ufcsIt->second != "push_back(values, 42)") {
        std::cerr << "Missing UFCS rewrite entry\n";
        return 1;
    }
    auto forIt = rewriteMap.find(forKey);
    if (forIt == rewriteMap.end() ||
        forIt->second != "for (auto&& item : items) { total += item; }") {
        std::cerr << "Missing for-loop rewrite entry\n";
        return 1;
    }
    auto whileIt = rewriteMap.find(whileKey);
    if (whileIt == rewriteMap.end() ||
        whileIt->second != "while (running) { tick(); }") {
        std::cerr << "Missing while-loop rewrite entry\n";
        return 1;
    }
    auto loopIt = rewriteMap.find(loopKey);
    if (loopIt == rewriteMap.end() ||
        loopIt->second != "while (true) { work(); }") {
        std::cerr << "Missing loop-forever rewrite entry\n";
        return 1;
    }
    auto inspectIt = rewriteMap.find(inspectKey);
    const std::string expectedInspect =
        "/* inspect */\n"
        "auto&& __inspect_value = value;\n"
        "(void)__inspect_value;\n"
        "// cases:\n"
        "handle_case();\n";
    if (inspectIt == rewriteMap.end() || inspectIt->second != expectedInspect) {
        std::cerr << "Missing inspect rewrite entry\n";
        return 1;
    }
    auto inspectCaseIt = rewriteMap.find(inspectCaseKey);
    if (inspectCaseIt == rewriteMap.end() ||
        inspectCaseIt->second != "// inspect case: is Value => handle_case();") {
        std::cerr << "Missing inspect case rewrite entry\n";
        return 1;
    }
    std::cout << "✓ cpp2_mlir_rewriter produced expected output\n";
    return 0;
}
