#include "cpp2_mlir_assembler.h"

#include <algorithm>
#include <cctype>
#include <string>
#include <vector>

#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/IR/Builders.h>
#include <llvm/ADT/StringRef.h>

#include "evidence.h"
#include "cpp2_mlir_loader.h"

#include <unistd.h>

namespace cppfort::stage0 {
using cppfort::stage0::sanitize_symbol;
namespace {

struct ParsedFunction {
    std::string name;
    std::string params;
    std::string returnType;
    std::string body;
    std::size_t nameStart = 0;
    std::size_t nameEnd = 0;
    std::size_t paramsStart = 0;
    std::size_t paramsEnd = 0;
    std::size_t returnStart = 0;
    std::size_t returnEnd = 0;
    std::size_t bodyStart = 0;
    std::size_t bodyEnd = 0;
};

std::string trim(std::string_view text) {
    std::size_t begin = 0;
    std::size_t end = text.size();
    while (begin < end && std::isspace(static_cast<unsigned char>(text[begin]))) {
        ++begin;
    }
    while (end > begin && std::isspace(static_cast<unsigned char>(text[end - 1]))) {
        --end;
    }
    return std::string(text.substr(begin, end - begin));
}

bool parseMinimalFunction(std::string_view source, ParsedFunction& out) {
    const std::size_t colonPos = source.find(':');
    if (colonPos == std::string_view::npos) {
        return false;
    }

    const std::size_t paramsOpen = source.find('(', colonPos);
    const std::size_t paramsClose = source.find(')', paramsOpen);
    if (paramsOpen == std::string_view::npos || paramsClose == std::string_view::npos) {
        return false;
    }

    const std::size_t arrowPos = source.find("->", paramsClose);
    if (arrowPos == std::string_view::npos) {
        return false;
    }

    const std::size_t equalsPos = source.find('=', arrowPos);
    if (equalsPos == std::string_view::npos) {
        return false;
    }

    const std::size_t braceOpen = source.find('{', equalsPos);
    const std::size_t braceClose = source.rfind('}');
    if (braceOpen == std::string_view::npos || braceClose == std::string_view::npos || braceClose <= braceOpen) {
        return false;
    }

    out.nameStart = 0;
    out.nameEnd = colonPos;
    out.name = trim(source.substr(out.nameStart, out.nameEnd - out.nameStart));

    out.paramsStart = paramsOpen + 1;
    out.paramsEnd = paramsClose;
    out.params = trim(source.substr(out.paramsStart, out.paramsEnd - out.paramsStart));

    out.returnStart = arrowPos + 2;
    out.returnEnd = equalsPos;
    out.returnType = trim(source.substr(out.returnStart, out.returnEnd - out.returnStart));

    out.bodyStart = braceOpen + 1;
    out.bodyEnd = braceClose;
    out.body = trim(source.substr(out.bodyStart, out.bodyEnd - out.bodyStart));
    return !out.name.empty();
}

void emitCutout(mlir::OpBuilder& builder,
                mlir::Location loc,
                llvm::StringRef label,
                llvm::StringRef text,
                std::size_t start,
                std::size_t end,
                std::size_t ordinal) {
    mlir::OperationState state(loc, "cppfort.cutout");
    state.addAttribute("cutout_name", builder.getStringAttr(label));
    state.addAttribute("text", builder.getStringAttr(text));
    state.addAttribute("start", builder.getI64IntegerAttr(static_cast<int64_t>(start)));
    state.addAttribute("end", builder.getI64IntegerAttr(static_cast<int64_t>(end)));
    state.addAttribute("hierarchy_ordinal", builder.getI64IntegerAttr(static_cast<int64_t>(ordinal)));
    builder.create(state);
}

std::string ordinalToAlpha(std::size_t ordinal) {
    std::string label;
    std::size_t value = ordinal + 1;
    while (value > 0) {
        value -= 1;
        const char ch = static_cast<char>('a' + (value % 26));
        label.push_back(ch);
        value /= 26;
    }
    std::reverse(label.begin(), label.end());
    return label;
}

} // namespace

mlir::OwningOpRef<mlir::ModuleOp> Cpp2MlirAssembler::buildMinimalFunctionModule(
    std::string_view source,
    mlir::MLIRContext& context,
    const std::string& moduleName) {

    ParsedFunction parsed;
    if (!parseMinimalFunction(source, parsed)) {
        return mlir::OwningOpRef<mlir::ModuleOp>();
    }

    mlir::OpBuilder builder(&context);
    mlir::Location loc = builder.getUnknownLoc();
    mlir::ModuleOp module = mlir::ModuleOp::create(loc, moduleName);
    builder.setInsertionPointToStart(module.getBody());

    mlir::OperationState state(loc, "cppfort.func");
    state.addAttribute("name", builder.getStringAttr(parsed.name));
    state.addAttribute("params", builder.getStringAttr(parsed.params));
    state.addAttribute("return_type", builder.getStringAttr(parsed.returnType));
    state.addAttribute("body", builder.getStringAttr(parsed.body));
    state.addAttribute("source", builder.getStringAttr(std::string(source)));
    state.addRegion(std::make_unique<mlir::Region>());
    auto* funcOp = builder.create(state);

    auto& region = funcOp->getRegion(0);
    auto* block = new mlir::Block();
    region.push_back(block);

    mlir::OpBuilder regionBuilder(block, block->end());
    emitCutout(regionBuilder, loc, "func_name", parsed.name, parsed.nameStart, parsed.nameEnd, 0);
    emitCutout(regionBuilder, loc, "params", parsed.params, parsed.paramsStart, parsed.paramsEnd, 1);
    emitCutout(regionBuilder, loc, "return_type", parsed.returnType, parsed.returnStart, parsed.returnEnd, 2);
    emitCutout(regionBuilder, loc, "body", parsed.body, parsed.bodyStart, parsed.bodyEnd, 3);

    return mlir::OwningOpRef<mlir::ModuleOp>(module);
}

mlir::OwningOpRef<mlir::ModuleOp> Cpp2MlirAssembler::buildModuleFromMatches(
    std::string_view source,
    const std::vector<cppfort::pijul::OrbitMatchInfo>& matches,
    mlir::MLIRContext& context,
    const std::string& moduleName) {

    if (matches.empty()) {
        return mlir::OwningOpRef<mlir::ModuleOp>();
    }

    std::vector<NamedCutout> cutouts;
    cutouts.reserve(matches.size());

    const std::size_t sourceSize = source.size();
    for (std::size_t i = 0; i < matches.size(); ++i) {
        const auto& match = matches[i];

        NamedCutout cutout;
        const std::string baseName = match.patternName.empty()
            ? ("cutout_" + std::to_string(i))
            : match.patternName;
        cutout.name = sanitize_symbol(baseName);
        cutout.context = match.context;

        const std::size_t start = std::min(match.context.start_pos, sourceSize);
        const std::size_t end = std::min(match.context.end_pos, sourceSize);
        std::string snippet;
        if (end > start) {
            snippet.assign(source.substr(start, end - start));
        }

        EvidenceSpan span(start, end, snippet);
        span.pattern_name = match.patternName;
        span.match_length = end > start ? (end - start) : 0;
        span.confidence = 1.0;
        span.nesting_depth = match.context.depth_hint;
        cutout.span = std::move(span);

        if (cutout.context.hierarchy_label.empty()) {
            cutout.context.hierarchy_label = ordinalToAlpha(cutout.context.hierarchy_ordinal);
        }

        cutouts.push_back(std::move(cutout));
    }

    Cpp2MlirLoader loader(context);
    const std::string sanitizedModule = sanitize_symbol(moduleName.empty() ? "cpp2_capture" : moduleName);
    return loader.buildModule(sanitizedModule, cutouts, Cpp2MlirLoader::Options{});
}

} // namespace cppfort::stage0
