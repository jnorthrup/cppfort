#include "cpp2_mlir_loader.h"

#include <algorithm>
#include <cctype>
#include <cstddef>
#include <cstdint>
#include <string>
#include <utility>

#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/IR/Builders.h>
#include <mlir/IR/BuiltinTypes.h>
#include <mlir/IR/Operation.h>
#include <mlir/IR/Value.h>

#include <llvm/ADT/SmallVector.h>
#include <llvm/Support/raw_ostream.h>

namespace cppfort::stage0 {
namespace {

mlir::ArrayAttr makeIntegerArrayAttr(mlir::OpBuilder& builder,
                                     const std::vector<int64_t>& values) {
    llvm::SmallVector<mlir::Attribute, 8> attrs;
    attrs.reserve(values.size());
    for (int64_t value : values) {
        attrs.push_back(builder.getI64IntegerAttr(value));
    }
    return builder.getArrayAttr(attrs);
}

template <typename Container>
mlir::ArrayAttr makeIntegerArrayAttr(mlir::OpBuilder& builder,
                                     const Container& values) {
    llvm::SmallVector<mlir::Attribute, 8> attrs;
    attrs.reserve(values.size());
    for (auto value : values) {
        attrs.push_back(builder.getI64IntegerAttr(
            static_cast<int64_t>(value)));
    }
    return builder.getArrayAttr(attrs);
}

mlir::DictionaryAttr makeSpanAttr(mlir::OpBuilder& builder,
                                  const EvidenceSpan& span) {
    llvm::SmallVector<mlir::NamedAttribute, 4> attrs;
    attrs.emplace_back(builder.getStringAttr("start"),
                       builder.getI64IntegerAttr(
                           static_cast<int64_t>(span.start_pos)));
    attrs.emplace_back(builder.getStringAttr("end"),
                       builder.getI64IntegerAttr(
                           static_cast<int64_t>(span.end_pos)));
    if (!span.content.empty()) {
        attrs.emplace_back(builder.getStringAttr("snippet"),
                           builder.getStringAttr(span.content));
    }
    attrs.emplace_back(builder.getStringAttr("confidence"),
                       builder.getF64FloatAttr(span.confidence));
    return builder.getDictionaryAttr(attrs);
}

mlir::DictionaryAttr makeHierarchyAttr(mlir::OpBuilder& builder,
                                       const cppfort::pijul::NodeContext& ctx) {
    llvm::SmallVector<mlir::NamedAttribute, 8> attrs;
    attrs.emplace_back(builder.getStringAttr("path"),
                       makeIntegerArrayAttr(builder, ctx.hierarchy_index));
    attrs.emplace_back(builder.getStringAttr("depth"),
                       builder.getI64IntegerAttr(
                           static_cast<int64_t>(ctx.hierarchy_depth)));
    attrs.emplace_back(builder.getStringAttr("ordinal"),
                       builder.getI64IntegerAttr(
                           static_cast<int64_t>(ctx.hierarchy_ordinal)));
    attrs.emplace_back(builder.getStringAttr("label"),
                       builder.getStringAttr(ctx.hierarchy_label));
    attrs.emplace_back(builder.getStringAttr("start"),
                       builder.getI64IntegerAttr(
                           static_cast<int64_t>(ctx.start_pos)));
    attrs.emplace_back(builder.getStringAttr("end"),
                       builder.getI64IntegerAttr(
                           static_cast<int64_t>(ctx.end_pos)));
    if (!ctx.orbit_signature.empty()) {
        attrs.emplace_back(builder.getStringAttr("signature"),
                           builder.getStringAttr(ctx.orbit_signature));
    }
    return builder.getDictionaryAttr(attrs);
}

void attachArgumentAttributes(mlir::func::FuncOp func,
                              mlir::OpBuilder& builder,
                              const NamedCutout& cutout,
                              std::size_t index,
                              const Cpp2MlirLoader::Options& options) {
    const auto hierarchyAttr = makeHierarchyAttr(builder, cutout.context);
    func.setArgAttr(index, "hierarchy_index", hierarchyAttr);
    func.setArgAttr(index, "cutout_name", builder.getStringAttr(cutout.name));
    func.setArgAttr(index, "source_span", makeSpanAttr(builder, cutout.span));
    if (options.attach_orbit_metadata) {
        func.setArgAttr(index, "orbit_hashes",
                        makeIntegerArrayAttr(builder, cutout.context.orbit_hashes));
        func.setArgAttr(index, "orbit_counts",
                        makeIntegerArrayAttr(builder, cutout.context.orbit_counts));
    }
}

void emitCutoutOperation(mlir::OpBuilder& builder,
                         const NamedCutout& cutout,
                         std::size_t index,
                         const Cpp2MlirLoader::Options& options) {
    mlir::OperationState state(builder.getUnknownLoc(), "cppfort.cutout");
    state.addAttribute("cutout_name", builder.getStringAttr(cutout.name));
    state.addAttribute("argument_index",
                       builder.getI64IntegerAttr(static_cast<int64_t>(index)));
    state.addAttribute("hierarchy_index", makeHierarchyAttr(builder, cutout.context));
    state.addAttribute("source_span", makeSpanAttr(builder, cutout.span));
    if (options.attach_orbit_metadata) {
        state.addAttribute("orbit_hashes",
                           makeIntegerArrayAttr(builder, cutout.context.orbit_hashes));
        state.addAttribute("orbit_counts",
                           makeIntegerArrayAttr(builder, cutout.context.orbit_counts));
    }
    state.addAttribute("hierarchy_label",
                       builder.getStringAttr(cutout.context.hierarchy_label));
    builder.create(state);
}

} // namespace

Cpp2MlirLoader::Cpp2MlirLoader(mlir::MLIRContext& context)
    : context_(context) {
    context_.allowUnregisteredDialects(true);
    context_.loadDialect<mlir::func::FuncDialect>();
}

mlir::OwningOpRef<mlir::ModuleOp> Cpp2MlirLoader::buildModule(
    const std::string& moduleName,
    const std::vector<NamedCutout>& cutouts,
    const Options& options) {

    mlir::OpBuilder builder(&context_);
    const auto location = builder.getUnknownLoc();
    mlir::ModuleOp module = mlir::ModuleOp::create(location);

    mlir::Type argType = builder.getIndexType();
    llvm::SmallVector<mlir::Type, 8> argumentTypes(cutouts.size(), argType);
    auto functionType = builder.getFunctionType(argumentTypes, {});

    builder.setInsertionPointToStart(module.getBody());
    const std::string symbol = moduleName.empty() ? "cpp2_capture" : moduleName;
    auto func = builder.create<mlir::func::FuncOp>(location, symbol, functionType);

    auto* entryBlock = func.addEntryBlock();
    mlir::OpBuilder attrBuilder(&context_);
    std::size_t index = 0;
    for (const auto& cutout : cutouts) {
        attachArgumentAttributes(func, attrBuilder, cutout, index, options);
        ++index;
    }

    mlir::OpBuilder bodyBuilder(&context_);
    bodyBuilder.setInsertionPointToStart(entryBlock);
    for (std::size_t i = 0; i < cutouts.size(); ++i) {
        emitCutoutOperation(bodyBuilder, cutouts[i], i, options);
    }
    bodyBuilder.create<mlir::func::ReturnOp>(location);

    module->setAttr("cppfort.cutout_count",
                    builder.getI64IntegerAttr(static_cast<int64_t>(cutouts.size())));
    func->setAttr("cppfort.cutout_count",
                  builder.getI64IntegerAttr(static_cast<int64_t>(cutouts.size())));

    if (!moduleName.empty()) {
        module->setAttr("cppfort.module_name", builder.getStringAttr(moduleName));
        func->setAttr("cppfort.module_name", builder.getStringAttr(moduleName));
    }

    return mlir::OwningOpRef<mlir::ModuleOp>(module);
}

std::string Cpp2MlirLoader::moduleToString(mlir::ModuleOp module) const {
    std::string buffer;
    llvm::raw_string_ostream os(buffer);
    module->print(os);
    os.flush();
    return buffer;
}

std::string sanitize_symbol(std::string_view input) {
    std::string sanitized;
    sanitized.reserve(input.size() + 1);
    for (const char ch : input) {
        if (std::isalnum(static_cast<unsigned char>(ch)) || ch == '_') {
            sanitized.push_back(ch);
        } else {
            sanitized.push_back('_');
        }
    }
    if (sanitized.empty()) {
        sanitized = "module";
    }
    if (std::isdigit(static_cast<unsigned char>(sanitized.front()))) {
        sanitized.insert(sanitized.begin(), '_');
    }
    return sanitized;
}

} // namespace cppfort::stage0
