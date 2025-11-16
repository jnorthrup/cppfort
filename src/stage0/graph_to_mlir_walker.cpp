#include "graph_to_mlir_walker.h"
#include <mlir/IR/Verifier.h>
#include <mlir/IR/BuiltinTypes.h>
#include <llvm/Support/raw_ostream.h>
#include <iostream>

namespace cppfort {
namespace stage0 {

GraphToMlirWalker::WalkerResult GraphToMlirWalker::generateModule(
    const ir::mlir::RegionNode& root,
    const std::string& moduleName) {

    WalkerResult result;

    // Reset state
    valueMap_.clear();
    regionMap_.clear();
    opsGenerated_ = 0;
    regionsProcessed_ = 0;

    // Create module operation
    mlir::Location loc = builder_->getUnknownLoc();
    mlir::OwningOpRef<mlir::ModuleOp> moduleOp = mlir::ModuleOp::create(loc, moduleName);

    if (!moduleOp) {
        result.errorMessage = "Failed to create module operation";
        return result;
    }

    // Get module body block
    mlir::Block* moduleBlock = moduleOp->getBody();

    // Walk the region tree and generate operations
    try {
        walkRegion(root, moduleBlock);
    } catch (const std::exception& e) {
        result.errorMessage = std::string("Exception during IR generation: ") + e.what();
        return result;
    }

    // Validate generated module if requested
    if (config_.validateIR) {
        std::string validationError;
        if (!validateModule(*moduleOp, validationError)) {
            result.errorMessage = "IR validation failed: " + validationError;
            return result;
        }
    }

    // Success
    result.module = std::move(moduleOp);
    result.success = true;
    result.opsGenerated = opsGenerated_;
    result.regionsProcessed = regionsProcessed_;

    return result;
}

void GraphToMlirWalker::walkRegion(const ir::mlir::RegionNode& node, mlir::Block* parentBlock) {
    regionsProcessed_++;

    // Set insertion point
    builder_->setInsertionPointToEnd(parentBlock);

    // Generate appropriate MLIR construct based on region type
    switch (node.getType()) {
        case ir::mlir::RegionNode::RegionType::FUNCTION: {
            auto funcOp = generateFunction(node, parentBlock);
            if (funcOp) {
                opsGenerated_++;
            }
            break;
        }

        case ir::mlir::RegionNode::RegionType::BLOCK:
        case ir::mlir::RegionNode::RegionType::CONDITIONAL:
        case ir::mlir::RegionNode::RegionType::LOOP: {
            // For non-function regions, generate operations directly
            for (const auto& op : node.getOperations()) {
                auto mlirOp = generateOperation(op, parentBlock, node);
                if (mlirOp) {
                    opsGenerated_++;
                }
            }

            // Recursively process child regions
            for (const auto& child : node.getChildren()) {
                walkRegion(*child, parentBlock);
            }
            break;
        }

        default:
            // Unknown or unhandled region type
            break;
    }
}

mlir::func::FuncOp GraphToMlirWalker::generateFunction(
    const ir::mlir::RegionNode& node,
    mlir::Block* moduleBlock) {

    builder_->setInsertionPointToEnd(moduleBlock);

    // Extract function name
    std::string funcName = node.getName();
    if (funcName.empty()) {
        funcName = "anonymous_func_" + std::to_string(regionsProcessed_);
    }

    // Build function type
    llvm::SmallVector<mlir::Type, 4> argTypes;
    llvm::SmallVector<mlir::Type, 1> resultTypes;

    // Parse arguments from region
    for (const auto& arg : node.getArguments()) {
        // For now, default to i32 type
        // TODO: Parse actual types from argument strings
        argTypes.push_back(builder_->getI32Type());
    }

    // Parse return type from attributes or default to void
    auto attrs = node.getMlirAttributes();
    if (attrs.count("return_type") > 0) {
        const auto& returnTypeStr = attrs.at("return_type");
        mlir::Type returnType = mapTypeString(returnTypeStr);
        if (returnType) {
            resultTypes.push_back(returnType);
        }
    }

    mlir::FunctionType funcType = builder_->getFunctionType(argTypes, resultTypes);
    mlir::Location loc = generateLocation(node);

    // Create function operation
    auto funcOp = builder_->create<mlir::func::FuncOp>(loc, funcName, funcType);

    // Create entry block
    mlir::Block* entryBlock = funcOp.addEntryBlock();

    // Generate operations in function body
    builder_->setInsertionPointToEnd(entryBlock);

    for (const auto& op : node.getOperations()) {
        // Skip the func.func operation itself if it's in the operations list
        if (op.name == "func.func") {
            continue;
        }

        generateOperation(op, entryBlock, node);
    }

    // Process child regions (function body blocks)
    for (const auto& child : node.getChildren()) {
        if (child->getType() == ir::mlir::RegionNode::RegionType::BLOCK) {
            // Generate block contents directly into entry block
            for (const auto& childOp : child->getOperations()) {
                generateOperation(childOp, entryBlock, *child);
            }
        }
    }

    // Add terminator if missing
    if (entryBlock->empty() || !entryBlock->back().hasTrait<mlir::OpTrait::IsTerminator>()) {
        if (resultTypes.empty()) {
            builder_->create<mlir::func::ReturnOp>(loc);
        }
    }

    return funcOp;
}

mlir::Block* GraphToMlirWalker::generateBlock(
    const ir::mlir::RegionNode& node,
    mlir::Region& parentRegion) {

    mlir::Block* block = new mlir::Block();
    parentRegion.push_back(block);

    // Add block arguments
    for (const auto& arg : node.getArguments()) {
        // Default to i32 for now
        mlir::Type argType = builder_->getI32Type();
        block->addArgument(argType, builder_->getUnknownLoc());
    }

    builder_->setInsertionPointToEnd(block);

    // Generate operations
    for (const auto& op : node.getOperations()) {
        generateOperation(op, block, node);
    }

    regionMap_[&node] = &parentRegion;

    return block;
}

mlir::Operation* GraphToMlirWalker::generateOperation(
    const ir::mlir::RegionNode::Operation& op,
    mlir::Block* block,
                                      const ir::mlir::RegionNode& parentRegion) {

    builder_->setInsertionPointToEnd(block);
    mlir::Location loc = builder_->getUnknownLoc();

    // Map operation name to MLIR operation
    if (op.name.empty()) {
        return nullptr;
    }

    // Handle specific operation types
    if (op.name == "arith.addi") {
        // Example: create integer addition
        // Would need actual operands from valueMap_
        // For now, create placeholder constants
        auto lhs = builder_->create<mlir::arith::ConstantIntOp>(loc, 0, builder_->getI32Type());
        auto rhs = builder_->create<mlir::arith::ConstantIntOp>(loc, 0, builder_->getI32Type());
        auto addOp = builder_->create<mlir::arith::AddIOp>(loc, lhs, rhs);
        return addOp.getOperation();
    }

    if (op.name == "func.return") {
        // Create return operation
        llvm::SmallVector<mlir::Value, 1> returnValues;
        // TODO: Map operand indices to actual values
        auto returnOp = builder_->create<mlir::func::ReturnOp>(loc, returnValues);
        return returnOp.getOperation();
    }

    // Generic operation creation (for custom dialects)
    mlir::OperationState state(loc, op.name);

    // Add operands
    for (size_t operandIdx : op.operand_indices) {
        if (valueMap_.count(operandIdx) > 0) {
            state.addOperands(valueMap_[operandIdx]);
        }
    }

    // Add result types
    if (op.result_index != SIZE_MAX) {
        const auto* resultVal = parentRegion.getValue(op.result_index);
        if (resultVal) {
            mlir::Type resultType = mapTypeString(resultVal->type);
            if (resultType) {
                state.addTypes(resultType);
            }
        }
    }

    // Add attributes
    for (const auto& [key, value] : op.attributes) {
        // Convert string attributes to MLIR attributes
        mlir::Attribute attr = builder_->getStringAttr(value);
        state.addAttribute(key, attr);
    }

    mlir::Operation* mlirOp = builder_->create(state);

    // Map result to value map
    if (mlirOp->getNumResults() > 0 && op.result_index != SIZE_MAX) {
        valueMap_[op.result_index] = mlirOp->getResult(0);
    }

    return mlirOp;
}

mlir::Value GraphToMlirWalker::createValue(
    const ir::mlir::RegionNode::Value& val,
    mlir::Block* block) {

    builder_->setInsertionPointToEnd(block);

    // Create a constant or block argument based on value characteristics
    mlir::Type type = mapTypeString(val.type);
    if (!type) {
        type = builder_->getI32Type();  // Default fallback
    }

    // For now, create a placeholder constant
    // In a real implementation, this would depend on the value's defining operation
    if (type.isInteger(32)) {
        auto constOp = builder_->create<mlir::arith::ConstantIntOp>(
            builder_->getUnknownLoc(), 0, type);
        return constOp.getResult();
    }

    return mlir::Value();
}

mlir::Operation* GraphToMlirWalker::createRegionOperation(
    const ir::mlir::RegionNode& node,
    mlir::Block* parentBlock) {

    // This method creates operations that themselves contain regions
    // Examples: scf.if, scf.while, etc.

    builder_->setInsertionPointToEnd(parentBlock);
    mlir::Location loc = generateLocation(node);

    // For now, return nullptr for unsupported region operations
    return nullptr;
}

mlir::Location GraphToMlirWalker::generateLocation(const ir::mlir::RegionNode& node) {
    if (config_.emitDebugInfo) {
        // Create file location with line/column information
        // For now, use source positions as line numbers
        size_t line = node.getSourceStart();
        size_t col = 0;

        std::string filename = "cppfort_source.cpp2";
        return mlir::FileLineColLoc::get(builder_->getStringAttr(filename), line, col);
    }

    return builder_->getUnknownLoc();
}

mlir::Type GraphToMlirWalker::mapTypeString(const std::string& typeStr) {
    // Map string type representations to MLIR types
    if (typeStr.empty() || typeStr == "unknown") {
        return nullptr;
    }

    if (typeStr == "i32" || typeStr == "int") {
        return builder_->getI32Type();
    }
    if (typeStr == "i64" || typeStr == "long") {
        return builder_->getI64Type();
    }
    if (typeStr == "f32" || typeStr == "float") {
        return builder_->getF32Type();
    }
    if (typeStr == "f64" || typeStr == "double") {
        return builder_->getF64Type();
    }
    if (typeStr == "void") {
        return builder_->getNoneType();
    }
    if (typeStr == "bool") {
        return builder_->getI1Type();
    }

    // Default: try to parse as integer type
    if (typeStr.length() > 1 && typeStr[0] == 'i') {
        try {
            unsigned width = std::stoi(typeStr.substr(1));
            return builder_->getIntegerType(width);
        } catch (...) {
            // Fall through
        }
    }

    return nullptr;
}

bool GraphToMlirWalker::validateModule(mlir::ModuleOp module, std::string& outError) {
    // Use MLIR's built-in verification
    if (mlir::failed(mlir::verify(module))) {
        std::string errorStr;
        llvm::raw_string_ostream errorStream(errorStr);
        module.print(errorStream);
        errorStream.flush();
        outError = "MLIR verification failed. Module:\n" + errorStr;
        return false;
    }

    return true;
}

// Standalone function
GraphToMlirWalker::WalkerResult generateMlirFromGraph(
    const ir::mlir::RegionNode& root,
    mlir::MLIRContext& context,
    const std::string& moduleName,
    const GraphToMlirWalker::WalkerConfig& config) {

    GraphToMlirWalker walker(context, config);
    return walker.generateModule(root, moduleName);
}

} // namespace stage0
} // namespace cppfort
