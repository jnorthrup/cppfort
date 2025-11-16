#pragma once

#include <mlir_region_node.h>
#include <mlir/IR/Builders.h>
#include <mlir/IR/BuiltinOps.h>
#include <mlir/IR/MLIRContext.h>
#include <mlir/IR/OwningOpRef.h>
#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/Dialect/Arith/IR/Arith.h>
#include <memory>
#include <string>
#include <unordered_map>

namespace cppfort {
namespace stage0 {

/**
 * GraphToMlirWalker: Step 5 from TODO.md
 *
 * This component performs a deterministic walk over the RegionNode graph
 * to generate MLIR. It traverses the terraced field structure and uses
 * mlir::OpBuilder to create corresponding MLIR constructs.
 *
 * The walker implements the final assembly step:
 *   RegionNode graph -> MLIR IR
 */
class GraphToMlirWalker {
public:
    /**
     * Configuration for MLIR generation
     */
    struct WalkerConfig {
        bool emitDebugInfo = true;           // Emit location information
        bool validateIR = true;              // Validate generated IR
        bool enableOptimizations = false;    // Apply basic optimizations
        std::string targetDialect = "func";  // Primary dialect to target

        WalkerConfig() = default;
    };

    /**
     * Result of MLIR generation
     */
    struct WalkerResult {
        mlir::OwningOpRef<mlir::ModuleOp> module;
        bool success = false;
        std::string errorMessage;
        size_t opsGenerated = 0;
        size_t regionsProcessed = 0;

        WalkerResult() = default;
    };

private:
    mlir::MLIRContext& context_;
    WalkerConfig config_;
    std::unique_ptr<mlir::OpBuilder> builder_;

    // Value mapping: RegionNode::Value index -> MLIR Value
    std::unordered_map<size_t, mlir::Value> valueMap_;

    // Region mapping: RegionNode* -> mlir::Region*
    std::unordered_map<const ir::mlir::RegionNode*, mlir::Region*> regionMap_;

    // Operation statistics
    size_t opsGenerated_ = 0;
    size_t regionsProcessed_ = 0;

public:
    /**
     * Constructor
     * @param context MLIR context for IR generation
     * @param config Walker configuration
     */
    explicit GraphToMlirWalker(mlir::MLIRContext& context,
                              const WalkerConfig& config = WalkerConfig())
        : context_(context), config_(config) {
        builder_ = std::make_unique<mlir::OpBuilder>(&context);
    }

    /**
     * Generate MLIR module from RegionNode graph
     * @param root Root of the RegionNode tree
     * @param moduleName Name for the generated module
     * @return WalkerResult containing MLIR module or error
     */
    WalkerResult generateModule(const ir::mlir::RegionNode& root,
                               const std::string& moduleName = "cppfort_module");

    /**
     * Get current configuration
     */
    const WalkerConfig& getConfig() const { return config_; }

    /**
     * Update configuration
     */
    void setConfig(const WalkerConfig& config) { config_ = config; }

private:
    /**
     * Walk region tree recursively and generate MLIR
     */
    void walkRegion(const ir::mlir::RegionNode& node, mlir::Block* parentBlock);

    /**
     * Generate MLIR function from function region
     */
    mlir::func::FuncOp generateFunction(const ir::mlir::RegionNode& node,
                                       mlir::Block* moduleBlock);

    /**
     * Generate MLIR block from block region
     */
    mlir::Block* generateBlock(const ir::mlir::RegionNode& node, mlir::Region& parentRegion);

    /**
     * Generate MLIR operation from OpStub
     */
    mlir::Operation* generateOperation(const ir::mlir::RegionNode::Operation& op,
                                      mlir::Block* block,
                                      const ir::mlir::RegionNode& parentRegion);

    /**
     * Create MLIR value from ValueStub
     */
    mlir::Value createValue(const ir::mlir::RegionNode::Value& val,
                           mlir::Block* block);

    /**
     * Map RegionNode::RegionType to appropriate MLIR operation
     */
    mlir::Operation* createRegionOperation(const ir::mlir::RegionNode& node,
                                          mlir::Block* parentBlock);

    /**
     * Generate location information from source positions
     */
    mlir::Location generateLocation(const ir::mlir::RegionNode& node);

    /**
     * Map dialect name to MLIR type
     */
    mlir::Type mapTypeString(const std::string& typeStr);

    /**
     * Validate generated MLIR
     */
    bool validateModule(mlir::ModuleOp module, std::string& outError);
};

/**
 * Standalone function: Generate MLIR from RegionNode tree
 */
GraphToMlirWalker::WalkerResult generateMlirFromGraph(
    const ir::mlir::RegionNode& root,
    mlir::MLIRContext& context,
    const std::string& moduleName = "cppfort_module",
    const GraphToMlirWalker::WalkerConfig& config = GraphToMlirWalker::WalkerConfig());

} // namespace stage0
} // namespace cppfort
