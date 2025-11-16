#pragma once

#include "wide_scanner.h"
#include "rbcursive_regions.h"
#include "pattern_applier.h"
#include "graph_to_mlir_walker.h"
#include <mlir_region_node.h>

#include <mlir/IR/MLIRContext.h>
#include <mlir/IR/BuiltinOps.h>
#include <mlir/IR/OwningOpRef.h>
#include <string>
#include <memory>
#include <filesystem>

namespace cppfort {
namespace stage0 {

/**
 * SemanticPipeline: Integrated 4-stage pipeline
 *
 * This orchestrates the complete transformation flow from TODO.md:
 *   1. WideScanner: Character plasma -> enriched BoundaryEvent stream
 *   2. RBCursiveRegions: Boundary stream -> terraced RegionNode graph via confix inference
 *   3. PatternApplier: RegionNode graph -> semantically labeled graph
 *   4. GraphToMlirWalker: Labeled graph -> MLIR module
 *
 * This replaces the brittle text-based approach in cpp2_emitter.cpp
 */
class SemanticPipeline {
public:
    /**
     * Pipeline configuration
     */
    struct PipelineConfig {
        // Stage 1: WideScanner
        WideScanner::ScanConfig scanConfig;

        // Stage 2: RBCursiveRegions
        ir::RBCursiveRegions::CarveConfig carveConfig;

        // Stage 3: PatternApplier
        std::filesystem::path patternsPath;
        double patternConfidenceThreshold = 0.6;

        // Stage 4: GraphToMlirWalker
        GraphToMlirWalker::WalkerConfig walkerConfig;

        // General
        bool enableDebug = false;
        bool validateIntermediate = true;

        PipelineConfig() {
            // Set default patterns path
            patternsPath = "patterns/cppfort_core_patterns.yaml";
        }
    };

    /**
     * Pipeline execution result
     */
    struct PipelineResult {
        bool success = false;
        std::string errorMessage;
        std::string failedStage;  // Which stage failed: "scan", "carve", "label", "generate"

        // Intermediate products (for debugging/inspection)
        std::vector<WideScanner::BoundaryEvent> boundaryEvents;
        std::unique_ptr<ir::mlir::RegionNode> regionGraph;
        size_t regionsLabeled = 0;

        // Final product
        mlir::OwningOpRef<mlir::ModuleOp> mlirModule;

        // Statistics
        size_t boundaryCount = 0;
        size_t regionCount = 0;
        size_t opsGenerated = 0;

        PipelineResult() = default;
    };

private:
    mlir::MLIRContext& context_;
    PipelineConfig config_;

    // Pipeline stages
    std::unique_ptr<WideScanner> scanner_;
    std::unique_ptr<ir::RBCursiveRegions> carver_;
    std::unique_ptr<PatternApplier> applier_;
    std::unique_ptr<GraphToMlirWalker> walker_;

public:
    /**
     * Constructor
     * @param context MLIR context for IR generation
     * @param config Pipeline configuration
     */
    explicit SemanticPipeline(mlir::MLIRContext& context,
                             const PipelineConfig& config = PipelineConfig())
        : context_(context), config_(config) {
        initializeStages();
    }

    /**
     * Execute full pipeline: source -> MLIR
     * @param source Cpp2 source code
     * @param moduleName Name for generated MLIR module
     * @return PipelineResult with MLIR module or error
     */
    PipelineResult execute(const std::string& source,
                          const std::string& moduleName = "cppfort_module");

    /**
     * Execute pipeline up to a specific stage (for debugging)
     * @param source Cpp2 source code
     * @param stopAfter Stop after this stage: "scan", "carve", "label", or "generate"
     * @return Partial pipeline result
     */
    PipelineResult executePartial(const std::string& source,
                                  const std::string& stopAfter);

    /**
     * Get configuration
     */
    const PipelineConfig& getConfig() const { return config_; }

    /**
     * Update configuration
     */
    void setConfig(const PipelineConfig& config) {
        config_ = config;
        initializeStages();
    }

private:
    /**
     * Initialize pipeline stages based on configuration
     */
    void initializeStages();

    /**
     * Stage 1: Scan source into boundary events
     */
    bool executeScan(const std::string& source, PipelineResult& result);

    /**
     * Stage 2: Carve regions from boundary stream
     */
    bool executeCarve(const std::string& source, PipelineResult& result);

    /**
     * Stage 3: Apply patterns to label regions
     */
    bool executeLabel(const std::string& source, PipelineResult& result);

    /**
     * Stage 4: Generate MLIR from labeled graph
     */
    bool executeGenerate(const std::string& moduleName, PipelineResult& result);
};

/**
 * Convenience function: Execute complete pipeline in one call
 */
SemanticPipeline::PipelineResult transpileCpp2ToMlir(
    const std::string& source,
    mlir::MLIRContext& context,
    const std::string& moduleName = "cppfort_module",
    const SemanticPipeline::PipelineConfig& config = SemanticPipeline::PipelineConfig());

} // namespace stage0
} // namespace cppfort
