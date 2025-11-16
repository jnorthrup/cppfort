#include "semantic_pipeline.h"
#include <iostream>

namespace cppfort {
namespace stage0 {

void SemanticPipeline::initializeStages() {
    // Stage 1: WideScanner
    scanner_ = std::make_unique<WideScanner>(config_.scanConfig);

    // Stage 2: RBCursiveRegions
    carver_ = std::make_unique<ir::RBCursiveRegions>(config_.carveConfig);

    // Stage 3: PatternApplier
    if (!config_.patternsPath.empty()) {
        applier_ = std::make_unique<PatternApplier>(config_.patternsPath);
        applier_->setConfidenceThreshold(config_.patternConfidenceThreshold);
        applier_->setDebug(config_.enableDebug);

        if (!applier_->initialize()) {
            std::cerr << "[SemanticPipeline] Warning: Failed to initialize PatternApplier\n";
            applier_.reset();
        }
    }

    // Stage 4: GraphToMlirWalker
    walker_ = std::make_unique<GraphToMlirWalker>(context_, config_.walkerConfig);
}

bool SemanticPipeline::executeScan(const std::string& source, PipelineResult& result) {
    if (config_.enableDebug) {
        std::cerr << "[SemanticPipeline] Stage 1: Scanning source (" << source.length() << " bytes)\n";
    }

    // Execute WideScanner to produce enriched boundary stream
    result.boundaryEvents = scanner_->scanAnchorsWithOrbits(source);
    result.boundaryCount = result.boundaryEvents.size();

    if (result.boundaryEvents.empty()) {
        result.errorMessage = "WideScanner produced no boundary events";
        result.failedStage = "scan";
        return false;
    }

    if (config_.enableDebug) {
        std::cerr << "[SemanticPipeline] Stage 1 complete: " << result.boundaryCount << " boundaries\n";
    }

    return true;
}

bool SemanticPipeline::executeCarve(const std::string& source, PipelineResult& result) {
    if (config_.enableDebug) {
        std::cerr << "[SemanticPipeline] Stage 2: Carving regions from "
                  << result.boundaryCount << " boundaries\n";
    }

    // Execute RBCursiveRegions to carve semantic regions
    auto carveResult = carver_->carveRegions(result.boundaryEvents, source);

    if (!carveResult.success) {
        result.errorMessage = "Region carving failed: " + carveResult.errorMessage;
        result.failedStage = "carve";
        return false;
    }

    result.regionGraph = std::move(carveResult.rootRegion);
    result.regionCount = carveResult.regionCount;

    if (!result.regionGraph) {
        result.errorMessage = "Region carving produced no root region";
        result.failedStage = "carve";
        return false;
    }

    if (config_.enableDebug) {
        std::cerr << "[SemanticPipeline] Stage 2 complete: " << result.regionCount << " regions\n";
        ir::RBCursiveRegions::printCarvedRegions(*result.regionGraph);
    }

    return true;
}

bool SemanticPipeline::executeLabel(const std::string& source, PipelineResult& result) {
    if (config_.enableDebug) {
        std::cerr << "[SemanticPipeline] Stage 3: Labeling " << result.regionCount << " regions\n";
    }

    if (!applier_) {
        result.errorMessage = "PatternApplier not initialized";
        result.failedStage = "label";
        return false;
    }

    // Apply patterns to entire region tree
    result.regionsLabeled = applier_->applyPatternsToTree(*result.regionGraph, source);

    if (config_.enableDebug) {
        std::cerr << "[SemanticPipeline] Stage 3 complete: " << result.regionsLabeled
                  << " regions labeled\n";
    }

    // Validate labeled tree if requested
    if (config_.validateIntermediate) {
        std::vector<std::string> validationErrors;
        if (!validateLabeledTree(*result.regionGraph, validationErrors)) {
            result.errorMessage = "Labeled tree validation failed: " +
                                 std::to_string(validationErrors.size()) + " errors";
            result.failedStage = "label";

            if (config_.enableDebug) {
                for (const auto& err : validationErrors) {
                    std::cerr << "  Validation error: " << err << "\n";
                }
            }

            // Don't fail on validation errors, just warn
            if (config_.enableDebug) {
                std::cerr << "[SemanticPipeline] Warning: Validation errors found, continuing anyway\n";
            }
        }
    }

    return true;
}

bool SemanticPipeline::executeGenerate(const std::string& moduleName, PipelineResult& result) {
    if (config_.enableDebug) {
        std::cerr << "[SemanticPipeline] Stage 4: Generating MLIR module '" << moduleName << "'\n";
    }

    // Execute GraphToMlirWalker to generate MLIR
    auto walkerResult = walker_->generateModule(*result.regionGraph, moduleName);

    if (!walkerResult.success) {
        result.errorMessage = "MLIR generation failed: " + walkerResult.errorMessage;
        result.failedStage = "generate";
        return false;
    }

    result.mlirModule = std::move(walkerResult.module);
    result.opsGenerated = walkerResult.opsGenerated;

    if (config_.enableDebug) {
        std::cerr << "[SemanticPipeline] Stage 4 complete: " << result.opsGenerated
                  << " operations generated\n";
    }

    return true;
}

SemanticPipeline::PipelineResult SemanticPipeline::execute(
    const std::string& source,
    const std::string& moduleName) {

    PipelineResult result;

    // Stage 1: Scan
    if (!executeScan(source, result)) {
        return result;
    }

    // Stage 2: Carve
    if (!executeCarve(source, result)) {
        return result;
    }

    // Stage 3: Label
    if (!executeLabel(source, result)) {
        return result;
    }

    // Stage 4: Generate
    if (!executeGenerate(moduleName, result)) {
        return result;
    }

    // Success
    result.success = true;

    if (config_.enableDebug) {
        std::cerr << "[SemanticPipeline] Pipeline complete:\n"
                  << "  Boundaries: " << result.boundaryCount << "\n"
                  << "  Regions: " << result.regionCount << "\n"
                  << "  Labeled: " << result.regionsLabeled << "\n"
                  << "  Operations: " << result.opsGenerated << "\n";
    }

    return result;
}

SemanticPipeline::PipelineResult SemanticPipeline::executePartial(
    const std::string& source,
    const std::string& stopAfter) {

    PipelineResult result;

    // Stage 1: Scan
    if (!executeScan(source, result)) {
        return result;
    }
    if (stopAfter == "scan") {
        result.success = true;
        return result;
    }

    // Stage 2: Carve
    if (!executeCarve(source, result)) {
        return result;
    }
    if (stopAfter == "carve") {
        result.success = true;
        return result;
    }

    // Stage 3: Label
    if (!executeLabel(source, result)) {
        return result;
    }
    if (stopAfter == "label") {
        result.success = true;
        return result;
    }

    // Stage 4: Generate
    if (!executeGenerate("partial_module", result)) {
        return result;
    }

    result.success = true;
    return result;
}

// Convenience function
SemanticPipeline::PipelineResult transpileCpp2ToMlir(
    const std::string& source,
    mlir::MLIRContext& context,
    const std::string& moduleName,
    const SemanticPipeline::PipelineConfig& config) {

    SemanticPipeline pipeline(context, config);
    return pipeline.execute(source, moduleName);
}

} // namespace stage0
} // namespace cppfort
