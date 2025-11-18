#pragma once

#include <vector>
#include <string>
#include <string_view>
#include <cstddef>
#include "wide_scanner.h"
#include <mlir_region_node.h>

namespace cppfort {
namespace ir {

/**
 * RBCursiveRegions: Structural inference engine for carving regions from boundary streams
 * 
 * This implements Step 3 from TODO.md: "The Confix Inference Engine — Tuning the Frequency"
 * Repurposes RBCursiveScanner from text-matching to structural region carving.
 */
class RBCursiveRegions {
public:
    /**
     * Configuration for region carving
     */
    struct CarveConfig {
        double minConfidence = 0.6;      // Minimum confidence for region boundary
        size_t minRegionSize = 10;       // Minimum region size in characters
        bool enableWobbling = true;      // Enable wobbling window for confix deduction
        size_t maxNestingDepth = 100;    // Maximum nested region depth
        bool validateOrbitDepth = true;  // Validate using orbit confix depth
    };
    
    struct CarveResult {
        std::unique_ptr<mlir::RegionNode> rootRegion;
        std::vector<const mlir::RegionNode*> allRegions;
        size_t regionCount = 0;
        bool success = false;
        std::string errorMessage;
        
        CarveResult() = default;
    };
    
private:
    CarveConfig config_;
    std::vector<size_t> confixDepthStack_;  // Stack for tracking confix depth
    size_t currentDepth_ = 0;
    std::vector<std::pair<char, char>> confixPairs_ = {
        {'{', '}'},  // Object/scope boundaries
        {'[', ']'},  // Array boundaries
        {'(', ')'},  // Parameter boundaries
        {'<', '>'},  // Template/generic boundaries
    };
    
    // Helper: Check if a character is an opening confix
    bool isOpeningConfix(char ch) const;
    
    // Helper: Check if a character is a closing confix matching an opening
    bool isMatchingClosingConfix(char open, char close) const;
    
    // Helper: Get confix depth change for a character
    int getConfixDepthChange(char ch) const;
    
    // Wobbling window: Find valid region boundary by perturbing end position
    size_t wobbleFindBoundary(const std::vector<WideScanner::BoundaryEvent>& events,
                             size_t startIdx,
                             size_t initialEndIdx,
                             const std::string& source) const;
    
    // Recursive region carving
    std::unique_ptr<mlir::RegionNode> carveRegionRecursive(
        const std::vector<WideScanner::BoundaryEvent>& events,
        const std::string& source,
        size_t& currentIdx,
        size_t endIdx,
        mlir::RegionNode::RegionType parentType = mlir::RegionNode::RegionType::UNKNOWN);
    
    // Determine region type from boundary evidence
    mlir::RegionNode::RegionType inferRegionType(
        const WideScanner::BoundaryEvent& startBoundary,
        const std::vector<WideScanner::BoundaryEvent>& followingEvents) const;
    
    // Calculate confidence score for a region boundary
    double calculateBoundaryConfidence(
        const WideScanner::BoundaryEvent& event,
        size_t depth) const;
    
public:
    /**
     * Constructor
     */
    explicit RBCursiveRegions(const CarveConfig& config)
        : config_(config) {}
    
    /**
     * Core method: Carve regions from boundary event stream
     * 
     * @param events Enriched boundary stream from WideScanner
     * @param source Original source text for context
     * @return CarveResult containing region tree or error
     */
    CarveResult carveRegions(const std::vector<WideScanner::BoundaryEvent>& events,
                            const std::string& source);
    
    /**
     * Carve regions from a substring range
     * @param events Boundary event stream
     * @param source Source text
     * @param startPos Starting character position
     * @param endPos Ending character position
     * @return CarveResult
     */
    CarveResult carveRegionsInSpan(const std::vector<WideScanner::BoundaryEvent>& events,
                                  const std::string& source,
                                  size_t startPos,
                                  size_t endPos);
    
    /**
     * Get configuration
     */
    const CarveConfig& getConfig() const { return config_; }
    
    /**
     * Update configuration
     */
    void updateConfig(const CarveConfig& config) { config_ = config; }
    
    /**
     * Debug helper: Print carved region tree
     */
    static void printCarvedRegions(const mlir::RegionNode& root, size_t indent = 0);
};

/**
 * Helper function to create a BoundaryEvent stream from source text
 * Convenience function for when you have raw text without pre-computed boundaries
 */
std::vector<WideScanner::BoundaryEvent> generateBoundaryEventsFromSource(
    const std::string& source,
    bool includeStructuralChars = true);

/**
 * Standalone carving function for quick region extraction
 */
RBCursiveRegions::CarveResult carveRegionsStandalone(
    const std::string& source,
    const RBCursiveRegions::CarveConfig& config = RBCursiveRegions::CarveConfig());

} // namespace ir
} // namespace cppfort
