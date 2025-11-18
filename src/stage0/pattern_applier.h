#pragma once

#include <string>
#include <vector>
#include <unordered_map>
#include <memory>
#include <filesystem>
#include <mlir_region_node.h>
#include "rbcursive.h"
#include "pijul_graph_matcher.h"
#include "pijul_parameter_graph.h"
#include "pijul_orbit_builder.h"

namespace cppfort {
namespace stage0 {

struct PatternData;  // Forward declaration

/**
 * PatternApplier: Step 4 from TODO.md
 * 
 * This component operates on the text content of a RegionNode and uses
 * YAML/JSON patterns not for brittle text substitution, but for robust
 * semantic labeling of the regions.
 */
class PatternApplier {
public:
    struct ApplicationResult {
        bool success = false;
        std::string matchedPatternName;
        std::unordered_map<std::string, std::string> capturedSpans;
        std::unordered_map<std::string, double> confidenceScores;
        std::string errorMessage;
        
        ApplicationResult() = default;
    };
    
private:
    std::filesystem::path patternsPath_;
    std::vector<PatternData> patterns_;
    double confidenceThreshold_ = 0.6;
    bool enableDebug_ = false;
    
    // Pattern matching engines
    std::unique_ptr<cppfort::ir::RBCursiveScanner> rbcursiveScanner_;
    // Graph matchers per pattern
    std::vector<std::unique_ptr<PijulGraphMatcher>> graphMatchers_;
    // Parameter graph to collect semantic anchors
    ::cppfort::pijul::ParameterGraph parameterGraph_;
    
    // Helper: Load patterns from JSON/YAML files
    bool loadPatterns();
    
    // Helper: Find best matching pattern for a region
    const PatternData* findBestPattern(
        const std::string& regionContent,
        double& outConfidence) const;
    
    // Helper: Extract evidence spans using alternating anchors
    std::unordered_map<std::string, std::string> extractEvidenceSpans(
        const std::string& content,
        const PatternData& pattern) const;
    
    // Helper: Apply pattern to populate RegionNode
    void populateRegionFromPattern(
        ir::mlir::RegionNode& region,
        const PatternData& pattern,
        const std::unordered_map<std::string, std::string>& captures);
    
    // Helper: Map pattern name to RegionType
    ir::mlir::RegionNode::RegionType patternNameToRegionType(const std::string& patternName) const;
    
    // Helper: Infer MLIR dialect from pattern and captures
    std::string inferMlirDialect(
        const PatternData& pattern,
        const std::unordered_map<std::string, std::string>& captures) const;
    
public:
    /**
     * Constructor
     * @param patternsPath Directory or file containing pattern definitions
     */
    explicit PatternApplier(const std::filesystem::path& patternsPath);
    
    /**
     * Destructor
     */
    ~PatternApplier();
    
    /**
     * Initialize the applier - load and validate patterns
     * @return true if successful
     */
    bool initialize();
    
    /**
     * Apply patterns to classify and populate a RegionNode
     * This is the core method for semantic labeling
     * 
     * @param region RegionNode to populate (text content must be accessible via source)
     * @param source Full source text for context
     * @param contextHints Optional hints about expected type (e.g., "function", "parameter")
     * @return ApplicationResult with captured data or error
     */
    ApplicationResult applyPatternToRegion(
        ir::mlir::RegionNode& region,
        const std::string& source,
        const std::vector<std::string>& contextHints = {});
    
    /**
     * Apply patterns to all regions in a tree recursively
     * @param root Root of the RegionNode tree
     * @param source Full source text
     * @return Number of regions successfully classified
     */
    size_t applyPatternsToTree(
        ir::mlir::RegionNode& root,
        const std::string& source);
    
    /**
     * Apply a specific pattern by name to a region
     * @param region Region to populate
     * @param patternName Name of pattern to apply
     * @param source Full source text
     * @return ApplicationResult
     */
    ApplicationResult applySpecificPattern(
        ir::mlir::RegionNode& region,
        const std::string& patternName,
        const std::string& source);
    
    /**
     * Get loaded patterns
     */
    const std::vector<PatternData>& getPatterns() const { return patterns_; }
    
    /**
     * Update confidence threshold
     */
    void setConfidenceThreshold(double threshold) { confidenceThreshold_ = threshold; }
    double getConfidenceThreshold() const { return confidenceThreshold_; }
    
    /**
     * Enable/disable debug output
     */
    void setDebug(bool enable) { enableDebug_ = enable; }
    bool getDebug() const { return enableDebug_; }

    // Retrieve parameter graph constructed by pattern applier
    const ::cppfort::pijul::ParameterGraph& getParameterGraph() const { return parameterGraph_; }
};

/**
 * Helper function for quick pattern application
 */
PatternApplier::ApplicationResult applyPatternStandalone(
    const std::string& content,
    const std::filesystem::path& patternsPath,
    const std::vector<std::string>& contextHints = {});

/**
 * Validation function: Check if a RegionNode tree is properly labeled
 */
bool validateLabeledTree(
    const ir::mlir::RegionNode& root,
    std::vector<std::string>& outErrors);

} // namespace stage0
} // namespace cppfort
