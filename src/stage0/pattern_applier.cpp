#include "pattern_applier.h"
#include "pattern_loader.h"
#include "rbcursive.h"
#include <mlir_region_node.h>
#include "iterpeeps.h"
#include "pijul_parameter_graph.h"
#include "pijul_orbit_builder.h"
#include <iostream>
#include <algorithm>
#include <fstream>

namespace cppfort {
namespace stage0 {

PatternApplier::PatternApplier(const std::filesystem::path& patternsPath)
    : patternsPath_(patternsPath) {
    rbcursiveScanner_ = std::make_unique<cppfort::ir::RBCursiveScanner>();
}

PatternApplier::~PatternApplier() = default;

bool PatternApplier::loadPatterns() {
    patterns_.clear();

    PatternLoader loader;

    // Handle both directory and file paths
    if (std::filesystem::is_directory(patternsPath_)) {
        // Load all YAML pattern files in directory
        for (const auto& entry : std::filesystem::directory_iterator(patternsPath_)) {
            if (entry.path().extension() == ".yaml" || entry.path().extension() == ".yml") {
                if (enableDebug_) {
                    std::cerr << "[PatternApplier] Loading patterns from: " << entry.path() << "\n";
                }
                if (!loader.load_yaml(entry.path().string())) {
                    std::cerr << "[PatternApplier] Failed to load patterns from: " << entry.path() << "\n";
                }
            }
        }
    } else if (std::filesystem::is_regular_file(patternsPath_)) {
        // Load single pattern file
        if (!loader.load_yaml(patternsPath_.string())) {
            std::cerr << "[PatternApplier] Failed to load patterns from: " << patternsPath_ << "\n";
            return false;
        }
    } else {
        std::cerr << "[PatternApplier] Invalid patterns path: " << patternsPath_ << "\n";
        return false;
    }

    patterns_ = loader.patterns();
    // Create graph matchers for loaded patterns
    graphMatchers_.clear();
    for (const auto& p : patterns_) {
        graphMatchers_.push_back(std::make_unique<PijulGraphMatcher>(p));
    }

    if (enableDebug_) {
        std::cerr << "[PatternApplier] Loaded " << patterns_.size() << " patterns\n";
    }

    return !patterns_.empty();
}

bool PatternApplier::initialize() {
    return loadPatterns();
}

const PatternData* PatternApplier::findBestPattern(
    const std::string& regionContent,
    double& outConfidence) const {

    const PatternData* bestPattern = nullptr;
    double bestConfidence = 0.0;
    int bestPriority = -1;

    for (size_t i = 0; i < patterns_.size(); ++i) {
        const auto& pattern = patterns_[i];
        // Skip patterns without alternating anchors (old-style patterns)
        if (!pattern.use_alternating || pattern.alternating_anchors.empty()) {
            continue;
        }

        // Attempt to match pattern using rbcursive scanner
        double confidence = 0.0;

        // Use GraphMatcher if available for more robust matching
        if (i < graphMatchers_.size() && graphMatchers_[i]) {
            auto matches = graphMatchers_[i]->find_matches(regionContent);
            if (!matches.empty()) {
                // Basic confidence: total matched length / region length, weighted by pattern weight
                size_t totalLen = 0;
                for (const auto& m : matches) totalLen += (m.end_pos - m.start_pos);
                confidence = static_cast<double>(totalLen) / std::max<size_t>(1, regionContent.size());
                confidence *= pattern.weight;
                if (pattern.priority > 0) confidence += pattern.priority * 0.001;
            }
        } else {
            // Fallback to simple anchor heuristic
            size_t anchorMatches = 0;
            for (const auto& anchor : pattern.alternating_anchors) {
                if (regionContent.find(anchor) != std::string::npos) {
                    anchorMatches++;
                }
            }
            if (anchorMatches > 0) {
                confidence = static_cast<double>(anchorMatches) / pattern.alternating_anchors.size();
                confidence *= pattern.weight;
                if (pattern.priority > 0) {
                    confidence += pattern.priority * 0.001;  // Small boost for priority
                }
            }
        }

        // Update best match if this is better
        if (confidence > bestConfidence ||
            (confidence == bestConfidence && pattern.priority > bestPriority)) {
            bestConfidence = confidence;
            bestPattern = &pattern;
            bestPriority = pattern.priority;
        }
    }

    outConfidence = bestConfidence;
    return bestPattern;
}

std::unordered_map<std::string, std::string> PatternApplier::extractEvidenceSpans(
    const std::string& content,
    const PatternData& pattern) const {

    std::unordered_map<std::string, std::string> captures;

    if (!pattern.use_alternating || pattern.alternating_anchors.empty()) {
        return captures;
    }

    // Use rbcursive to extract spans between alternating anchors
    size_t currentPos = 0;

    for (size_t i = 0; i < pattern.alternating_anchors.size(); ++i) {
        const auto& anchor = pattern.alternating_anchors[i];

        // Find anchor position
        size_t anchorPos = content.find(anchor, currentPos);
        if (anchorPos == std::string::npos) {
            break;  // Anchor not found, stop extraction
        }

        // Extract span before this anchor (if not the first anchor)
        if (i > 0 && currentPos < anchorPos) {
            std::string spanContent = content.substr(currentPos, anchorPos - currentPos);

            // Trim whitespace
            size_t start = spanContent.find_first_not_of(" \t\n\r");
            size_t end = spanContent.find_last_not_of(" \t\n\r");
            if (start != std::string::npos && end != std::string::npos) {
                spanContent = spanContent.substr(start, end - start + 1);
            }

            // Store with evidence type name if available
            if (i - 1 < pattern.evidence_types.size()) {
                captures[pattern.evidence_types[i - 1]] = spanContent;
            } else {
                captures["span_" + std::to_string(i - 1)] = spanContent;
            }
        }

        // Move past the anchor
        currentPos = anchorPos + anchor.length();
    }

    // Extract final span after last anchor
    if (currentPos < content.length() && !pattern.evidence_types.empty()) {
        size_t lastEvidenceIdx = pattern.evidence_types.size() - 1;
        std::string spanContent = content.substr(currentPos);

        // Trim whitespace
        size_t start = spanContent.find_first_not_of(" \t\n\r");
        size_t end = spanContent.find_last_not_of(" \t\n\r");
        if (start != std::string::npos && end != std::string::npos) {
            spanContent = spanContent.substr(start, end - start + 1);
        }

        if (lastEvidenceIdx < pattern.evidence_types.size()) {
            captures[pattern.evidence_types[lastEvidenceIdx]] = spanContent;
        }
    }

    return captures;
}

ir::mlir::RegionNode::RegionType PatternApplier::patternNameToRegionType(const std::string& patternName) const {
    // Map pattern names to region types
    if (patternName.find("function") != std::string::npos) {
        return ir::mlir::RegionNode::RegionType::FUNCTION;
    }
    if (patternName.find("block") != std::string::npos) {
        return ir::mlir::RegionNode::RegionType::BLOCK;
    }
    if (patternName.find("conditional") != std::string::npos ||
        patternName.find("if") != std::string::npos) {
        return ir::mlir::RegionNode::RegionType::CONDITIONAL;
    }
    if (patternName.find("loop") != std::string::npos ||
        patternName.find("while") != std::string::npos ||
        patternName.find("for") != std::string::npos) {
        return ir::mlir::RegionNode::RegionType::LOOP;
    }
    if (patternName.find("return") != std::string::npos) {
        return ir::mlir::RegionNode::RegionType::RETURN_REGION;
    }

    return ir::mlir::RegionNode::RegionType::BLOCK;
}

std::string PatternApplier::inferMlirDialect(
    const PatternData& pattern,
    const std::unordered_map<std::string, std::string>& captures) const {

    // Infer MLIR dialect from pattern characteristics
    if (pattern.name.find("function") != std::string::npos) {
        return "func";
    }
    if (pattern.name.find("arith") != std::string::npos) {
        return "arith";
    }
    if (pattern.name.find("cf") != std::string::npos) {
        return "cf";
    }

    // Default to standard dialect
    return "std";
}

void PatternApplier::populateRegionFromPattern(
    ir::mlir::RegionNode& region,
    const PatternData& pattern,
    const std::unordered_map<std::string, std::string>& captures) {

    // Set region type based on pattern
    region.setType(patternNameToRegionType(pattern.name));

    // Set MLIR dialect
    region.setMlirDialect(inferMlirDialect(pattern, captures));

    // Extract and set region name if available
    if (captures.count("identifier") > 0) {
        region.setName(captures.at("identifier"));
    } else if (captures.count("name") > 0) {
        region.setName(captures.at("name"));
    }

    // Create operations based on captured evidence
    if (region.getType() == ir::mlir::RegionNode::RegionType::FUNCTION) {
        // Create function declaration operation
            ir::mlir::RegionNode::Operation funcOp("func.func");

        // Add function name as attribute
        if (!region.getName().empty()) {
            funcOp.attributes["sym_name"] = region.getName();
        }

        // Add parameters as block arguments if available
        if (captures.count("parameters") > 0) {
            const auto& paramsStr = captures.at("parameters");
            // Parse parameters and add to region
            // Simple split by comma for now
            size_t start = 0;
            size_t comma = paramsStr.find(',');
            while (start < paramsStr.length()) {
                size_t end = (comma != std::string::npos) ? comma : paramsStr.length();
                std::string param = paramsStr.substr(start, end - start);

                // Trim whitespace
                size_t paramStart = param.find_first_not_of(" \t\n\r");
                size_t paramEnd = param.find_last_not_of(" \t\n\r");
                if (paramStart != std::string::npos && paramEnd != std::string::npos) {
                    param = param.substr(paramStart, paramEnd - paramStart + 1);
                    region.addArgument(param);
                }

                if (comma == std::string::npos) break;
                start = comma + 1;
                comma = paramsStr.find(',', start);
            }
        }

        // Add return type as attribute if available
        if (captures.count("return_type") > 0) {
            funcOp.attributes["return_type"] = captures.at("return_type");
        }

        region.addOperation(funcOp);
    }

    // Create value stubs for identifiers found in evidence
    for (const auto& [evidenceType, evidenceText] : captures) {
        if (evidenceType == "identifier" || evidenceType == "name") {
            ir::mlir::RegionNode::Value val;
            val.name = evidenceText;
            val.type = "unknown";  // Type inference needed
            region.addValue(val);
        } else if (evidenceType == "type" || evidenceType == "type_expression") {
            // Store type information for later use
            region.setMlirAttribute("inferred_type", evidenceText);
        }
    }

    // Register a semantic orbit anchor into the ParameterGraph
    try {
        ::cppfort::pijul::OrbitMatchInfo info;
        info.key = pattern.name + "@" + region.getName();
        info.patternName = pattern.name;
        info.context.start_pos = region.getSourceStart();
        info.context.end_pos = region.getSourceEnd();
        info.context.depth_hint = 0; // best effort
        info.context.grammar_type = 0; // unknown

        std::string source_fragment = "";
        if (info.context.end_pos > info.context.start_pos) {
            source_fragment = region.getSourceStart() < region.getSourceEnd() ? std::string("<source fragment>") : std::string("");
        }

        ::cppfort::pijul::ParameterAnchor anchor = ::cppfort::pijul::make_anchor(info,
            source_fragment,
            std::string_view{}, captures,
            "pattern-applier-semantic-anchor");
        parameterGraph_.add_anchor(anchor);
    } catch (...) {
        if (enableDebug_) std::cerr << "[PatternApplier] Failed to register parameter anchor" << std::endl;
    }
}

PatternApplier::ApplicationResult PatternApplier::applyPatternToRegion(
    ir::mlir::RegionNode& region,
    const std::string& source,
    const std::vector<std::string>& contextHints) {

    ApplicationResult result;

    // Extract region content from source
    size_t start = region.getSourceStart();
    size_t end = region.getSourceEnd();

    if (start >= source.length() || end > source.length() || start >= end) {
        result.errorMessage = "Invalid region source location";
        return result;
    }

    std::string regionContent = source.substr(start, end - start);

    // Find best matching pattern
    double confidence = 0.0;
    const PatternData* bestPattern = findBestPattern(regionContent, confidence);

    if (!bestPattern) {
        result.errorMessage = "No matching pattern found";
        return result;
    }

    if (confidence < confidenceThreshold_) {
        result.errorMessage = "Pattern confidence below threshold: " + std::to_string(confidence);
        return result;
    }

    // Extract evidence spans
    auto captures = extractEvidenceSpans(regionContent, *bestPattern);

    // Populate region from pattern
    populateRegionFromPattern(region, *bestPattern, captures);

    // Build result
    result.success = true;
    result.matchedPatternName = bestPattern->name;
    result.capturedSpans = std::move(captures);
    result.confidenceScores["overall"] = confidence;

    if (enableDebug_) {
        std::cerr << "[PatternApplier] Matched pattern '" << bestPattern->name
                  << "' with confidence " << confidence << "\n";
    }

    // Run peephole optimizations for this region if any
    // Note: iterpeeps expects Node* but we're working with RegionNode*
    // This is a temporary stub - proper integration needed
    // try {
    //     ::cppfort::ir::IterPeeps peephole;
    //     peephole.iterate(&region);
    // } catch (...) {
    //     if (enableDebug_) std::cerr << "[PatternApplier] Peephole iteration failed or no peepholes available\n";
    // }

    return result;
}

size_t PatternApplier::applyPatternsToTree(
    ir::mlir::RegionNode& root,
    const std::string& source) {

    size_t successCount = 0;

    // Apply pattern to this node
    auto result = applyPatternToRegion(root, source);
    if (result.success) {
        successCount++;
    }

    // Recursively apply to children
    for (auto& child : root.getChildren()) {
        successCount += applyPatternsToTree(*child, source);
    }

    return successCount;
}

PatternApplier::ApplicationResult PatternApplier::applySpecificPattern(
    ir::mlir::RegionNode& region,
    const std::string& patternName,
    const std::string& source) {

    ApplicationResult result;

    // Find pattern by name
    const PatternData* targetPattern = nullptr;
    for (const auto& pattern : patterns_) {
        if (pattern.name == patternName) {
            targetPattern = &pattern;
            break;
        }
    }

    if (!targetPattern) {
        result.errorMessage = "Pattern not found: " + patternName;
        return result;
    }

    // Extract region content
    size_t start = region.getSourceStart();
    size_t end = region.getSourceEnd();

    if (start >= source.length() || end > source.length() || start >= end) {
        result.errorMessage = "Invalid region source location";
        return result;
    }

    std::string regionContent = source.substr(start, end - start);

    // Extract evidence spans
    auto captures = extractEvidenceSpans(regionContent, *targetPattern);

    // Populate region
    populateRegionFromPattern(region, *targetPattern, captures);

    result.success = true;
    result.matchedPatternName = targetPattern->name;
    result.capturedSpans = std::move(captures);
    result.confidenceScores["overall"] = 1.0;  // Forced match

    return result;
}

// Standalone helper function
PatternApplier::ApplicationResult applyPatternStandalone(
    const std::string& content,
    const std::filesystem::path& patternsPath,
    const std::vector<std::string>& contextHints) {

    PatternApplier applier(patternsPath);
    if (!applier.initialize()) {
        PatternApplier::ApplicationResult result;
        result.errorMessage = "Failed to initialize pattern applier";
        return result;
    }

    // Create a dummy region spanning the entire content
    ir::mlir::RegionNode dummyRegion;
    dummyRegion.setSourceLocation(0, content.length());

    return applier.applyPatternToRegion(dummyRegion, content, contextHints);
}

// Validation function
bool validateLabeledTree(
    const ir::mlir::RegionNode& root,
    std::vector<std::string>& outErrors) {

    bool valid = true;

    // Check basic validation
    if (!root.validate()) {
        outErrors.push_back("Region validation failed at: " + root.getName());
        valid = false;
    }

    // Check if region has a type set
    if (root.getType() == ir::mlir::RegionNode::RegionType::UNKNOWN) {
        outErrors.push_back("Region has unknown type: " + root.getName());
        valid = false;
    }

    // Check if function regions have operations
    if (root.getType() == ir::mlir::RegionNode::RegionType::FUNCTION) {
        if (root.getOperations().empty()) {
            outErrors.push_back("Function region has no operations: " + root.getName());
            valid = false;
        }
    }

    // Recursively validate children
    for (const auto& child : root.getChildren()) {
        if (!validateLabeledTree(*child, outErrors)) {
            valid = false;
        }
    }

    return valid;
}

} // namespace stage0
} // namespace cppfort
