#pragma once

#include <map>
#include <string>
#include <string_view>
#include <vector>

#include <yaml-cpp/yaml.h>

#include "orbit_mask.h"

namespace cppfort::stage0 {

// Anchor segment: text segment relative to anchor position
struct AnchorSegment {
    std::string name;       // e.g., "name", "params", "body"
    int ordinal;            // Position in semantic structure
    int offset_from_anchor; // Offset from anchor (negative = before, positive = after)
    std::string delimiter_start; // Start delimiter (e.g., "{", "(", ":")
    std::string delimiter_end;   // End delimiter (e.g., "}", ")", "")
};

struct EvidenceConstraint {
    std::string kind;                         // evidence_types entry to which this applies
    std::vector<std::string> require_tokens;  // tokens that must appear (positive evidence)
    std::vector<std::string> forbid_tokens;   // hard negation: tokens that must NOT appear
    bool enforce_type_evidence = false;       // enable TypeEvidence lattice validation
};

struct PatternData {
    std::string name;
    int orbit_id = 0;
    std::vector<std::string> signature_patterns;  // Anchors to match
    double weight = 1.0;
    int priority = 0;             // Optional priority for disambiguation
    int grammar_modes = 7;  // All modes by default
    int lattice_filter = 65535;  // All classes by default
    std::vector<std::string> prev_tokens;
    std::vector<std::string> next_tokens;
    std::string scope_requirement = "any";
    int confix_mask = 63;  // All scopes by default

    // Alternating anchor/evidence pattern system
    std::vector<std::string> alternating_anchors;     // Fixed anchor strings
    std::vector<std::string> evidence_types;          // Types for evidence spans between anchors
    std::vector<EvidenceConstraint> evidence_constraints; // Optional fine-grained constraints
    bool use_alternating = false;                     // Whether to use alternating system

    // Legacy segment-based system (for backward compatibility)
    std::vector<AnchorSegment> segments;

    // Substitution templates per grammar
    // Uses $0, $1, $2 for segments and @ANCHOR@ for anchor masking
    std::map<int, std::string> substitution_templates;
};

class PatternLoader {
public:
    PatternLoader() = default;

    bool load_yaml(const std::string& path);

    const std::vector<PatternData>& patterns() const { return patterns_; }
    std::vector<PatternData>& patterns() { return patterns_; }

    size_t pattern_count() const { return patterns_.size(); }

private:
    std::vector<PatternData> patterns_;
    
    bool load_pattern(const YAML::Node& pattern_node);

};

} // namespace cppfort::stage0
