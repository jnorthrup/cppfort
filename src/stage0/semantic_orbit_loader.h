#pragma once

#include <cstdint>
#include <string>
#include <unordered_map>
#include <vector>

namespace cppfort::stage0 {

// Semantic orbit pattern with masking ring support
struct SemanticOrbitPattern {
    std::string name;
    std::string orbit_type;
    uint16_t masking_ring;  // 16-bit mask for confidence scoring

    struct Evidence {
        std::string pattern;
        double weight;
        std::string semantic;  // Semantic meaning tag
    };
    std::vector<Evidence> evidence;

    std::vector<std::string> children;
    std::unordered_map<std::string, std::string> metadata;

    // Compute confidence with masking ring
    double compute_confidence(const std::string& text, uint16_t context_mask = 0xFFFF) const {
        double base_confidence = 0.0;

        // Apply evidence patterns
        for (const auto& e : evidence) {
            // TODO: Actual pattern matching
            base_confidence += e.weight * 0.5;  // Placeholder
        }

        // Apply masking ring
        uint16_t effective_mask = masking_ring & context_mask;
        double mask_factor = __builtin_popcount(effective_mask) / 16.0;

        return base_confidence * mask_factor;
    }
};

// Context-sensitive masking combinations
struct MaskingContext {
    std::string context_name;
    std::unordered_map<std::string, uint16_t> pattern_masks;

    uint16_t get_mask(const std::string& pattern_name) const {
        auto it = pattern_masks.find(pattern_name);
        return it != pattern_masks.end() ? it->second : 0xFFFF;
    }
};

// Composition rules for nested orbits
struct CompositionRule {
    std::string parent;
    std::vector<std::string> contains;
};

class SemanticOrbitLoader {
public:
    // Load orbit patterns from YAML files
    bool load_patterns(const std::string& cpp2_path,
                      const std::string& cpp_path,
                      const std::string& c_path);

    // Get patterns for a specific language
    std::vector<SemanticOrbitPattern> get_patterns(const std::string& language) const;

    // Get context-specific masking
    MaskingContext get_context(const std::string& context_name) const;

    // Check composition rules
    bool can_contain(const std::string& parent, const std::string& child) const;

    // Find best matching pattern
    const SemanticOrbitPattern* find_best_pattern(
        const std::string& text,
        const std::string& language,
        const std::string& context = "") const;

private:
    std::unordered_map<std::string, std::vector<SemanticOrbitPattern>> patterns_;
    std::unordered_map<std::string, MaskingContext> contexts_;
    std::vector<CompositionRule> composition_rules_;

    bool parse_yaml_file(const std::string& path, const std::string& language);
};

} // namespace cppfort::stage0