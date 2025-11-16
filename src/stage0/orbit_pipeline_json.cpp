#include "orbit_pipeline.h"
#include "json_pattern_loader.h"
#include <iostream>

namespace cppfort::stage0 {

// Load patterns from JSON file
bool OrbitPipeline::load_patterns(const std::string& path) {
    // Try JSON loader first (for .json files)
    if (path.size() >= 5 && path.substr(path.size() - 5) == ".json") {
        JsonPatternLoader json_loader;
        auto json_patterns = json_loader.load_from_file(path);
        
        if (json_patterns.empty()) {
            std::cerr << "Failed to load JSON patterns: " << json_loader.get_last_error() << "\n";
            return false;
        }
        
        // Convert JSON patterns to PatternData
        loader_.patterns().clear();
        for (const auto& jp : json_patterns) {
            PatternData pd;
            pd.name = jp.name;
            pd.use_alternating = jp.use_alternating;
            pd.alternating_anchors = jp.alternating_anchors;
            pd.evidence_types = jp.evidence_types;
            pd.grammar_modes = jp.grammar_modes;
            pd.priority = jp.priority;
            // Convert unordered_map<int, string> to std::map<int, string>
            for (const auto &tpl : jp.templates) {
                pd.substitution_templates[tpl.first] = tpl.second;
            }
            
            // Extract signature patterns from alternating anchors
            if (!jp.alternating_anchors.empty()) {
                // Use the first anchor as the signature pattern
                pd.signature_patterns.push_back(jp.alternating_anchors[0]);
            }
            
            loader_.patterns().push_back(pd);
        }
        
        // Build grammar tree
        grammar_tree_.clear();
        for (const auto& pattern : loader_.patterns()) {
            grammar_tree_.insert(pattern);
        }
        
        return true;
    }
    
    // Fallback to YAML loader
    const bool ok = loader_.load_yaml(path);
    grammar_tree_.clear();
    if (ok) {
        for (const auto& pattern : loader_.patterns()) {
            grammar_tree_.insert(pattern);
        }
    }
    return ok;
}

} // namespace cppfort::stage0