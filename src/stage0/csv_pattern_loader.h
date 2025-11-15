#pragma once

#include <vector>
#include <string>
#include <string_view>
#include <unordered_map>
#include <optional>

namespace cppfort::stage0 {

/**
 * CSV-based pattern loader for dogfooding quaternion orbit detector
 * Replaces YAML pattern loading with CSV materialization of semantic features:
 * - Alternating anchor tuples (anchor_1, anchor_2, anchor_3)
 * - Type evidence spans (evidence_type_1..4) 
 * - Transformation templates (cpp2_template, cpp_template, c_template)
 * - Grammar mode flags
 */
struct CSVParsedPattern {
    std::string name;
    bool use_alternating;
    std::vector<std::string> alternating_anchors;  // Up to 3 anchors
    std::vector<std::string> evidence_types;       // Up to 4 evidence types
    int grammar_modes;
    std::unordered_map<int, std::string> templates; // 1=C, 2=CPP, 4=CPP2
    
    CSVParsedPattern() : use_alternating(false), grammar_modes(0) {}
    
    // Check if pattern has required alternating structure
    bool is_valid_alternating() const {
        return use_alternating && !alternating_anchors.empty() && !evidence_types.empty();
    }
};

class CSVPatternLoader {
public:
    CSVPatternLoader() = default;
    
    /**
     * Load patterns from CSV file
     * Returns empty vector on error, patterns on success
     */
    std::vector<CSVParsedPattern> load_patterns_from_file(const std::string& filepath);
    
    /**
     * Parse patterns from CSV string content
     * Used for testing with embedded CSV data
     */
    std::vector<CSVParsedPattern> parse_patterns_from_string(std::string_view csv_content);
    
    /**
     * Get last error message
     */
    std::string get_last_error() const { return last_error; }
    
    /**
     * Validate that loaded patterns match semantic structure of YAML
     */
    bool validate_semantic_equivalence(const CSVParsedPattern& csv_pattern, 
                                     const std::string& yaml_pattern_name) const;

private:
    std::string last_error;
    
    // Parse a single CSV line respecting quoted fields
    std::vector<std::string> parse_csv_line(std::string_view line);
    
    // Extract template from string, handling escaped quotes
    std::string unescape_template(std::string_view escaped);
    
    // Trim whitespace
    std::string_view trim(std::string_view s);
};

} // namespace cppfort::stage0