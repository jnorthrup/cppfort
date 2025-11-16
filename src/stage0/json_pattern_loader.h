#pragma once

#include <vector>
#include <string>
#include <string_view>
#include <unordered_map>
#include <optional>

namespace cppfort::stage0 {

/**
 * JSON-based pattern loader
 * Loads patterns from JSON format matching cppfort_core_patterns.json structure
 */
struct JsonParsedPattern {
    std::string name;
    bool use_alternating;
    std::vector<std::string> alternating_anchors;
    std::vector<std::string> evidence_types;
    int grammar_modes;
    std::unordered_map<int, std::string> templates; // 1=C, 2=CPP, 4=CPP2
    int priority;
    
    JsonParsedPattern() : use_alternating(false), grammar_modes(0), priority(100) {}
};

class JsonPatternLoader {
public:
    JsonPatternLoader() = default;
    
    /**
     * Load patterns from JSON file
     * Returns empty vector on error, patterns on success
     */
    std::vector<JsonParsedPattern> load_from_file(const std::string& filepath);
    
    /**
     * Parse patterns from JSON string content
     */
    std::vector<JsonParsedPattern> load_from_string(const std::string& json_content);
    
    /**
     * Get last error message
     */
    std::string get_last_error() const { return last_error; }
    
private:
    std::string last_error;
    
    // Simple JSON parsing helpers (minimal implementation)
    std::string_view trim(std::string_view str) const;
    size_t skip_whitespace(const std::string& json, size_t pos) const;
    size_t find_matching_brace(const std::string& json, size_t start) const;
    std::string extract_string(const std::string& json, size_t& pos);
    int extract_int(const std::string& json, size_t& pos);
    std::vector<std::string> extract_string_array(const std::string& json, size_t& pos);
    std::unordered_map<int, std::string> extract_templates_object(const std::string& json, size_t& pos);
};

} // namespace cppfort::stage0