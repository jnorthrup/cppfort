#include "csv_pattern_loader.h"
#include <fstream>
#include <sstream>
#include <algorithm>
#include <cctype>

namespace cppfort::stage0 {

std::vector<CSVParsedPattern> CSVPatternLoader::load_patterns_from_file(const std::string& filepath) {
    std::ifstream file(filepath);
    if (!file.is_open()) {
        last_error = "Failed to open file: " + filepath;
        return {};
    }
    
    std::stringstream buffer;
    buffer << file.rdbuf();
    return parse_patterns_from_string(buffer.str());
}

std::vector<CSVParsedPattern> CSVPatternLoader::parse_patterns_from_string(std::string_view csv_content) {
    std::vector<CSVParsedPattern> patterns;
    
    if (csv_content.empty()) {
        last_error = "Empty CSV content";
        return {};
    }
    
    // Split into lines
    std::vector<std::string_view> lines;
    size_t pos = 0;
    while (pos < csv_content.size()) {
        size_t newline = csv_content.find('\n', pos);
        if (newline == std::string_view::npos) {
            lines.push_back(csv_content.substr(pos));
            break;
        }
        lines.push_back(csv_content.substr(pos, newline - pos));
        pos = newline + 1;
    }
    
    if (lines.empty()) {
        last_error = "No lines in CSV";
        return {};
    }
    
    // Validate header (optional but recommended)
    std::string_view header = trim(lines[0]);
    bool has_header = header.find("name") != std::string_view::npos && 
                     header.find("anchor_1") != std::string_view::npos;
    
    size_t start_line = has_header ? 1 : 0;
    
    // Parse each pattern row
    for (size_t i = start_line; i < lines.size(); ++i) {
        std::string_view line = trim(lines[i]);
        if (line.empty()) continue; // Skip empty lines
        
        std::vector<std::string> fields = parse_csv_line(line);
        if (fields.size() < 13) {
            // Skip malformed lines but continue parsing
            continue;
        }
        
        CSVParsedPattern pattern;
        
        // name, field 0
        pattern.name = trim(fields[0]);
        
        // use_alternating, field 1
        pattern.use_alternating = (trim(fields[1]) == "true");
        
        // alternating_anchors, fields 2-4
        for (int j = 2; j <= 4; ++j) {
            std::string_view anchor = trim(fields[j]);
            if (!anchor.empty()) {
                pattern.alternating_anchors.push_back(std::string(anchor));
            }
        }
        
        // grammar_modes, field 5
        std::string_view modes = trim(fields[5]);
        if (!modes.empty()) {
            pattern.grammar_modes = std::stoi(std::string(modes));
        }
        
        // evidence_types, fields 6-9
        for (int j = 6; j <= 9; ++j) {
            std::string_view ev = trim(fields[j]);
            if (!ev.empty()) {
                pattern.evidence_types.push_back(std::string(ev));
            }
        }
        
        // transformation templates, fields 10-12
        // cpp2_template (grammar mode 4)
        std::string_view cpp2 = trim(fields[10]);
        if (!cpp2.empty()) {
            pattern.templates[4] = std::string(cpp2);
        }
        
        // cpp_template (grammar mode 2)
        std::string_view cpp = trim(fields[11]);
        if (!cpp.empty()) {
            pattern.templates[2] = std::string(cpp);
        }
        
        // c_template (grammar mode 1)
        std::string_view c = trim(fields[12]);
        if (!c.empty()) {
            pattern.templates[1] = std::string(c);
        }
        
        patterns.push_back(std::move(pattern));
    }
    
    if (patterns.empty() && lines.size() > start_line) {
        last_error = "Failed to parse any valid patterns from CSV";
    }
    
    return patterns;
}

std::vector<std::string> CSVPatternLoader::parse_csv_line(std::string_view line) {
    std::vector<std::string> fields;
    fields.reserve(13); // We expect 13 columns
    
    size_t pos = 0;
    bool in_quotes = false;
    std::string current;
    
    while (pos < line.size()) {
        char ch = line[pos];
        
        if (ch == '"') {
            if (in_quotes && pos + 1 < line.size() && line[pos + 1] == '"') {
                // Escaped quote ("")
                current += '"';
                pos += 2;
                continue;
            }
            in_quotes = !in_quotes;
            pos++;
        } else if (ch == ',' && !in_quotes) {
            // End of field
            fields.push_back(std::move(current));
            current.clear();
            pos++;
        } else {
            current += ch;
            pos++;
        }
    }
    
    // Add final field
    fields.push_back(std::move(current));
    
    return fields;
}

std::string_view CSVPatternLoader::trim(std::string_view s) {
    // Trim leading whitespace
    size_t start = 0;
    while (start < s.size() && std::isspace(static_cast<unsigned char>(s[start]))) {
        start++;
    }
    
    // Trim trailing whitespace
    size_t end = s.size();
    while (end > start && std::isspace(static_cast<unsigned char>(s[end - 1]))) {
        end--;
    }
    
    return s.substr(start, end - start);
}

bool CSVPatternLoader::validate_semantic_equivalence(const CSVParsedPattern& csv_pattern, 
                                                   const std::string& yaml_pattern_name) const {
    // Basic validation: check that alternating patterns have required structure
    if (csv_pattern.name != yaml_pattern_name) {
        return false;
    }
    
    if (csv_pattern.use_alternating) {
        // Alternating patterns must have at least one anchor and evidence type
        if (csv_pattern.alternating_anchors.empty() || csv_pattern.evidence_types.empty()) {
            return false;
        }
    }
    
    // Check that templates are present for defined grammar modes
    for (const auto& [mode, template_str] : csv_pattern.templates) {
        if (template_str.empty()) {
            // Template expected but missing - validation failure
            return false;
        }
    }
    
    return true;
}

} // namespace cppfort::stage0