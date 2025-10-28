#pragma once

// Lightweight CPP2 pattern extractor (header-only)
// - Loads YAML produced by scripts/extract_cpp2_patterns.py
// - Extracts pattern names and stores raw YAML for later processing
// - Minimal dependencies to keep integration simple for now

#include <string>
#include <vector>
#include <fstream>
#include <sstream>
#include <iostream>
#include <optional>
#include <algorithm>
#include "rbcursive.h"

namespace cppfort {
namespace stage0 {

class CPP2PatternExtractor {
public:
    CPP2PatternExtractor() = default;

    // Load a YAML file (output from scripts/extract_cpp2_patterns.py)
    // Returns true on success, false on failure.
    bool loadPatternsYaml(const std::string& yamlPath) {
        raw_yaml_.clear();
        pattern_names_.clear();

        std::ifstream in(yamlPath);
        if (!in) {
            std::cerr << "CPP2PatternExtractor: failed to open " << yamlPath << std::endl;
            return false;
        }

        std::ostringstream ss;
        ss << in.rdbuf();
        raw_yaml_ = ss.str();

        // Extract pattern names from YAML assuming entries contain a "name:" field
        // This is intentionally permissive (handles name: foo or name: "foo")
        std::istringstream lines(raw_yaml_);
        std::string line;
        while (std::getline(lines, line)) {
            // Find "name:" prefix
            size_t name_pos = line.find("name:");
            if (name_pos == std::string::npos) continue;

            // Skip to value after "name:"
            size_t value_start = name_pos + 5;
            while (value_start < line.size() && std::isspace(line[value_start])) {
                value_start++;
            }
            if (value_start >= line.size()) continue;

            // Strip quotes if present
            size_t value_end = line.size();
            while (value_end > value_start && std::isspace(line[value_end - 1])) {
                value_end--;
            }

            if (line[value_start] == '"' || line[value_start] == '\'') {
                value_start++;
            }
            if (value_end > value_start && (line[value_end - 1] == '"' || line[value_end - 1] == '\'')) {
                value_end--;
            }

            if (value_end > value_start) {
                std::string nm = line.substr(value_start, value_end - value_start);
                pattern_names_.push_back(nm);
            }
        }

        // De-duplicate while preserving order
        std::vector<std::string> unique;
        unique.reserve(pattern_names_.size());
        for (const auto& p : pattern_names_) {
            if (std::find(unique.begin(), unique.end(), p) == unique.end())
                unique.push_back(p);
        }
        pattern_names_.swap(unique);

        return true;
    }

    // Returns the raw YAML string loaded (empty if none)
    const std::string& rawYaml() const { return raw_yaml_; }

    // Return pattern names discovered in YAML
    const std::vector<std::string>& patternNames() const { return pattern_names_; }

    // Try to find the YAML key entry for a given pattern name and return a small excerpt
    // This is a convenience for debugging / integration until a full YAML parser is introduced.
    std::optional<std::string> getPatternSnippet(const std::string& patternName, size_t contextLines = 4) const {
        if (raw_yaml_.empty()) return std::nullopt;

        // Find the line "name: patternName"
        std::vector<std::string> lines;
        std::istringstream line_stream(raw_yaml_);
        std::string line;
        while (std::getline(line_stream, line)) {
            lines.push_back(line);
        }

        for (size_t i = 0; i < lines.size(); ++i) {
            // match name: patternName possibly quoted
            size_t name_pos = lines[i].find("name:");
            if (name_pos == std::string::npos) continue;

            size_t value_start = name_pos + 5;
            while (value_start < lines[i].size() && std::isspace(lines[i][value_start])) {
                value_start++;
            }
            if (value_start >= lines[i].size()) continue;

            // Extract value with optional quotes
            size_t value_end = lines[i].size();
            while (value_end > value_start && std::isspace(lines[i][value_end - 1])) {
                value_end--;
            }

            bool start_quote = (lines[i][value_start] == '"' || lines[i][value_start] == '\'');
            bool end_quote = (value_end > value_start &&
                            (lines[i][value_end - 1] == '"' || lines[i][value_end - 1] == '\''));

            size_t actual_start = start_quote ? value_start + 1 : value_start;
            size_t actual_end = end_quote ? value_end - 1 : value_end;

            if (actual_end > actual_start) {
                std::string value = lines[i].substr(actual_start, actual_end - actual_start);
                if (value == patternName) {
                    // collect context
                    size_t start = (i >= contextLines) ? (i - contextLines) : 0;
                    size_t end_index = std::min(lines.size(), i + contextLines + 1);
                    std::ostringstream out;
                    for (size_t j = start; j < end_index; ++j) {
                        out << lines[j] << "\n";
                    }
                    return out.str();
                }
            }
        }
        return std::nullopt;
    }

    // Simple helper to print discovered pattern names (debug)
    void dumpPatternNames() const {
        std::cout << "CPP2PatternExtractor: discovered " << pattern_names_.size() << " patterns\n";
        for (const auto& n : pattern_names_) {
            std::cout << "  - " << n << "\n";
        }
    }

private:
    std::string raw_yaml_;
    std::vector<std::string> pattern_names_;
};

} // namespace stage0
} // namespace cppfort