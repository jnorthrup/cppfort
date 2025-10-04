#pragma once

// Lightweight CPP2 pattern extractor (header-only)
// - Loads YAML produced by scripts/extract_cpp2_patterns.py
// - Extracts pattern names and stores raw YAML for later processing
// - Minimal dependencies to keep integration simple for now

#include <string>
#include <vector>
#include <fstream>
#include <sstream>
#include <regex>
#include <iostream>
#include <optional>
#include <algorithm>

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
        std::regex name_re(R"(^\s*name\s*:\s*["']?([A-Za-z0-9_\-:./]+)["']?\s*$)",
                           std::regex::icase | std::regex::multiline);
        std::smatch m;
        std::string s = raw_yaml_;
        auto begin = s.cbegin();
        auto end = s.cend();
        while (std::regex_search(begin, end, m, name_re)) {
            if (m.size() >= 2) {
                std::string nm = m[1].str();
                pattern_names_.push_back(nm);
            }
            begin = m.suffix().first;
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
        std::regex line_re("\\n");
        std::vector<std::string> lines;
        {
            std::sregex_token_iterator it(raw_yaml_.begin(), raw_yaml_.end(), line_re, -1);
            std::sregex_token_iterator end;
            for (; it != end; ++it) lines.push_back(it->str());
        }

        for (size_t i = 0; i < lines.size(); ++i) {
            // match name: patternName possibly quoted
            std::regex name_match(R"(^\s*name\s*:\s*["']?" + std::regex_replace(patternName, std::regex(R"([.^$|()\\[\]{}*+?])"), "\\$&") + R"(["']?\s*$))");
            try {
                if (std::regex_search(lines[i], name_match)) {
                    // collect context
                    size_t start = (i >= contextLines) ? (i - contextLines) : 0;
                    size_t end_index = std::min(lines.size(), i + contextLines + 1);
                    std::ostringstream out;
                    for (size_t j = start; j < end_index; ++j) {
                        out << lines[j] << "\n";
                    }
                    return out.str();
                }
            } catch (const std::regex_error&) {
                // fall through on regex construction issues
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