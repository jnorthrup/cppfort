#include "pattern_loader.h"

#include <cctype>
#include <fstream>
#include <iostream>
#include <sstream>

namespace cppfort::stage0 {
namespace {

std::string trim(std::string_view text) {
    size_t begin = 0;
    size_t end = text.size();
    while (begin < end && std::isspace(static_cast<unsigned char>(text[begin]))) {
        ++begin;
    }
    while (end > begin && std::isspace(static_cast<unsigned char>(text[end - 1]))) {
        --end;
    }
    return std::string{text.substr(begin, end - begin)};
}

std::string strip_quotes(std::string value) {
    // Strip comments first (everything after #)
    size_t comment_pos = value.find('#');
    if (comment_pos != std::string::npos) {
        value = value.substr(0, comment_pos);
    }

    // Trim whitespace
    value = trim(value);

    // Strip quotes
    if (value.size() >= 2 && ((value.front() == '"' && value.back() == '"') ||
                              (value.front() == '\'' && value.back() == '\''))) {
        return value.substr(1, value.size() - 2);
    }
    return value;
}

::cppfort::ir::GrammarType parseGrammarType(const std::string& key) {
    using ::cppfort::ir::GrammarType;
    if (key == "C") return GrammarType::C;
    if (key == "CPP" || key == "C++") return GrammarType::CPP;
    if (key == "CPP2") return GrammarType::CPP2;
    return GrammarType::UNKNOWN;
}

} // namespace

bool PatternLoader::load_yaml(const std::string& path) {
    patterns_.clear();

    std::ifstream input(path);
    if (!input.is_open()) {
        return false;
    }

    PatternData current;
    std::string line;
    bool in_pattern = false;
    std::string current_section;

    // Temp storage for segments during parsing
    std::map<int, AnchorSegment> temp_segments;

    while (std::getline(input, line)) {
        if (line.empty()) {
            continue;
        }

        const std::string trimmed = trim(line);
        if (trimmed.empty() || trimmed.rfind("#", 0) == 0) {
            continue;
        }

        // Check for pattern separator (---)
        if (trimmed == "---") {
            if (in_pattern && !current.name.empty()) {
                // Finalize segments
                for (auto& [ord, seg] : temp_segments) {
                    current.segments.push_back(seg);
                }

                patterns_.push_back(current);
                current = PatternData{};
                temp_segments.clear();
            }
            in_pattern = true;
            current_section.clear();
            continue;
        }

        if (!in_pattern) {
            continue;
        }

        // Check for list items first (before key-value pairs)
        if (trimmed.rfind("-", 0) == 0) {
            if (!current_section.empty()) {
                // List item
                std::string item = trim(trimmed.substr(1));
                item = strip_quotes(item);
                // Strip comment
                size_t comment_pos = item.find('#');
                if (comment_pos != std::string::npos) {
                    item = trim(item.substr(0, comment_pos));
                }
                item = strip_quotes(item);  // Strip quotes again after comment removal

                if (current_section == "signature_patterns") {
                    current.signature_patterns.push_back(item);
                } else if (current_section == "prev_tokens") {
                    current.prev_tokens.push_back(item);
                } else if (current_section == "next_tokens") {
                    current.next_tokens.push_back(item);
                } else if (current_section == "alternating_anchors") {
                    current.alternating_anchors.push_back(item);
                } else if (current_section == "evidence_types") {
                    current.evidence_types.push_back(item);
                }
            }
        }
        // Parse key-value pairs
        else if (auto colon_pos = trimmed.find(':'); colon_pos != std::string::npos) {
            std::string key = trim(trimmed.substr(0, colon_pos));
            std::string value = trim(trimmed.substr(colon_pos + 1));
            if (key == "name") {
                current.name = strip_quotes(value);
            } else if (key == "orbit_id") {
                current.orbit_id = std::stoi(value);
            } else if (key == "weight") {
                current.weight = std::stod(value);
            } else if (key == "grammar_modes") {
                current.grammar_modes = std::stoi(value);
            } else if (key == "lattice_filter") {
                current.lattice_filter = std::stoi(value);
            } else if (key == "scope_requirement") {
                current.scope_requirement = strip_quotes(value);
            } else if (key == "confix_mask") {
                current.confix_mask = std::stoi(value);
            } else if (key == "signature_patterns" || key == "prev_tokens" || key == "next_tokens" || 
                       key == "alternating_anchors" || key == "evidence_types" || key == "transformation_templates") {
                current_section = key;
            } else if (key == "use_alternating") {
                current.use_alternating = (strip_quotes(value) == "true");
            } else if (current_section == "transformation_templates") {
                // Parse grammar_mode: template
                int grammar_mode = std::stoi(key);
                current.substitution_templates[grammar_mode] = strip_quotes(value);
            } else if (key.rfind("segment_", 0) == 0) {
                // Parse segment_X_Y format
                size_t first_underscore = key.find('_', 8);
                if (first_underscore != std::string::npos) {
                    int segment_idx = std::stoi(key.substr(8, first_underscore - 8));
                    std::string field = key.substr(first_underscore + 1);

                    if (temp_segments.find(segment_idx) == temp_segments.end()) {
                        temp_segments[segment_idx] = AnchorSegment{};
                        temp_segments[segment_idx].ordinal = segment_idx;
                    }

                    if (field == "name") {
                        temp_segments[segment_idx].name = strip_quotes(value);
                    } else if (field == "offset") {
                        temp_segments[segment_idx].offset_from_anchor = std::stoi(value);
                    } else if (field == "delim_start") {
                        temp_segments[segment_idx].delimiter_start = strip_quotes(value);
                    } else if (field == "delim_end") {
                        temp_segments[segment_idx].delimiter_end = strip_quotes(value);
                    }
                }
            } else if (key.rfind("substitute_", 0) == 0) {
                // Parse substitute_X format
                int grammar_mode = std::stoi(key.substr(11));
                current.substitution_templates[grammar_mode] = strip_quotes(value);
            }
        }
    }

    // Add the last pattern if it exists
    if (in_pattern && !current.name.empty()) {
        // Finalize segments
        for (auto& [ord, seg] : temp_segments) {
            current.segments.push_back(seg);
        }
        patterns_.push_back(current);
    }

    return !patterns_.empty();
}

} // namespace cppfort::stage0

