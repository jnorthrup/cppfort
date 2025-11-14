#include "pattern_loader.h"

#include <cctype>
#include <charconv>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <sstream>
#include <optional>

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

std::optional<int> parse_int_value(std::string_view text) {
    std::string cleaned = trim(text);
    if (auto comment_pos = cleaned.find('#'); comment_pos != std::string::npos) {
        cleaned = cleaned.substr(0, comment_pos);
        cleaned = trim(cleaned);
    }
    if (cleaned.empty()) {
        return std::nullopt;
    }
    int value = 0;
    const char* begin = cleaned.data();
    const char* end = begin + cleaned.size();
    auto [ptr, ec] = std::from_chars(begin, end, value);
    if (ec != std::errc{} || ptr != end) {
        return std::nullopt;
    }
    return value;
}

std::optional<double> parse_double_value(std::string_view text) {
    std::string cleaned = trim(text);
    if (auto comment_pos = cleaned.find('#'); comment_pos != std::string::npos) {
        cleaned = cleaned.substr(0, comment_pos);
        cleaned = trim(cleaned);
    }
    if (cleaned.empty()) {
        return std::nullopt;
    }
    // strtod tolerates whitespace and stops at first invalid char
    char* parse_end = nullptr;
    const double result = std::strtod(cleaned.c_str(), &parse_end);
    if (!parse_end || parse_end != cleaned.c_str() + cleaned.size()) {
        return std::nullopt;
    }
    return result;
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
                // Strip comment first (before stripping quotes)
                size_t comment_pos = item.find('#');
                if (comment_pos != std::string::npos) {
                    item = trim(item.substr(0, comment_pos));
                }
                item = strip_quotes(item);  // Strip quotes once

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
                } else if (current_section == "require_tokens" && !current.evidence_constraints.empty()) {
                    current.evidence_constraints.back().require_tokens.push_back(item);
                } else if (current_section == "forbid_tokens" && !current.evidence_constraints.empty()) {
                    current.evidence_constraints.back().forbid_tokens.push_back(item);
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
                if (auto parsed = parse_int_value(value)) {
                    current.orbit_id = *parsed;
                } else {
                    std::cerr << "PatternLoader: invalid orbit_id value '" << value << "'\n";
                }
            } else if (key == "weight") {
                if (auto parsed = parse_double_value(value)) {
                    current.weight = *parsed;
                } else {
                    std::cerr << "PatternLoader: invalid weight value '" << value << "'\n";
                }
            } else if (key == "grammar_modes") {
                if (auto parsed = parse_int_value(value)) {
                    current.grammar_modes = *parsed;
                } else {
                    std::cerr << "PatternLoader: invalid grammar_modes value '" << value << "'\n";
                }
            } else if (key == "lattice_filter") {
                if (auto parsed = parse_int_value(value)) {
                    current.lattice_filter = *parsed;
                } else {
                    std::cerr << "PatternLoader: invalid lattice_filter value '" << value << "'\n";
                }
            } else if (key == "scope_requirement") {
                current.scope_requirement = strip_quotes(value);
            } else if (key == "confix_mask") {
                if (auto parsed = parse_int_value(value)) {
                    current.confix_mask = *parsed;
                } else {
                    std::cerr << "PatternLoader: invalid confix_mask value '" << value << "'\n";
                }
            } else if (key == "priority") {
                if (auto parsed = parse_int_value(value)) {
                    current.priority = *parsed;
                } else {
                    std::cerr << "PatternLoader: invalid priority value '" << value << "'\n";
                }
                current_section.clear();
            } else if (key == "signature_patterns" || key == "prev_tokens" || key == "next_tokens" ||
                       key == "alternating_anchors" || key == "evidence_types" || key == "transformation_templates") {
                current_section = key;
            } else if (key == "use_alternating") {
                current.use_alternating = (strip_quotes(value) == "true");
            } else if (key == "evidence_constraints") {
                current.evidence_constraints.emplace_back();
                current_section = key;
            } else if (current_section == "transformation_templates") {
                // Parse grammar_mode: template
                if (auto parsed_mode = parse_int_value(key)) {
                    current.substitution_templates[*parsed_mode] = strip_quotes(value);
                } else {
                    std::cerr << "PatternLoader: invalid transformation template key '" << key << "'\n";
                }
            } else if (current_section == "evidence_constraints" && !current.evidence_constraints.empty()) {
                auto& constraint = current.evidence_constraints.back();
                if (key == "kind") {
                    constraint.kind = strip_quotes(value);
                } else if (key == "enforce_type_evidence") {
                    std::string val = strip_quotes(value);
                    constraint.enforce_type_evidence = (val == "true" || val == "1");
                } else if (key == "require_tokens") {
                    current_section = "require_tokens";
                } else if (key == "forbid_tokens") {
                    current_section = "forbid_tokens";
                }
            } else if (key.rfind("segment_", 0) == 0) {
                // Parse segment_X_Y format
                size_t first_underscore = key.find('_', 8);
                if (first_underscore != std::string::npos) {
                    std::string ordinal_text = key.substr(8, first_underscore - 8);
                    auto parsed_idx = parse_int_value(ordinal_text);
                    if (!parsed_idx) {
                        std::cerr << "PatternLoader: invalid segment ordinal '" << ordinal_text << "'\n";
                        continue;
                    }
                    std::string field = key.substr(first_underscore + 1);

                    if (temp_segments.find(*parsed_idx) == temp_segments.end()) {
                        temp_segments[*parsed_idx] = AnchorSegment{};
                        temp_segments[*parsed_idx].ordinal = *parsed_idx;
                    }

                    if (field == "name") {
                        temp_segments[*parsed_idx].name = strip_quotes(value);
                    } else if (field == "offset") {
                        if (auto parsed_offset = parse_int_value(value)) {
                            temp_segments[*parsed_idx].offset_from_anchor = *parsed_offset;
                        } else {
                            std::cerr << "PatternLoader: invalid segment offset '" << value << "'\n";
                        }
                    } else if (field == "delim_start") {
                        temp_segments[*parsed_idx].delimiter_start = strip_quotes(value);
                    } else if (field == "delim_end") {
                        temp_segments[*parsed_idx].delimiter_end = strip_quotes(value);
                    }
                }
            } else if (key.rfind("substitute_", 0) == 0) {
                // Parse substitute_X format
                if (auto parsed_mode = parse_int_value(key.substr(11))) {
                    current.substitution_templates[*parsed_mode] = strip_quotes(value);
                } else {
                    std::cerr << "PatternLoader: invalid substitute key '" << key << "'\n";
                }
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
