#include "pattern_loader.h"

#include <yaml-cpp/yaml.h>
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

    try {
        std::vector<YAML::Node> documents = YAML::LoadAllFromFile(path);
        
        if (documents.empty()) {
            std::cerr << "PatternLoader: No documents found in " << path << std::endl;
            return false;
        }

        // If single document and it's a sequence, use it directly
        if (documents.size() == 1 && documents[0].IsSequence()) {
            for (const auto& pattern_node : documents[0]) {
                if (load_pattern(pattern_node)) {
                    // Pattern loaded successfully
                }
            }
            return true;
        }
        
        // If multiple documents, treat each as a pattern
        for (const auto& doc : documents) {
            if (load_pattern(doc)) {
                // Pattern loaded successfully
            }
        }
        
        return !patterns_.empty();
    } catch (const YAML::Exception& e) {
        std::cerr << "PatternLoader: YAML parsing error in " << path << ": " << e.what() << std::endl;
        return false;
    }
}

bool PatternLoader::load_pattern(const YAML::Node& pattern_node) {
    PatternData pattern;

    if (pattern_node["name"]) {
        pattern.name = pattern_node["name"].as<std::string>();
    } else {
        std::cerr << "PatternLoader: Pattern missing name" << std::endl;
        return false;
    }

    if (pattern_node["orbit_id"]) {
        pattern.orbit_id = pattern_node["orbit_id"].as<int>();
    }

    if (pattern_node["weight"]) {
        pattern.weight = pattern_node["weight"].as<double>();
    }

    if (pattern_node["priority"]) {
        pattern.priority = pattern_node["priority"].as<int>();
    }

    if (pattern_node["grammar_modes"]) {
        pattern.grammar_modes = pattern_node["grammar_modes"].as<int>();
    }

    if (pattern_node["lattice_filter"]) {
        pattern.lattice_filter = pattern_node["lattice_filter"].as<int>();
    }

    if (pattern_node["scope_requirement"]) {
        pattern.scope_requirement = pattern_node["scope_requirement"].as<std::string>();
    }

    if (pattern_node["confix_mask"]) {
        pattern.confix_mask = pattern_node["confix_mask"].as<int>();
    }

    if (pattern_node["use_alternating"]) {
        pattern.use_alternating = pattern_node["use_alternating"].as<bool>();
    }

    if (pattern_node["alternating_anchors"] && pattern_node["alternating_anchors"].IsSequence()) {
        for (const auto& anchor : pattern_node["alternating_anchors"]) {
            pattern.alternating_anchors.push_back(anchor.as<std::string>());
        }
    }

    if (pattern_node["evidence_types"] && pattern_node["evidence_types"].IsSequence()) {
        for (const auto& type : pattern_node["evidence_types"]) {
            pattern.evidence_types.push_back(type.as<std::string>());
        }
    }

    if (pattern_node["transformation_templates"] && pattern_node["transformation_templates"].IsMap()) {
        for (const auto& template_pair : pattern_node["transformation_templates"]) {
            int mode = template_pair.first.as<int>();
            std::string template_str = template_pair.second.as<std::string>();
            pattern.substitution_templates[mode] = template_str;
        }
    }

    patterns_.push_back(pattern);
    return true;
}

} // namespace cppfort::stage0
