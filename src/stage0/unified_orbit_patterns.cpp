#include "unified_orbit_patterns.h"
#include <yaml-cpp/yaml.h>
#include <fstream>
#include <iostream>
#include <algorithm>

namespace cppfort {
namespace ir {

namespace {

// Grammar mode constants
constexpr uint8_t GRAMMAR_C = 0x01;
constexpr uint8_t GRAMMAR_CPP = 0x02;
constexpr uint8_t GRAMMAR_CPP2 = 0x04;

// Category string to enum mapping
::std::unordered_map<::std::string, UnifiedOrbitCategory> categoryMap = {
    {"declaration", UnifiedOrbitCategory::Declaration},
    {"function", UnifiedOrbitCategory::Function},
    {"type_definition", UnifiedOrbitCategory::TypeDefinition},
    {"namespace", UnifiedOrbitCategory::Namespace},
    {"control_flow", UnifiedOrbitCategory::ControlFlow},
    {"expression", UnifiedOrbitCategory::Expression},
    {"parameter_list", UnifiedOrbitCategory::ParameterList},
    {"member_access", UnifiedOrbitCategory::MemberAccess},
    {"scope_resolution", UnifiedOrbitCategory::ScopeResolution},
    {"cpp2_only", UnifiedOrbitCategory::CPP2_Only},
    {"cpp_only", UnifiedOrbitCategory::CPP_Only},
    {"legacy_c", UnifiedOrbitCategory::Legacy_C}
};

// Parse grammar modes from string
uint8_t parseGrammarModes(const ::std::string& modes_str) {
    uint8_t modes = 0;
    if (modes_str.find("c") != ::std::string::npos) modes |= GRAMMAR_C;
    if (modes_str.find("cpp") != ::std::string::npos) modes |= GRAMMAR_CPP;
    if (modes_str.find("cpp2") != ::std::string::npos) modes |= GRAMMAR_CPP2;
    return modes;
}

} // anonymous namespace

bool UnifiedOrbitDatabase::loadUnifiedPatterns(const ::std::string& patternFile) {
    try {
        YAML::Node root = YAML::LoadFile(patternFile);

        if (!root.IsMap()) {
            ::std::cerr << "Error: Root node is not a map in " << patternFile << ::std::endl;
            return false;
        }

        // Load unified trunk patterns
        if (root["unified_trunk_patterns"] && root["unified_trunk_patterns"].IsSequence()) {
            for (const auto& pattern_node : root["unified_trunk_patterns"]) {
                UnifiedOrbitPattern pattern;

                pattern.name = pattern_node["name"].as<::std::string>();
                pattern.orbit_id = pattern_node["orbit_id"].as<uint32_t>();
                pattern.ast_node_type = pattern_node["ast_node_type"].as<::std::string>();

                // Parse category
                ::std::string category_str = pattern_node["category"].as<::std::string>();
                auto cat_it = categoryMap.find(category_str);
                if (cat_it != categoryMap.end()) {
                    pattern.category = cat_it->second;
                } else {
                    ::std::cerr << "Warning: Unknown category '" << category_str << "' for pattern '" << pattern.name << "'" << ::std::endl;
                    continue;
                }

                // Parse unified signatures
                if (pattern_node["unified_signatures"] && pattern_node["unified_signatures"].IsSequence()) {
                    for (const auto& sig : pattern_node["unified_signatures"]) {
                        pattern.unified_signatures.push_back(sig.as<::std::string>());
                    }
                }

                // Parse grammar variants
                if (pattern_node["grammar_variants"]) {
                    for (const auto& variant : pattern_node["grammar_variants"]) {
                        ::std::string grammar = variant.first.as<::std::string>();
                        uint8_t grammar_mode = 0;
                        if (grammar == "c") grammar_mode = GRAMMAR_C;
                        else if (grammar == "cpp") grammar_mode = GRAMMAR_CPP;
                        else if (grammar == "cpp2") grammar_mode = GRAMMAR_CPP2;

                        if (grammar_mode != 0 && variant.second.IsSequence()) {
                            ::std::vector<::std::string> variants;
                            for (const auto& var : variant.second) {
                                variants.push_back(var.as<::std::string>());
                            }
                            pattern.grammar_variants[grammar_mode] = variants;
                        }
                    }
                }

                // Parse optional properties
                if (pattern_node["weight"]) pattern.weight = pattern_node["weight"].as<double>();
                if (pattern_node["grammar_modes"]) {
                    pattern.grammar_modes = parseGrammarModes(pattern_node["grammar_modes"].as<::std::string>());
                }
                if (pattern_node["lattice_filter"]) pattern.lattice_filter = pattern_node["lattice_filter"].as<uint16_t>();
                if (pattern_node["confix_mask"]) pattern.confix_mask = pattern_node["confix_mask"].as<uint8_t>();
                if (pattern_node["scope_requirement"]) pattern.scope_requirement = pattern_node["scope_requirement"].as<::std::string>();

                // Store pattern
                _patterns[pattern.name] = pattern;
                _patternsByCategory[pattern.category].push_back(pattern);
            }
        }

        ::std::cout << "Loaded " << _patterns.size() << " unified orbit patterns from " << patternFile << ::std::endl;
        return true;

    } catch (const YAML::Exception& e) {
        ::std::cerr << "YAML parsing error in " << patternFile << ": " << e.what() << ::std::endl;
        return false;
    } catch (const ::std::exception& e) {
        ::std::cerr << "Error loading patterns from " << patternFile << ": " << e.what() << ::std::endl;
        return false;
    }
}

const UnifiedOrbitPattern* UnifiedOrbitDatabase::getPattern(const ::std::string& name) const {
    auto it = _patterns.find(name);
    return it != _patterns.end() ? &it->second : nullptr;
}

::std::vector<UnifiedOrbitPattern> UnifiedOrbitDatabase::getPatternsByCategory(UnifiedOrbitCategory category) const {
    auto it = _patternsByCategory.find(category);
    return it != _patternsByCategory.end() ? it->second : ::std::vector<UnifiedOrbitPattern>{};
}

::std::vector<UnifiedOrbitPattern> UnifiedOrbitDatabase::getPatternsForGrammar(uint8_t grammar_mode) const {
    ::std::vector<UnifiedOrbitPattern> result;
    for (const auto& pair : _patterns) {
        if (pair.second.supportsGrammar(grammar_mode)) {
            result.push_back(pair.second);
        }
    }
    return result;
}

::std::string UnifiedOrbitDatabase::getUnifiedASTNodeType(const ::std::string& pattern_name) const {
    const UnifiedOrbitPattern* pattern = getPattern(pattern_name);
    return pattern ? pattern->ast_node_type : "";
}

UnifiedOrbitDatabase::UnificationStats UnifiedOrbitDatabase::getUnificationStats() const {
    UnificationStats stats = {0, 0, 0, 0.0};

    stats.total_patterns = _patterns.size();

    for (const auto& pair : _patterns) {
        const UnifiedOrbitPattern& pattern = pair.second;
        if (pattern.category < UnifiedOrbitCategory::CPP2_Only) {
            // Common trunk patterns
            stats.unified_patterns++;
        } else {
            // Grammar-specific patterns
            stats.grammar_specific_patterns++;
        }
    }

    if (stats.total_patterns > 0) {
        stats.unification_ratio = static_cast<double>(stats.unified_patterns) / stats.total_patterns;
    }

    return stats;
}

} // namespace ir
} // namespace cppfort