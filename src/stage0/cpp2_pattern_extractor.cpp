#include "cpp2_pattern_extractor.h"
#include <yaml-cpp/yaml.h>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <sstream>

namespace cppfort {
namespace stage0 {

CPP2PatternExtractor::CPP2PatternExtractor() = default;

CPP2PatternExtractor::~CPP2PatternExtractor() = default;

bool CPP2PatternExtractor::loadPatternsFromYAML(const std::filesystem::path& yamlPath) {
    try {
        m_patterns.clear();

        if (!std::filesystem::exists(yamlPath)) {
            std::cerr << "YAML file not found: " << yamlPath << std::endl;
            return false;
        }

        YAML::Node root = YAML::LoadFile(yamlPath.string());

        if (!root.IsMap()) {
            std::cerr << "Invalid YAML structure: expected map at root" << std::endl;
            return false;
        }

        // Load CPP2 patterns
        if (root["cpp2_patterns"] && root["cpp2_patterns"].IsSequence()) {
            for (const auto& patternNode : root["cpp2_patterns"]) {
                CPP2Pattern pattern;
                if (parsePatternNode(patternNode, pattern)) {
                    m_patterns.push_back(pattern);
                }
            }
        }

        // Load peer mappings
        if (root["peer_mappings"] && root["peer_mappings"].IsSequence()) {
            for (const auto& mappingNode : root["peer_mappings"]) {
                PeerMapping mapping;
                if (parsePeerMappingNode(mappingNode, mapping)) {
                    m_peerMappings.push_back(mapping);
                }
            }
        }

        std::cout << "Loaded " << m_patterns.size() << " CPP2 patterns and "
                  << m_peerMappings.size() << " peer mappings" << std::endl;
        return true;

    } catch (const YAML::Exception& e) {
        std::cerr << "YAML parsing error: " << e.what() << std::endl;
        return false;
    } catch (const std::exception& e) {
        std::cerr << "Error loading YAML: " << e.what() << std::endl;
        return false;
    }
}

bool CPP2PatternExtractor::parsePatternNode(const YAML::Node& node, CPP2Pattern& pattern) {
    if (!node.IsMap()) {
        return false;
    }

    try {
        pattern.name = node["name"].as<std::string>("");
        pattern.canonical_form = node["canonical_form"].as<std::string>("");
        pattern.semantic_context = node["semantic_context"].as<std::string>("");
        pattern.scope_requirement = node["scope_requirement"].as<std::string>("any");
        pattern.similarity_threshold = node["similarity_threshold"].as<double>(0.8);
        pattern.confidence_modifier = node["confidence_modifier"].as<double>(1.0);

        if (node["lattice_filter"] && node["lattice_filter"].IsSequence()) {
            for (const auto& filter : node["lattice_filter"]) {
                pattern.lattice_filter.push_back(filter.as<std::string>());
            }
        }

        if (node["prev_tokens"] && node["prev_tokens"].IsSequence()) {
            for (const auto& token : node["prev_tokens"]) {
                pattern.prev_tokens.push_back(token.as<std::string>());
            }
        }

        if (node["next_tokens"] && node["next_tokens"].IsSequence()) {
            for (const auto& token : node["next_tokens"]) {
                pattern.next_tokens.push_back(token.as<std::string>());
            }
        }

        return !pattern.name.empty() && !pattern.canonical_form.empty();

    } catch (const YAML::Exception& e) {
        std::cerr << "Error parsing pattern node: " << e.what() << std::endl;
        return false;
    }
}

bool CPP2PatternExtractor::parsePeerMappingNode(const YAML::Node& node, PeerMapping& mapping) {
    if (!node.IsMap()) {
        return false;
    }

    try {
        mapping.cpp2_key = node["cpp2_key"].as<std::string>("");
        mapping.c_pattern = node["c_pattern"].as<std::string>("");
        mapping.cpp_pattern = node["cpp_pattern"].as<std::string>("");
        mapping.similarity_score = node["similarity_score"].as<double>(0.0);

        return !mapping.cpp2_key.empty();

    } catch (const YAML::Exception& e) {
        std::cerr << "Error parsing peer mapping node: " << e.what() << std::endl;
        return false;
    }
}

const std::vector<CPP2Pattern>& CPP2PatternExtractor::getPatterns() const {
    return m_patterns;
}

const std::vector<PeerMapping>& CPP2PatternExtractor::getPeerMappings() const {
    return m_peerMappings;
}

std::vector<CPP2Pattern> CPP2PatternExtractor::findPatternsByContext(const std::string& context) const {
    std::vector<CPP2Pattern> matches;

    for (const auto& pattern : m_patterns) {
        if (pattern.semantic_context.find(context) != std::string::npos) {
            matches.push_back(pattern);
        }
    }

    return matches;
}

std::vector<PeerMapping> CPP2PatternExtractor::findPeersForKey(const std::string& cpp2Key) const {
    std::vector<PeerMapping> matches;

    for (const auto& mapping : m_peerMappings) {
        if (mapping.cpp2_key == cpp2Key) {
            matches.push_back(mapping);
        }
    }

    return matches;
}

bool CPP2PatternExtractor::validatePatterns() const {
    for (const auto& pattern : m_patterns) {
        if (pattern.name.empty() || pattern.canonical_form.empty()) {
            std::cerr << "Invalid pattern: missing name or canonical form" << std::endl;
            return false;
        }

        if (pattern.similarity_threshold < 0.0 || pattern.similarity_threshold > 1.0) {
            std::cerr << "Invalid similarity threshold for pattern " << pattern.name << std::endl;
            return false;
        }
    }

    return true;
}

void CPP2PatternExtractor::printSummary() const {
    std::cout << "CPP2 Pattern Extractor Summary:" << std::endl;
    std::cout << "  Patterns: " << m_patterns.size() << std::endl;
    std::cout << "  Peer Mappings: " << m_peerMappings.size() << std::endl;

    if (!m_patterns.empty()) {
        std::cout << "  Pattern categories:" << std::endl;
        std::unordered_map<std::string, size_t> categories;
        for (const auto& pattern : m_patterns) {
            categories[pattern.semantic_context]++;
        }
        for (const auto& [category, count] : categories) {
            std::cout << "    " << category << ": " << count << std::endl;
        }
    }
}

} // namespace stage0
} // namespace cppfort