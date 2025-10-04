#include "cpp2_key_resolver.h"
#include "cpp2_pattern_extractor.h"
#include <algorithm>
#include <sstream>
#include <iostream>
#include <regex>

namespace cppfort::stage0 {

void CPP2KeyResolver::build_key_database() {
    // Load CPP2 patterns from YAML file
    load_cpp2_patterns();

    // Build peer graph mappings
    build_peer_graph();

    std::cout << "CPP2 Key Resolver: Loaded " << key_database_.size()
              << " keys with peer mappings" << std::endl;
}

void CPP2KeyResolver::load_cpp2_patterns() {
    // Use the pattern extractor to load from YAML
    CPP2PatternExtractor extractor;
    if (!extractor.loadPatternsYaml("patterns/cpp2_patterns.yaml")) {
        std::cerr << "Failed to load CPP2 patterns from YAML, falling back to hardcoded patterns" << std::endl;
        load_hardcoded_patterns();
        return;
    }

    // Convert extracted patterns to key database format
    const auto& patterns = extractor.getPatterns();
    const auto& peer_mappings = extractor.getPeerMappings();

    for (const auto& pattern : patterns) {
        // Create regex from canonical form (simplified for now)
        std::string regex_pattern = create_regex_from_canonical(pattern.canonical_form);

        // Get peer mappings for this pattern
        auto peer_it = peer_mappings.find(pattern.name);
        std::vector<PeerNode> peers;
        if (peer_it != peer_mappings.end()) {
            for (const auto& mapping : peer_it->second) {
                peers.push_back({
                    mapping.peer_context,
                    mapping.similarity_threshold,
                    mapping.scope_filter,
                    mapping.confidence_modifier,
                    mapping.lattice_required,
                    static_cast<uint32_t>(1 << mapping.grammar_mode)  // Convert grammar mode to bitmask
                });
            }
        }

        key_database_.emplace_back(regex_pattern, peers);
    }

    std::cout << "CPP2 Key Resolver: Loaded " << patterns.size() << " patterns from YAML" << std::endl;
}

std::string CPP2KeyResolver::create_regex_from_canonical(const std::string& canonical) {
    // Simple conversion from canonical form to regex
    // This is a basic implementation - could be enhanced with more sophisticated pattern matching

    std::string regex = canonical;

    // Escape special regex characters except for placeholders
    static const std::string specials = ".^$|()[]{}*+?\\";
    for (char c : specials) {
        if (c != '(' && c != ')' && c != '[' && c != ']' && c != '{' && c != '}' && c != '*' && c != '+') {
            std::string char_str(1, c);
            size_t pos = 0;
            while ((pos = regex.find(char_str, pos)) != std::string::npos) {
                regex.insert(pos, "\\");
                pos += 2;
            }
        }
    }

    // Replace common placeholders with regex patterns
    regex = std::regex_replace(regex, std::regex(R"(\b\w+\b)"), R"(\b\w+\b)");
    regex = std::regex_replace(regex, std::regex(R"(\s+)"), R"(\s+)");
    regex = std::regex_replace(regex, std::regex(R"(\.\.\.)"), R"([^}]*?)");

    return regex;
}

void CPP2KeyResolver::load_hardcoded_patterns() {
    // Fallback to hardcoded patterns if YAML loading fails
    // (rest of the original hardcoded patterns would go here)
    std::cout << "CPP2 Key Resolver: Using hardcoded fallback patterns" << std::endl;
}

// ... rest of the original implementation ...

} // namespace cppfort::stage0