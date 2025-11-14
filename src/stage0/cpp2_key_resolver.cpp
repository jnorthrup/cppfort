#include "cpp2_key_resolver.h"
#include <algorithm>
#include <sstream>
#include <iostream>

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
    // For now, hardcode key CPP2 patterns based on documentation analysis
    // In production, this would load from patterns/cpp2_patterns.yaml

    // Type annotation pattern: x: int = 5
    key_database_.emplace_back(
        R"(\b\w+\s*:\s*(?!namespace|type)\w+(\s*=\s*[^;]+)?)",
        std::vector<PeerNode>{
            {
                "C_variable_declaration",
                0.7,  // similarity threshold
                {"function_body", "global"},
                0.8,  // confidence modifier
                0xFFFF,  // any lattice
                0x01   // C mode
            },
            {
                "CPP_auto_declaration",
                0.8,
                {"function_body", "global"},
                1.2,
                0xFFFF,
                0x02  // CPP mode
            }
        }
    );

    // Function signature pattern: f: (x: int) -> int = body
    key_database_.emplace_back(
        R"(\b\w+\s*:\s*\([^)]*\)\s*(->\s*\w+)?\s*=)",
        std::vector<PeerNode>{
            {
                "C_function_definition",
                0.6,
                {"global"},
                0.7,
                0xFFFF,
                0x01
            },
            {
                "CPP_lambda_definition",
                0.9,
                {"function_body", "global"},
                1.1,
                0xFFFF,
                0x02
            }
        }
    );

    // Type definition pattern: T: type = {...}
    key_database_.emplace_back(
        R"(\b\w+\s*:\s*type\s*=)",
        std::vector<PeerNode>{
            {
                "C_struct_definition",
                0.8,
                {"global"},
                0.9,
                0xFFFF,
                0x01
            },
            {
                "CPP_class_definition",
                0.85,
                {"global", "namespace_body"},
                1.0,
                0xFFFF,
                0x02
            }
        }
    );

    // Namespace pattern: ns: namespace = {...}
    key_database_.emplace_back(
        R"(\b\w+\s*:\s*namespace\s*=)",
        std::vector<PeerNode>{
            {
                "C_no_direct_equivalent",
                0.0,  // Not applicable in C
                {},
                0.0,
                0xFFFF,
                0x01
            },
            {
                "CPP_namespace_definition",
                0.95,
                {"global"},
                1.0,
                0xFFFF,
                0x02
            }
        }
    );

    // Object construction pattern: T{...} or T(...)
    key_database_.emplace_back(
        R"(\b\w+\s*[{(][^}]*[})])",
        std::vector<PeerNode>{
            {
                "C_struct_initialization",
                0.8,
                {"function_body", "global"},
                1.0,
                0xFFFF,
                0x01
            },
            {
                "CPP_uniform_initialization",
                0.95,
                {"function_body", "global"},
                1.3,
                0xFFFF,
                0x02
            }
        }
    );

    // Member access pattern: obj.field
    key_database_.emplace_back(
        R"(\b\w+\.\w+)",
        std::vector<PeerNode>{
            {
                "C_struct_member_access",
                0.9,
                {"function_body"},
                1.0,
                0xFFFF,
                0x01
            },
            {
                "CPP_member_access",
                0.95,
                {"function_body", "class_body"},
                1.1,
                0xFFFF,
                0x02
            }
        }
    );

    // Range-for pattern: for (x: container) = {...}
    key_database_.emplace_back(
        R"(\bfor\s*\(\s*\w+\s*:\s*[^)]+\)\s*=)",
        std::vector<PeerNode>{
            {
                "C_no_direct_equivalent",
                0.0,
                {},
                0.0,
                0xFFFF,
                0x01
            },
            {
                "CPP_range_based_for",
                0.9,
                {"function_body"},
                1.2,
                0xFFFF,
                0x02
            }
        }
    );

    // Contract pattern: pre: (...) = ..., post: (...) = ...
    key_database_.emplace_back(
        R"(\b(pre|post|assert)\s*:\s*\([^)]*\)\s*=)",
        std::vector<PeerNode>{
            {
                "C_no_direct_equivalent",
                0.0,
                {},
                0.0,
                0xFFFF,
                0x01
            },
            {
                "CPP_contract_specification",
                0.85,
                {"function_body"},
                1.0,
                0xFFFF,
                0x02
            }
        }
    );
}

void CPP2KeyResolver::build_peer_graph() {
    // Peer graph is built during key_database_ construction
    // Each CPP2Key contains its peer mappings
    std::cout << "CPP2 Peer Graph: " << key_database_.size() << " keys mapped" << std::endl;
}

double CPP2KeyResolver::compute_cpp2_similarity(const std::string& token_sequence,
                                               const CPP2Key& key) const {
    // RBCursive combinator-based pattern matching (no regex)
    cppfort::ir::RBCursiveScanner scanner;

    // Use glob matching for pattern similarity
    if (scanner.matchGlob(token_sequence, key.pattern)) {
        // Exact glob match
        return 1.0;
    }

    // Partial match: count matching characters
    size_t matches = 0;
    size_t pattern_idx = 0;

    for (size_t i = 0; i < token_sequence.length() && pattern_idx < key.pattern.length(); ++i) {
        if (key.pattern[pattern_idx] == '*') {
            matches++;
            continue;
        }
        if (token_sequence[i] == key.pattern[pattern_idx]) {
            matches++;
            pattern_idx++;
        }
    }

    double match_ratio = static_cast<double>(matches) / token_sequence.length();
    return std::min(match_ratio, 1.0);
}

bool CPP2KeyResolver::peer_constraints_satisfied(const PeerNode& peer,
                                                const std::string& scope_type,
                                                uint16_t lattice_mask) const {
    // Check scope filter
    if (!peer.scope_filter.empty()) {
        bool scope_match = std::find(peer.scope_filter.begin(),
                                    peer.scope_filter.end(),
                                    scope_type) != peer.scope_filter.end();
        if (!scope_match) {
            return false;
        }
    }

    // Check lattice requirements
    if (peer.lattice_required != 0xFFFF) {
        if ((lattice_mask & peer.lattice_required) == 0) {
            return false;
        }
    }

    return true;
}

std::vector<cppfort::ir::OrbitPattern> CPP2KeyResolver::resolve_with_cpp2_keys(
    const std::string& token_sequence,
    std::vector<cppfort::ir::OrbitPattern> candidates,
    const std::string& scope_type,
    uint16_t lattice_mask
) const {
    std::vector<cppfort::ir::OrbitPattern> refined_candidates = candidates;

    // Find best CPP2 key match
    double best_similarity = 0.0;
    const CPP2Key* best_key = nullptr;

    for (const auto& key : key_database_) {
        double similarity = compute_cpp2_similarity(token_sequence, key);
        if (similarity > best_similarity) {
            best_similarity = similarity;
            best_key = &key;
        }
    }

    // If we found a good CPP2 match, activate peers
    if (best_key && best_similarity >= 0.5) {  // Minimum threshold
        for (const auto& peer : best_key->peers) {
            // Check if peer constraints are satisfied
            if (!peer_constraints_satisfied(peer, scope_type, lattice_mask)) {
                continue;
            }

            // Check similarity threshold
            if (best_similarity < peer.similarity_threshold) {
                continue;
            }

            // Apply peer activation to candidates
            for (auto& candidate : refined_candidates) {
                // Check if candidate matches peer context
                if (candidate_matches_peer(candidate, peer)) {
                    // Apply confidence modifier
                    candidate.weight *= peer.confidence_modifier;

                    // Ensure weight stays in valid range
                    candidate.weight = std::max(0.0, std::min(1.0, candidate.weight));
                }
            }
        }
    }

    return refined_candidates;
}

bool CPP2KeyResolver::candidate_matches_peer(const cppfort::ir::OrbitPattern& candidate,
                                           const PeerNode& peer) const {
    // Simple string matching for peer context
    // In production, this would be more sophisticated

    std::string candidate_context = candidate.name;

    // Extract context from candidate name (simplified)
    if (candidate_context.find(peer.context_name) != std::string::npos) {
        return true;
    }

    // Check grammar mode compatibility
    if ((candidate.grammar_modes & peer.grammar_mode) != 0) {
        return true;
    }

    return false;
}

} // namespace cppfort::stage0