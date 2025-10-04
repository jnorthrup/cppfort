#pragma once

#include <cstdint>
#include <string>
#include <vector>
#include <unordered_map>
#include <regex>
#include "tblgen_patterns.h"
#include "lattice_classes.h"

namespace cppfort::stage0 {

/**
 * CPP2-Keyed Peer Resolution System
 *
 * Uses CPP2 canonical patterns as index keys to activate
 * C/C++ peer context nodes for disambiguation.
 *
 * Architecture:
 *   CPP2 Pattern (key) → Similarity Score → Peer Activation → Confidence Boost/Penalty
 */

struct PeerNode {
    std::string context_name;           // "C_bitfield", "C++_uniform_init", etc.
    double similarity_threshold;         // Min CPP2 similarity to activate (0.0-1.0)
    std::vector<std::string> scope_filter;  // Required scopes or empty for "any"
    double confidence_modifier;          // Multiplier for confidence (>1.0 = boost, <1.0 = penalty)
    uint16_t lattice_required;          // Optional lattice constraint (0xFFFF = any)
    uint8_t grammar_mode;               // OrbitPattern::GrammarMode bits
};

struct CPP2Key {
    std::string canonical_pattern;      // CPP2 pattern as regex
    std::vector<PeerNode> peers;        // C/C++ contexts keyed by this pattern
    std::regex compiled_pattern;        // Precompiled for fast matching

    CPP2Key(const std::string& pattern, const std::vector<PeerNode>& peer_list)
        : canonical_pattern(pattern)
        , peers(peer_list)
        , compiled_pattern(pattern, std::regex::optimize)
    {}
};

class CPP2KeyResolver {
public:
    CPP2KeyResolver() {
        build_key_database();
    }

    /**
     * Resolve orbit candidates using CPP2 key matching
     *
     * Returns refined candidates with adjusted confidence scores
     * based on CPP2 canonical pattern similarity
     */
    std::vector<cppfort::ir::OrbitPattern> resolve_with_cpp2_keys(
        const std::string& token_sequence,
        std::vector<cppfort::ir::OrbitPattern> candidates,
        const std::string& scope_type,
        uint16_t lattice_mask
    ) const;

private:
    std::vector<CPP2Key> key_database_;

    /**
     * Build CPP2 key database from canonical patterns
     * Extracted from docs/cpp2/*.md documentation
     */
    void build_key_database();

    /**
     * Compute CPP2 canonical pattern similarity
     * Returns 0.0-1.0 score indicating how well token sequence matches CPP2 form
     */
    double compute_cpp2_similarity(const std::string& token_sequence, const CPP2Key& key) const;

    /**
     * Check if peer node constraints are satisfied
     */
    bool peer_constraints_satisfied(
        const PeerNode& peer,
        const std::string& scope_type,
        uint16_t lattice_mask
    ) const;
};

} // namespace cppfort::stage0
