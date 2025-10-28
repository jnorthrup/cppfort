#pragma once

#include <string>
#include <vector>
#include "orbit_ring.h"

namespace cppfort::stage0 {

/**
 * Speculative match result from pattern matching
 */
struct SpeculativeMatch {
    size_t match_length;        // Length of the match
    double confidence;          // Match confidence (0.0-1.0)
    std::string pattern_name;   // Name of the matched pattern
    OrbitFragment result;       // Resulting fragment

    SpeculativeMatch() = default;
    SpeculativeMatch(size_t len, double conf, std::string name, OrbitFragment frag)
        : match_length(len), confidence(conf), pattern_name(std::move(name)), result(std::move(frag)) {}
};

} // namespace cppfort::stage0