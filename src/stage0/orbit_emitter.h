#pragma once

#include <string>
#include <string_view>
#include <vector>

#include "orbit_ring.h"
#include "orbit_iterator.h"

namespace cppfort::stage0 {

// Reconstructs tokens from orbit evidence
class OrbitEmitter {
public:
    struct Token {
        size_t start_pos;
        size_t end_pos;
        std::string text;
        double confidence;
        std::string orbit_type;
    };

    // Emit tokens from an orbit, preserving input structure
    std::vector<Token> emit_tokens(Orbit* orbit, std::string_view source) const;

    // Full pipeline: iterate orbits and emit token stream
    std::string reconstruct_source(OrbitIterator& iterator, std::string_view source) const;

private:
    // Extract the most confident evidence text for position range
    std::string extract_text(Orbit* orbit, std::string_view source) const;

    // Determine token boundaries from orbit structure
    std::vector<std::pair<size_t, size_t>> compute_boundaries(Orbit* orbit) const;
};

} // namespace cppfort::stage0