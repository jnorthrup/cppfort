#pragma once

#include <array>
#include <cstdint>
#include <string>
#include <vector>

namespace cppfort::ir {

class OrbitContext;

/**
 * Chapter 19: Rabin-Karp Rolling Hash Implementation
 *
 * Implements hierarchical orbit-based rolling hash for deep pattern matching.
 * Uses orbit anchor and parameter counts for semantic hashing with hierarchical depth analysis.
 *
 * Based on ororoboros-couchduck reference implementation.
 * Prime = 31 for hash calculations.
 */
class RabinKarp {
private:
    static constexpr uint64_t PRIME = 31;
    // Orbit element types: brace, bracket, angle, paren, quote, number
    static constexpr size_t NUM_ORBIT_TYPES = 6;
    
    ::std::array<uint64_t, NUM_ORBIT_TYPES> _orbit_hashes;
    ::std::array<uint64_t, NUM_ORBIT_TYPES> _orbit_powers;
    ::std::array<size_t, NUM_ORBIT_TYPES> _orbit_counts;

public:
    /**
     * Constructor for orbit-based hierarchical hashing.
     */
    RabinKarp();

    /**
     * Process orbit context and compute hierarchical hashes based on anchor/parameter counts.
     * Returns array of hashes for each orbit type: [brace, bracket, angle, paren, quote, number]
     */
    ::std::array<uint64_t, NUM_ORBIT_TYPES> processOrbitContext(const OrbitContext& context);

    /**
     * Update hashes with new orbit context.
     */
    void updateOrbitContext(const OrbitContext& context);

    /**
     * Get hash value for specific orbit type (0-5).
     */
    uint64_t hashAt(size_t orbitType) const;

    /**
     * Get orbit count for specific type.
     */
    size_t orbitCount(size_t orbitType) const;

    /**
     * Reset all hashes to initial state.
     */
    void reset();
};

} // namespace cppfort::ir