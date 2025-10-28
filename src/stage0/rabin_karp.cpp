#include "rabin_karp.h"
#include "orbit_mask.h"
#include <algorithm>

namespace cppfort::ir {

// Static constexpr definitions
static constexpr size_t NUM_ORBIT_TYPES = 6;
static constexpr uint64_t PRIME = 31;

RabinKarp::RabinKarp() {
    _orbit_counts.fill(0);
    _orbit_hashes.fill(0);
    _orbit_powers.fill(1);
    
    // Precompute powers for orbit hashing
    for (size_t i = 1; i < NUM_ORBIT_TYPES; ++i) {
        _orbit_powers[i] = (_orbit_powers[i - 1] * PRIME) % UINT64_MAX;
    }
}

::std::array<uint64_t, NUM_ORBIT_TYPES> RabinKarp::processOrbitContext(const OrbitContext& context) {
    ::std::array<uint64_t, NUM_ORBIT_TYPES> orbit_hashes = {0};

    // Get orbit counts from context
    _orbit_counts = context.getCounts();

    // Compute hierarchical hashes for each orbit type
    for (size_t type = 0; type < NUM_ORBIT_TYPES; ++type) {
        uint64_t hash = 0;
        size_t count = _orbit_counts[type];
        
        // Hierarchical hash: each level contributes to the hash
        for (size_t level = 0; level < count; ++level) {
            hash = (hash + _orbit_powers[level % NUM_ORBIT_TYPES]) % UINT64_MAX;
        }
        
        orbit_hashes[type] = hash;
        _orbit_hashes[type] = hash;
    }

    return orbit_hashes;
}

void RabinKarp::updateOrbitContext(const OrbitContext& context) {
    ::std::array<uint64_t, NUM_ORBIT_TYPES> hashes = {0};

    // Get orbit counts from context
    _orbit_counts = context.getCounts();

    // Compute hierarchical hashes for each orbit type
    for (size_t type = 0; type < NUM_ORBIT_TYPES; ++type) {
        uint64_t hash = 0;
        size_t count = _orbit_counts[type];
        
        // Hierarchical hash: each level contributes to the hash
        for (size_t level = 0; level < count; ++level) {
            hash = (hash + _orbit_powers[level % NUM_ORBIT_TYPES]) % UINT64_MAX;
        }
        
        hashes[type] = hash;
        _orbit_hashes[type] = hash;
    }
}

uint64_t RabinKarp::hashAt(size_t orbitType) const {
    if (orbitType >= NUM_ORBIT_TYPES) return 0;
    return _orbit_hashes[orbitType];
}

size_t RabinKarp::orbitCount(size_t orbitType) const {
    if (orbitType >= NUM_ORBIT_TYPES) return 0;
    return _orbit_counts[orbitType];
}

void RabinKarp::reset() {
    _orbit_counts.fill(0);
    _orbit_hashes.fill(0);
}

} // namespace cppfort::ir
