#pragma once

#include <cstddef>
#include <unordered_map>

#include "orbit_ring.h"

namespace cppfort::stage0 {

struct PackratEntry {
    size_t position = 0;
    OrbitType orbit_id = OrbitType::Confix;
    double result = 0.0;
};

class PackratCache {
public:
    bool has_cached(size_t pos, OrbitType type) const;
    PackratEntry* get_cached(size_t pos, OrbitType type);
    const PackratEntry* get_cached(size_t pos, OrbitType type) const;
    void store_cache(size_t pos, OrbitType type, double result);
    void clear();

private:
    static size_t make_key(size_t pos, OrbitType type);

    std::unordered_map<size_t, PackratEntry> cache_;
};

} // namespace cppfort::stage0

