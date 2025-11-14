#include "packrat_cache.h"

namespace cppfort::stage0 {

namespace {
constexpr size_t type_shift = 56; // assume OrbitType fits in 8 bits
}

size_t PackratCache::make_key(size_t pos, OrbitType type) {
    return (static_cast<size_t>(static_cast<unsigned>(type)) << type_shift) ^ pos;
}

bool PackratCache::has_cached(size_t pos, OrbitType type) const {
    return cache_.find(make_key(pos, type)) != cache_.end();
}

PackratEntry* PackratCache::get_cached(size_t pos, OrbitType type) {
    auto it = cache_.find(make_key(pos, type));
    return it == cache_.end() ? nullptr : &it->second;
}

const PackratEntry* PackratCache::get_cached(size_t pos, OrbitType type) const {
    auto it = cache_.find(make_key(pos, type));
    return it == cache_.end() ? nullptr : &it->second;
}

void PackratCache::store_cache(size_t pos, OrbitType type, double result) {
    cache_[make_key(pos, type)] = PackratEntry{pos, type, result};
}

void PackratCache::clear() {
    cache_.clear();
}

} // namespace cppfort::stage0

