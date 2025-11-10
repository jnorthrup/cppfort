#include "packrat_cache.h"
#include <cassert>
#include <iostream>

namespace cppfort::stage0 {

namespace {
constexpr size_t type_shift = 56; // assume OrbitType fits in 8 bits
}

size_t PackratCache::make_key(size_t pos, OrbitType type) {
    return (static_cast<size_t>(static_cast<unsigned>(type)) << type_shift) ^ pos;
}

PackratCache::PackratCache(const CacheConfig& config)
    : config_(config) {
    // Reserve space to reduce rehashing
    if (config.max_entries > 0) {
        cache_.reserve(config.max_entries);
        lru_iterators_.reserve(config.max_entries);
    }
}

bool PackratCache::has_cached(size_t pos, OrbitType type) const {
    size_t key = make_key(pos, type);
    auto it = cache_.find(key);
    
    if (it != cache_.end()) {
        // Update access tracking for LRU
        update_access(const_cast<PackratEntry&>(it->second), key);
        metrics_.record_hit();
        return true;
    }
    
    metrics_.record_miss();
    return false;
}

PackratEntry* PackratCache::get_cached(size_t pos, OrbitType type) {
    size_t key = make_key(pos, type);
    auto it = cache_.find(key);
    
    if (it != cache_.end()) {
        update_access(it->second, key);
        metrics_.record_hit();
        return &it->second;
    }
    
    metrics_.record_miss();
    return nullptr;
}

const PackratEntry* PackratCache::get_cached(size_t pos, OrbitType type) const {
    size_t key = make_key(pos, type);
    auto it = cache_.find(key);
    
    if (it != cache_.end()) {
        update_access(const_cast<PackratEntry&>(it->second), key);
        metrics_.record_hit();
        return &it->second;
    }
    
    metrics_.record_miss();
    return nullptr;
}

void PackratCache::store_cache(size_t pos, OrbitType type, double result) {
    size_t key = make_key(pos, type);
    
    // Check if we're updating an existing entry
    auto it = cache_.find(key);
    if (it != cache_.end()) {
        // Update existing entry
        it->second.result = result;
        update_access(it->second, key);
        metrics_.record_update();  // Count as update, not new entry
        return;
    }
    
    // Check if we need to evict before storing
    if (config_.enable_eviction && cache_.size() >= config_.max_entries) {
        evict_lru();
    }
    
    // Store new entry
    PackratEntry entry;
    entry.position = pos;
    entry.orbit_id = type;
    entry.result = result;
    entry.key = key;
    entry.access_count = 1;
    entry.last_access = std::chrono::steady_clock::now();
    
    cache_[key] = std::move(entry);
    metrics_.record_store();
    add_to_lru_front(key);
}

void PackratCache::clear() {
    cache_.clear();
    lru_order_.clear();
    lru_iterators_.clear();
    
    // Reset metrics but keep historical counts
    metrics_.current_entries = 0;
}

void PackratCache::update_access(PackratEntry& entry, size_t key) const {
    entry.access_count++;
    entry.last_access = std::chrono::steady_clock::now();
    
    // Update LRU order - move to front (most recent)
    remove_from_lru(key);
    add_to_lru_front(key);
}

void PackratCache::remove_from_lru(size_t key) const {
    auto it = lru_iterators_.find(key);
    if (it != lru_iterators_.end()) {
        lru_order_.erase(it->second);
        lru_iterators_.erase(it);
    }
}

void PackratCache::add_to_lru_front(size_t key) const {
    // Remove if already exists (shouldn't happen in normal flow, but be safe)
    remove_from_lru(key);
    
    // Add to front of list (most recently used)
    lru_order_.push_front(key);
    lru_iterators_[key] = lru_order_.begin();
}

void PackratCache::evict_lru() {
    if (lru_order_.empty()) {
        return;  // Nothing to evict
    }
    
    // Remove from back of list (least recently used)
    size_t lru_key = lru_order_.back();
    
    // Remove from cache and LRU tracking
    cache_.erase(lru_key);
    lru_order_.pop_back();
    lru_iterators_.erase(lru_key);
    
    metrics_.record_eviction();
    
    if (config_.log_evictions) {
        std::cerr << "[PackratCache] Evicted LRU entry (key=" << lru_key << ")\n";
    }
}

size_t PackratCache::evict_if_needed() {
    if (!config_.enable_eviction) {
        return 0;
    }
    
    size_t evicted = 0;
    while (cache_.size() > config_.max_entries && !lru_order_.empty()) {
        evict_lru();
        evicted++;
    }
    
    return evicted;
}

size_t PackratCache::force_evict(size_t target_count) {
    if (target_count >= cache_.size()) {
        size_t evicted = cache_.size();
        clear();
        return evicted;
    }
    
    size_t evicted = 0;
    size_t target_size = cache_.size() - target_count;
    
    while (cache_.size() > target_size && !lru_order_.empty()) {
        evict_lru();
        evicted++;
    }
    
    return evicted;
}

} // namespace cppfort::stage0