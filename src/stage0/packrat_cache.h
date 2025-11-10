#pragma once

#include <cstddef>
#include <chrono>
#include <list>
#include <unordered_map>
#include <vector>

#include "orbit_ring.h"

namespace cppfort::stage0 {

// Eviction policy selection
enum class EvictionPolicy {
    LRU,    // Least Recently Used
    LFU,    // Least Frequently Used (placeholder for future)
    TTL     // Time To Live (placeholder for future)
};

// Configuration for cache behavior
struct CacheConfig {
    size_t max_entries = 100000;  // Reasonable default for 1GB heap limit
    bool enable_eviction = true;
    std::chrono::milliseconds entry_ttl{0};  // 0 = no expiration
    EvictionPolicy policy = EvictionPolicy::LRU;
    bool log_evictions = false;  // For debugging
};

// Metrics for cache monitoring
struct CacheMetrics {
    size_t current_entries = 0;
    size_t peak_entries = 0;
    size_t total_hits = 0;
    size_t total_misses = 0;
    size_t evictions = 0;
    size_t stores = 0;
    
    double hit_rate() const {
        const size_t total_requests = total_hits + total_misses;
        return total_requests > 0 ? static_cast<double>(total_hits) / total_requests : 0.0;
    }
    
    void record_hit() { total_hits++; }
    void record_miss() { total_misses++; }
    void record_store() { 
        stores++; 
        current_entries++;
        if (current_entries > peak_entries) {
            peak_entries = current_entries;
        }
    }
    void record_update() { 
        stores++;  // Count as a store operation but don't change entry count
    }
    void record_eviction(size_t count = 1) { 
        evictions += count;
        current_entries -= count;
    }
};

// Enhanced cache entry with metadata for eviction
struct PackratEntry {
    size_t position = 0;
    OrbitType orbit_id = OrbitType::Confix;
    double result = 0.0;
    
    // For LRU/LFU/TTL tracking
    std::chrono::steady_clock::time_point last_access;
    size_t access_count = 0;
    size_t key = 0;  // Store the cache key for easy eviction
};

class PackratCache {
public:
    explicit PackratCache(const CacheConfig& config = {});
    
    bool has_cached(size_t pos, OrbitType type) const;
    PackratEntry* get_cached(size_t pos, OrbitType type);
    const PackratEntry* get_cached(size_t pos, OrbitType type) const;
    void store_cache(size_t pos, OrbitType type, double result);
    void clear();
    
    // Metrics access
    const CacheMetrics& metrics() const { return metrics_; }
    
    // Manual eviction control
    size_t evict_if_needed();
    size_t force_evict(size_t target_count);
    
    // Configuration access
    const CacheConfig& config() const { return config_; }

private:
    static size_t make_key(size_t pos, OrbitType type);
    void evict_lru();
    void update_access(PackratEntry& entry, size_t key) const;
    void remove_from_lru(size_t key) const;
    void add_to_lru_front(size_t key) const;
    
    CacheConfig config_;
    mutable CacheMetrics metrics_;  // mutable to allow updates in const methods
    std::unordered_map<size_t, PackratEntry> cache_;
    
    // LRU tracking: keys in access order (front = most recent, back = least recent)
    // Marked mutable to allow updates in const methods (has_cached, get_cached)
    mutable std::list<size_t> lru_order_;
    mutable std::unordered_map<size_t, std::list<size_t>::iterator> lru_iterators_;
};

} // namespace cppfort::stage0

