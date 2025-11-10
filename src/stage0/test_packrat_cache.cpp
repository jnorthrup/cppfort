#include <iostream>
#include <cassert>
#include <vector>
#include <chrono>
#include <thread>

#include "packrat_cache.h"

using namespace cppfort::stage0;

void test_default_config() {
    std::cout << "Testing default cache configuration...\n";
    
    PackratCache cache;  // Uses default config
    
    const auto& config = cache.config();
    assert(config.max_entries == 100000);
    assert(config.enable_eviction == true);
    assert(config.policy == EvictionPolicy::LRU);
    
    const auto& metrics = cache.metrics();
    assert(metrics.current_entries == 0);
    assert(metrics.peak_entries == 0);
    assert(metrics.total_hits == 0);
    assert(metrics.total_misses == 0);
    assert(metrics.evictions == 0);
    assert(metrics.hit_rate() == 0.0);
    
    std::cout << "✓ Default config test passed\n";
}

void test_custom_config() {
    std::cout << "Testing custom cache configuration...\n";
    
    CacheConfig config;
    config.max_entries = 100;
    config.enable_eviction = true;
    config.policy = EvictionPolicy::LRU;
    config.log_evictions = true;
    
    PackratCache cache(config);
    
    assert(cache.config().max_entries == 100);
    assert(cache.metrics().current_entries == 0);
    
    std::cout << "✓ Custom config test passed\n";
}

void test_basic_storage_and_retrieval() {
    std::cout << "Testing basic cache storage and retrieval...\n";
    
    CacheConfig config;
    config.max_entries = 1000;
    PackratCache cache(config);
    
    // Store some entries
    cache.store_cache(10, OrbitType::Confix, 1.0);
    cache.store_cache(20, OrbitType::Keyword, 2.0);
    cache.store_cache(30, OrbitType::Operator, 3.0);
    
    // Verify metrics
    const auto& metrics = cache.metrics();
    assert(metrics.current_entries == 3);
    assert(metrics.stores == 3);
    assert(metrics.peak_entries == 3);
    
    // Retrieve entries
    auto* entry1 = cache.get_cached(10, OrbitType::Confix);
    assert(entry1 != nullptr);
    assert(entry1->position == 10);
    assert(entry1->result == 1.0);
    
    auto* entry2 = cache.get_cached(20, OrbitType::Keyword);
    assert(entry2 != nullptr);
    assert(entry2->result == 2.0);
    
    // Test miss
    auto* entry_miss = cache.get_cached(999, OrbitType::Confix);
    assert(entry_miss == nullptr);
    
    // Verify hit/miss metrics
    assert(metrics.total_hits == 2);
    assert(metrics.total_misses == 1);
    assert(metrics.hit_rate() == 2.0 / 3.0);
    
    std::cout << "✓ Basic storage and retrieval test passed\n";
}

void test_lru_eviction() {
    std::cout << "Testing LRU eviction policy...\n";
    
    CacheConfig config;
    config.max_entries = 3;  // Very small limit for testing
    config.enable_eviction = true;
    config.log_evictions = false;  // Set to true for debugging
    
    PackratCache cache(config);
    
    // Store entries in order: A, B, C
    cache.store_cache(10, OrbitType::Confix, 1.0);      // A
    cache.store_cache(20, OrbitType::Keyword, 2.0);     // B
    cache.store_cache(30, OrbitType::Operator, 3.0);    // C
    
    // Verify all three are in cache
    assert(cache.get_cached(10, OrbitType::Confix) != nullptr);
    assert(cache.get_cached(20, OrbitType::Keyword) != nullptr);
    assert(cache.get_cached(30, OrbitType::Operator) != nullptr);
    
    // Access A to make it recently used (order: A, C, B)
    cache.get_cached(10, OrbitType::Confix);
    
    // Store D - should evict B (least recently used)
    cache.store_cache(40, OrbitType::Identifier, 4.0);  // D
    
    // Verify D is in cache
    assert(cache.get_cached(40, OrbitType::Identifier) != nullptr);
    
    // Verify A and C are still in cache
    assert(cache.get_cached(10, OrbitType::Confix) != nullptr);
    assert(cache.get_cached(30, OrbitType::Operator) != nullptr);
    
    // Verify B was evicted
    assert(cache.get_cached(20, OrbitType::Keyword) == nullptr);
    
    // Verify eviction metrics
    const auto& metrics = cache.metrics();
    assert(metrics.evictions == 1);
    assert(metrics.current_entries == 3);
    
    std::cout << "✓ LRU eviction test passed\n";
}

void test_update_existing_entry() {
    std::cout << "Testing update of existing cache entry...\n";
    
    CacheConfig config;
    config.max_entries = 100;
    PackratCache cache(config);
    
    // Store an entry
    cache.store_cache(10, OrbitType::Confix, 1.0);
    
    // Update the same entry
    cache.store_cache(10, OrbitType::Confix, 2.0);
    
    // Should only have one entry
    const auto& metrics = cache.metrics();
    assert(metrics.current_entries == 1);
    assert(metrics.stores == 2);  // Two store operations
    
    // Verify the value was updated
    auto* entry = cache.get_cached(10, OrbitType::Confix);
    assert(entry != nullptr);
    assert(entry->result == 2.0);
    
    std::cout << "✓ Update existing entry test passed\n";
}

void test_multiple_orbit_types_same_position() {
    std::cout << "Testing multiple orbit types at same position...\n";
    
    CacheConfig config;
    config.max_entries = 100;
    PackratCache cache(config);
    
    // Store different orbit types at same position
    cache.store_cache(100, OrbitType::Confix, 1.0);
    cache.store_cache(100, OrbitType::Keyword, 2.0);
    cache.store_cache(100, OrbitType::Operator, 3.0);
    cache.store_cache(100, OrbitType::Identifier, 4.0);
    cache.store_cache(100, OrbitType::Literal, 5.0);
    
    // Verify all are stored and distinct
    const auto& metrics = cache.metrics();
    assert(metrics.current_entries == 5);
    
    assert(cache.get_cached(100, OrbitType::Confix)->result == 1.0);
    assert(cache.get_cached(100, OrbitType::Keyword)->result == 2.0);
    assert(cache.get_cached(100, OrbitType::Operator)->result == 3.0);
    assert(cache.get_cached(100, OrbitType::Identifier)->result == 4.0);
    assert(cache.get_cached(100, OrbitType::Literal)->result == 5.0);
    
    std::cout << "✓ Multiple orbit types test passed\n";
}

void test_eviction_under_pressure() {
    std::cout << "Testing eviction under pressure (many entries)...\n";
    
    CacheConfig config;
    config.max_entries = 10;
    config.enable_eviction = true;
    config.log_evictions = false;
    
    PackratCache cache(config);
    
    // Store more entries than the limit
    for (size_t i = 0; i < 50; ++i) {
        cache.store_cache(i * 10, OrbitType::Confix, static_cast<double>(i));
    }
    
    const auto& metrics = cache.metrics();
    assert(metrics.current_entries <= config.max_entries);
    assert(metrics.evictions >= 40);  // At least 40 evictions (50 - 10)
    assert(metrics.stores == 50);
    
    std::cout << "✓ Eviction under pressure test passed\n";
    std::cout << "  - Stores: " << metrics.stores << "\n";
    std::cout << "  - Evictions: " << metrics.evictions << "\n";
    std::cout << "  - Final size: " << metrics.current_entries << "\n";
}

void test_clear_functionality() {
    std::cout << "Testing clear functionality...\n";
    
    CacheConfig config;
    config.max_entries = 100;
    PackratCache cache(config);
    
    // Store some entries
    for (size_t i = 0; i < 10; ++i) {
        cache.store_cache(i * 10, OrbitType::Confix, static_cast<double>(i));
    }
    
    // Clear the cache
    cache.clear();
    
    const auto& metrics = cache.metrics();
    assert(metrics.current_entries == 0);
    assert(cache.get_cached(10, OrbitType::Confix) == nullptr);
    
    // Historical counts should be preserved
    assert(metrics.stores == 10);
    assert(metrics.peak_entries == 10);
    
    std::cout << "✓ Clear functionality test passed\n";
}

void test_hit_rate_calculation() {
    std::cout << "Testing hit rate calculation...\n";
    
    CacheConfig config;
    config.max_entries = 100;
    PackratCache cache(config);
    
    // Start with all misses
    for (size_t i = 0; i < 10; ++i) {
        cache.get_cached(i * 10, OrbitType::Confix);  // All misses
    }
    
    // Store some entries
    for (size_t i = 0; i < 5; ++i) {
        cache.store_cache(i * 10, OrbitType::Confix, static_cast<double>(i));
    }
    
    // Now we should have hits
    for (size_t i = 0; i < 5; ++i) {
        cache.get_cached(i * 10, OrbitType::Confix);  // Should be hits
    }
    
    const auto& metrics = cache.metrics();
    assert(metrics.total_hits == 5);
    assert(metrics.total_misses == 10);
    assert(metrics.hit_rate() == 5.0 / 15.0);
    
    std::cout << "✓ Hit rate calculation test passed\n";
    std::cout << "  - Hit rate: " << metrics.hit_rate() << "\n";
}

void test_force_evict() {
    std::cout << "Testing force eviction...\n";
    
    CacheConfig config;
    config.max_entries = 50;
    PackratCache cache(config);
    
    // Fill the cache
    for (size_t i = 0; i < 30; ++i) {
        cache.store_cache(i * 10, OrbitType::Confix, static_cast<double>(i));
    }
    
    const auto& metrics_before = cache.metrics();
    assert(metrics_before.current_entries == 30);
    
    // Force evict 20 entries (leaving 10)
    size_t evicted = cache.force_evict(20);
    
    const auto& metrics_after = cache.metrics();
    assert(metrics_after.current_entries == 10);
    assert(evicted == 20);  // Should have evicted 20 entries
    assert(metrics_after.evictions == 20);
    
    std::cout << "✓ Force eviction test passed\n";
}

void test_lru_with_access_patterns() {
    std::cout << "Testing LRU with specific access patterns...\n";
    
    CacheConfig config;
    config.max_entries = 3;
    config.enable_eviction = true;
    PackratCache cache(config);
    
    // Store A, B, C
    cache.store_cache(10, OrbitType::Confix, 1.0);    // A
    cache.store_cache(20, OrbitType::Keyword, 2.0);   // B
    cache.store_cache(30, OrbitType::Operator, 3.0);  // C
    
    // Access pattern: B, A, C (B becomes most recent, then A, then C)
    cache.get_cached(20, OrbitType::Keyword);   // Access B
    cache.get_cached(10, OrbitType::Confix);    // Access A
    cache.get_cached(30, OrbitType::Operator);  // Access C
    
    // Store D - should evict B (least recently used)
    cache.store_cache(40, OrbitType::Identifier, 4.0);  // D
    
    // Verify D is in cache
    assert(cache.get_cached(40, OrbitType::Identifier) != nullptr);
    
    // Verify A and C are still in cache (they were accessed more recently than B)
    assert(cache.get_cached(10, OrbitType::Confix) != nullptr);
    assert(cache.get_cached(30, OrbitType::Operator) != nullptr);
    
    // Verify B was evicted (it was least recently used)
    assert(cache.get_cached(20, OrbitType::Keyword) == nullptr);
    
    std::cout << "✓ LRU with access patterns test passed\n";
}

int main() {
    std::cout << "Packrat Cache Unit Tests\n";
    std::cout << "========================\n\n";
    
    try {
        test_default_config();
        test_custom_config();
        test_basic_storage_and_retrieval();
        test_lru_eviction();
        test_update_existing_entry();
        test_multiple_orbit_types_same_position();
        test_eviction_under_pressure();
        test_clear_functionality();
        test_hit_rate_calculation();
        test_force_evict();
        test_lru_with_access_patterns();
        
        std::cout << "\n✅ All tests passed!\n";
        return 0;
        
    } catch (const std::exception& e) {
        std::cerr << "\n❌ Test failed with exception: " << e.what() << "\n";
        return 1;
    } catch (...) {
        std::cerr << "\n❌ Test failed with unknown exception\n";
        return 1;
    }
}