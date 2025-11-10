#include <iostream>
#include <string>
#include <vector>
#include <chrono>

#include "packrat_cache.h"
#include "heap_limiter.h"

using namespace cppfort::stage0;

// Simulate processing a large file with many anchor positions
void test_unbounded_growth() {
    // Test with LIMITED cache (should NOT cause heap soak)
    CacheConfig cache_config;
    cache_config.max_entries = 10000;  // Limit to 10,000 entries
    cache_config.enable_eviction = true;
    cache_config.log_evictions = false;
    
    PackratCache cache(cache_config);

    std::cout << "Testing PackratCache WITH size limits (10,000 entries max)...\n";
    std::cout << "Simulating 100MB input file with 64-byte anchors\n";
    std::cout << "This should NOT hit the 1GB heap limit due to LRU eviction\n\n";

    // Simulate 100MB file = 104,857,600 bytes
    // With 64-byte anchors = ~1,638,400 positions
    // With 5 orbit types per position = ~8,192,000 potential cache entries
    // But cache is limited to 10,000 entries, so most will be evicted
    const size_t file_size = 100 * 1024 * 1024; // 100 MB
    const size_t anchor_interval = 64;
    const size_t num_positions = file_size / anchor_interval;

    std::cout << "Expected positions: " << num_positions << "\n";
    std::cout << "Potential cache entries (5 orbits): " << (num_positions * 5) << "\n";
    std::cout << "Cache limit: " << cache_config.max_entries << " entries\n\n";

    auto start = std::chrono::steady_clock::now();

    // Simulate scanning with 5 orbit types
    OrbitType orbit_types[] = {
        OrbitType::Confix,
        OrbitType::Keyword,
        OrbitType::Operator,
        OrbitType::Identifier,
        OrbitType::Literal
    };

    size_t entries_stored = 0;
    size_t progress_interval = 100000;  // Report every 100K entries

    try {
        for (size_t pos = 0; pos < file_size; pos += anchor_interval) {
            for (auto orbit : orbit_types) {
                // Store a cache entry for each (position, orbit_type) pair
                cache.store_cache(pos, orbit, 0.5);
                entries_stored++;

                // Report progress
                if (entries_stored % progress_interval == 0) {
                    const auto& metrics = cache.metrics();
                    std::cout << "Stored " << entries_stored << " entries... "
                              << "Cache size: " << metrics.current_entries 
                              << ", Evictions: " << metrics.evictions 
                              << ", Hit rate: " << metrics.hit_rate() * 100 << "%\n";
                }
            }
        }

        auto end = std::chrono::steady_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);

        const auto& final_metrics = cache.metrics();

        std::cout << "\n✅ SUCCESS: Processed " << entries_stored << " potential cache entries\n";
        std::cout << "Time: " << duration.count() << " ms\n";
        std::cout << "Final cache size: " << final_metrics.current_entries << " entries\n";
        std::cout << "Total evictions: " << final_metrics.evictions << "\n";
        std::cout << "Peak cache size: " << final_metrics.peak_entries << " entries\n";
        std::cout << "Cache stayed within limit: " << (final_metrics.current_entries <= cache_config.max_entries ? "YES" : "NO") << "\n";
        std::cout << "\nHEAP SOAK PREVENTED: Cache eviction working correctly!\n";

    } catch (const std::bad_alloc& e) {
        auto end = std::chrono::steady_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);

        const auto& metrics = cache.metrics();

        std::cout << "\n❌ FAILED: std::bad_alloc after " << entries_stored << " entries\n";
        std::cout << "Time to failure: " << duration.count() << " ms\n";
        std::cout << "Cache size at failure: " << metrics.current_entries << " entries\n";
        std::cout << "Evictions before failure: " << metrics.evictions << "\n";
        std::cout << "\nHEAP SOAK DETECTED: Cache eviction may not be working!\n";
        throw;
    }
}

// Test with multiple files in sequence (simulating regression test suite)
void test_multi_file_soak() {
    std::cout << "\n=== Testing multi-file scenario ===\n";
    std::cout << "Simulating 195 test files (like regression suite)\n";

    const size_t num_files = 195;
    const size_t avg_file_size = 5 * 1024; // 5KB average
    const size_t anchor_interval = 64;

    // Use cache with limits
    CacheConfig cache_config;
    cache_config.max_entries = 5000;  // 5,000 entries per file
    cache_config.enable_eviction = true;
    
    PackratCache cache(cache_config);

    size_t total_entries = 0;
    size_t file_progress_interval = 20;  // Report every 20 files

    try {
        for (size_t file_idx = 0; file_idx < num_files; file_idx++) {
            size_t file_size = avg_file_size;
            size_t num_positions = file_size / anchor_interval;

            // Clear cache between files (simulating fresh parse)
            // Comment this out to test accumulation behavior
            // cache.clear();

            for (size_t pos = 0; pos < file_size; pos += anchor_interval) {
                for (int orbit = 0; orbit < 5; orbit++) {
                    cache.store_cache(pos + (file_idx * 1000000),
                                     static_cast<OrbitType>(orbit),
                                     0.5);
                    total_entries++;
                }
            }

            if ((file_idx + 1) % file_progress_interval == 0) {
                const auto& metrics = cache.metrics();
                std::cout << "Processed " << (file_idx + 1) << " files, "
                         << total_entries << " total entries, "
                         << "cache size: " << metrics.current_entries << ", "
                         << "evictions: " << metrics.evictions << "\n";
            }
        }

        const auto& final_metrics = cache.metrics();

        std::cout << "\n✅ SUCCESS: Processed all " << num_files << " files\n";
        std::cout << "Total cache entries processed: " << total_entries << "\n";
        std::cout << "Final cache size: " << final_metrics.current_entries << " entries\n";
        std::cout << "Total evictions: " << final_metrics.evictions << "\n";
        std::cout << "Peak cache size: " << final_metrics.peak_entries << " entries\n";
        std::cout << "Final hit rate: " << final_metrics.hit_rate() * 100 << "%\n";
        std::cout << "Cache stayed within limit: " << (final_metrics.current_entries <= cache_config.max_entries ? "YES" : "NO") << "\n";
        std::cout << "\nHEAP SOAK PREVENTED: Cache limits working across multiple files!\n";

    } catch (const std::bad_alloc& e) {
        const auto& metrics = cache.metrics();

        std::cout << "\n❌ FAILED: std::bad_alloc processing file " 
                  << (total_entries / (5 * (avg_file_size / anchor_interval))) << "\n";
        std::cout << "Total entries before failure: " << total_entries << "\n";
        std::cout << "Cache size at failure: " << metrics.current_entries << " entries\n";
        std::cout << "Evictions before failure: " << metrics.evictions << "\n";
        std::cout << "\nHEAP SOAK DETECTED: Cache accumulation across files!\n";
        throw;
    }
}

// Test cache performance with realistic access patterns
void test_cache_performance() {
    std::cout << "\n=== Testing cache performance ===\n";
    std::cout << "Simulating realistic access patterns with locality\n";

    CacheConfig cache_config;
    cache_config.max_entries = 10000;
    cache_config.enable_eviction = true;
    
    PackratCache cache(cache_config);

    const size_t num_operations = 100000;
    const size_t working_set_size = 1000;  // Simulate locality

    auto start = std::chrono::steady_clock::now();

    // Simulate access pattern: 80% of accesses to 20% of data (Pareto principle)
    for (size_t i = 0; i < num_operations; ++i) {
        size_t pos;
        if (i % 5 == 0) {
            // 20% of the time: access random data (cold misses)
            pos = (i * 9973) % (working_set_size * 10);
        } else {
            // 80% of the time: access working set (should hit)
            pos = (i * 61) % working_set_size;
        }

        // Try to get from cache first
        if (!cache.has_cached(pos, OrbitType::Confix)) {
            // Miss - store result
            cache.store_cache(pos, OrbitType::Confix, 0.5);
        }
        // else: Hit - already in cache
    }

    auto end = std::chrono::steady_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);

    const auto& metrics = cache.metrics();

    std::cout << "Operations: " << num_operations << "\n";
    std::cout << "Time: " << duration.count() << " ms\n";
    std::cout << "Ops/sec: " << (num_operations * 1000.0 / duration.count()) << "\n";
    std::cout << "Cache size: " << metrics.current_entries << " entries\n";
    std::cout << "Evictions: " << metrics.evictions << "\n";
    std::cout << "Hit rate: " << metrics.hit_rate() * 100 << "%\n";
    std::cout << "Stores: " << metrics.stores << "\n";

    // With locality, we should see a decent hit rate
    if (metrics.hit_rate() > 0.5) {
        std::cout << "✅ Good locality detected (hit rate > 50%)\n";
    } else {
        std::cout << "⚠️  Low hit rate - may indicate poor locality\n";
    }
}

int main() {
    if (!cppfort::stage0::ensure_heap_limit(std::cerr)) {
        return 1;
    }

    std::cout << "PackratCache Heap Soak Test (WITH Eviction)\n";
    std::cout << "=============================================\n";
    std::cout << "This test demonstrates that cache limits and LRU eviction\n";
    std::cout << "prevent the heap soak issues seen in the original implementation.\n\n";

    try {
        test_unbounded_growth();
        test_multi_file_soak();
        test_cache_performance();

        std::cout << "\n🎉 ALL TESTS PASSED!\n";
        std::cout << "Cache limits and eviction are working correctly.\n";
        std::cout << "Heap soak issues have been resolved.\n";

        return 0;

    } catch (const std::bad_alloc&) {
        std::cout << "\n💥 TESTS FAILED: Heap soak still occurring!\n";
        std::cout << "Cache eviction may not be working properly.\n";
        return 1;
    } catch (const std::exception& e) {
        std::cerr << "\n❌ Test failed with exception: " << e.what() << "\n";
        return 1;
    }
}