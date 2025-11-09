#include <iostream>
#include <string>
#include <vector>
#include <chrono>

#include "packrat_cache.h"
#include "heap_limiter.h"

using namespace cppfort::stage0;

// Simulate processing a large file with many anchor positions
void test_unbounded_growth() {
    PackratCache cache;

    std::cout << "Testing PackratCache unbounded growth...\n";
    std::cout << "Simulating 100MB input file with 64-byte anchors\n";
    std::cout << "This will demonstrate hitting the 1GB heap limit\n";

    // Simulate 100MB file = 104,857,600 bytes
    // With 64-byte anchors = ~1,638,400 positions
    // With 5 orbit types per position = ~8,192,000 cache entries
    // Each entry ~24 bytes = ~196 MB of cache data alone
    const size_t file_size = 100 * 1024 * 1024; // 100 MB
    const size_t anchor_interval = 64;
    const size_t num_positions = file_size / anchor_interval;

    std::cout << "Expected positions: " << num_positions << "\n";
    std::cout << "Expected cache entries (5 orbits): " << (num_positions * 5) << "\n";

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

    try {
        for (size_t pos = 0; pos < file_size; pos += anchor_interval) {
            for (auto orbit : orbit_types) {
                // Store a cache entry for each (position, orbit_type) pair
                cache.store_cache(pos, orbit, 0.5);
                entries_stored++;

                // Report progress every 10,000 entries
                if (entries_stored % 10000 == 0) {
                    std::cout << "Stored " << entries_stored << " entries...\n";
                }
            }
        }

        auto end = std::chrono::steady_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);

        std::cout << "\nSUCCESS: Stored " << entries_stored << " cache entries\n";
        std::cout << "Time: " << duration.count() << " ms\n";
        std::cout << "Estimated memory: " << (entries_stored * sizeof(PackratEntry)) << " bytes\n";

    } catch (const std::bad_alloc& e) {
        auto end = std::chrono::steady_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);

        std::cout << "\nFAILED: std::bad_alloc after " << entries_stored << " entries\n";
        std::cout << "Time to failure: " << duration.count() << " ms\n";
        std::cout << "Memory consumed: ~" << (entries_stored * sizeof(PackratEntry)) << " bytes\n";
        std::cout << "\nHEAP SOAK CONFIRMED: PackratCache grows unbounded\n";
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

    PackratCache cache; // Note: cache NOT cleared between files!

    size_t total_entries = 0;

    try {
        for (size_t file_idx = 0; file_idx < num_files; file_idx++) {
            size_t file_size = avg_file_size;
            size_t num_positions = file_size / anchor_interval;

            for (size_t pos = 0; pos < file_size; pos += anchor_interval) {
                for (int orbit = 0; orbit < 5; orbit++) {
                    cache.store_cache(pos + (file_idx * 1000000),
                                     static_cast<OrbitType>(orbit),
                                     0.5);
                    total_entries++;
                }
            }

            if ((file_idx + 1) % 20 == 0) {
                std::cout << "Processed " << (file_idx + 1) << " files, "
                         << total_entries << " cache entries\n";
            }
        }

        std::cout << "\nSUCCESS: Processed all " << num_files << " files\n";
        std::cout << "Total cache entries: " << total_entries << "\n";

    } catch (const std::bad_alloc& e) {
        std::cout << "\nFAILED: std::bad_alloc processing file pattern\n";
        std::cout << "Total entries before failure: " << total_entries << "\n";
        std::cout << "\nHEAP SOAK CONFIRMED: Cache accumulates across files\n";
        throw;
    }
}

int main() {
    if (!cppfort::stage0::ensure_heap_limit(std::cerr)) {
        return 1;
    }

    std::cout << "PackratCache Heap Soak Test\n";
    std::cout << "============================\n\n";

    try {
        test_unbounded_growth();
    } catch (const std::bad_alloc&) {
        std::cout << "\nTest 1 FAILED as expected (heap limit reached)\n";
    }

    std::cout << "\n";

    try {
        test_multi_file_soak();
    } catch (const std::bad_alloc&) {
        std::cout << "\nTest 2 FAILED as expected (heap limit reached)\n";
        return 1; // This demonstrates the actual regression test failure
    }

    return 0;
}
