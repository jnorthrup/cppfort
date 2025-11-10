#include <iostream>
#include <cassert>
#include <string>

#include "rbcursive.h"

using namespace cppfort::stage0;

void test_default_config() {
    std::cout << "Testing default pool configuration...\n";
    
    CombinatorPool pool;  // Uses default config
    
    const auto& metrics = pool.metrics();
    assert(metrics.current_size == 16);  // Default initial size
    assert(metrics.peak_size == 16);
    assert(metrics.total_allocations == 0);
    assert(metrics.allocation_failures == 0);
    assert(metrics.growth_events == 0);
    
    std::cout << "✓ Default config test passed\n";
}

void test_custom_config() {
    std::cout << "Testing custom pool configuration...\n";
    
    PoolConfig config;
    config.initial_size = 8;
    config.max_size = 32;
    config.growth_factor = 2.0f;
    config.allow_growth = true;
    
    CombinatorPool pool(config);
    
    const auto& metrics = pool.metrics();
    assert(metrics.current_size == 8);
    assert(metrics.peak_size == 8);
    assert(pool.capacity() == 8);
    
    std::cout << "✓ Custom config test passed\n";
}

void test_linear_growth() {
    std::cout << "Testing linear growth behavior...\n";
    
    PoolConfig config;
    config.initial_size = 4;
    config.max_size = 16;
    config.growth_factor = 1.25f;  // Small growth factor
    config.log_growth = true;
    
    CombinatorPool pool(config);
    
    // Allocate all initial slots
    std::vector<RBCursiveScanner*> scanners;
    for (int i = 0; i < 4; ++i) {
        auto* scanner = pool.allocate();
        assert(scanner != nullptr);
        scanners.push_back(scanner);
    }
    
    // Try to allocate more - should trigger growth
    auto* extra_scanner = pool.allocate();
    assert(extra_scanner != nullptr);
    scanners.push_back(extra_scanner);
    
    const auto& metrics = pool.metrics();
    assert(metrics.growth_events == 1);
    assert(metrics.current_size > 4);  // Should have grown
    assert(metrics.peak_size >= metrics.current_size);
    
    std::cout << "✓ Linear growth test passed\n";
    
    // Cleanup
    for (auto* scanner : scanners) {
        pool.release(scanner);
    }
}

void test_exponential_growth() {
    std::cout << "Testing exponential growth behavior...\n";
    
    PoolConfig config;
    config.initial_size = 8;
    config.max_size = 100;
    config.growth_factor = 2.0f;  // Double each time
    
    CombinatorPool pool(config);
    
    std::vector<RBCursiveScanner*> scanners;
    
    // Keep allocating until we trigger multiple growth events
    size_t initial_capacity = pool.capacity();
    size_t allocations = 0;
    
    while (pool.capacity() < 50 && allocations < 100) {
        auto* scanner = pool.allocate();
        if (scanner) {
            scanners.push_back(scanner);
            allocations++;
        } else {
            break;  // Reached max size
        }
    }
    
    const auto& metrics = pool.metrics();
    assert(metrics.growth_events >= 2);  // Should have grown multiple times
    assert(metrics.current_size > initial_capacity);
    
    // Verify exponential growth pattern
    // With factor 2.0: 8 -> 16 -> 32 -> 64 (stops before 100)
    assert(pool.capacity() == 64);  // Should hit this exact size
    
    std::cout << "✓ Exponential growth test passed (" 
              << metrics.growth_events << " growth events)\n";
    
    // Cleanup
    for (auto* scanner : scanners) {
        pool.release(scanner);
    }
}

void test_max_size_limit() {
    std::cout << "Testing maximum size enforcement...\n";
    
    PoolConfig config;
    config.initial_size = 4;
    config.max_size = 8;  // Very small limit
    config.growth_factor = 1.5f;
    
    CombinatorPool pool(config);
    
    std::vector<RBCursiveScanner*> scanners;
    
    // Try to allocate more than max_size
    for (int i = 0; i < 20; ++i) {
        auto* scanner = pool.allocate();
        if (scanner) {
            scanners.push_back(scanner);
        } else {
            break;  // Should fail when we hit max_size and all are used
        }
    }
    
    const auto& metrics = pool.metrics();
    assert(pool.capacity() <= config.max_size);
    assert(metrics.allocation_failures > 0);  // Should have some failures
    
    std::cout << "✓ Max size limit test passed (" 
              << metrics.allocation_failures << " allocation failures)\n";
    
    // Cleanup
    for (auto* scanner : scanners) {
        pool.release(scanner);
    }
}

void test_resize_functionality() {
    std::cout << "Testing manual resize functionality...\n";
    
    PoolConfig config;
    config.initial_size = 8;
    config.max_size = 64;
    
    CombinatorPool pool(config);
    
    // Test growing
    assert(pool.resize(16));
    assert(pool.capacity() == 16);
    assert(pool.metrics().current_size == 16);
    
    // Test shrinking (should work when no scanners in use)
    assert(pool.resize(12));
    assert(pool.capacity() == 12);
    
    // Allocate some scanners
    auto* scanner1 = pool.allocate();
    auto* scanner2 = pool.allocate();
    assert(scanner1 != nullptr && scanner2 != nullptr);
    
    // Try to shrink below used capacity (should fail)
    assert(!pool.resize(8));  // Can't shrink when scanners in use
    assert(pool.capacity() == 12);  // Should remain unchanged
    
    // Release scanners
    pool.release(scanner1);
    pool.release(scanner2);
    
    // Now shrinking should work
    assert(pool.resize(8));
    assert(pool.capacity() == 8);
    
    // Test exceeding max_size
    assert(!pool.resize(100));  // Exceeds config.max_size
    assert(pool.capacity() == 8);  // Should remain unchanged
    
    std::cout << "✓ Resize functionality test passed\n";
}

void test_metrics_accuracy() {
    std::cout << "Testing metrics accuracy...\n";
    
    PoolConfig config;
    config.initial_size = 4;
    config.max_size = 16;
    
    CombinatorPool pool(config);
    
    // Allocate and release to generate metrics
    std::vector<RBCursiveScanner*> scanners;
    
    for (int round = 0; round < 3; ++round) {
        // Allocate all we can
        while (true) {
            auto* scanner = pool.allocate();
            if (scanner) {
                scanners.push_back(scanner);
            } else {
                break;
            }
        }
        
        // Release all
        for (auto* scanner : scanners) {
            pool.release(scanner);
        }
        scanners.clear();
    }
    
    const auto& metrics = pool.metrics();
    assert(metrics.total_allocations > 0);
    assert(metrics.peak_size >= metrics.current_size);
    assert(metrics.peak_size <= config.max_size);
    
    std::cout << "✓ Metrics accuracy test passed\n";
    std::cout << "  - Total allocations: " << metrics.total_allocations << "\n";
    std::cout << "  - Peak size: " << metrics.peak_size << "\n";
    std::cout << "  - Growth events: " << metrics.growth_events << "\n";
    std::cout << "  - Failures: " << metrics.allocation_failures << "\n";
}

void test_legacy_constructor() {
    std::cout << "Testing legacy constructor compatibility...\n";
    
    // Test the legacy constructor that takes just initial size
    CombinatorPool pool(32);  // Should create pool with initial size 32
    
    assert(pool.capacity() == 32);
    
    auto* scanner = pool.allocate();
    assert(scanner != nullptr);
    
    pool.release(scanner);
    
    std::cout << "✓ Legacy constructor test passed\n";
}

int main() {
    std::cout << "Combinator Pool Unit Tests\n";
    std::cout << "==========================\n\n";
    
    try {
        test_default_config();
        test_custom_config();
        test_linear_growth();
        test_exponential_growth();
        test_max_size_limit();
        test_resize_functionality();
        test_metrics_accuracy();
        test_legacy_constructor();
        
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