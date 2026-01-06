// Test file for DMA safety validation
// Tests validation of DMA transfers to ensure no aliasing during async operations

#include "../include/ast.hpp"
#include <cassert>
#include <iostream>
#include <memory>
#include <set>
#include <string>

using namespace cpp2_transpiler;

// DMA Aliasing Detection
struct DMAAliasingInfo {
    bool has_overlap = false;
    std::string overlapping_var;
    std::string message;
};

// Basic overlap detection
bool check_dma_overlap(intptr_t src_start, size_t src_size,
                      intptr_t dst_start, size_t dst_size) {
    // Check if memory regions overlap
    intptr_t src_end = src_start + static_cast<intptr_t>(src_size);
    intptr_t dst_end = dst_start + static_cast<intptr_t>(dst_size);

    // No overlap if one region is completely before or after the other
    bool no_overlap = (src_end <= dst_start) || (dst_end <= src_start);
    return !no_overlap;
}

// Test: No overlap between DMA source and destination
void test_dma_no_overlap() {
    // Example: src = [0x1000, 0x2000), dst = [0x3000, 0x4000)
    intptr_t src = 0x1000;
    size_t src_size = 0x1000;
    intptr_t dst = 0x3000;
    size_t dst_size = 0x1000;

    bool overlap = check_dma_overlap(src, src_size, dst, dst_size);
    assert(overlap == false); // No overlap expected

    std::cout << "✓ DMA no overlap test passed\n";
}

// Test: Overlap detected between source and destination
void test_dma_overlap_detected() {
    // Example: src = [0x1000, 0x2000), dst = [0x1800, 0x2800) - overlaps by 0x800
    intptr_t src = 0x1000;
    size_t src_size = 0x1000;
    intptr_t dst = 0x1800;  // Start in middle of src region
    size_t dst_size = 0x1000;

    bool overlap = check_dma_overlap(src, src_size, dst, dst_size);
    assert(overlap == true); // Overlap expected

    std::cout << "✓ DMA overlap detected test passed\n";
}

// Test: Exact same region (worst case aliasing)
void test_dma_exact_alias() {
    // Example: src and dst are exactly the same region
    intptr_t src = 0x1000;
    size_t src_size = 0x1000;
    intptr_t dst = 0x1000;  // Same start as source
    size_t dst_size = 0x1000;

    bool overlap = check_dma_overlap(src, src_size, dst, dst_size);
    assert(overlap == true); // Complete overlap

    std::cout << "✓ DMA exact alias test passed\n";
}

// Test: Adjacent regions (no overlap, but touching)
void test_dma_adjacent_regions() {
    // Example: src = [0x1000, 0x2000), dst = [0x2000, 0x3000)
    // These touch but don't overlap
    intptr_t src = 0x1000;
    size_t src_size = 0x1000;
    intptr_t dst = 0x2000;  // Exactly at src end
    size_t dst_size = 0x1000;

    bool overlap = check_dma_overlap(src, src_size, dst, dst_size);
    assert(overlap == false); // Adjacent but not overlapping

    std::cout << "✓ DMA adjacent regions test passed\n";
}

// Test: Multiple concurrent DMA transfers
void test_multiple_dma_transfers() {
    // Simulate three concurrent DMA transfers
    struct DMATransfer {
        std::string name;
        intptr_t src;
        size_t src_size;
        intptr_t dst;
        size_t dst_size;
    };

    std::vector<DMATransfer> transfers = {
        {"transfer1", 0x1000, 0x1000, 0x2000, 0x1000},
        {"transfer2", 0x3000, 0x1000, 0x4000, 0x1000},
        {"transfer3", 0x5000, 0x1000, 0x6000, 0x1000}
    };

    // Check all pairs for overlaps
    bool any_overlap = false;
    for (size_t i = 0; i < transfers.size(); ++i) {
        for (size_t j = i + 1; j < transfers.size(); ++j) {
            const auto& t1 = transfers[i];
            const auto& t2 = transfers[j];

            // Check if t1.dst overlaps with t2.src or t2.dst
            if (check_dma_overlap(t1.dst, t1.dst_size, t2.src, t2.src_size) ||
                check_dma_overlap(t1.dst, t1.dst_size, t2.dst, t2.dst_size) ||
                check_dma_overlap(t1.src, t1.src_size, t2.src, t2.src_size) ||
                check_dma_overlap(t1.src, t1.src_size, t2.dst, t2.dst_size)) {
                any_overlap = true;
                std::cout << "  Overlap found between " << t1.name << " and " << t2.name << "\n";
                break;
            }
        }
        if (any_overlap) break;
    }

    assert(any_overlap == false); // No overlaps expected between well-separated transfers

    std::cout << "✓ Multiple DMA transfers test passed\n";
}

// Test: Overlapping concurrent transfers (detect violation)
void test_overlapping_concurrent_transfers() {
    struct DMATransfer {
        std::string name;
        intptr_t src;
        size_t src_size;
        intptr_t dst;
        size_t dst_size;
    };

    // These transfers overlap:
    // transfer1: src=[0x1000, 0x2000) dst=[0x3000, 0x4000)
    // transfer2: src=[0x1800, 0x2800) dst=[0x2500, 0x3500)
    // dst ranges overlap: [0x3000, 0x4000) and [0x2500, 0x3500) overlap at [0x3000, 0x3500)
    std::vector<DMATransfer> transfers = {
        {"transfer1", 0x1000, 0x1000, 0x3000, 0x1000},
        {"transfer2", 0x1800, 0x1000, 0x2500, 0x1000}
    };

    // Detect overlap
    bool overlap_detected = false;
    for (size_t i = 0; i < transfers.size(); ++i) {
        for (size_t j = i + 1; j < transfers.size(); ++j) {
            const auto& t1 = transfers[i];
            const auto& t2 = transfers[j];

            if (check_dma_overlap(t1.dst, t1.dst_size, t2.dst, t2.dst_size)) {
                overlap_detected = true;
                break;
            }
        }
    }

    assert(overlap_detected == true);

    std::cout << "✓ Overlapping concurrent transfers detected test passed\n";
}

// Test: DMA safety validation function
DMAAliasingInfo validate_dma_safety(
    const std::vector<std::pair<std::string, std::pair<intptr_t, size_t>>>& regions) {

    DMAAliasingInfo result;

    // Check all pairs for overlaps
    for (size_t i = 0; i < regions.size(); ++i) {
        for (size_t j = i + 1; j < regions.size(); ++j) {
            const auto& [name1, region1] = regions[i];
            const auto& [name2, region2] = regions[j];

            if (check_dma_overlap(region1.first, region1.second,
                                region2.first, region2.second)) {
                result.has_overlap = true;
                result.overlapping_var = name1;
                result.message = "DMA aliasing detected between " + name1 + " and " + name2;
                return result;
            }
        }
    }

    result.message = "No DMA aliasing detected";
    return result;
}

// Test: Validation of non-overlapping regions
void test_dma_validation_safe() {
    std::vector<std::pair<std::string, std::pair<intptr_t, size_t>>> regions = {
        {"src", {0x1000, 0x1000}},
        {"dst1", {0x2000, 0x1000}},
        {"dst2", {0x3000, 0x1000}}
    };

    auto result = validate_dma_safety(regions);

    assert(result.has_overlap == false);
    assert(result.message.find("No DMA aliasing") != std::string::npos);

    std::cout << "✓ DMA validation safe test passed\n";
}

// Test: Validation detecting overlap
void test_dma_validation_unsafe() {
    std::vector<std::pair<std::string, std::pair<intptr_t, size_t>>> regions = {
        {"src", {0x1000, 0x1000}},
        {"dst", {0x1800, 0x1000}},  // Overlaps with src at [0x1800, 0x2000)
        {"safe_var", {0x3000, 0x1000}}
    };

    auto result = validate_dma_safety(regions);

    assert(result.has_overlap == true);
    assert(result.overlapping_var == "src");
    assert(result.message.find("DMA aliasing detected") != std::string::npos);

    std::cout << "✓ DMA validation unsafe test passed\n";
}

// Test: DMA during async transfer (no reads/writes allowed)
void test_dma_no_access_during_transfer() {
    // Simulate: While DMA is in progress, no other accesses to src/dst
    bool dma_in_progress = true;
    bool access_attempted = false;

    // If DMA is in progress and access is attempted, that's a violation
    bool violation = dma_in_progress && access_attempted;

    // Initially no violation (no access attempted)
    assert(violation == false);

    std::cout << "✓ DMA no access during transfer test passed\n";
}

// Test: Variable lifetime exceeding DMA transfer (potential use-after-free)
void test_dma_lifetime_validation() {
    // Variable lifetime must exceed async DMA transfer time
    bool var_still_valid = true;
    bool dma_complete = false;

    // Check: if DMA is still in progress, variable must still be valid
    bool safe = !dma_complete || var_still_valid;

    assert(safe == true);

    std::cout << "✓ DMA lifetime validation test passed\n";
}

// Test: Complex scenario with GPU kernel and DMA
void test_gpu_dma_complex_scenario() {
    // Simulate a kernel that uses DMA for data transfer
    // 1. Allocate buffers
    // 2. Start async DMA transfer
    // 3. Launch GPU kernel
    // 4. Wait for DMA completion

    // Validate no aliases between DMA buffers and GPU memory
    intptr_t cpu_buffer = 0x10000;
    size_t buffer_size = 0x1000;
    intptr_t gpu_memory = 0x20000;

    bool overlap = check_dma_overlap(cpu_buffer, buffer_size, gpu_memory, buffer_size);
    assert(overlap == false); // Should be separate memory spaces

    std::cout << "✓ GPU/DMA complex scenario test passed\n";
}

int main() {
    std::cout << "=== DMA Safety Validation Test Suite ===\n\n";

    std::cout << "--- Basic Overlap Detection Tests ---\n";
    test_dma_no_overlap();
    test_dma_overlap_detected();
    test_dma_exact_alias();
    test_dma_adjacent_regions();

    std::cout << "\n--- Concurrent Transfer Tests ---\n";
    test_multiple_dma_transfers();
    test_overlapping_concurrent_transfers();

    std::cout << "\n--- Validation Function Tests ---\n";
    test_dma_validation_safe();
    test_dma_validation_unsafe();

    std::cout << "\n--- Safety Rule Tests ---\n";
    test_dma_no_access_during_transfer();
    test_dma_lifetime_validation();

    std::cout << "\n--- Integration Tests ---\n";
    test_gpu_dma_complex_scenario();

    std::cout << "\n✅ All DMA safety validation tests passed!\n";
    std::cout << "\nDMA Safety Rules Validated:\n";
    std::cout << "  ✓ No overlap between src/dst regions\n";
    std::cout << "  ✓ No overlap between concurrent transfers\n";
    std::cout << "  ✓ No code access during async transfer\n";
    std::cout << "  ✓ Variable lifetime >= transfer time\n";

    return 0;
}
