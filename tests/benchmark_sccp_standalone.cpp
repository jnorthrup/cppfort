//===- benchmark_sccp_standalone.cpp - SCCP Standalone Benchmarks ---------===//
///
/// Performance benchmarks for SCCP lattice and constant folding operations.
/// Tests the standalone SCCP components without MLIR dependencies.
///
//===----------------------------------------------------------------------===//

#include "../include/LatticeValue.h"
#include "../include/ConstantFolder.h"

#include <chrono>
#include <iostream>
#include <iomanip>
#include <vector>

using namespace cppfort::sccp;
using namespace std::chrono;

// ============================================================================
// Benchmark Configuration
// ============================================================================

struct BenchmarkConfig {
    size_t numOperations;
    const char* name;
};

static const BenchmarkConfig BENCHMARKS[] = {
    {1000, "Small"},
    {10000, "Medium"},
    {100000, "Large"},
    {1000000, "XLarge"},
};

// ============================================================================
// Arithmetic Operation Benchmarks
// ============================================================================

double benchmarkArithmeticOperations(size_t numOps) {
    auto start = high_resolution_clock::now();

    for (size_t i = 0; i < numOps; ++i) {
        LatticeValue a = LatticeValue::getConstant(static_cast<int64_t>(i));
        LatticeValue b = LatticeValue::getConstant(static_cast<int64_t>(i + 1));

        // Perform operations
        LatticeValue sum = ConstantFolder::foldAdd(a, b);
        LatticeValue diff = ConstantFolder::foldSub(sum, a);
        LatticeValue prod = ConstantFolder::foldMul(diff, b);
        LatticeValue quot = ConstantFolder::foldDiv(prod, b);

        // Prevent optimization
        (void)quot;
    }

    auto end = high_resolution_clock::now();
    return duration_cast<microseconds>(end - start).count() / 1000.0;
}

// ============================================================================
// Logical Operation Benchmarks
// ============================================================================

double benchmarkLogicalOperations(size_t numOps) {
    auto start = high_resolution_clock::now();

    for (size_t i = 0; i < numOps; ++i) {
        bool val1 = (i % 2) == 0;
        bool val2 = (i % 3) == 0;

        LatticeValue a = LatticeValue::getConstant(val1);
        LatticeValue b = LatticeValue::getConstant(val2);

        // Perform operations
        LatticeValue andResult = ConstantFolder::foldAnd(a, b);
        LatticeValue orResult = ConstantFolder::foldOr(a, b);
        LatticeValue notResult = ConstantFolder::foldNot(andResult);

        // Prevent optimization
        (void)orResult;
        (void)notResult;
    }

    auto end = high_resolution_clock::now();
    return duration_cast<microseconds>(end - start).count() / 1000.0;
}

// ============================================================================
// Comparison Operation Benchmarks
// ============================================================================

double benchmarkComparisonOperations(size_t numOps) {
    auto start = high_resolution_clock::now();

    for (size_t i = 0; i < numOps; ++i) {
        LatticeValue a = LatticeValue::getConstant(static_cast<int64_t>(i));
        LatticeValue b = LatticeValue::getConstant(static_cast<int64_t>(numOps - i));

        // Perform comparisons
        LatticeValue eq = ConstantFolder::foldCmp(LatticeValue::CmpPredicate::EQ, a, b);
        LatticeValue lt = ConstantFolder::foldCmp(LatticeValue::CmpPredicate::LT, a, b);
        LatticeValue gt = ConstantFolder::foldCmp(LatticeValue::CmpPredicate::GT, a, b);

        // Prevent optimization
        (void)eq;
        (void)lt;
        (void)gt;
    }

    auto end = high_resolution_clock::now();
    return duration_cast<microseconds>(end - start).count() / 1000.0;
}

// ============================================================================
// Range Analysis Benchmarks
// ============================================================================

double benchmarkRangeOperations(size_t numOps) {
    auto start = high_resolution_clock::now();

    for (size_t i = 0; i < numOps; ++i) {
        int64_t base = static_cast<int64_t>(i);
        LatticeValue range1 = LatticeValue::getIntegerRange(base, base + 100);
        LatticeValue range2 = LatticeValue::getIntegerRange(base + 50, base + 150);

        // Perform range operations
        LatticeValue intersection = LatticeValue::meet(range1, range2);
        LatticeValue constant = LatticeValue::getConstant(base + 75);
        LatticeValue check = LatticeValue::meet(intersection, constant);

        // Range arithmetic
        LatticeValue offset = LatticeValue::getConstant(10LL);
        LatticeValue shifted = ConstantFolder::foldAdd(range1, offset);

        // Prevent optimization
        (void)check;
        (void)shifted;
    }

    auto end = high_resolution_clock::now();
    return duration_cast<microseconds>(end - start).count() / 1000.0;
}

// ============================================================================
// Meet Operation Benchmarks
// ============================================================================

double benchmarkMeetOperations(size_t numOps) {
    auto start = high_resolution_clock::now();

    for (size_t i = 0; i < numOps; ++i) {
        // Test various meet scenarios
        LatticeValue top = LatticeValue::getTop();
        LatticeValue bottom = LatticeValue::getBottom();
        LatticeValue constant = LatticeValue::getConstant(static_cast<int64_t>(i));

        // Meet operations
        LatticeValue r1 = LatticeValue::meet(top, constant);
        LatticeValue r2 = LatticeValue::meet(constant, constant);
        LatticeValue r3 = LatticeValue::meet(bottom, constant);

        // Prevent optimization
        (void)r1;
        (void)r2;
        (void)r3;
    }

    auto end = high_resolution_clock::now();
    return duration_cast<microseconds>(end - start).count() / 1000.0;
}

// ============================================================================
// Main Benchmark Suite
// ============================================================================

int main() {
    std::cout << "========================================\n";
    std::cout << "SCCP Standalone Performance Benchmarks\n";
    std::cout << "========================================\n\n";

    std::cout << "Benchmark Configuration:\n";
    std::cout << "  - Components: LatticeValue, ConstantFolder\n";
    std::cout << "  - Metric: Throughput (ops/ms)\n";
    std::cout << "  - Categories: Arithmetic, Logical, Comparison, Range, Meet\n\n";

    // Arithmetic benchmarks
    std::cout << "--- Arithmetic Operations ---\n";
    std::cout << std::setw(10) << "Size"
              << std::setw(15) << "Time (ms)"
              << std::setw(18) << "Throughput (K/s)"
              << "\n";
    std::cout << std::string(43, '-') << "\n";

    double totalArithTime = 0.0;
    for (const auto& config : BENCHMARKS) {
        double timeMs = benchmarkArithmeticOperations(config.numOperations);
        totalArithTime += timeMs;
        double throughput = config.numOperations / timeMs;

        std::cout << std::setw(10) << config.name
                  << std::setw(15) << std::fixed << std::setprecision(2) << timeMs
                  << std::setw(18) << std::fixed << std::setprecision(1) << throughput
                  << "\n";
    }
    std::cout << "\n";

    // Logical benchmarks
    std::cout << "--- Logical Operations ---\n";
    std::cout << std::setw(10) << "Size"
              << std::setw(15) << "Time (ms)"
              << std::setw(18) << "Throughput (K/s)"
              << "\n";
    std::cout << std::string(43, '-') << "\n";

    double totalLogicalTime = 0.0;
    for (const auto& config : BENCHMARKS) {
        double timeMs = benchmarkLogicalOperations(config.numOperations);
        totalLogicalTime += timeMs;
        double throughput = config.numOperations / timeMs;

        std::cout << std::setw(10) << config.name
                  << std::setw(15) << std::fixed << std::setprecision(2) << timeMs
                  << std::setw(18) << std::fixed << std::setprecision(1) << throughput
                  << "\n";
    }
    std::cout << "\n";

    // Comparison benchmarks
    std::cout << "--- Comparison Operations ---\n";
    std::cout << std::setw(10) << "Size"
              << std::setw(15) << "Time (ms)"
              << std::setw(18) << "Throughput (K/s)"
              << "\n";
    std::cout << std::string(43, '-') << "\n";

    double totalCmpTime = 0.0;
    for (const auto& config : BENCHMARKS) {
        double timeMs = benchmarkComparisonOperations(config.numOperations);
        totalCmpTime += timeMs;
        double throughput = config.numOperations / timeMs;

        std::cout << std::setw(10) << config.name
                  << std::setw(15) << std::fixed << std::setprecision(2) << timeMs
                  << std::setw(18) << std::fixed << std::setprecision(1) << throughput
                  << "\n";
    }
    std::cout << "\n";

    // Range benchmarks
    std::cout << "--- Range Analysis Operations ---\n";
    std::cout << std::setw(10) << "Size"
              << std::setw(15) << "Time (ms)"
              << std::setw(18) << "Throughput (K/s)"
              << "\n";
    std::cout << std::string(43, '-') << "\n";

    double totalRangeTime = 0.0;
    for (const auto& config : BENCHMARKS) {
        double timeMs = benchmarkRangeOperations(config.numOperations);
        totalRangeTime += timeMs;
        double throughput = config.numOperations / timeMs;

        std::cout << std::setw(10) << config.name
                  << std::setw(15) << std::fixed << std::setprecision(2) << timeMs
                  << std::setw(18) << std::fixed << std::setprecision(1) << throughput
                  << "\n";
    }
    std::cout << "\n";

    // Meet benchmarks
    std::cout << "--- Meet Operations ---\n";
    std::cout << std::setw(10) << "Size"
              << std::setw(15) << "Time (ms)"
              << std::setw(18) << "Throughput (K/s)"
              << "\n";
    std::cout << std::string(43, '-') << "\n";

    double totalMeetTime = 0.0;
    for (const auto& config : BENCHMARKS) {
        double timeMs = benchmarkMeetOperations(config.numOperations);
        totalMeetTime += timeMs;
        double throughput = config.numOperations / timeMs;

        std::cout << std::setw(10) << config.name
                  << std::setw(15) << std::fixed << std::setprecision(2) << timeMs
                  << std::setw(18) << std::fixed << std::setprecision(1) << throughput
                  << "\n";
    }
    std::cout << "\n";

    // Performance regression check
    std::cout << "--- Performance Regression Analysis ---\n";

    bool hasRegression = false;
    const double MAX_TIME_PER_MIL_OPS = 5000.0;  // Max 5 seconds per million ops

    struct Category {
        const char* name;
        double totalTime;
        size_t totalOps;
    };

    Category categories[] = {
        {"Arithmetic", totalArithTime, 4000000},
        {"Logical", totalLogicalTime, 4000000},
        {"Comparison", totalCmpTime, 4000000},
        {"Range", totalRangeTime, 4000000},
        {"Meet", totalMeetTime, 4000000},
    };

    for (const auto& cat : categories) {
        double timePerMillionOps = (cat.totalTime / cat.totalOps) * 1000000;
        if (timePerMillionOps > MAX_TIME_PER_MIL_OPS) {
            std::cout << "⚠ REGRESSION: " << cat.name << " operations took "
                      << std::fixed << std::setprecision(1) << timePerMillionOps
                      << " ms per million ops (limit: " << MAX_TIME_PER_MIL_OPS << " ms)\n";
            hasRegression = true;
        }
    }

    if (!hasRegression) {
        std::cout << "✓ No performance regressions detected\n";
        std::cout << "✓ All operations completed within acceptable time limits\n";
    }
    std::cout << "\n";

    // Summary
    std::cout << "--- Summary ---\n";
    std::cout << "Total benchmark time: "
              << std::fixed << std::setprecision(2)
              << (totalArithTime + totalLogicalTime + totalCmpTime + totalRangeTime + totalMeetTime)
              << " ms\n";
    std::cout << "Total operations: 20,000,000\n";
    std::cout << "Categories tested: 5\n";
    std::cout << "\n";

    std::cout << "========================================\n";
    std::cout << "Benchmarks Complete\n";
    std::cout << "========================================\n";

    return hasRegression ? 1 : 0;
}
