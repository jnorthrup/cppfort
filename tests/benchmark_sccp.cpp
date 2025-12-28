//===- benchmark_sccp.cpp - SCCP Performance Benchmarks -------------------===//
///
/// Performance benchmarks for SCCP pass on FIR dialect.
/// Measures compilation time and optimization effectiveness.
///
//===----------------------------------------------------------------------===//

#include "../include/Cpp2Passes.h"
#include "../include/Cpp2FIRDialect.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"

#include <chrono>
#include <iostream>
#include <iomanip>

using namespace mlir;
using namespace std::chrono;

// ============================================================================
// Benchmark Configuration
// ============================================================================

struct BenchmarkConfig {
    size_t numOperations;
    const char* name;
    const char* description;
};

static const BenchmarkConfig BENCHMARKS[] = {
    {10, "Tiny", "10 constant operations"},
    {100, "Small", "100 constant operations"},
    {500, "Medium", "500 constant operations"},
    {1000, "Large", "1000 constant operations"},
    {5000, "XLarge", "5000 constant operations"},
};

// ============================================================================
// Helper Functions
// ============================================================================

/// Count operations in a module
size_t countOperations(ModuleOp module) {
    size_t count = 0;
    module.walk([&](Operation* op) {
        if (!isa<ModuleOp, mlir::cpp2fir::FuncOp>(op)) {
            count++;
        }
    });
    return count;
}

/// Count constant operations in a module
size_t countConstants(ModuleOp module) {
    size_t count = 0;
    module.walk([&](mlir::cpp2fir::ConstantOp op) {
        count++;
    });
    return count;
}

/// Generate a chain of constant additions: c1 + c2 + c3 + ... + cN
ModuleOp generateConstantChain(MLIRContext* context, size_t length) {
    OpBuilder builder(context);
    auto loc = builder.getUnknownLoc();

    auto module = ModuleOp::create(loc);
    builder.setInsertionPointToEnd(module.getBody());

    auto i64Type = builder.getI64Type();
    auto funcType = builder.getFunctionType({}, {i64Type});

    auto func = builder.create<mlir::cpp2fir::FuncOp>(
        loc, builder.getStringAttr("constant_chain"), TypeAttr::get(funcType));

    Block* entry = func.addEntryBlock();
    builder.setInsertionPointToStart(entry);

    // Create chain: 1 + 2 + 3 + ... + length
    auto result = builder.create<mlir::cpp2fir::ConstantOp>(
        loc, i64Type, builder.getI64IntegerAttr(1)).getResult();

    for (size_t i = 2; i <= length; ++i) {
        auto constant = builder.create<mlir::cpp2fir::ConstantOp>(
            loc, i64Type, builder.getI64IntegerAttr(i));
        result = builder.create<mlir::cpp2fir::AddOp>(
            loc, result, constant.getResult()).getResult();
    }

    builder.create<mlir::cpp2fir::ReturnOp>(loc, ValueRange{result});

    return module;
}

/// Generate a binary tree of constant operations
ModuleOp generateConstantTree(MLIRContext* context, size_t numLeaves) {
    OpBuilder builder(context);
    auto loc = builder.getUnknownLoc();

    auto module = ModuleOp::create(loc);
    builder.setInsertionPointToEnd(module.getBody());

    auto i64Type = builder.getI64Type();
    auto funcType = builder.getFunctionType({}, {i64Type});

    auto func = builder.create<mlir::cpp2fir::FuncOp>(
        loc, builder.getStringAttr("constant_tree"), TypeAttr::get(funcType));

    Block* entry = func.addEntryBlock();
    builder.setInsertionPointToStart(entry);

    // Create leaf constants
    std::vector<Value> values;
    for (size_t i = 0; i < numLeaves; ++i) {
        auto constant = builder.create<mlir::cpp2fir::ConstantOp>(
            loc, i64Type, builder.getI64IntegerAttr(i + 1));
        values.push_back(constant.getResult());
    }

    // Build binary tree by combining pairs
    while (values.size() > 1) {
        std::vector<Value> nextLevel;
        for (size_t i = 0; i + 1 < values.size(); i += 2) {
            auto add = builder.create<mlir::cpp2fir::AddOp>(
                loc, values[i], values[i + 1]);
            nextLevel.push_back(add.getResult());
        }
        // Handle odd case
        if (values.size() % 2 == 1) {
            nextLevel.push_back(values.back());
        }
        values = std::move(nextLevel);
    }

    builder.create<mlir::cpp2fir::ReturnOp>(loc, ValueRange{values[0]});

    return module;
}

// ============================================================================
// Benchmark Runner
// ============================================================================

struct BenchmarkResult {
    const char* name;
    size_t numOperations;
    size_t opsBeforeOptim;
    size_t constantsBeforeOptim;
    size_t opsAfterOptim;
    size_t constantsAfterOptim;
    double optimTimeMs;
    double improvement;
};

BenchmarkResult runBenchmark(const BenchmarkConfig& config, bool useTree) {
    MLIRContext context;
    context.loadDialect<mlir::cpp2fir::Cpp2FIRDialect>();
    context.loadDialect<mlir::func::FuncDialect>();

    BenchmarkResult result;
    result.name = config.name;
    result.numOperations = config.numOperations;

    // Generate test module
    ModuleOp module = useTree
        ? generateConstantTree(&context, config.numOperations)
        : generateConstantChain(&context, config.numOperations);

    // Measure before optimization
    result.opsBeforeOptim = countOperations(module);
    result.constantsBeforeOptim = countConstants(module);

    // Run SCCP pass and measure time
    PassManager pm(&context);
    pm.addPass(mlir::cpp2::createFIRSCCPPass());

    auto start = high_resolution_clock::now();

    if (failed(pm.run(module))) {
        std::cerr << "SCCP pass failed for " << config.name << "\n";
        result.optimTimeMs = -1.0;
        return result;
    }

    auto end = high_resolution_clock::now();

    // Measure after optimization
    result.opsAfterOptim = countOperations(module);
    result.constantsAfterOptim = countConstants(module);
    result.optimTimeMs = duration_cast<microseconds>(end - start).count() / 1000.0;

    // Calculate improvement (reduction in operations)
    if (result.opsBeforeOptim > 0) {
        result.improvement = 100.0 * (result.opsBeforeOptim - result.opsAfterOptim)
                           / result.opsBeforeOptim;
    } else {
        result.improvement = 0.0;
    }

    module.erase();

    return result;
}

// ============================================================================
// Main Benchmark Suite
// ============================================================================

int main() {
    std::cout << "========================================\n";
    std::cout << "SCCP Performance Benchmarks\n";
    std::cout << "========================================\n\n";

    std::cout << "Benchmark Configuration:\n";
    std::cout << "  - Pattern: Constant chains and binary trees\n";
    std::cout << "  - Metric: Compilation time (ms), operation count reduction\n";
    std::cout << "  - Pass: FIR SCCP optimization\n\n";

    // Run chain benchmarks
    std::cout << "--- Chain Benchmarks (Linear) ---\n";
    std::cout << std::setw(10) << "Size"
              << std::setw(12) << "Ops Before"
              << std::setw(12) << "Ops After"
              << std::setw(12) << "Time (ms)"
              << std::setw(12) << "Reduction %"
              << "\n";
    std::cout << std::string(58, '-') << "\n";

    std::vector<BenchmarkResult> chainResults;
    for (const auto& config : BENCHMARKS) {
        auto result = runBenchmark(config, false);
        chainResults.push_back(result);

        std::cout << std::setw(10) << result.name
                  << std::setw(12) << result.opsBeforeOptim
                  << std::setw(12) << result.opsAfterOptim
                  << std::setw(12) << std::fixed << std::setprecision(2) << result.optimTimeMs
                  << std::setw(11) << std::fixed << std::setprecision(1) << result.improvement << "%"
                  << "\n";
    }
    std::cout << "\n";

    // Run tree benchmarks
    std::cout << "--- Tree Benchmarks (Binary Tree) ---\n";
    std::cout << std::setw(10) << "Size"
              << std::setw(12) << "Ops Before"
              << std::setw(12) << "Ops After"
              << std::setw(12) << "Time (ms)"
              << std::setw(12) << "Reduction %"
              << "\n";
    std::cout << std::string(58, '-') << "\n";

    std::vector<BenchmarkResult> treeResults;
    for (const auto& config : BENCHMARKS) {
        auto result = runBenchmark(config, true);
        treeResults.push_back(result);

        std::cout << std::setw(10) << result.name
                  << std::setw(12) << result.opsBeforeOptim
                  << std::setw(12) << result.opsAfterOptim
                  << std::setw(12) << std::fixed << std::setprecision(2) << result.optimTimeMs
                  << std::setw(11) << std::fixed << std::setprecision(1) << result.improvement << "%"
                  << "\n";
    }
    std::cout << "\n";

    // Performance regression check
    std::cout << "--- Performance Regression Analysis ---\n";

    bool hasRegression = false;
    const double MAX_TIME_MS = 100.0;  // Max acceptable time for largest benchmark

    for (const auto& result : chainResults) {
        if (result.optimTimeMs > MAX_TIME_MS) {
            std::cout << "⚠ REGRESSION: " << result.name << " chain took "
                      << result.optimTimeMs << " ms (limit: " << MAX_TIME_MS << " ms)\n";
            hasRegression = true;
        }
    }

    for (const auto& result : treeResults) {
        if (result.optimTimeMs > MAX_TIME_MS) {
            std::cout << "⚠ REGRESSION: " << result.name << " tree took "
                      << result.optimTimeMs << " ms (limit: " << MAX_TIME_MS << " ms)\n";
            hasRegression = true;
        }
    }

    if (!hasRegression) {
        std::cout << "✓ No performance regressions detected\n";
        std::cout << "✓ All benchmarks completed within acceptable time limits\n";
    }
    std::cout << "\n";

    // Summary statistics
    std::cout << "--- Summary ---\n";

    double avgChainTime = 0.0;
    double avgTreeTime = 0.0;
    double avgChainImprovement = 0.0;
    double avgTreeImprovement = 0.0;

    for (const auto& result : chainResults) {
        avgChainTime += result.optimTimeMs;
        avgChainImprovement += result.improvement;
    }
    avgChainTime /= chainResults.size();
    avgChainImprovement /= chainResults.size();

    for (const auto& result : treeResults) {
        avgTreeTime += result.optimTimeMs;
        avgTreeImprovement += result.improvement;
    }
    avgTreeTime /= treeResults.size();
    avgTreeImprovement /= treeResults.size();

    std::cout << "Chain benchmarks:\n";
    std::cout << "  Average time: " << std::fixed << std::setprecision(2) << avgChainTime << " ms\n";
    std::cout << "  Average reduction: " << std::fixed << std::setprecision(1) << avgChainImprovement << "%\n\n";

    std::cout << "Tree benchmarks:\n";
    std::cout << "  Average time: " << std::fixed << std::setprecision(2) << avgTreeTime << " ms\n";
    std::cout << "  Average reduction: " << std::fixed << std::setprecision(1) << avgTreeImprovement << "%\n\n";

    std::cout << "========================================\n";
    std::cout << "Benchmarks Complete\n";
    std::cout << "========================================\n";

    return hasRegression ? 1 : 0;
}
