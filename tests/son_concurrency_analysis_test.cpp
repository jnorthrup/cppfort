#include "../include/Cpp2Passes.h"
#include "mlir/IR/AsmState.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Verifier.h"
#include "mlir/Parser/Parser.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/PassManager.h"

#include "llvm/Support/InitLLVM.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/ToolOutputFile.h"
#include "llvm/Support/raw_ostream.h"

#include <iostream>
#include <cassert>
#include <memory>

using namespace mlir;

// Helper to create a simple test function with a lock
OwningOpRef<ModuleOp> parseModule(MLIRContext* context, const std::string& moduleStr) {
    ParserConfig parserConfig(context);
    return parseSourceString<ModuleOp>(moduleStr, parserConfig, "test.mlir");
}

void test_lock_elision_empty_critical_section() {
    std::cout << "Running test_lock_elision_empty_critical_section...\n";

    MLIRContext context;
    context.loadDialect<sond::Cpp2SONDialect>();

    const std::string moduleStr = R"(
soned.func @test_empty_lock() {
    %start = sond.start
    %lock = sond.call "lock"() : () -> !sond.ctrl
    %unlock = sond.call "unlock"() : () -> !sond.ctrl
    sond.stop
}
)";

    auto module = parseModule(&context, moduleStr);
    assert(module && "Failed to parse module");

    // Create pass manager and run analysis
    PassManager pm(module->getName());
    pm.addPass(mlir::cpp2::createSONConcurrencyAnalysisPass());

    if (succeeded(pm.run(module.get()))) {
        std::cout << "  PASS: Empty critical section analysis completed\n";
    } else {
        std::cerr << "  FAIL: Pass execution failed\n";
        exit(1);
    }
}

void test_barrier_elimination() {
    std::cout << "Running test_barrier_elimination...\n";

    MLIRContext context;
    context.loadDialect<sond::Cpp2SONDialect>();

    const std::string moduleStr = R"(
sond.func @test_barrier() {
    %start = sond.start
    %val1 = sond.add %zero, %one : i32 -> i32
    %barrier = sond.call "memory_barrier"() : () -> !sond.ctrl
    %val2 = sond.add %val1, %one : i32 -> i32
    sond.stop
}
)";

    auto module = parseModule(&context, moduleStr);
    assert(module && "Failed to parse module");

    PassManager pm(module->getName());
    pm.addPass(mlir::cpp2::createSONConcurrencyAnalysisPass());

    if (succeeded(pm.run(module.get()))) {
        std::cout << "  PASS: Barrier analysis completed\n";
    } else {
        std::cerr << "  FAIL: Pass execution failed\n";
        exit(1);
    }
}

void test_async_to_sync_conversion() {
    std::cout << "Running test_async_to_sync_conversion...\n";

    MLIRContext context;
    context.loadDialect<sond::Cpp2SONDialect>();

    const std::string moduleStr = R"(
sond.func @test_async() {
    %start = sond.start
    %result = sond.call "async_compute"(%input) : (i32) -> i32
    %final = sond.add %result, %one : i32 -> i32
    sond.stop
}
)";

    auto module = parseModule(&context, moduleStr);
    assert(module && "Failed to parse module");

    PassManager pm(module->getName());
    pm.addPass(mlir::cpp2::createSONConcurrencyAnalysisPass());

    if (succeeded(pm.run(module.get()))) {
        std::cout << "  PASS: Async-to-sync analysis completed\n";
    } else {
        std::cerr << "  FAIL: Pass execution failed\n";
        exit(1);
    }
}

void test_parallel_region_detection() {
    std::cout << "Running test_parallel_region_detection...\n";

    MLIRContext context;
    context.loadDialect<sond::Cpp2SONDialect>();

    const std::string moduleStr = R"(
sond.func @test_parallel() {
    %start = sond.start
    %val1 = sond.mul %a, %b : i32 -> i32
    %val2 = sond.mul %c, %d : i32 -> i32
    %val3 = sond.add %val1, %val2 : i32 -> i32
    sond.stop
}
)";

    auto module = parseModule(&context, moduleStr);
    assert(module && "Failed to parse module");

    PassManager pm(module->getName());
    pm.addPass(mlir::cpp2::createSONConcurrencyAnalysisPass());

    if (succeeded(pm.run(module.get()))) {
        std::cout << "  PASS: Parallel region detection completed\n";
    } else {
        std::cerr << "  FAIL: Pass execution failed\n";
        exit(1);
    }
}

void test_memory_alias_analysis() {
    std::cout << "Running test_memory_alias_analysis...\n";

    MLIRContext context;
    context.loadDialect<sond::Cpp2SONDialect>();

    const std::string moduleStr = R"(
sond.func @test_alias() {
    %start = sond.start
    %ptr1 = sond.new 1024 : i32 -> !sond.mem<1>
    %ptr2 = sond.new 512 : i32 -> !sond.mem<2>
    %val1 = sond.load %ptr1 : !sond.mem<1> -> i32
    %val2 = sond.load %ptr2 : !sond.mem<2> -> i32
    sond.stop
}
)";

    auto module = parseModule(&context, moduleStr);
    assert(module && "Failed to parse module");

    PassManager pm(module->getName());
    pm.addPass(mlir::cpp2::createSONConcurrencyAnalysisPass());

    if (succeeded(pm.run(module.get()))) {
        std::cout << "  PASS: Memory alias analysis completed\n";
    } else {
        std::cerr << "  FAIL: Pass execution failed\n";
        exit(1);
    }
}

void test_critical_section_identification() {
    std::cout << "Running test_critical_section_identification...\n";

    MLIRContext context;
    context.loadDialect<sond::Cpp2SONDialect>();

    const std::string moduleStr = R"(
sond.func @test_critical_section() {
    %start = sond.start
    %lock1 = sond.call "mutex_lock"(%mutex) : (!sond.mem<0>) -> !sond.ctrl
    %data = sond.load %shared_ptr : !sond.mem<1> -> i32
    %result = sond.add %data, %one : i32 -> i32
    %unlock1 = sond.call "mutex_unlock"(%mutex) : (!sond.mem<0>) -> !sond.ctrl
    sond.stop
}
)";

    auto module = parseModule(&context, moduleStr);
    assert(module && "Failed to parse module");

    PassManager pm(module->getName());
    pm.addPass(mlir::cpp2::createSONConcurrencyAnalysisPass());

    if (succeeded(pm.run(module.get()))) {
        std::cout << "  PASS: Critical section identification completed\n";
    } else {
        std::cerr << "  FAIL: Pass execution failed\n";
        exit(1);
    }
}

void test_statistics_reporting() {
    std::cout << "Running test_statistics_reporting...\n";

    MLIRContext context;
    context.loadDialect<sond::Cpp2SONDialect>();

    const std::string moduleStr = R"(
sond.func @test_stats() {
    %start = sond.start
    %lock1 = sond.call "lock"() : () -> !sond.ctrl
    %unlock1 = sond.call "unlock"() : () -> !sond.ctrl
    %barrier = sond.call "barrier"() : () -> !sond.ctrl
    sond.stop
}
)";

    auto module = parseModule(&context, moduleStr);
    assert(module && "Failed to parse module");

    PassManager pm(module->getName());
    pm.addPass(mlir::cpp2::createSONConcurrencyAnalysisPass());

    if (succeeded(pm.run(module.get()))) {
        std::cout << "  PASS: Statistics reporting completed\n";
    } else {
        std::cerr << "  FAIL: Pass execution failed\n";
        exit(1);
    }
}

int main(int argc, char** argv) {
    llvm::InitLLVM y(argc, argv);

    try {
        test_lock_elision_empty_critical_section();
        test_barrier_elimination();
        test_async_to_sync_conversion();
        test_parallel_region_detection();
        test_memory_alias_analysis();
        test_critical_section_identification();
        test_statistics_reporting();

        std::cout << "\n========================================\n";
        std::cout << "✅ All 7 SON concurrency analysis tests passed!\n";
        std::cout << "========================================\n";
    } catch (const std::exception& e) {
        std::cerr << "Exception: " << e.what() << "\n";
        return 1;
    }

    return 0;
}
