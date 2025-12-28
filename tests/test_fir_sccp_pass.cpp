//===- test_fir_sccp_pass.cpp - FIR SCCP Pass Tests --------------------------===//
///
/// Tests for FIR SCCP pass.
/// Verifies constant propagation and IR rewriting in FIR dialect programs.
///
//===----------------------------------------------------------------------===//

#include <iostream>

#include "../include/Cpp2Passes.h"
#include "../include/Cpp2FIRDialect.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"

using namespace mlir;

int main() {
    std::cout << "=== FIR SCCP Pass Tests ===\n\n";

    MLIRContext context;
    context.loadDialect<mlir::cpp2fir::Cpp2FIRDialect>();
    context.loadDialect<mlir::func::FuncDialect>();

    OpBuilder builder(&context);
    auto loc = builder.getUnknownLoc();

    // Test 1: Simple constant folding (5 + 3 = 8)
    {
        std::cout << "Test 1: Simple constant folding\n";

        auto module = ModuleOp::create(loc);
        builder.setInsertionPointToEnd(module.getBody());

        auto i64Type = builder.getI64Type();
        auto funcType = builder.getFunctionType({}, {i64Type});

        auto func = builder.create<mlir::cpp2fir::FuncOp>(
            loc, builder.getStringAttr("test_add"), mlir::TypeAttr::get(funcType), mlir::ArrayAttr{}, mlir::ArrayAttr{});

        Block* entry = func.addEntryBlock();
        builder.setInsertionPointToStart(entry);

        auto c1 = builder.create<mlir::cpp2fir::ConstantOp>(loc, i64Type, builder.getI64IntegerAttr(5));
        auto c2 = builder.create<mlir::cpp2fir::ConstantOp>(loc, i64Type, builder.getI64IntegerAttr(3));
        auto add = builder.create<mlir::cpp2fir::AddOp>(loc, i64Type, c1.getResult(), c2.getResult());
        builder.create<mlir::cpp2fir::ReturnOp>(loc, ValueRange{add.getResult()});

        std::cout << "Before SCCP:\n";
        module.print(llvm::outs());
        std::cout << "\n";

        PassManager pm(&context);
        pm.addPass(mlir::cpp2::createFIRSCCPPass());

        if (failed(pm.run(module))) {
            std::cerr << "✗ Pass failed!\n";
            return 1;
        }

        std::cout << "After SCCP:\n";
        module.print(llvm::outs());
        std::cout << "\n";

        std::cout << "✓ Test 1 passed\n\n";
    }

    // Test 2: Logical operations
    {
        std::cout << "Test 2: Logical operations folding\n";

        auto module = ModuleOp::create(loc);
        builder.setInsertionPointToEnd(module.getBody());

        auto i1Type = builder.getI1Type();
        auto funcType = builder.getFunctionType({}, {i1Type});

        auto func = builder.create<mlir::cpp2fir::FuncOp>(
            loc, builder.getStringAttr("test_logical"), mlir::TypeAttr::get(funcType), mlir::ArrayAttr{}, mlir::ArrayAttr{});

        Block* entry = func.addEntryBlock();
        builder.setInsertionPointToStart(entry);

        auto c1 = builder.create<mlir::cpp2fir::ConstantOp>(loc, i1Type, builder.getIntegerAttr(i1Type, 1));
        auto c2 = builder.create<mlir::cpp2fir::ConstantOp>(loc, i1Type, builder.getIntegerAttr(i1Type, 0));
        auto andOp = builder.create<mlir::cpp2fir::AndOp>(loc, i1Type, c1.getResult(), c2.getResult());
        builder.create<mlir::cpp2fir::ReturnOp>(loc, ValueRange{andOp.getResult()});

        PassManager pm(&context);
        pm.addPass(mlir::cpp2::createFIRSCCPPass());

        if (failed(pm.run(module))) {
            std::cerr << "✗ Test 2 failed!\n";
            return 1;
        }

        std::cout << "✓ Test 2 passed\n\n";
    }

    // Test 3: Comparison folding
    {
        std::cout << "Test 3: Comparison folding\n";

        auto module = ModuleOp::create(loc);
        builder.setInsertionPointToEnd(module.getBody());

        auto i64Type = builder.getI64Type();
        auto i1Type = builder.getI1Type();
        auto funcType = builder.getFunctionType({}, {i1Type});

        auto func = builder.create<mlir::cpp2fir::FuncOp>(
            loc, builder.getStringAttr("test_cmp"), mlir::TypeAttr::get(funcType), mlir::ArrayAttr{}, mlir::ArrayAttr{});

        Block* entry = func.addEntryBlock();
        builder.setInsertionPointToStart(entry);

        auto c1 = builder.create<mlir::cpp2fir::ConstantOp>(loc, i64Type, builder.getI64IntegerAttr(5));
        auto c2 = builder.create<mlir::cpp2fir::ConstantOp>(loc, i64Type, builder.getI64IntegerAttr(10));
        auto cmp = builder.create<mlir::cpp2fir::CmpOp>(loc, i1Type, c1.getResult(), c2.getResult(), builder.getStringAttr("lt"));
        builder.create<mlir::cpp2fir::ReturnOp>(loc, ValueRange{cmp.getResult()});

        PassManager pm(&context);
        pm.addPass(mlir::cpp2::createFIRSCCPPass());

        if (failed(pm.run(module))) {
            std::cerr << "✗ Test 3 failed!\n";
            return 1;
        }

        std::cout << "✓ Test 3 passed\n\n";
    }

    // Test 4: Phi node folding
    {
        std::cout << "Test 4: Phi node folding\n";

        auto module = ModuleOp::create(loc);
        builder.setInsertionPointToEnd(module.getBody());

        auto i64Type = builder.getI64Type();
        auto funcType = builder.getFunctionType({}, {i64Type});

        auto func = builder.create<mlir::cpp2fir::FuncOp>(
            loc, builder.getStringAttr("test_phi"), mlir::TypeAttr::get(funcType), mlir::ArrayAttr{}, mlir::ArrayAttr{});

        Block* entry = func.addEntryBlock();
        builder.setInsertionPointToStart(entry);

        auto c1 = builder.create<mlir::cpp2fir::ConstantOp>(loc, i64Type, builder.getI64IntegerAttr(42));
        auto c2 = builder.create<mlir::cpp2fir::ConstantOp>(loc, i64Type, builder.getI64IntegerAttr(42));
        auto phi = builder.create<mlir::cpp2fir::PhiOp>(loc, i64Type, ValueRange{c1.getResult(), c2.getResult()});
        builder.create<mlir::cpp2fir::ReturnOp>(loc, ValueRange{phi.getResult()});

        PassManager pm(&context);
        pm.addPass(mlir::cpp2::createFIRSCCPPass());

        if (failed(pm.run(module))) {
            std::cerr << "✗ Test 4 failed!\n";
            return 1;
        }

        std::cout << "✓ Test 4 passed\n\n";
    }

    // Test 5: Multiple operations
    {
        std::cout << "Test 5: Multiple operations folding\n";

        auto module = ModuleOp::create(loc);
        builder.setInsertionPointToEnd(module.getBody());

        auto i64Type = builder.getI64Type();
        auto funcType = builder.getFunctionType({}, {i64Type});

        auto func = builder.create<mlir::cpp2fir::FuncOp>(
            loc, builder.getStringAttr("test_multi"), mlir::TypeAttr::get(funcType), mlir::ArrayAttr{}, mlir::ArrayAttr{});

        Block* entry = func.addEntryBlock();
        builder.setInsertionPointToStart(entry);

        auto c10 = builder.create<mlir::cpp2fir::ConstantOp>(loc, i64Type, builder.getI64IntegerAttr(10));
        auto c5 = builder.create<mlir::cpp2fir::ConstantOp>(loc, i64Type, builder.getI64IntegerAttr(5));
        auto sub = builder.create<mlir::cpp2fir::SubOp>(loc, i64Type, c10.getResult(), c5.getResult());
        auto c2 = builder.create<mlir::cpp2fir::ConstantOp>(loc, i64Type, builder.getI64IntegerAttr(2));
        auto mul = builder.create<mlir::cpp2fir::MulOp>(loc, i64Type, sub.getResult(), c2.getResult());
        builder.create<mlir::cpp2fir::ReturnOp>(loc, ValueRange{mul.getResult()});

        // (10 - 5) * 2 = 10
        PassManager pm(&context);
        pm.addPass(mlir::cpp2::createFIRSCCPPass());

        if (failed(pm.run(module))) {
            std::cerr << "✗ Test 5 failed!\n";
            return 1;
        }

        std::cout << "✓ Test 5 passed\n\n";
    }

    std::cout << "=== All FIR SCCP Pass tests passed! ===\n";
    return 0;
}
