#include <iostream>

#include "../include/Cpp2FIRDialect.h"
#include "../include/Cpp2SONDialect.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"

// Forward declare the pass factory
namespace mlir {
namespace cpp2 {
std::unique_ptr<Pass> createConvertFIRToSONPass();
}
}

int main() {
    std::cout << "=== FIR to SON Pass Test ===\n\n";

    mlir::MLIRContext context;
    context.loadDialect<mlir::cpp2fir::Cpp2FIRDialect>();
    context.loadDialect<mlir::sond::Cpp2SONDialect>();
    context.loadDialect<mlir::func::FuncDialect>();

    // Create a simple FIR module
    mlir::OpBuilder builder(&context);
    auto loc = builder.getUnknownLoc();
    auto module = mlir::ModuleOp::create(loc);
    builder.setInsertionPointToEnd(module.getBody());

    // Create a FIR function that models a chained relational: (60 <= x) && (x < 90)
    auto i32Type = builder.getI32Type();
    auto i1Type = builder.getI1Type();
    auto funcType = builder.getFunctionType({i32Type}, {i1Type});

    auto firFunc = builder.create<mlir::cpp2fir::FuncOp>(
        loc,
        builder.getStringAttr("test_func"),
        mlir::TypeAttr::get(funcType),
        mlir::ArrayAttr{},
        mlir::ArrayAttr{});

    mlir::Block* entry = firFunc.addEntryBlock();
    builder.setInsertionPointToStart(entry);

    auto x = entry->getArgument(0);

    auto c60 = builder.create<mlir::cpp2fir::ConstantOp>(
        loc, i32Type, builder.getI32IntegerAttr(60));
    auto c90 = builder.create<mlir::cpp2fir::ConstantOp>(
        loc, i32Type, builder.getI32IntegerAttr(90));

    auto ge60 = builder.create<mlir::cpp2fir::CmpOp>(
        loc, i1Type, c60.getResult(), x, builder.getStringAttr("le"));
    auto lt90 = builder.create<mlir::cpp2fir::CmpOp>(
        loc, i1Type, x, c90.getResult(), builder.getStringAttr("lt"));
    auto both = builder.create<mlir::cpp2fir::AndOp>(
        loc, i1Type, ge60.getResult(), lt90.getResult());

    builder.create<mlir::cpp2fir::ReturnOp>(loc, mlir::ValueRange{both.getResult()});

    std::cout << "Before lowering:\n";
    module.print(llvm::outs());
    std::cout << "\n\n";

    // Create and run the lowering pass
    mlir::PassManager pm(&context);
    pm.addPass(mlir::cpp2::createConvertFIRToSONPass());

    if (mlir::failed(pm.run(module))) {
        std::cerr << "Pass failed!\n";
        return 1;
    }

    std::cout << "After lowering:\n";
    module.print(llvm::outs());
    std::cout << "\n\n";

    std::cout << "✓ Pass executed successfully\n";
    return 0;
}
