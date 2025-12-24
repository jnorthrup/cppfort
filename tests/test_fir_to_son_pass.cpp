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

    // Create a FIR function with a constant
    auto i32Type = builder.getI32Type();
    auto funcType = builder.getFunctionType({}, {i32Type});

    auto firFunc = builder.create<mlir::cpp2fir::FuncOp>(
        loc,
        builder.getStringAttr("test_func"),
        mlir::TypeAttr::get(funcType),
        mlir::ArrayAttr{},
        mlir::ArrayAttr{});

    mlir::Block* entry = firFunc.addEntryBlock();
    builder.setInsertionPointToStart(entry);

    auto constOp = builder.create<mlir::cpp2fir::ConstantOp>(
        loc, i32Type, builder.getI32IntegerAttr(42));

    builder.create<mlir::cpp2fir::ReturnOp>(loc, mlir::ValueRange{constOp.getResult()});

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
