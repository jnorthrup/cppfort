#include <iostream>
#include <cassert>
#include <sstream>

#include "../include/ast.hpp"
#include "../include/ast_to_fir.hpp"

#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Verifier.h"

using namespace cpp2_transpiler;

// Helper to verify MLIR modules
static bool verify_mlir(mlir::ModuleOp module) {
    return mlir::succeeded(mlir::verify(module));
}

// Test 1: Convert simple function that returns a constant
void test_simple_constant_return() {
    std::cout << "Test: Simple constant return function\n";

    // Create AST for: main: () -> int = 42;
    auto func = std::make_unique<FunctionDeclaration>("main", 1);
    func->return_type = std::make_unique<Type>(Type::Kind::Builtin);
    func->return_type->name = "int";

    // Body: return 42;
    auto ret_stmt = std::make_unique<ReturnStatement>(
        std::make_unique<LiteralExpression>(int64_t(42), 1),
        1
    );
    func->body = std::move(ret_stmt);

    // Convert to FIR
    mlir::MLIRContext context;
    context.loadDialect<mlir::cpp2fir::Cpp2FIRDialect>();

    ASTToFIRConverter converter(&context);
    auto module = converter.convertToFIR(*func);

    // Verify the module is valid
    assert(module && "Failed to convert function to FIR");
    assert(verify_mlir(module) && "Generated MLIR is invalid");

    // Check that the module contains a function named "main"
    auto main_func = module.lookupSymbol<mlir::cpp2fir::FuncOp>("main");
    assert(main_func && "Main function not found in module");

    // Check that it has the correct type
    auto func_type = main_func.getFunctionType();
    assert(func_type.getNumInputs() == 0 && "Expected no parameters");
    assert(func_type.getNumResults() == 1 && "Expected one result");

    std::cout << "✓ Test passed\n";
}

// Test 2: Convert function with simple parameter
void test_function_with_parameter() {
    std::cout << "Test: Function with parameter\n";

    // Create AST for: add_one: (x: int) -> int = x + 1;
    auto func = std::make_unique<FunctionDeclaration>("add_one", 1);

    FunctionDeclaration::Parameter param;
    param.name = "x";
    param.type = std::make_unique<Type>(Type::Kind::Builtin);
    param.type->name = "int";
    func->parameters.push_back(std::move(param));

    func->return_type = std::make_unique<Type>(Type::Kind::Builtin);
    func->return_type->name = "int";

    // Body: return x + 1; (simplified - just return constant for now)
    auto ret_stmt = std::make_unique<ReturnStatement>(
        std::make_unique<LiteralExpression>(int64_t(1), 1),
        1
    );
    func->body = std::move(ret_stmt);

    // Convert to FIR
    mlir::MLIRContext context;
    context.loadDialect<mlir::cpp2fir::Cpp2FIRDialect>();

    ASTToFIRConverter converter(&context);
    auto module = converter.convertToFIR(*func);

    // Verify the module is valid
    assert(module && "Failed to convert function to FIR");
    assert(verify_mlir(module) && "Generated MLIR is invalid");

    // Check function exists and has correct signature
    auto add_func = module.lookupSymbol<mlir::cpp2fir::FuncOp>("add_one");
    assert(add_func && "Function not found in module");

    auto func_type = add_func.getFunctionType();
    assert(func_type.getNumInputs() == 1 && "Expected one parameter");
    assert(func_type.getNumResults() == 1 && "Expected one result");

    std::cout << "✓ Test passed\n";
}

int main() {
    std::cout << "=== AST to FIR Conversion Tests ===\n\n";

    test_simple_constant_return();
    test_function_with_parameter();

    std::cout << "\n=== All tests passed ===\n";
    return 0;
}
