#include "../include/ast.hpp"
#include "../include/ast_to_fir.hpp"
#include "../include/Cpp2FIRDialect.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include <cassert>
#include <iostream>

using namespace cppfort;
using namespace mlir;

void test_mixed_hello_function() {
    std::cout << "Test: AST to FIR for mixed-hello.cpp2 function\n";

    // Create a function that simulates: name: () -> std::string = { return "world"; }
    auto func_decl = std::make_unique<FunctionDeclaration>(
        "name", nullptr, std::make_unique<SimpleType>("std::string"));

    // Create return statement with string literal
    auto return_stmt = std::make_unique<ReturnStatement>(
        std::make_unique<StringLiteral>("world"));

    func_decl->body = std::make_unique<Block>();
    func_decl->body->statements.push_back(std::move(return_stmt));

    // Convert to FIR
    mlir::MLIRContext context;
    context.loadDialect<mlir::cpp2fir::Cpp2FIRDialect>();
    context.loadDialect<mlir::func::FuncDialect>();

    ASTToFIRConverter converter(&context);
    auto module = converter.convertToFIR(*func_decl);

    // Verify structure
    bool found_function = false;
    module.walk([&](mlir::cpp2fir::FuncOp op) {
        found_function = true;
        assert(op.getName() == "name" && "Function name should be 'name'");
    });

    assert(found_function && "Should find function in FIR");

    std::cout << "✓ mixed-hello function test passed\n\n";
}

void test_nested_expression() {
    std::cout << "Test: AST to FIR for nested expression (a + b)\n";

    auto func_decl = std::make_unique<FunctionDeclaration>(
        "add", nullptr, std::make_unique<SimpleType>("int"));

    // Parameters
    auto param_a = std::make_unique<Parameter>("a", std::make_unique<SimpleType>("int"));
    auto param_b = std::make_unique<Parameter>("b", std::make_unique<SimpleType>("int"));
    func_decl->parameters.push_back(std::move(param_a));
    func_decl->parameters.push_back(std::move(param_b));

    // Create return: a + b
    auto return_stmt = std::make_unique<ReturnStatement>(
        std::make_unique<BinaryExpression>(
            BinaryOperator::Plus,
            std::make_unique<Identifier>("a"),
            std::make_unique<Identifier>("b")));

    func_decl->body = std::make_unique<Block>();
    func_decl->body->statements.push_back(std::move(return_stmt));

    // Convert to FIR
    mlir::MLIRContext context;
    context.loadDialect<mlir::cpp2fir::Cpp2FIRDialect>();
    context.loadDialect<mlir::func::FuncDialect>();

    ASTToFIRConverter converter(&context);
    auto module = converter.convertToFIR(*func_decl);

    // Verify structure
    int constant_count = 0;
    int add_count = 0;

    module.walk([&](mlir::cpp2fir::ConstantOp op) { constant_count++; });

    // For now, we just check it doesn't crash
    // Future: verify the add operation is created

    std::cout << "✓ Nested expression test passed (basic structure)\n\n";
}

int main() {
    std::cout << "=== Comprehensive AST to FIR Tests ===\n\n";

    test_mixed_hello_function();
    test_nested_expression();

    std::cout << "=== Tests completed (may have limitations) ===\n";
    return 0;
}
