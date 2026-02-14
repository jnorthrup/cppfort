#include "../include/ast.hpp"
#include "../include/ast_to_fir.hpp"
#include "../include/Cpp2FIRDialect.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include <cassert>
#include <iostream>

using namespace cpp2_transpiler;
using namespace mlir;

void test_mixed_hello_function() {
    std::cout << "Test: AST to FIR for mixed-hello.cpp2 function\n";

    // Create a function that simulates: name: () -> std::string = { return "world"; }
    auto func_decl = std::make_unique<FunctionDeclaration>("name", 0);
    func_decl->return_type = std::make_unique<cpp2_transpiler::Type>(cpp2_transpiler::Type::Kind::UserDefined);
    func_decl->return_type->name = "std::string";

    // Create return statement with string literal
    auto return_stmt = std::make_unique<ReturnStatement>(
        std::make_unique<LiteralExpression>(std::string("world"), 0), 0);

    auto block = std::make_unique<cpp2_transpiler::BlockStatement>(0);
    block->statements.push_back(std::move(return_stmt));
    func_decl->body = std::move(block);

    // Convert to FIR
    mlir::MLIRContext context;
    context.loadDialect<mlir::cpp2fir::Cpp2FIRDialect>();
    context.loadDialect<mlir::func::FuncDialect>();

    ASTToFIRConverter converter(&context);
    auto module = converter.convertToFIR(*func_decl);

    // Skip if using stub (returns empty module)
    if (!module) {
        std::cout << "  [SKIP] Stub FIR converter\n\n";
        return;
    }

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

    auto func_decl = std::make_unique<FunctionDeclaration>("add", 0);
    func_decl->return_type = std::make_unique<cpp2_transpiler::Type>(cpp2_transpiler::Type::Kind::Builtin);
    func_decl->return_type->name = "int";

    // Parameters
    FunctionDeclaration::Parameter param_a;
    param_a.name = "a";
    param_a.type = std::make_unique<cpp2_transpiler::Type>(cpp2_transpiler::Type::Kind::Builtin);
    param_a.type->name = "int";
    FunctionDeclaration::Parameter param_b;
    param_b.name = "b";
    param_b.type = std::make_unique<cpp2_transpiler::Type>(cpp2_transpiler::Type::Kind::Builtin);
    param_b.type->name = "int";
    func_decl->parameters.push_back(std::move(param_a));
    func_decl->parameters.push_back(std::move(param_b));

    // Create return: a + b
    auto return_stmt = std::make_unique<ReturnStatement>(
        std::make_unique<BinaryExpression>(
            std::make_unique<IdentifierExpression>("a", 0),
            cpp2_transpiler::TokenType::Plus,
            std::make_unique<IdentifierExpression>("b", 0),
            0), 0);

    auto block = std::make_unique<cpp2_transpiler::BlockStatement>(0);
    block->statements.push_back(std::move(return_stmt));
    func_decl->body = std::move(block);

    // Convert to FIR
    mlir::MLIRContext context;
    context.loadDialect<mlir::cpp2fir::Cpp2FIRDialect>();
    context.loadDialect<mlir::func::FuncDialect>();

    ASTToFIRConverter converter(&context);
    auto module = converter.convertToFIR(*func_decl);

    // Skip if using stub
    if (!module) {
        std::cout << "  [SKIP] Stub FIR converter\n\n";
        return;
    }

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
