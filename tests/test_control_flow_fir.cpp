#include <iostream>
#include <cassert>
#include <sstream>

#include "../include/ast.hpp"
#include "../include/ast_to_fir.hpp"

#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Verifier.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Arith/IR/Arith.h"

using namespace cpp2_transpiler;
using mlir::MLIRContext;

// Helper to verify MLIR modules
static bool verify_mlir(mlir::ModuleOp module) {
    return mlir::succeeded(mlir::verify(module));
}

// Helper to get MLIR as string
static std::string getModuleAsString(mlir::ModuleOp module) {
    std::string str;
    llvm::raw_string_ostream os(str);
    module.print(os);
    return str;
}

// Test 1: Simple if statement without else
void test_if_without_else() {
    std::cout << "Test: If statement without else\n";

    MLIRContext context;
    context.loadDialect<mlir::cpp2fir::Cpp2FIRDialect>();
    context.loadDialect<mlir::func::FuncDialect>();
    context.loadDialect<mlir::arith::ArithDialect>();
    context.loadDialect<mlir::scf::SCFDialect>();

    auto func = std::make_unique<FunctionDeclaration>("test_if", 1);
    func->return_type = std::make_unique<Type>(Type::Kind::Builtin);
    func->return_type->name = "int";

    // Body: if (true) { return 1; } return 0;
    auto ifStmt = std::make_unique<IfStatement>(
        std::make_unique<LiteralExpression>(int64_t(1), 1),  // condition: true (1)
        std::make_unique<ReturnStatement>(  // then: return 1
            std::make_unique<LiteralExpression>(int64_t(1), 1),
            1
        ),
        nullptr,  // no else
        1
    );

    auto ret = std::make_unique<ReturnStatement>(
        std::make_unique<LiteralExpression>(int64_t(0), 1),
        1
    );

    // Create a block to hold both statements
    auto block = std::make_unique<BlockStatement>(1);
    block->statements.push_back(std::move(ifStmt));
    block->statements.push_back(std::move(ret));
    func->body = std::move(block);

    ASTToFIRConverter converter(&context);
    auto module = converter.convertToFIR(*func);

    assert(module && "Failed to convert function to FIR");

    std::cout << "DEBUG: Generated MLIR:" << std::endl;
    module.print(llvm::outs());
    std::cout << "DEBUG: End MLIR" << std::endl;

    assert(verify_mlir(module) && "Generated MLIR is invalid");

    std::string moduleStr = getModuleAsString(module);
    // Should contain scf.if operation
    assert(moduleStr.find("scf.if") != std::string::npos &&
           "Expected scf.if operation");

    std::cout << "✓ Test passed\n";
}

// Test 2: If statement with else
void test_if_with_else() {
    std::cout << "Test: If statement with else\n";

    MLIRContext context;
    context.loadDialect<mlir::cpp2fir::Cpp2FIRDialect>();
    context.loadDialect<mlir::func::FuncDialect>();
    context.loadDialect<mlir::arith::ArithDialect>();
    context.loadDialect<mlir::scf::SCFDialect>();

    auto func = std::make_unique<FunctionDeclaration>("test_if_else", 1);
    func->return_type = std::make_unique<Type>(Type::Kind::Builtin);
    func->return_type->name = "int";

    // Body: if (x) { return 1; } else { return 2; }
    auto ifStmt = std::make_unique<IfStatement>(
        std::make_unique<IdentifierExpression>("x", 1),  // condition
        std::make_unique<ReturnStatement>(  // then: return 1
            std::make_unique<LiteralExpression>(int64_t(1), 1),
            1
        ),
        std::make_unique<ReturnStatement>(  // else: return 2
            std::make_unique<LiteralExpression>(int64_t(2), 1),
            1
        ),
        1
    );

    auto block = std::make_unique<BlockStatement>(1);
    block->statements.push_back(std::move(ifStmt));
    func->body = std::move(block);

    ASTToFIRConverter converter(&context);
    auto module = converter.convertToFIR(*func);

    assert(module && "Failed to convert function to FIR");

    std::string moduleStr = getModuleAsString(module);
    assert(moduleStr.find("scf.if") != std::string::npos &&
           "Expected scf.if operation");

    std::cout << "✓ Test passed\n";
}

// Test 3: While loop
void test_while_loop() {
    std::cout << "Test: While loop\n";

    MLIRContext context;
    context.loadDialect<mlir::cpp2fir::Cpp2FIRDialect>();
    context.loadDialect<mlir::func::FuncDialect>();
    context.loadDialect<mlir::arith::ArithDialect>();
    context.loadDialect<mlir::scf::SCFDialect>();

    auto func = std::make_unique<FunctionDeclaration>("test_while", 1);
    func->return_type = std::make_unique<Type>(Type::Kind::Builtin);
    func->return_type->name = "int";

    // Body: while (false) { } return 0;  (never executes)
    auto whileStmt = std::make_unique<WhileStatement>(
        std::make_unique<LiteralExpression>(int64_t(0), 1),  // condition: false
        nullptr,  // no body
        1
    );

    auto ret = std::make_unique<ReturnStatement>(
        std::make_unique<LiteralExpression>(int64_t(0), 1),
        1
    );

    auto block = std::make_unique<BlockStatement>(1);
    block->statements.push_back(std::move(whileStmt));
    block->statements.push_back(std::move(ret));
    func->body = std::move(block);

    ASTToFIRConverter converter(&context);
    auto module = converter.convertToFIR(*func);

    assert(module && "Failed to convert function to FIR");

    std::cout << "DEBUG: Generated MLIR:" << std::endl;
    module.print(llvm::outs());
    std::cout << "DEBUG: End MLIR" << std::endl;

    assert(verify_mlir(module) && "Generated MLIR is invalid");

    std::string moduleStr = getModuleAsString(module);
    // Should contain scf.while operation
    assert((moduleStr.find("scf.while") != std::string::npos ||
            moduleStr.find("scf.condition") != std::string::npos) &&
           "Expected scf.while operation");

    std::cout << "✓ Test passed\n";
}

// Test 4: Nested if statements
void test_nested_if() {
    std::cout << "Test: Nested if statements\n";

    MLIRContext context;
    context.loadDialect<mlir::cpp2fir::Cpp2FIRDialect>();
    context.loadDialect<mlir::func::FuncDialect>();
    context.loadDialect<mlir::arith::ArithDialect>();
    context.loadDialect<mlir::scf::SCFDialect>();

    auto func = std::make_unique<FunctionDeclaration>("test_nested_if", 1);
    func->return_type = std::make_unique<Type>(Type::Kind::Builtin);
    func->return_type->name = "int";

    // Body: if (a) { if (b) { return 1; } }
    auto innerIf = std::make_unique<IfStatement>(
        std::make_unique<IdentifierExpression>("b", 1),
        std::make_unique<ReturnStatement>(
            std::make_unique<LiteralExpression>(int64_t(1), 1),
            1
        ),
        nullptr,
        1
    );

    auto innerBlock = std::make_unique<BlockStatement>(1);
    innerBlock->statements.push_back(std::move(innerIf));

    auto outerIf = std::make_unique<IfStatement>(
        std::make_unique<IdentifierExpression>("a", 1),
        std::move(innerBlock),
        nullptr,
        1
    );

    auto block = std::make_unique<BlockStatement>(1);
    block->statements.push_back(std::move(outerIf));
    func->body = std::move(block);

    ASTToFIRConverter converter(&context);
    auto module = converter.convertToFIR(*func);

    assert(module && "Failed to convert function to FIR");

    std::string moduleStr = getModuleAsString(module);
    assert(moduleStr.find("scf.if") != std::string::npos &&
           "Expected scf.if operation");

    std::cout << "✓ Test passed\n";
}

int main() {
    std::cout << "=== Control Flow FIR Conversion Tests ===\n\n";

    test_if_without_else();
    test_if_with_else();
    test_while_loop();
    test_nested_if();

    std::cout << "\n=== All tests passed ===\n";
    return 0;
}
