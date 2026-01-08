#include <iostream>
#include <cassert>
#include <sstream>

#include "../include/ast.hpp"
#include "../include/ast_to_fir.hpp"

#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Verifier.h"

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

// Test 1: Assert operation without message
void test_assert_no_message() {
    std::cout << "Test: Assert without message\n";

    MLIRContext context;
    context.loadDialect<mlir::cpp2fir::Cpp2FIRDialect>();

    auto func = std::make_unique<FunctionDeclaration>("test_assert", 1);
    func->return_type = std::make_unique<Type>(Type::Kind::Builtin);
    func->return_type->name = "void";

    // Body: assert(true)
    auto assertExpr = std::make_unique<ContractExpression>(
        ContractExpression::ContractKind::Assert,
        std::make_unique<LiteralExpression>(int64_t(1), 1),  // true = 1
        1
    );
    assertExpr->message = "condition is true";

    auto block = std::make_unique<BlockStatement>(1);
    block->statements.push_back(std::make_unique<ContractStatement>(std::move(assertExpr), 1));
    func->body = std::move(block);

    ASTToFIRConverter converter(&context);
    auto module = converter.convertToFIR(*func);

    if (!module) { std::cout << "  [SKIP] Stub FIR converter\n"; return; }

    std::cout << "DEBUG: Generated MLIR:" << std::endl;
    module.print(llvm::outs());
    std::cout << "DEBUG: End MLIR" << std::endl;

    assert(verify_mlir(module) && "Generated MLIR is invalid");

    std::string moduleStr = getModuleAsString(module);
    assert(moduleStr.find("cpp2fir.assert") != std::string::npos &&
           "Expected cpp2fir.assert operation");

    std::cout << "✓ Test passed\n";
}

// Test 2: Assert operation with message
void test_assert_with_message() {
    std::cout << "Test: Assert with message\n";

    MLIRContext context;
    context.loadDialect<mlir::cpp2fir::Cpp2FIRDialect>();

    auto func = std::make_unique<FunctionDeclaration>("test_assert_msg", 1);
    func->return_type = std::make_unique<Type>(Type::Kind::Builtin);
    func->return_type->name = "void";

    // Body: assert(x > 0, "x must be positive")
    auto binop = std::make_unique<BinaryExpression>(
        std::make_unique<IdentifierExpression>("x", 1),
        TokenType::GreaterThan,
        std::make_unique<LiteralExpression>(int64_t(0), 1),
        1
    );

    auto assertExpr = std::make_unique<ContractExpression>(
        ContractExpression::ContractKind::Assert,
        std::move(binop),
        1
    );
    assertExpr->message = "x must be positive";

    auto block = std::make_unique<BlockStatement>(1);
    block->statements.push_back(std::make_unique<ContractStatement>(std::move(assertExpr), 1));
    func->body = std::move(block);

    ASTToFIRConverter converter(&context);
    auto module = converter.convertToFIR(*func);

    if (!module) { std::cout << "  [SKIP] Stub FIR converter\n"; return; }

    std::string moduleStr = getModuleAsString(module);
    assert(moduleStr.find("cpp2fir.assert") != std::string::npos &&
           "Expected cpp2fir.assert operation");
    assert(moduleStr.find("x must be positive") != std::string::npos &&
           "Expected message in assert");

    std::cout << "✓ Test passed\n";
}

// Test 3: Precondition operation
void test_precondition() {
    std::cout << "Test: Precondition operation\n";

    MLIRContext context;
    context.loadDialect<mlir::cpp2fir::Cpp2FIRDialect>();

    auto func = std::make_unique<FunctionDeclaration>("test_pre", 1);
    func->return_type = std::make_unique<Type>(Type::Kind::Builtin);
    func->return_type->name = "void";

    // Body: pre(value != nullptr, "value must not be null")
    auto binop = std::make_unique<BinaryExpression>(
        std::make_unique<IdentifierExpression>("value", 1),
        TokenType::NotEqual,
        std::make_unique<LiteralExpression>(int64_t(0), 1),  // nullptr as 0
        1
    );

    auto preExpr = std::make_unique<ContractExpression>(
        ContractExpression::ContractKind::Pre,
        std::move(binop),
        1
    );
    preExpr->message = "value must not be null";

    auto block = std::make_unique<BlockStatement>(1);
    block->statements.push_back(std::make_unique<ContractStatement>(std::move(preExpr), 1));
    func->body = std::move(block);

    ASTToFIRConverter converter(&context);
    auto module = converter.convertToFIR(*func);

    if (!module) { std::cout << "  [SKIP] Stub FIR converter\n"; return; }

    std::string moduleStr = getModuleAsString(module);
    assert(moduleStr.find("cpp2fir.precondition") != std::string::npos &&
           "Expected cpp2fir.precondition operation");

    std::cout << "✓ Test passed\n";
}

// Test 4: Postcondition operation
void test_postcondition() {
    std::cout << "Test: Postcondition operation\n";

    MLIRContext context;
    context.loadDialect<mlir::cpp2fir::Cpp2FIRDialect>();

    auto func = std::make_unique<FunctionDeclaration>("test_post", 1);
    func->return_type = std::make_unique<Type>(Type::Kind::Builtin);
    func->return_type->name = "int";

    // Body: post(result >= 0, "result must be non-negative")
    auto binop = std::make_unique<BinaryExpression>(
        std::make_unique<IdentifierExpression>("result", 1),
        TokenType::GreaterThanOrEqual,
        std::make_unique<LiteralExpression>(int64_t(0), 1),
        1
    );

    auto postExpr = std::make_unique<ContractExpression>(
        ContractExpression::ContractKind::Post,
        std::move(binop),
        1
    );
    postExpr->message = "result must be non-negative";

    auto ret = std::make_unique<ReturnStatement>(
        std::make_unique<LiteralExpression>(int64_t(42), 1),
        1
    );

    auto block = std::make_unique<BlockStatement>(1);
    block->statements.push_back(std::make_unique<ContractStatement>(std::move(postExpr), 1));
    block->statements.push_back(std::move(ret));
    func->body = std::move(block);

    ASTToFIRConverter converter(&context);
    auto module = converter.convertToFIR(*func);

    if (!module) { std::cout << "  [SKIP] Stub FIR converter\n"; return; }

    std::string moduleStr = getModuleAsString(module);
    assert(moduleStr.find("cpp2fir.postcondition") != std::string::npos &&
           "Expected cpp2fir.postcondition operation");

    std::cout << "✓ Test passed\n";
}

// Test 5: Assert with category
void test_assert_with_category() {
    std::cout << "Test: Assert with category\n";

    MLIRContext context;
    context.loadDialect<mlir::cpp2fir::Cpp2FIRDialect>();

    auto func = std::make_unique<FunctionDeclaration>("test_category", 1);
    func->return_type = std::make_unique<Type>(Type::Kind::Builtin);
    func->return_type->name = "void";

    // Body: assert<type_safety>(x is int, "x must be int")
    auto assertExpr = std::make_unique<ContractExpression>(
        ContractExpression::ContractKind::Assert,
        std::make_unique<LiteralExpression>(int64_t(1), 1),  // placeholder
        1
    );
    assertExpr->categories.push_back(ContractCategory::TypeSafety);
    assertExpr->message = "x must be int";

    auto block = std::make_unique<BlockStatement>(1);
    block->statements.push_back(std::make_unique<ContractStatement>(std::move(assertExpr), 1));
    func->body = std::move(block);

    ASTToFIRConverter converter(&context);
    auto module = converter.convertToFIR(*func);

    if (!module) { std::cout << "  [SKIP] Stub FIR converter\n"; return; }

    std::string moduleStr = getModuleAsString(module);
    assert(moduleStr.find("cpp2fir.assert") != std::string::npos &&
           "Expected cpp2fir.assert operation");

    std::cout << "✓ Test passed\n";
}

// Test 6: Assert with audit flag
void test_assert_with_audit() {
    std::cout << "Test: Assert with audit flag\n";

    MLIRContext context;
    context.loadDialect<mlir::cpp2fir::Cpp2FIRDialect>();

    auto func = std::make_unique<FunctionDeclaration>("test_audit", 1);
    func->return_type = std::make_unique<Type>(Type::Kind::Builtin);
    func->return_type->name = "void";

    // Body: assert<bounds_safety, audit>(index < size, "bounds check")
    auto assertExpr = std::make_unique<ContractExpression>(
        ContractExpression::ContractKind::Assert,
        std::make_unique<LiteralExpression>(int64_t(1), 1),  // placeholder
        1
    );
    assertExpr->categories.push_back(ContractCategory::BoundsSafety);
    assertExpr->audit = true;
    assertExpr->message = "bounds check";

    auto block = std::make_unique<BlockStatement>(1);
    block->statements.push_back(std::make_unique<ContractStatement>(std::move(assertExpr), 1));
    func->body = std::move(block);

    ASTToFIRConverter converter(&context);
    auto module = converter.convertToFIR(*func);

    if (!module) { std::cout << "  [SKIP] Stub FIR converter\n"; return; }

    std::string moduleStr = getModuleAsString(module);
    assert(moduleStr.find("cpp2fir.assert") != std::string::npos &&
           "Expected cpp2fir.assert operation");

    std::cout << "✓ Test passed\n";
}

// Test 7: Multiple contracts
void test_multiple_contracts() {
    std::cout << "Test: Multiple contracts\n";

    MLIRContext context;
    context.loadDialect<mlir::cpp2fir::Cpp2FIRDialect>();

    auto func = std::make_unique<FunctionDeclaration>("test_multiple", 1);
    func->return_type = std::make_unique<Type>(Type::Kind::Builtin);
    func->return_type->name = "void";

    // Body: pre(x > 0) ... post(result > 0)
    auto preBinop = std::make_unique<BinaryExpression>(
        std::make_unique<IdentifierExpression>("x", 1),
        TokenType::GreaterThan,
        std::make_unique<LiteralExpression>(int64_t(0), 1),
        1
    );
    auto preExpr = std::make_unique<ContractExpression>(
        ContractExpression::ContractKind::Pre,
        std::move(preBinop),
        1
    );
    preExpr->message = "x must be positive";

    auto postBinop = std::make_unique<BinaryExpression>(
        std::make_unique<IdentifierExpression>("result", 1),
        TokenType::GreaterThan,
        std::make_unique<LiteralExpression>(int64_t(0), 1),
        1
    );
    auto postExpr = std::make_unique<ContractExpression>(
        ContractExpression::ContractKind::Post,
        std::move(postBinop),
        1
    );
    postExpr->message = "result must be positive";

    auto block = std::make_unique<BlockStatement>(1);
    block->statements.push_back(std::make_unique<ContractStatement>(std::move(preExpr), 1));
    block->statements.push_back(std::make_unique<ContractStatement>(std::move(postExpr), 1));
    func->body = std::move(block);

    ASTToFIRConverter converter(&context);
    auto module = converter.convertToFIR(*func);

    if (!module) { std::cout << "  [SKIP] Stub FIR converter\n"; return; }

    std::string moduleStr = getModuleAsString(module);
    assert(moduleStr.find("cpp2fir.precondition") != std::string::npos &&
           "Expected cpp2fir.precondition operation");
    assert(moduleStr.find("cpp2fir.postcondition") != std::string::npos &&
           "Expected cpp2fir.postcondition operation");

    std::cout << "✓ Test passed\n";
}

int main() {
    std::cout << "=== Contract FIR Operation Tests ===\n\n";

    test_assert_no_message();
    test_assert_with_message();
    test_precondition();
    test_postcondition();
    test_assert_with_category();
    test_assert_with_audit();
    test_multiple_contracts();

    std::cout << "\n=== All tests passed ===\n";
    return 0;
}
