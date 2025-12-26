#include <iostream>
#include <cassert>
#include <sstream>

#include "../include/ast.hpp"
#include "../include/ast_to_fir.hpp"

#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Verifier.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"

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

// Test 1: Simple UFCS call - method(object)
void test_ufcs_simple_call() {
    std::cout << "Test: Simple UFCS call\n";

    MLIRContext context;
    context.loadDialect<mlir::cpp2fir::Cpp2FIRDialect>();
    context.getOrLoadDialect<mlir::func::FuncDialect>();

    auto func = std::make_unique<FunctionDeclaration>("test_ufcs", 1);
    func->return_type = std::make_unique<Type>(Type::Kind::Builtin);
    func->return_type->name = "void";

    // Body: length("hello") - UFCS call
    // In UFCS, the object becomes the first argument
    auto callee = std::make_unique<IdentifierExpression>("length", 1);
    auto obj = std::make_unique<LiteralExpression>(std::string("hello"), 1);

    auto callExpr = std::make_unique<CallExpression>(std::move(callee), 1);
    callExpr->is_ufcs = true;
    callExpr->args.push_back(std::move(obj));

    auto exprStmt = std::make_unique<ExpressionStatement>(std::move(callExpr), 1);

    auto block = std::make_unique<BlockStatement>(1);
    block->statements.push_back(std::move(exprStmt));
    func->body = std::move(block);

    ASTToFIRConverter converter(&context);
    auto module = converter.convertToFIR(*func);

    assert(module && "Failed to convert function to FIR");

    std::cout << "DEBUG: Generated MLIR:" << std::endl;
    module.print(llvm::outs());
    std::cout << "DEBUG: End MLIR" << std::endl;

    assert(verify_mlir(module) && "Generated MLIR is invalid");

    std::string moduleStr = getModuleAsString(module);
    assert(moduleStr.find("cpp2fir.ufcs_call") != std::string::npos &&
           "Expected cpp2fir.ufcs_call operation");

    std::cout << "✓ Test passed\n";
}

// Test 2: UFCS call with multiple arguments
void test_ufcs_call_with_args() {
    std::cout << "Test: UFCS call with arguments\n";

    MLIRContext context;
    context.loadDialect<mlir::cpp2fir::Cpp2FIRDialect>();
    context.getOrLoadDialect<mlir::func::FuncDialect>();

    auto func = std::make_unique<FunctionDeclaration>("test_ufcs_args", 1);
    func->return_type = std::make_unique<Type>(Type::Kind::Builtin);
    func->return_type->name = "void";

    // Body: substring(str, 0, 5) - UFCS call with object + 2 arguments
    // In UFCS, the object is the first argument, followed by the method arguments
    auto callee = std::make_unique<IdentifierExpression>("substring", 1);
    auto obj = std::make_unique<IdentifierExpression>("str", 1);
    auto arg1 = std::make_unique<LiteralExpression>(int64_t(0), 1);
    auto arg2 = std::make_unique<LiteralExpression>(int64_t(5), 1);

    auto callExpr = std::make_unique<CallExpression>(std::move(callee), 1);
    callExpr->is_ufcs = true;
    callExpr->args.push_back(std::move(obj));
    callExpr->args.push_back(std::move(arg1));
    callExpr->args.push_back(std::move(arg2));

    auto exprStmt = std::make_unique<ExpressionStatement>(std::move(callExpr), 1);

    auto block = std::make_unique<BlockStatement>(1);
    block->statements.push_back(std::move(exprStmt));
    func->body = std::move(block);

    ASTToFIRConverter converter(&context);
    auto module = converter.convertToFIR(*func);

    assert(module && "Failed to convert function to FIR");

    std::string moduleStr = getModuleAsString(module);
    assert(moduleStr.find("cpp2fir.ufcs_call") != std::string::npos &&
           "Expected cpp2fir.ufcs_call operation");

    std::cout << "✓ Test passed\n";
}

// Test 3: Regular call (non-UFCS) for comparison
void test_regular_call() {
    std::cout << "Test: Regular call (non-UFCS)\n";

    MLIRContext context;
    context.loadDialect<mlir::cpp2fir::Cpp2FIRDialect>();
    context.getOrLoadDialect<mlir::func::FuncDialect>();

    auto func = std::make_unique<FunctionDeclaration>("test_regular", 1);
    func->return_type = std::make_unique<Type>(Type::Kind::Builtin);
    func->return_type->name = "void";

    // Body: printf("hello") - regular function call
    auto callee = std::make_unique<IdentifierExpression>("printf", 1);
    auto arg = std::make_unique<LiteralExpression>(std::string("hello"), 1);

    auto callExpr = std::make_unique<CallExpression>(std::move(callee), 1);
    callExpr->is_ufcs = false;  // regular call
    callExpr->args.push_back(std::move(arg));

    auto exprStmt = std::make_unique<ExpressionStatement>(std::move(callExpr), 1);

    auto block = std::make_unique<BlockStatement>(1);
    block->statements.push_back(std::move(exprStmt));
    func->body = std::move(block);

    ASTToFIRConverter converter(&context);
    auto module = converter.convertToFIR(*func);

    assert(module && "Failed to convert function to FIR");

    std::string moduleStr = getModuleAsString(module);
    // Regular call should use func::CallOp, not cpp2fir.ufcs_call
    assert(moduleStr.find("func.call") != std::string::npos &&
           "Expected func.call operation for regular call");
    assert(moduleStr.find("ufcs_call") == std::string::npos &&
           "Should not have ufcs_call for regular call");

    std::cout << "✓ Test passed\n";
}

// Test 4: Chained UFCS calls
void test_chained_ufcs_calls() {
    std::cout << "Test: Chained UFCS calls\n";

    MLIRContext context;
    context.loadDialect<mlir::cpp2fir::Cpp2FIRDialect>();
    context.getOrLoadDialect<mlir::func::FuncDialect>();

    auto func = std::make_unique<FunctionDeclaration>("test_chained", 1);
    func->return_type = std::make_unique<Type>(Type::Kind::Builtin);
    func->return_type->name = "void";

    // Body: length(trim(str)) - chained UFCS
    // inner UFCS: trim(str)
    auto trimCallee = std::make_unique<IdentifierExpression>("trim", 1);
    auto innerCall = std::make_unique<CallExpression>(std::move(trimCallee), 1);
    innerCall->is_ufcs = true;
    innerCall->args.push_back(std::make_unique<IdentifierExpression>("str", 1));

    // outer UFCS: length(trim(str))
    auto lengthCallee = std::make_unique<IdentifierExpression>("length", 1);
    auto outerCall = std::make_unique<CallExpression>(std::move(lengthCallee), 1);
    outerCall->is_ufcs = true;
    outerCall->args.push_back(std::move(innerCall));  // result of inner call as object

    auto exprStmt = std::make_unique<ExpressionStatement>(std::move(outerCall), 1);

    auto block = std::make_unique<BlockStatement>(1);
    block->statements.push_back(std::move(exprStmt));
    func->body = std::move(block);

    ASTToFIRConverter converter(&context);
    auto module = converter.convertToFIR(*func);

    assert(module && "Failed to convert function to FIR");

    std::string moduleStr = getModuleAsString(module);
    assert(moduleStr.find("cpp2fir.ufcs_call") != std::string::npos &&
           "Expected cpp2fir.ufcs_call operation");

    std::cout << "✓ Test passed\n";
}

int main() {
    std::cout << "=== UFCS FIR Operation Tests ===\n\n";

    test_ufcs_simple_call();
    test_ufcs_call_with_args();
    test_regular_call();
    test_chained_ufcs_calls();

    std::cout << "\n=== All tests passed ===\n";
    return 0;
}
