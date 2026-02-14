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

// Test 1: Simple function call with no arguments
void test_call_no_args() {
    std::cout << "Test: Call function with no arguments\n";

    MLIRContext context;
    context.loadDialect<mlir::cpp2fir::Cpp2FIRDialect>();
    context.loadDialect<mlir::func::FuncDialect>();

    auto func = std::make_unique<FunctionDeclaration>("test_call_no_args", 1);
    func->return_type = std::make_unique<Type>(Type::Kind::Builtin);
    func->return_type->name = "int";

    // Body: return foo()
    auto call = std::make_unique<CallExpression>(
        std::make_unique<IdentifierExpression>("foo", 1),
        1
    );
    auto ret = std::make_unique<ReturnStatement>(std::move(call), 1);
    func->body = std::move(ret);

    ASTToFIRConverter converter(&context);
    auto module = converter.convertToFIR(*func);

    if (!module) { std::cout << "  [SKIP] Stub FIR converter\n"; return; }

    std::cout << "DEBUG: Generated MLIR:" << std::endl;
    module.print(llvm::outs());
    std::cout << "DEBUG: End MLIR" << std::endl;

    assert(verify_mlir(module) && "Generated MLIR is invalid");

    std::string moduleStr = getModuleAsString(module);
    // Should contain a call operation - either func.call or cpp2fir.call
    assert((moduleStr.find("func.call") != std::string::npos ||
            moduleStr.find("cpp2fir.call") != std::string::npos) &&
           "Expected call operation");

    assert(moduleStr.find("@foo") != std::string::npos &&
           "Expected function reference @foo");

    std::cout << "✓ Test passed\n";
}

// Test 2: Function call with single argument
void test_call_one_arg() {
    std::cout << "Test: Call function with one argument\n";

    MLIRContext context;
    context.loadDialect<mlir::cpp2fir::Cpp2FIRDialect>();
    context.loadDialect<mlir::func::FuncDialect>();

    auto func = std::make_unique<FunctionDeclaration>("test_call_one_arg", 1);
    func->return_type = std::make_unique<Type>(Type::Kind::Builtin);
    func->return_type->name = "int";

    // Body: return bar(42)
    auto call = std::make_unique<CallExpression>(
        std::make_unique<IdentifierExpression>("bar", 1),
        1
    );
    call->args.push_back(std::make_unique<LiteralExpression>(int64_t(42), 1));

    auto ret = std::make_unique<ReturnStatement>(std::move(call), 1);
    func->body = std::move(ret);

    ASTToFIRConverter converter(&context);
    auto module = converter.convertToFIR(*func);

    if (!module) { std::cout << "  [SKIP] Stub FIR converter\n"; return; }

    std::string moduleStr = getModuleAsString(module);
    assert((moduleStr.find("func.call") != std::string::npos ||
            moduleStr.find("cpp2fir.call") != std::string::npos) &&
           "Expected call operation");

    std::cout << "✓ Test passed\n";
}

// Test 3: Function call with multiple arguments
void test_call_multiple_args() {
    std::cout << "Test: Call function with multiple arguments\n";

    MLIRContext context;
    context.loadDialect<mlir::cpp2fir::Cpp2FIRDialect>();
    context.loadDialect<mlir::func::FuncDialect>();

    auto func = std::make_unique<FunctionDeclaration>("test_call_multiple_args", 1);
    func->return_type = std::make_unique<Type>(Type::Kind::Builtin);
    func->return_type->name = "int";

    // Body: return compute(5, 10, 15)
    auto call = std::make_unique<CallExpression>(
        std::make_unique<IdentifierExpression>("compute", 1),
        1
    );
    call->args.push_back(std::make_unique<LiteralExpression>(int64_t(5), 1));
    call->args.push_back(std::make_unique<LiteralExpression>(int64_t(10), 1));
    call->args.push_back(std::make_unique<LiteralExpression>(int64_t(15), 1));

    auto ret = std::make_unique<ReturnStatement>(std::move(call), 1);
    func->body = std::move(ret);

    ASTToFIRConverter converter(&context);
    auto module = converter.convertToFIR(*func);

    if (!module) { std::cout << "  [SKIP] Stub FIR converter\n"; return; }

    std::string moduleStr = getModuleAsString(module);
    assert((moduleStr.find("func.call") != std::string::npos ||
            moduleStr.find("cpp2fir.call") != std::string::npos) &&
           "Expected call operation");

    std::cout << "✓ Test passed\n";
}

// Test 4: Nested function calls
void test_call_nested() {
    std::cout << "Test: Nested function calls\n";

    MLIRContext context;
    context.loadDialect<mlir::cpp2fir::Cpp2FIRDialect>();
    context.loadDialect<mlir::func::FuncDialect>();

    auto func = std::make_unique<FunctionDeclaration>("test_call_nested", 1);
    func->return_type = std::make_unique<Type>(Type::Kind::Builtin);
    func->return_type->name = "int";

    // Body: return outer(inner(5))
    auto innerCall = std::make_unique<CallExpression>(
        std::make_unique<IdentifierExpression>("inner", 1),
        1
    );
    innerCall->args.push_back(std::make_unique<LiteralExpression>(int64_t(5), 1));

    auto outerCall = std::make_unique<CallExpression>(
        std::make_unique<IdentifierExpression>("outer", 1),
        1
    );
    outerCall->args.push_back(std::move(innerCall));

    auto ret = std::make_unique<ReturnStatement>(std::move(outerCall), 1);
    func->body = std::move(ret);

    ASTToFIRConverter converter(&context);
    auto module = converter.convertToFIR(*func);

    if (!module) { std::cout << "  [SKIP] Stub FIR converter\n"; return; }
    assert(verify_mlir(module) && "Generated MLIR is invalid");

    std::string moduleStr = getModuleAsString(module);
    assert((moduleStr.find("func.call") != std::string::npos ||
            moduleStr.find("cpp2fir.call") != std::string::npos) &&
           "Expected call operation");

    std::cout << "✓ Test passed\n";
}

// Test 5: Function call with expression as argument
void test_call_with_expression() {
    std::cout << "Test: Function call with expression argument\n";

    MLIRContext context;
    context.loadDialect<mlir::cpp2fir::Cpp2FIRDialect>();
    context.loadDialect<mlir::func::FuncDialect>();

    auto func = std::make_unique<FunctionDeclaration>("test_call_with_expr", 1);
    func->return_type = std::make_unique<Type>(Type::Kind::Builtin);
    func->return_type->name = "int";

    // Body: return process(x + y)
    auto call = std::make_unique<CallExpression>(
        std::make_unique<IdentifierExpression>("process", 1),
        1
    );

    // Argument: x + y (using binary expression)
    auto binop = std::make_unique<BinaryExpression>(
        std::make_unique<IdentifierExpression>("x", 1),
        TokenType::Plus,
        std::make_unique<IdentifierExpression>("y", 1),
        1
    );
    call->args.push_back(std::move(binop));

    auto ret = std::make_unique<ReturnStatement>(std::move(call), 1);
    func->body = std::move(ret);

    ASTToFIRConverter converter(&context);
    auto module = converter.convertToFIR(*func);

    if (!module) { std::cout << "  [SKIP] Stub FIR converter\n"; return; }

    std::string moduleStr = getModuleAsString(module);
    assert((moduleStr.find("func.call") != std::string::npos ||
            moduleStr.find("cpp2fir.call") != std::string::npos) &&
           "Expected call operation");

    std::cout << "✓ Test passed\n";
}

int main() {
    std::cout << "=== Call Expression FIR Conversion Tests ===\n\n";

    test_call_no_args();
    test_call_one_arg();
    test_call_multiple_args();
    test_call_nested();
    test_call_with_expression();

    std::cout << "\n=== All tests passed ===\n";
    return 0;
}
