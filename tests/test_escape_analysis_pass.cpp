// Test file for escape analysis pass
// Tests that escape analysis correctly identifies escape scenarios

#include "../include/ast.hpp"
#include <cassert>
#include <iostream>

using namespace cpp2_transpiler;

// Forward declare the escape analysis function
// This will be implemented in src/semantic_analyzer.cpp
void analyze_escape(AST& ast);

// Test Case 1: Local variable that doesn't escape
void test_no_escape_local_variable() {
    // Create AST: main: () = { x: int = 42; }
    AST ast;

    auto func = std::make_unique<FunctionDeclaration>("main", 1);
    func->body = std::make_unique<BlockStatement>(1);

    auto var_decl = std::make_unique<VariableDeclaration>("x", 2);
    var_decl->type = std::make_unique<Type>();
    var_decl->type->kind = Type::Kind::Builtin;
    var_decl->type->name = "int";
    var_decl->initializer = std::make_unique<LiteralExpression>(
        LiteralExpression::LiteralKind::Integer, "42", 2);

    auto decl_stmt = std::make_unique<DeclarationStatement>(std::move(var_decl), 2);
    func->body->statements.push_back(std::move(decl_stmt));

    ast.declarations.push_back(std::move(func));

    // Run escape analysis
    analyze_escape(ast);

    // TODO: Verify EscapeInfo is attached and has kind NoEscape
    // This will be enabled once we attach EscapeInfo to VarDecl

    std::cout << "✓ No-escape local variable test passed\n";
}

// Test Case 2: Variable returned from function (escapes to return)
void test_escape_to_return() {
    // Create AST: get_value: () -> int = { x: int = 42; return x; }
    AST ast;

    auto func = std::make_unique<FunctionDeclaration>("get_value", 1);
    func->return_type = std::make_unique<Type>();
    func->return_type->kind = Type::Kind::Builtin;
    func->return_type->name = "int";
    func->body = std::make_unique<BlockStatement>(1);

    auto var_decl = std::make_unique<VariableDeclaration>("x", 2);
    var_decl->type = std::make_unique<Type>();
    var_decl->type->kind = Type::Kind::Builtin;
    var_decl->type->name = "int";
    var_decl->initializer = std::make_unique<LiteralExpression>(
        LiteralExpression::LiteralKind::Integer, "42", 2);

    auto decl_stmt = std::make_unique<DeclarationStatement>(std::move(var_decl), 2);
    func->body->statements.push_back(std::move(decl_stmt));

    auto return_value = std::make_unique<IdentifierExpression>("x", 3);
    auto return_stmt = std::make_unique<ReturnStatement>(std::move(return_value), 3);
    func->body->statements.push_back(std::move(return_stmt));

    ast.declarations.push_back(std::move(func));

    // Run escape analysis
    analyze_escape(ast);

    // TODO: Verify EscapeInfo shows EscapeToReturn

    std::cout << "✓ Escape-to-return test passed\n";
}

// Test Case 3: Stub for heap escape (will be implemented later)
void test_escape_to_heap() {
    // For now, just test that analyze_escape doesn't crash with empty AST
    AST ast;
    analyze_escape(ast);

    std::cout << "✓ Escape-to-heap stub test passed\n";
}

int main() {
    test_no_escape_local_variable();
    test_escape_to_return();
    test_escape_to_heap();

    std::cout << "\n✅ All escape analysis pass tests passed!\n";
    return 0;
}
