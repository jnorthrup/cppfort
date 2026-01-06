// Test file for BorrowChecker validation class
// Tests borrow checking rules and ownership enforcement

#include "../include/safety_checker.hpp"
#include "../include/ast.hpp"
#include <cassert>
#include <iostream>
#include <memory>

using namespace cpp2_transpiler;

// Test that BorrowChecker class exists
void test_borrow_checker_exists() {
    BorrowChecker checker;
    std::cout << "✓ BorrowChecker class instantiated\n";
}

// Test no aliasing violations check
void test_no_aliasing_violations() {
    BorrowChecker checker;

    // Create a simple AST with a variable declaration
    auto ast = std::make_unique<AST>();
    auto var_decl = std::make_unique<VariableDeclaration>("x", 1);
    var_decl->type = std::make_unique<Type>(Type::Kind::Builtin);
    var_decl->type->name = "int";

    // Test that check doesn't crash
    checker.check_no_aliasing_violations(*ast);

    std::cout << "✓ check_no_aliasing_violations test passed\n";
}

// Test borrow outlives owner check
void test_borrow_outlives_owner() {
    BorrowChecker checker;

    auto ast = std::make_unique<AST>();

    // Test that check doesn't crash
    checker.check_borrow_outlives_owner(*ast);

    std::cout << "✓ check_borrow_outlives_owner test passed\n";
}

// Test move invalidates borrows check
void test_move_invalidates_borrows() {
    BorrowChecker checker;

    auto ast = std::make_unique<AST>();

    // Test that check doesn't crash
    checker.check_move_invalidates_borrows(*ast);

    std::cout << "✓ check_move_invalidates_borrows test passed\n";
}

// Test exclusive mutable borrow enforcement
void test_exclusive_mut_borrow() {
    BorrowChecker checker;

    auto ast = std::make_unique<AST>();

    // Test that check doesn't crash
    checker.enforce_exclusive_mut_borrow(*ast);

    std::cout << "✓ enforce_exclusive_mut_borrow test passed\n";
}

// Test full check method
void test_full_check() {
    BorrowChecker checker;

    auto ast = std::make_unique<AST>();

    // Add a simple variable declaration
    auto var_decl = std::make_unique<VariableDeclaration>("test_var", 1);
    var_decl->type = std::make_unique<Type>(Type::Kind::Builtin);
    var_decl->type->name = "int";
    var_decl->initializer = std::make_unique<LiteralExpression>(static_cast<int64_t>(42), 1);

    ast->declarations.push_back(std::move(var_decl));

    // Run full check
    checker.check(*ast);

    std::cout << "✓ Full check test passed\n";
}

int main() {
    test_borrow_checker_exists();
    test_no_aliasing_violations();
    test_borrow_outlives_owner();
    test_move_invalidates_borrows();
    test_exclusive_mut_borrow();
    test_full_check();

    std::cout << "\n✅ All BorrowChecker validation tests passed!\n";
    return 0;
}
