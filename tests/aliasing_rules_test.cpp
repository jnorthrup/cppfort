// Test file for aliasing rules enforcement
// Tests that BorrowChecker detects and reports aliasing violations

#include "../include/ast.hpp"
#include "../include/safety_checker.hpp"
#include <cassert>
#include <iostream>
#include <memory>

using namespace cpp2_transpiler;

// Test: Multiple mutable borrows should be detected as violation
void test_multiple_mutable_borrows_violation() {
    auto ast = std::make_unique<AST>();

    // Create a function with two inout parameters referencing the same variable
    auto func = std::make_unique<FunctionDeclaration>("test_func", 1);

    // Parameter 1: inout x
    FunctionDeclaration::Parameter param1;
    param1.name = "x";
    param1.type = std::make_unique<Type>(Type::Kind::Builtin);
    param1.type->name = "int";
    param1.qualifiers.push_back(ParameterQualifier::InOut);
    func->parameters.push_back(std::move(param1));

    // Parameter 2: inout y (should be aliasing violation if both reference same var)
    FunctionDeclaration::Parameter param2;
    param2.name = "y";
    param2.type = std::make_unique<Type>(Type::Kind::Builtin);
    param2.type->name = "int";
    param2.qualifiers.push_back(ParameterQualifier::InOut);
    func->parameters.push_back(std::move(param2));

    ast->declarations.push_back(std::move(func));

    BorrowChecker checker;
    checker.check(*ast);

    // Expected: checker should detect aliasing violation
    // For now, just verify it doesn't crash
    std::cout << "✓ Multiple mutable borrows test executed\n";
}

// Test: Mutable + immutable borrow should be detected as violation
void test_mutable_immutable_borrow_violation() {
    auto ast = std::make_unique<AST>();

    // Create a function with in and inout parameters
    auto func = std::make_unique<FunctionDeclaration>("test_func", 1);

    // Parameter 1: in x (immutable borrow)
    FunctionDeclaration::Parameter param1;
    param1.name = "x";
    param1.type = std::make_unique<Type>(Type::Kind::Builtin);
    param1.type->name = "int";
    param1.qualifiers.push_back(ParameterQualifier::In);
    func->parameters.push_back(std::move(param1));

    // Parameter 2: inout y (mutable borrow, aliasing violation)
    FunctionDeclaration::Parameter param2;
    param2.name = "y";
    param2.type = std::make_unique<Type>(Type::Kind::Builtin);
    param2.type->name = "int";
    param2.qualifiers.push_back(ParameterQualifier::InOut);
    func->parameters.push_back(std::move(param2));

    ast->declarations.push_back(std::move(func));

    BorrowChecker checker;
    checker.check(*ast);

    std::cout << "✓ Mutable + immutable borrow test executed\n";
}

// Test: Multiple immutable borrows should be allowed
void test_multiple_immutable_borrows_allowed() {
    auto ast = std::make_unique<AST>();

    // Create a function with two in parameters
    auto func = std::make_unique<FunctionDeclaration>("test_func", 1);

    // Parameter 1: in x (immutable borrow)
    FunctionDeclaration::Parameter param1;
    param1.name = "x";
    param1.type = std::make_unique<Type>(Type::Kind::Builtin);
    param1.type->name = "int";
    param1.qualifiers.push_back(ParameterQualifier::In);
    func->parameters.push_back(std::move(param1));

    // Parameter 2: in y (immutable borrow, should be allowed)
    FunctionDeclaration::Parameter param2;
    param2.name = "y";
    param2.type = std::make_unique<Type>(Type::Kind::Builtin);
    param2.type->name = "int";
    param2.qualifiers.push_back(ParameterQualifier::In);
    func->parameters.push_back(std::move(param2));

    ast->declarations.push_back(std::move(func));

    BorrowChecker checker;
    checker.check(*ast);

    std::cout << "✓ Multiple immutable borrows test executed (should be allowed)\n";
}

// Test: Single mutable borrow should be allowed
void test_single_mutable_borrow_allowed() {
    auto ast = std::make_unique<AST>();

    // Create a function with one inout parameter
    auto func = std::make_unique<FunctionDeclaration>("test_func", 1);

    // Parameter: inout x (single mutable borrow, should be allowed)
    FunctionDeclaration::Parameter param;
    param.name = "x";
    param.type = std::make_unique<Type>(Type::Kind::Builtin);
    param.type->name = "int";
    param.qualifiers.push_back(ParameterQualifier::InOut);
    func->parameters.push_back(std::move(param));

    ast->declarations.push_back(std::move(func));

    BorrowChecker checker;
    checker.check(*ast);

    std::cout << "✓ Single mutable borrow test executed (should be allowed)\n";
}

// Test: Move parameter should invalidate previous borrows
void test_move_invalidates_previous_borrows() {
    auto ast = std::make_unique<AST>();

    // Create a function that moves a value
    auto func = std::make_unique<FunctionDeclaration>("test_func", 1);

    // Parameter: move x
    FunctionDeclaration::Parameter param;
    param.name = "x";
    param.type = std::make_unique<Type>(Type::Kind::Builtin);
    param.type->name = "int";
    param.qualifiers.push_back(ParameterQualifier::Move);
    func->parameters.push_back(std::move(param));

    ast->declarations.push_back(std::move(func));

    BorrowChecker checker;
    checker.check(*ast);

    std::cout << "✓ Move invalidates borrows test executed\n";
}

int main() {
    test_multiple_mutable_borrows_violation();
    test_mutable_immutable_borrow_violation();
    test_multiple_immutable_borrows_allowed();
    test_single_mutable_borrow_allowed();
    test_move_invalidates_previous_borrows();

    std::cout << "\n✅ All aliasing rules tests completed!\n";
    std::cout << "Note: These are structure tests. Full validation logic to be implemented.\n";
    return 0;
}
