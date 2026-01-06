// Test file for parameter qualifier to ownership kind mapping
// Tests the semantic mapping from Cpp2 parameter qualifiers to ownership semantics

#include "../include/ast.hpp"
#include "../include/semantic_analyzer.hpp"
#include <cassert>
#include <iostream>

using namespace cpp2_transpiler;

// Test 'in' parameter maps to Borrowed
void test_in_parameter_mapping() {
    auto ownership = map_qualifier_to_ownership(ParameterQualifier::In);
    assert(ownership == OwnershipKind::Borrowed);
    std::cout << "✓ 'in' parameter maps to Borrowed\n";
}

// Test 'out' parameter maps to MutBorrowed
void test_out_parameter_mapping() {
    auto ownership = map_qualifier_to_ownership(ParameterQualifier::Out);
    assert(ownership == OwnershipKind::MutBorrowed);
    std::cout << "✓ 'out' parameter maps to MutBorrowed\n";
}

// Test 'inout' parameter maps to MutBorrowed
void test_inout_parameter_mapping() {
    auto ownership = map_qualifier_to_ownership(ParameterQualifier::InOut);
    assert(ownership == OwnershipKind::MutBorrowed);
    std::cout << "✓ 'inout' parameter maps to MutBorrowed\n";
}

// Test 'move' parameter maps to Moved
void test_move_parameter_mapping() {
    auto ownership = map_qualifier_to_ownership(ParameterQualifier::Move);
    assert(ownership == OwnershipKind::Moved);
    std::cout << "✓ 'move' parameter maps to Moved\n";
}

// Test 'forward' parameter maps to Moved (conditional)
void test_forward_parameter_mapping() {
    auto ownership = map_qualifier_to_ownership(ParameterQualifier::Forward);
    assert(ownership == OwnershipKind::Moved);
    std::cout << "✓ 'forward' parameter maps to Moved (default case)\n";
}

// Test 'None' parameter maps to Owned
void test_none_parameter_mapping() {
    auto ownership = map_qualifier_to_ownership(ParameterQualifier::None);
    assert(ownership == OwnershipKind::Owned);
    std::cout << "✓ 'None' parameter maps to Owned\n";
}

// Test ownership semantics on function parameters
void test_function_parameter_ownership() {
    // Create a function with various parameter qualifiers
    auto func = std::make_unique<FunctionDeclaration>("test_func", 1);

    // Add 'in' parameter
    FunctionDeclaration::Parameter in_param;
    in_param.name = "x";
    in_param.type = std::make_unique<Type>(Type::Kind::Builtin);
    in_param.type->name = "int";
    in_param.qualifiers.push_back(ParameterQualifier::In);
    func->parameters.push_back(std::move(in_param));

    // Add 'inout' parameter
    FunctionDeclaration::Parameter inout_param;
    inout_param.name = "y";
    inout_param.type = std::make_unique<Type>(Type::Kind::Builtin);
    inout_param.type->name = "int";
    inout_param.qualifiers.push_back(ParameterQualifier::InOut);
    func->parameters.push_back(std::move(inout_param));

    // Add 'move' parameter
    FunctionDeclaration::Parameter move_param;
    move_param.name = "z";
    move_param.type = std::make_unique<Type>(Type::Kind::Builtin);
    move_param.type->name = "int";
    move_param.qualifiers.push_back(ParameterQualifier::Move);
    func->parameters.push_back(std::move(move_param));

    // Verify parameters were set up correctly
    assert(func->parameters.size() == 3);
    assert(func->parameters[0].qualifiers[0] == ParameterQualifier::In);
    assert(func->parameters[1].qualifiers[0] == ParameterQualifier::InOut);
    assert(func->parameters[2].qualifiers[0] == ParameterQualifier::Move);

    // Verify ownership mapping
    assert(map_qualifier_to_ownership(func->parameters[0].qualifiers[0]) == OwnershipKind::Borrowed);
    assert(map_qualifier_to_ownership(func->parameters[1].qualifiers[0]) == OwnershipKind::MutBorrowed);
    assert(map_qualifier_to_ownership(func->parameters[2].qualifiers[0]) == OwnershipKind::Moved);

    std::cout << "✓ Function parameter ownership semantics verified\n";
}

int main() {
    test_in_parameter_mapping();
    test_out_parameter_mapping();
    test_inout_parameter_mapping();
    test_move_parameter_mapping();
    test_forward_parameter_mapping();
    test_none_parameter_mapping();
    test_function_parameter_ownership();

    std::cout << "\n✅ All parameter ownership mapping tests passed!\n";
    return 0;
}
