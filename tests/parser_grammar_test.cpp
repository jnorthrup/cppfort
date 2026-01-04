// Test file for parser_grammar.hpp (Boost Spirit-style EBNF type aliases)

#include "../include/parser_grammar.hpp"
#include <cassert>
#include <iostream>

using namespace cpp2_transpiler::parser::grammar;

// Test operator precedence calculations
void test_operator_precedence() {
    using namespace operators;

    // Test multiplicative operators (precedence 3)
    assert(precedence(binary_op::multiply) == 3);
    assert(precedence(binary_op::divide) == 3);
    assert(precedence(binary_op::modulo) == 3);

    // Test additive operators (precedence 4)
    assert(precedence(binary_op::add) == 4);
    assert(precedence(binary_op::subtract) == 4);

    // Test shift operators (precedence 5)
    assert(precedence(binary_op::shift_left) == 5);
    assert(precedence(binary_op::shift_right) == 5);

    // Test range operators (precedence 6)
    assert(precedence(binary_op::range_inclusive) == 6);
    assert(precedence(binary_op::range_exclusive) == 6);

    // Test comparison operators (precedence 7)
    assert(precedence(binary_op::less) == 7);
    assert(precedence(binary_op::greater) == 7);
    assert(precedence(binary_op::less_eq) == 7);
    assert(precedence(binary_op::greater_eq) == 7);

    // Test equality operators (precedence 8)
    assert(precedence(binary_op::equal) == 8);
    assert(precedence(binary_op::not_equal) == 8);

    // Test bitwise operators (precedence 9-11)
    assert(precedence(binary_op::bitwise_and) == 9);
    assert(precedence(binary_op::bitwise_xor) == 10);
    assert(precedence(binary_op::bitwise_or) == 11);

    // Test logical operators (precedence 12-13)
    assert(precedence(binary_op::logical_and) == 12);
    assert(precedence(binary_op::logical_or) == 13);

    // Test precedence ordering
    assert(precedence(binary_op::multiply) < precedence(binary_op::add));
    assert(precedence(binary_op::add) < precedence(binary_op::shift_left));
    assert(precedence(binary_op::shift_left) < precedence(binary_op::less));
    assert(precedence(binary_op::less) < precedence(binary_op::equal));
    assert(precedence(binary_op::equal) < precedence(binary_op::bitwise_and));
    assert(precedence(binary_op::bitwise_and) < precedence(binary_op::logical_and));

    std::cout << "✓ Operator precedence tests passed\n";
}

// Test operator associativity
void test_operator_associativity() {
    using namespace operators;

    // All binary operators are left-associative (ternary/assignment are handled separately)
    assert(assoc(binary_op::multiply) == associativity::left);
    assert(assoc(binary_op::add) == associativity::left);
    assert(assoc(binary_op::less) == associativity::left);
    assert(assoc(binary_op::equal) == associativity::left);
    assert(assoc(binary_op::logical_and) == associativity::left);
    assert(assoc(binary_op::logical_or) == associativity::left);

    std::cout << "✓ Operator associativity tests passed\n";
}

// Test that type aliases are defined correctly
void test_type_alias_specification() {
    // These tests verify that the type aliases are defined at the type level
    // NOTE: Many grammar types cannot be instantiated due to circular dependencies,
    //       but they serve as type-level EBNF grammar specification

    // Lexical structure types (can be instantiated)
    identifier id = "foo";
    integer_literal int_lit = "42";
    float_literal float_lit = "3.14";
    string_literal str_lit = "\"hello\"";
    char_literal chr_lit = "'x'";

    // Primitive combinator types (templates defined)
    static_assert(std::is_same_v<opt<int>, std::optional<int>>);
    static_assert(std::is_same_v<many<int>, std::vector<int>>);
    static_assert(std::is_same_v<seq<int, char>, std::tuple<int, char>>);
    static_assert(std::is_same_v<alt<int, char>, std::variant<int, char>>);

    // Enums defined (can be used)
    [[maybe_unused]] param_qualifier pq = param_qualifier::in;
    [[maybe_unused]] assignment_operator ao = assignment_operator::assign;
    [[maybe_unused]] binary_operator bo = binary_operator::add;
    [[maybe_unused]] unary_operator uo = unary_operator::plus;
    [[maybe_unused]] type_modifier tm = type_modifier::const_;
    [[maybe_unused]] contract_kind ck = contract_kind::assert_;

    std::cout << "✓ Type alias specification tests passed\n";
}

// Test that the grammar namespace structure is correct
void test_namespace_structure() {
    // Verify namespaces are accessible
    using namespace cpp2_transpiler::parser::grammar;
    using namespace cpp2_transpiler::parser::grammar::operators;

    // Verify type aliases exist at namespace level
    using test_id = identifier;
    using test_opt = opt<int>;
    using test_many = many<char>;

    // If we can access these namespaces and types, the structure is correct
    std::cout << "✓ Namespace structure tests passed\n";
}

// Example: Demonstrate EBNF to type encoding
void demonstrate_ebnf_mapping() {
    std::cout << "\nEBNF to Type Encoding Examples:\n";
    std::cout << "=================================\n";

    std::cout << "EBNF: parameter_list ::= parameter { \",\" parameter } [ \",\" ]\n";
    std::cout << "Type: using parameter_list = many<parameter>;\n\n";

    std::cout << "EBNF: declaration ::= namespace_declaration | template_declaration | ...\n";
    std::cout << "Type: struct declaration : alt<namespace_declaration, ...> {};\n\n";

    std::cout << "EBNF: function_signature ::= [ template_params ] \"(\" ...\n";
    std::cout << "Type: using function_signature = seq<opt<template_params>, ...>;\n\n";

    std::cout << "EBNF: while_loop ::= \"while\" expression [ \"next\" expression ] block_statement\n";
    std::cout << "Type: using while_loop = seq<expression, opt<expression>, block_statement>;\n\n";

    std::cout << "Pattern:\n";
    std::cout << "  EBNF alternation (|)     → alt<A, B> = std::variant<A, B>\n";
    std::cout << "  EBNF sequence (a b)      → seq<A, B> = std::tuple<A, B>\n";
    std::cout << "  EBNF optional ([ a ])    → opt<A> = std::optional<A>\n";
    std::cout << "  EBNF repetition ({ a })  → many<A> = std::vector<A>\n";
}

int main() {
    std::cout << "Parser Grammar Tests\n";
    std::cout << "====================\n\n";

    try {
        test_operator_precedence();
        test_operator_associativity();
        test_type_alias_specification();
        test_namespace_structure();
        demonstrate_ebnf_mapping();

        std::cout << "\n✅ All parser grammar tests passed!\n";
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "\n❌ Test failed: " << e.what() << "\n";
        return 1;
    }
}
