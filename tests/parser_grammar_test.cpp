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

// Test that type aliases compile correctly
void test_type_alias_compilation() {
    // These tests verify that the type aliases are defined and compile correctly

    // Lexical structure types
    identifier id = "foo";
    integer_literal int_lit = "42";
    float_literal float_lit = "3.14";
    string_literal str_lit = "\"hello\"";

    // Result types (verify they're defined and can be used in type declarations)
    [[maybe_unused]] declaration_result decl_result;
    [[maybe_unused]] statement_result stmt_result;
    [[maybe_unused]] expression_result expr_result;
    [[maybe_unused]] type_specifier_result type_result;

    // Template parameter types
    [[maybe_unused]] template_params_result tparams_result;
    [[maybe_unused]] parameter_list_result params_result;

    // Statement types
    [[maybe_unused]] if_statement_result if_result;
    [[maybe_unused]] while_loop_result while_result;
    [[maybe_unused]] for_loop_result for_result;

    // Expression types (various precedence levels)
    [[maybe_unused]] assignment_expression_result assign_result;
    [[maybe_unused]] ternary_expression_result ternary_result;
    [[maybe_unused]] logical_or_expression_result or_result;
    [[maybe_unused]] equality_expression_result eq_result;
    [[maybe_unused]] comparison_expression_result cmp_result;
    [[maybe_unused]] addition_expression_result add_result;
    [[maybe_unused]] multiplication_expression_result mul_result;

    // Pattern matching types
    [[maybe_unused]] inspect_expression_result inspect_result;
    [[maybe_unused]] pattern_result pattern_result;

    // Type specifier types
    [[maybe_unused]] function_type_result fn_type_result;
    [[maybe_unused]] pointer_type_result ptr_type_result;
    [[maybe_unused]] basic_type_result basic_type_result;

    std::cout << "✓ Type alias compilation tests passed\n";
}

// Test that the grammar namespace structure is correct
void test_namespace_structure() {
    // Verify namespaces are accessible
    using namespace cpp2_transpiler::parser::grammar;
    using namespace cpp2_transpiler::parser::grammar::operators;
    using namespace cpp2_transpiler::parser::grammar::combinators;

    // If we can access these namespaces, the structure is correct
    std::cout << "✓ Namespace structure tests passed\n";
}

// Example: Demonstrate EBNF to type alias mapping
void demonstrate_ebnf_mapping() {
    std::cout << "\nEBNF to Type Alias Mapping Examples:\n";
    std::cout << "=====================================\n";

    std::cout << "EBNF: expression ::= assignment_expression\n";
    std::cout << "Type: using expression_result = std::unique_ptr<Expression>;\n\n";

    std::cout << "EBNF: declaration ::= namespace_declaration | template_declaration | ...\n";
    std::cout << "Type: using declaration_result = std::unique_ptr<Declaration>;\n\n";

    std::cout << "EBNF: parameter_list ::= parameter { \",\" parameter } [ \",\" ]\n";
    std::cout << "Type: using parameter_list_result = std::vector<std::pair<std::string, std::unique_ptr<Type>>>;\n\n";

    std::cout << "EBNF: if_statement ::= \"if\" [ \"constexpr\" ] expression block_statement ...\n";
    std::cout << "Type: using if_statement_result = std::unique_ptr<IfStatement>;\n\n";
}

int main() {
    std::cout << "Parser Grammar Tests\n";
    std::cout << "====================\n\n";

    try {
        test_operator_precedence();
        test_operator_associativity();
        test_type_alias_compilation();
        test_namespace_structure();
        demonstrate_ebnf_mapping();

        std::cout << "\n✅ All parser grammar tests passed!\n";
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "\n❌ Test failed: " << e.what() << "\n";
        return 1;
    }
}
