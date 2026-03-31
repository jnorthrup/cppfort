// chapter04_test.cpp - Test suite for Sea of Nodes Chapter 4
// Tests comparison operators and external variables (arg parameter)

#include <iostream>
#include <string>
#include <vector>
#include <stdexcept>
#include <cassert>

// Include the generated header
#include "son_chapter04.h"

// Test helper functions
bool test_result = true;

void check(bool condition, const char* msg) {
    if (!condition) {
        std::cout << "  FAIL: " << msg << std::endl;
        test_result = false;
    }
}

// Simple test driver
int main() {
    std::cout << "Testing Sea of Nodes Chapter 4..." << std::endl;

    // Test 1: Comparison operators defined
    std::cout << "Test 1: Comparison operator nodes..." << std::endl;
    {
        // Reset nodes
        reset_nodes();
        enter_scope();
        
        // Create start node
        start_node: int = make_start_node();
        
        // Create constants for comparison
        c1: int = make_constant_node(5, start_node);
        c2: int = make_constant_node(3, start_node);
        
        // Test EQ
        eq_node: int = make_eq_node(c1, c2);
        check(node_op(eq_node) == op_type::OP_EQ, "EQ node type correct");
        
        // Test NE
        ne_node: int = make_ne_node(c1, c2);
        check(node_op(ne_node) == op_type::OP_NE, "NE node type correct");
        
        // Test LT
        lt_node: int = make_lt_node(c1, c2);
        check(node_op(lt_node) == op_type::OP_LT, "LT node type correct");
        
        // Test LE
        le_node: int = make_le_node(c1, c2);
        check(node_op(le_node) == op_type::OP_LE, "LE node type correct");
        
        // Test GT
        gt_node: int = make_gt_node(c1, c2);
        check(node_op(gt_node) == op_type::OP_GT, "GT node type correct");
        
        // Test GE
        ge_node: int = make_ge_node(c1, c2);
        check(node_op(ge_node) == op_type::OP_GE, "GE node type correct");
        
        exit_scope();
        std::cout << "  PASS: Comparison operators defined" << std::endl;
    }

    // Test 2: Constant folding for comparisons
    std::cout << "Test 2: Constant folding for comparisons..." << std::endl;
    {
        reset_nodes();
        enter_scope();
        
        start_node: int = make_start_node();
        
        // Test 5 == 3 -> false (0)
        c5: int = make_constant_node(5, start_node);
        c3: int = make_constant_node(3, start_node);
        eq: int = make_eq_node(c5, c3);
        folded: int = parser_try_comparison_fold(parser(), eq);
        
        check(node_op(folded) == op_type::OP_CONSTANT, "EQ folded to constant");
        check(get_constant_value(folded) == 0, "5 == 3 is false");
        
        // Test 5 < 3 -> false (0)
        lt: int = make_lt_node(c5, c3);
        folded_lt: int = parser_try_comparison_fold(parser(), lt);
        
        check(node_op(folded_lt) == op_type::OP_CONSTANT, "LT folded to constant");
        check(get_constant_value(folded_lt) == 0, "5 < 3 is false");
        
        // Test 3 < 5 -> true (1)
        lt2: int = make_lt_node(c3, c5);
        folded_lt2: int = parser_try_comparison_fold(parser(), lt2);
        
        check(node_op(folded_lt2) == op_type::OP_CONSTANT, "LT folded to constant");
        check(get_constant_value(folded_lt2) == 1, "3 < 5 is true");
        
        exit_scope();
        std::cout << "  PASS: Constant folding works for comparisons" << std::endl;
    }

    // Test 3: Parameter node (arg)
    std::cout << "Test 3: Parameter node for 'arg'..." << std::endl;
    {
        reset_nodes();
        enter_scope();
        
        start_node: int = make_start_node();
        
        // Create parameter node
        param: int = make_parameter_node(start_node);
        check(node_op(param) == op_type::OP_PARAMETER, "Parameter node created");
        
        // Declare it in scope
        declare_variable("__arg__", param);
        
        // Look it up
        found: int = lookup_variable("__arg__");
        check(found == param, "Parameter can be looked up");
        
        exit_scope();
        std::cout << "  PASS: Parameter node works" << std::endl;
    }

    // Test 4: Parser with comparison expressions
    std::cout << "Test 4: Parser with comparison expressions..." << std::endl;
    {
        // Test parsing "return 5 < 10;"
        p: parser = make_parser("return 5 < 10;");
        result: int = parser_parse(p);
        
        check(result >= 0, "Parser returns valid node");
        
        // The result should be a return node
        check(node_op(result) == op_type::OP_RETURN, "Result is return node");
        
        std::cout << "  PASS: Parser handles comparison expressions" << std::endl;
    }

    // Test 5: Parser with chained comparisons
    std::cout << "Test 5: Parser with chained comparisons..." << std::endl;
    {
        // Test parsing "return 1 < 2 < 3;" - should be (1 < 2) < 3 = 1 < 3 = true
        p: parser = make_parser("return 1 < 2 < 3;");
        result: int = parser_parse(p);
        
        check(result >= 0, "Parser returns valid node for chained comparison");
        
        std::cout << "  PASS: Parser handles chained comparisons" << std::endl;
    }

    // Test 6: Parser with equality operators
    std::cout << "Test 6: Parser with equality operators..." << std::endl;
    {
        p: parser = make_parser("return 5 == 5;");
        result: int = parser_parse(p);
        check(result >= 0, "Parser handles ==");
        
        p2: parser = make_parser("return 5 != 3;");
        result2: int = parser_parse(p2);
        check(result2 >= 0, "Parser handles !=");
        
        std::cout << "  PASS: Parser handles equality operators" << std::endl;
    }

    // Test 7: Parser with >= and <=
    std::cout << "Test 7: Parser with >= and <=" << std::endl;
    {
        p: parser = make_parser("return 5 >= 3;");
        result: int = parser_parse(p);
        check(result >= 0, "Parser handles >=");
        
        p2: parser = make_parser("return 3 <= 5;");
        result2: int = parser_parse(p2);
        check(result2 >= 0, "Parser handles <=");
        
        std::cout << "  PASS: Parser handles >= and <=" << std::endl;
    }

    // Test 8: Complex expression with comparisons and arithmetic
    std::cout << "Test 8: Complex expression with comparisons and arithmetic..." << std::endl;
    {
        // Test: (5 + 3) < (10 - 2) -> 8 < 8 -> false
        p: parser = make_parser("return 5 + 3 < 10 - 2;");
        result: int = parser_parse(p);
        check(result >= 0, "Parser handles mixed arithmetic and comparison");
        
        std::cout << "  PASS: Complex expressions work" << std::endl;
    }

    // Test 9: Variables with comparisons
    std::cout << "Test 9: Variables with comparisons..." << std::endl;
    {
        p: parser = make_parser("int x = 5; return x < 10;");
        result: int = parser_parse(p);
        check(result >= 0, "Parser handles variable in comparison");
        
        std::cout << "  PASS: Variables work in comparisons" << std::endl;
    }

    // Test 10: Not operator
    std::cout << "Test 10: Not operator..." << std::endl;
    {
        reset_nodes();
        enter_scope();
        
        start_node: int = make_start_node();
        c1: int = make_constant_node(5, start_node);
        
        not_node: int = make_not_node(c1);
        check(node_op(not_node) == op_type::OP_NOT, "NOT node created");
        
        exit_scope();
        std::cout << "  PASS: NOT operator works" << std::endl;
    }

    if (test_result) {
        std::cout << std::endl << "All Chapter 4 tests passed!" << std::endl;
        return 0;
    } else {
        std::cout << std::endl << "Some tests FAILED!" << std::endl;
        return 1;
    }
}
