#include "c_parser.h"
#include <iostream>
#include <cassert>

using namespace cppfort::c;

void test_primary_expressions() {
    std::cout << "=== Testing Primary Expressions ===" << std::endl;

    // Integer literal
    {
        CLexer lexer("42", "test.c");
        CParser parser(lexer.tokenize());
        auto expr = parser.parsePrimaryExpression();
        assert(expr->type == CASTNodeType::INTEGER_CONST);
        assert(expr->value == "42");
    }

    // Identifier
    {
        CLexer lexer("variable", "test.c");
        CParser parser(lexer.tokenize());
        auto expr = parser.parsePrimaryExpression();
        assert(expr->type == CASTNodeType::IDENTIFIER_REF);
        assert(expr->value == "variable");
    }

    // Parenthesized
    {
        CLexer lexer("(100)", "test.c");
        CParser parser(lexer.tokenize());
        auto expr = parser.parsePrimaryExpression();
        assert(expr->type == CASTNodeType::INTEGER_CONST);
    }

    std::cout << "✓ Primary expressions parsed correctly" << std::endl;
}

void test_arithmetic_expressions() {
    std::cout << "=== Testing Arithmetic Expressions ===" << std::endl;

    // Addition
    {
        CLexer lexer("a + b", "test.c");
        CParser parser(lexer.tokenize());
        auto expr = parser.parseExpression();
        assert(expr->type == CASTNodeType::BINARY_EXPR);
        assert(expr->value == "+");
        assert(expr->children.size() == 2);
    }

    // Multiplication precedence
    {
        CLexer lexer("a + b * c", "test.c");
        CParser parser(lexer.tokenize());
        auto expr = parser.parseExpression();
        // Should be: a + (b * c)
        assert(expr->type == CASTNodeType::BINARY_EXPR);
        assert(expr->value == "+");
        // Right child should be multiplication
        assert(expr->children[1]->type == CASTNodeType::BINARY_EXPR);
        assert(expr->children[1]->value == "*");
    }

    // Parentheses override precedence
    {
        CLexer lexer("(a + b) * c", "test.c");
        CParser parser(lexer.tokenize());
        auto expr = parser.parseExpression();
        // Should be: (a + b) * c
        assert(expr->type == CASTNodeType::BINARY_EXPR);
        assert(expr->value == "*");
        // Left child should be addition
        assert(expr->children[0]->type == CASTNodeType::BINARY_EXPR);
        assert(expr->children[0]->value == "+");
    }

    std::cout << "✓ Arithmetic expressions with correct precedence" << std::endl;
}

void test_comparison_expressions() {
    std::cout << "=== Testing Comparison Expressions ===" << std::endl;

    {
        CLexer lexer("x < y", "test.c");
        CParser parser(lexer.tokenize());
        auto expr = parser.parseExpression();
        assert(expr->type == CASTNodeType::BINARY_EXPR);
        assert(expr->value == "<");
    }

    {
        CLexer lexer("x == y", "test.c");
        CParser parser(lexer.tokenize());
        auto expr = parser.parseExpression();
        assert(expr->type == CASTNodeType::BINARY_EXPR);
        assert(expr->value == "==");
    }

    {
        CLexer lexer("x != y", "test.c");
        CParser parser(lexer.tokenize());
        auto expr = parser.parseExpression();
        assert(expr->type == CASTNodeType::BINARY_EXPR);
        assert(expr->value == "!=");
    }

    std::cout << "✓ Comparison expressions parsed correctly" << std::endl;
}

void test_logical_expressions() {
    std::cout << "=== Testing Logical Expressions ===" << std::endl;

    {
        CLexer lexer("a && b", "test.c");
        CParser parser(lexer.tokenize());
        auto expr = parser.parseExpression();
        assert(expr->type == CASTNodeType::BINARY_EXPR);
        assert(expr->value == "&&");
    }

    {
        CLexer lexer("a || b", "test.c");
        CParser parser(lexer.tokenize());
        auto expr = parser.parseExpression();
        assert(expr->type == CASTNodeType::BINARY_EXPR);
        assert(expr->value == "||");
    }

    // Precedence: && binds tighter than ||
    {
        CLexer lexer("a || b && c", "test.c");
        CParser parser(lexer.tokenize());
        auto expr = parser.parseExpression();
        // Should be: a || (b && c)
        assert(expr->type == CASTNodeType::BINARY_EXPR);
        assert(expr->value == "||");
        assert(expr->children[1]->value == "&&");
    }

    std::cout << "✓ Logical expressions with correct precedence" << std::endl;
}

void test_bitwise_expressions() {
    std::cout << "=== Testing Bitwise Expressions ===" << std::endl;

    {
        CLexer lexer("a & b", "test.c");
        CParser parser(lexer.tokenize());
        auto expr = parser.parseExpression();
        assert(expr->type == CASTNodeType::BINARY_EXPR);
        assert(expr->value == "&");
    }

    {
        CLexer lexer("a | b", "test.c");
        CParser parser(lexer.tokenize());
        auto expr = parser.parseExpression();
        assert(expr->type == CASTNodeType::BINARY_EXPR);
        assert(expr->value == "|");
    }

    {
        CLexer lexer("a ^ b", "test.c");
        CParser parser(lexer.tokenize());
        auto expr = parser.parseExpression();
        assert(expr->type == CASTNodeType::BINARY_EXPR);
        assert(expr->value == "^");
    }

    {
        CLexer lexer("a << 2", "test.c");
        CParser parser(lexer.tokenize());
        auto expr = parser.parseExpression();
        assert(expr->type == CASTNodeType::BINARY_EXPR);
        assert(expr->value == "<<");
    }

    {
        CLexer lexer("a >> 2", "test.c");
        CParser parser(lexer.tokenize());
        auto expr = parser.parseExpression();
        assert(expr->type == CASTNodeType::BINARY_EXPR);
        assert(expr->value == ">>");
    }

    std::cout << "✓ Bitwise expressions parsed correctly" << std::endl;
}

void test_assignment_expressions() {
    std::cout << "=== Testing Assignment Expressions ===" << std::endl;

    {
        CLexer lexer("x = 42", "test.c");
        CParser parser(lexer.tokenize());
        auto expr = parser.parseExpression();
        assert(expr->type == CASTNodeType::BINARY_EXPR);
        assert(expr->value == "=");
    }

    {
        CLexer lexer("x += 10", "test.c");
        CParser parser(lexer.tokenize());
        auto expr = parser.parseExpression();
        assert(expr->type == CASTNodeType::BINARY_EXPR);
        assert(expr->value == "+=");
    }

    // Right associativity
    {
        CLexer lexer("a = b = c", "test.c");
        CParser parser(lexer.tokenize());
        auto expr = parser.parseExpression();
        // Should be: a = (b = c)
        assert(expr->type == CASTNodeType::BINARY_EXPR);
        assert(expr->value == "=");
        assert(expr->children[1]->type == CASTNodeType::BINARY_EXPR);
        assert(expr->children[1]->value == "=");
    }

    std::cout << "✓ Assignment expressions with right associativity" << std::endl;
}

void test_unary_expressions() {
    std::cout << "=== Testing Unary Expressions ===" << std::endl;

    {
        CLexer lexer("-x", "test.c");
        CParser parser(lexer.tokenize());
        auto expr = parser.parseExpression();
        assert(expr->type == CASTNodeType::UNARY_EXPR);
        assert(expr->value == "-");
    }

    {
        CLexer lexer("!flag", "test.c");
        CParser parser(lexer.tokenize());
        auto expr = parser.parseExpression();
        assert(expr->type == CASTNodeType::UNARY_EXPR);
        assert(expr->value == "!");
    }

    {
        CLexer lexer("~bits", "test.c");
        CParser parser(lexer.tokenize());
        auto expr = parser.parseExpression();
        assert(expr->type == CASTNodeType::UNARY_EXPR);
        assert(expr->value == "~");
    }

    {
        CLexer lexer("*ptr", "test.c");
        CParser parser(lexer.tokenize());
        auto expr = parser.parseExpression();
        assert(expr->type == CASTNodeType::POINTER_DEREF);
    }

    {
        CLexer lexer("&var", "test.c");
        CParser parser(lexer.tokenize());
        auto expr = parser.parseExpression();
        assert(expr->type == CASTNodeType::ADDRESS_OF);
    }

    std::cout << "✓ Unary expressions parsed correctly" << std::endl;
}

void test_postfix_expressions() {
    std::cout << "=== Testing Postfix Expressions ===" << std::endl;

    // Array subscript
    {
        CLexer lexer("arr[i]", "test.c");
        CParser parser(lexer.tokenize());
        auto expr = parser.parseExpression();
        assert(expr->type == CASTNodeType::ARRAY_SUBSCRIPT);
    }

    // Function call
    {
        CLexer lexer("foo(a, b)", "test.c");
        CParser parser(lexer.tokenize());
        auto expr = parser.parseExpression();
        assert(expr->type == CASTNodeType::CALL_EXPR);
        assert(expr->children.size() == 3);  // func + 2 args
    }

    // Member access
    {
        CLexer lexer("obj.field", "test.c");
        CParser parser(lexer.tokenize());
        auto expr = parser.parseExpression();
        assert(expr->type == CASTNodeType::MEMBER_ACCESS);
    }

    // Pointer member access (desugared to dereference + member)
    {
        CLexer lexer("ptr->field", "test.c");
        CParser parser(lexer.tokenize());
        auto expr = parser.parseExpression();
        assert(expr->type == CASTNodeType::MEMBER_ACCESS);
        assert(expr->children[0]->type == CASTNodeType::POINTER_DEREF);
    }

    std::cout << "✓ Postfix expressions parsed correctly" << std::endl;
}

void test_conditional_expression() {
    std::cout << "=== Testing Conditional Expression ===" << std::endl;

    {
        CLexer lexer("x ? y : z", "test.c");
        CParser parser(lexer.tokenize());
        auto expr = parser.parseExpression();
        assert(expr->type == CASTNodeType::CONDITIONAL_EXPR);
        assert(expr->children.size() == 3);  // cond, then, else
    }

    std::cout << "✓ Conditional expression (ternary operator) parsed" << std::endl;
}

void test_complex_expression() {
    std::cout << "=== Testing Complex Expression ===" << std::endl;

    // a + b * c < d && e || f(g) ? h : i->j
    CLexer lexer("a + b * c < d && e || f(g) ? h : i->j", "test.c");
    CParser parser(lexer.tokenize());
    auto expr = parser.parseExpression();

    // Should parse without error and create proper tree structure
    assert(expr != nullptr);
    assert(expr->type == CASTNodeType::CONDITIONAL_EXPR);  // Top-level is ?:

    std::cout << "✓ Complex expression parsed with correct precedence" << std::endl;
}

int main() {
    std::cout << "Running C Parser Expression Tests" << std::endl;
    std::cout << "===================================" << std::endl << std::endl;

    test_primary_expressions();
    test_arithmetic_expressions();
    test_comparison_expressions();
    test_logical_expressions();
    test_bitwise_expressions();
    test_assignment_expressions();
    test_unary_expressions();
    test_postfix_expressions();
    test_conditional_expression();
    test_complex_expression();

    std::cout << std::endl << "All C parser expression tests passed! ✓" << std::endl;
    return 0;
}
