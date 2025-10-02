#include "son_parser.h"
#include <iostream>
#include <cassert>

using namespace cppfort::ir;

void test_addition() {
    ::std::cout << "Testing: return 1 + 2;\n";
    SoNParser parser;

    Node* result = parser.parse("return 1 + 2;");

    // With peephole optimization, 1+2 should be folded to 3
    ReturnNode* ret = dynamic_cast<ReturnNode*>(result);
    assert(ret != nullptr);

    ConstantNode* constant = dynamic_cast<ConstantNode*>(ret->value());
    assert(constant != nullptr);
    assert(constant->_value == 3);

    ::std::cout << "✓ Test passed - constant folding worked\n\n";
}

void test_complex_arithmetic() {
    ::std::cout << "Testing: return 1 + 2 * 3;\n";
    SoNParser parser;

    Node* result = parser.parse("return 1 + 2 * 3;");

    // Should fold to 7 (respecting precedence: 2*3=6, then 1+6=7)
    ReturnNode* ret = dynamic_cast<ReturnNode*>(result);
    assert(ret != nullptr);

    ConstantNode* constant = dynamic_cast<ConstantNode*>(ret->value());
    assert(constant != nullptr);
    assert(constant->_value == 7);

    ::std::cout << "✓ Test passed - precedence and folding worked\n\n";
}

void test_unary_minus() {
    ::std::cout << "Testing: return -5;\n";
    SoNParser parser;

    Node* result = parser.parse("return -5;");

    ReturnNode* ret = dynamic_cast<ReturnNode*>(result);
    assert(ret != nullptr);

    ConstantNode* constant = dynamic_cast<ConstantNode*>(ret->value());
    assert(constant != nullptr);
    assert(constant->_value == -5);

    ::std::cout << "✓ Test passed\n\n";
}

void test_complex_with_unary() {
    ::std::cout << "Testing: return 1 + 2 * 3 + -5;\n";
    SoNParser parser;

    Node* result = parser.parse("return 1 + 2 * 3 + -5;");

    // Should fold to 2 (1 + 6 + (-5) = 2)
    ReturnNode* ret = dynamic_cast<ReturnNode*>(result);
    assert(ret != nullptr);

    ConstantNode* constant = dynamic_cast<ConstantNode*>(ret->value());
    assert(constant != nullptr);
    assert(constant->_value == 2);

    ::std::cout << "✓ Test passed - complex expression with unary minus\n\n";
}

void test_subtraction() {
    ::std::cout << "Testing: return 10 - 3;\n";
    SoNParser parser;

    Node* result = parser.parse("return 10 - 3;");

    ReturnNode* ret = dynamic_cast<ReturnNode*>(result);
    assert(ret != nullptr);

    ConstantNode* constant = dynamic_cast<ConstantNode*>(ret->value());
    assert(constant != nullptr);
    assert(constant->_value == 7);

    ::std::cout << "✓ Test passed\n\n";
}

void test_division() {
    ::std::cout << "Testing: return 15 / 3;\n";
    SoNParser parser;

    Node* result = parser.parse("return 15 / 3;");

    ReturnNode* ret = dynamic_cast<ReturnNode*>(result);
    assert(ret != nullptr);

    ConstantNode* constant = dynamic_cast<ConstantNode*>(ret->value());
    assert(constant != nullptr);
    assert(constant->_value == 5);

    ::std::cout << "✓ Test passed\n\n";
}

void test_parentheses() {
    ::std::cout << "Testing: return (1 + 2) * 3;\n";
    SoNParser parser;

    Node* result = parser.parse("return (1 + 2) * 3;");

    // Should fold to 9 (parentheses override precedence)
    ReturnNode* ret = dynamic_cast<ReturnNode*>(result);
    assert(ret != nullptr);

    ConstantNode* constant = dynamic_cast<ConstantNode*>(ret->value());
    assert(constant != nullptr);
    assert(constant->_value == 9);

    ::std::cout << "✓ Test passed - parentheses work\n\n";
}

void test_nested_parentheses() {
    ::std::cout << "Testing: return ((1 + 2) * (3 + 4));\n";
    SoNParser parser;

    Node* result = parser.parse("return ((1 + 2) * (3 + 4));");

    // Should fold to 21 (3 * 7)
    ReturnNode* ret = dynamic_cast<ReturnNode*>(result);
    assert(ret != nullptr);

    ConstantNode* constant = dynamic_cast<ConstantNode*>(ret->value());
    assert(constant != nullptr);
    assert(constant->_value == 21);

    ::std::cout << "✓ Test passed - nested parentheses\n\n";
}

void test_double_unary_minus() {
    ::std::cout << "Testing: return --5;\n";
    SoNParser parser;

    Node* result = parser.parse("return --5;");

    // Double negation should result in 5
    ReturnNode* ret = dynamic_cast<ReturnNode*>(result);
    assert(ret != nullptr);

    ConstantNode* constant = dynamic_cast<ConstantNode*>(ret->value());
    assert(constant != nullptr);
    assert(constant->_value == 5);

    ::std::cout << "✓ Test passed - double unary minus\n\n";
}

void test_type_system() {
    ::std::cout << "Testing: Type system lattice\n";

    // Test constant types
    TypeInteger* c1 = TypeInteger::constant(42);
    TypeInteger* c2 = TypeInteger::constant(42);
    assert(c1 == c2);  // Should be same instance (cached)
    assert(c1->isConstant());
    assert(c1->value() == 42);

    // Test bottom type
    TypeInteger* bottom = TypeInteger::bottom();
    assert(bottom->isBottom());
    assert(!bottom->isConstant());

    ::std::cout << "✓ Type system tests passed\n\n";
}

int main() {
    ::std::cout << "=== Chapter 2: Sea of Nodes - Arithmetic & Peephole Optimization ===\n\n";

    try {
        test_addition();
        test_complex_arithmetic();
        test_unary_minus();
        test_complex_with_unary();
        test_subtraction();
        test_division();
        test_parentheses();
        test_nested_parentheses();
        test_double_unary_minus();
        test_type_system();

        ::std::cout << "=== All Chapter 2 tests passed! ===\n";
        ::std::cout << "\nSummary:\n";
        ::std::cout << "- Arithmetic nodes: Add, Sub, Mul, Div, Minus (unary)\n";
        ::std::cout << "- Type system with TypeInteger lattice\n";
        ::std::cout << "- Peephole optimization during parsing\n";
        ::std::cout << "- Constant folding and propagation\n";
        ::std::cout << "- Dead node elimination via kill()\n";
        ::std::cout << "- Proper operator precedence and associativity\n";
        ::std::cout << "- Parenthesized expressions\n";

    } catch (const ::std::exception& e) {
        ::std::cerr << "Test failed: " << e.what() << ::std::endl;
        return 1;
    }

    return 0;
}