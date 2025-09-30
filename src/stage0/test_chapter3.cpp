#include "son_parser.h"
#include <iostream>
#include <cassert>

using namespace cppfort::ir;

void test_simple_declaration() {
    std::cout << "Testing: int a=1; return a;\n";
    SoNParser parser;

    Node* result = parser.parse("int a=1; return a;");

    // Should return the value 1
    ReturnNode* ret = dynamic_cast<ReturnNode*>(result);
    assert(ret != nullptr);

    ConstantNode* constant = dynamic_cast<ConstantNode*>(ret->value());
    assert(constant != nullptr);
    assert(constant->_value == 1);

    std::cout << "✓ Test passed - simple declaration worked\n\n";
}

void test_multiple_declarations() {
    std::cout << "Testing: int a=1; int b=2; return a+b;\n";
    SoNParser parser;

    Node* result = parser.parse("int a=1; int b=2; return a+b;");

    // Should fold to 3
    ReturnNode* ret = dynamic_cast<ReturnNode*>(result);
    assert(ret != nullptr);

    ConstantNode* constant = dynamic_cast<ConstantNode*>(ret->value());
    assert(constant != nullptr);
    assert(constant->_value == 3);

    std::cout << "✓ Test passed - multiple declarations worked\n\n";
}

void test_variable_reassignment() {
    std::cout << "Testing: int a=1; a=5; return a;\n";
    SoNParser parser;

    Node* result = parser.parse("int a=1; a=5; return a;");

    // Should return 5
    ReturnNode* ret = dynamic_cast<ReturnNode*>(result);
    assert(ret != nullptr);

    ConstantNode* constant = dynamic_cast<ConstantNode*>(ret->value());
    assert(constant != nullptr);
    assert(constant->_value == 5);

    std::cout << "✓ Test passed - variable reassignment worked\n\n";
}

void test_nested_scope() {
    std::cout << "Testing nested scope from README:\n";
    std::cout << "  int a=1; int b=2; int c=0; { int b=3; c=a+b; } return c;\n";
    SoNParser parser;

    Node* result = parser.parse(
        "int a=1; "
        "int b=2; "
        "int c=0; "
        "{ "
        "    int b=3; "
        "    c=a+b; "
        "} "
        "return c;"
    );

    // Should return 4 (1 + 3)
    ReturnNode* ret = dynamic_cast<ReturnNode*>(result);
    assert(ret != nullptr);

    ConstantNode* constant = dynamic_cast<ConstantNode*>(ret->value());
    assert(constant != nullptr);
    assert(constant->_value == 4);

    std::cout << "✓ Test passed - nested scope with variable shadowing worked\n\n";
}

void test_complex_expression_with_vars() {
    std::cout << "Testing: int x=2; int y=3; return x*y + x;\n";
    SoNParser parser;

    Node* result = parser.parse("int x=2; int y=3; return x*y + x;");

    // Should fold to 8 (2*3 + 2)
    ReturnNode* ret = dynamic_cast<ReturnNode*>(result);
    assert(ret != nullptr);

    ConstantNode* constant = dynamic_cast<ConstantNode*>(ret->value());
    assert(constant != nullptr);
    assert(constant->_value == 8);

    std::cout << "✓ Test passed - complex expression with variables worked\n\n";
}

void test_distance_formula() {
    std::cout << "Testing distance formula from README:\n";
    std::cout << "  int x0=1; int y0=2; int x1=3; int y1=4;\n";
    std::cout << "  return (x0-x1)*(x0-x1) + (y0-y1)*(y0-y1);\n";
    SoNParser parser;

    Node* result = parser.parse(
        "int x0=1; "
        "int y0=2; "
        "int x1=3; "
        "int y1=4; "
        "return (x0-x1)*(x0-x1) + (y0-y1)*(y0-y1);"
    );

    // Should fold to 8
    // (1-3)*(1-3) + (2-4)*(2-4) = (-2)*(-2) + (-2)*(-2) = 4 + 4 = 8
    ReturnNode* ret = dynamic_cast<ReturnNode*>(result);
    assert(ret != nullptr);

    ConstantNode* constant = dynamic_cast<ConstantNode*>(ret->value());
    assert(constant != nullptr);
    assert(constant->_value == 8);

    std::cout << "✓ Test passed - distance formula with peephole optimization worked\n\n";
}

void test_multiple_scopes_at_same_level() {
    std::cout << "Testing multiple scopes at same level:\n";
    std::cout << "  int a=1; int b=2; int c=0;\n";
    std::cout << "  { int b=5; c=a+b; }\n";
    std::cout << "  { int e=6; c=a+e; }\n";
    std::cout << "  return c;\n";
    SoNParser parser;

    Node* result = parser.parse(
        "int a=1; "
        "int b=2; "
        "int c=0; "
        "{ "
        "    int b=5; "
        "    c=a+b; "
        "} "
        "{ "
        "    int e=6; "
        "    c=a+e; "
        "} "
        "return c;"
    );

    // Should return 7 (1 + 6, from the second block)
    ReturnNode* ret = dynamic_cast<ReturnNode*>(result);
    assert(ret != nullptr);

    ConstantNode* constant = dynamic_cast<ConstantNode*>(ret->value());
    assert(constant != nullptr);
    assert(constant->_value == 7);

    std::cout << "✓ Test passed - multiple scopes at same level worked\n\n";
}

void test_deeply_nested_scopes() {
    std::cout << "Testing deeply nested scopes:\n";
    std::cout << "  int a=1; { int b=2; { int c=3; a=a+b+c; } } return a;\n";
    SoNParser parser;

    Node* result = parser.parse(
        "int a=1; "
        "{ "
        "    int b=2; "
        "    { "
        "        int c=3; "
        "        a=a+b+c; "
        "    } "
        "} "
        "return a;"
    );

    // Should return 6 (1 + 2 + 3)
    ReturnNode* ret = dynamic_cast<ReturnNode*>(result);
    assert(ret != nullptr);

    ConstantNode* constant = dynamic_cast<ConstantNode*>(ret->value());
    assert(constant != nullptr);
    assert(constant->_value == 6);

    std::cout << "✓ Test passed - deeply nested scopes worked\n\n";
}

void test_scope_cleanup() {
    std::cout << "Testing scope cleanup (variable dies when scope ends):\n";
    std::cout << "  int a=1; { int b=a*a*a; } return a;\n";
    SoNParser parser;

    Node* result = parser.parse(
        "int a=1; "
        "{ "
        "    int b=a*a*a; "
        "} "
        "return a;"
    );

    // Should return 1 (b and the cube calculation should be cleaned up)
    ReturnNode* ret = dynamic_cast<ReturnNode*>(result);
    assert(ret != nullptr);

    ConstantNode* constant = dynamic_cast<ConstantNode*>(ret->value());
    assert(constant != nullptr);
    assert(constant->_value == 1);

    std::cout << "✓ Test passed - scope cleanup worked\n\n";
}

int main() {
    std::cout << "=== Chapter 3 Tests: Variable Declarations and Scopes ===\n\n";

    try {
        test_simple_declaration();
        test_multiple_declarations();
        test_variable_reassignment();
        test_nested_scope();
        test_complex_expression_with_vars();
        test_distance_formula();
        test_multiple_scopes_at_same_level();
        test_deeply_nested_scopes();
        test_scope_cleanup();

        std::cout << "=== All Chapter 3 tests passed! ===\n";
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "Test failed with error: " << e.what() << std::endl;
        return 1;
    }
}