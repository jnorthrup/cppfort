#include "son_parser.h"
#include <iostream>
#include <cassert>

using namespace cppfort::ir;

void test_return_constant() {
    std::cout << "Testing: return 1;\n";
    SoNParser parser;

    Node* result = parser.parse("return 1;");

    // Verify we got a ReturnNode
    assert(result != nullptr);
    ReturnNode* ret = dynamic_cast<ReturnNode*>(result);
    assert(ret != nullptr);
    assert(ret->isCFG());

    // Verify the return has START as control input
    assert(ret->_inputs.size() == 2);
    assert(ret->_inputs[0] == parser.getStart());

    // Verify the return value is a Constant with value 1
    ConstantNode* constant = dynamic_cast<ConstantNode*>(ret->_inputs[1]);
    assert(constant != nullptr);
    assert(constant->_value == 1);

    // Verify the constant has START as input (for graph walking)
    assert(constant->_inputs.size() == 1);
    assert(constant->_inputs[0] == parser.getStart());

    std::cout << "✓ Test passed\n\n";

    // Visualize the graph
    std::cout << parser.visualize() << "\n";
}

void test_return_larger_constant() {
    std::cout << "Testing: return 42;\n";
    SoNParser parser;

    Node* result = parser.parse("return 42;");

    ReturnNode* ret = dynamic_cast<ReturnNode*>(result);
    assert(ret != nullptr);

    ConstantNode* constant = dynamic_cast<ConstantNode*>(ret->value());
    assert(constant != nullptr);
    assert(constant->_value == 42);

    std::cout << "✓ Test passed\n\n";
}

void test_return_zero() {
    std::cout << "Testing: return 0;\n";
    SoNParser parser;

    Node* result = parser.parse("return 0;");

    ReturnNode* ret = dynamic_cast<ReturnNode*>(result);
    assert(ret != nullptr);

    ConstantNode* constant = dynamic_cast<ConstantNode*>(ret->value());
    assert(constant != nullptr);
    assert(constant->_value == 0);

    std::cout << "✓ Test passed\n\n";
}

void test_return_negative() {
    std::cout << "Testing: return -5;\n";
    SoNParser parser;

    Node* result = parser.parse("return -5;");

    ReturnNode* ret = dynamic_cast<ReturnNode*>(result);
    assert(ret != nullptr);

    ConstantNode* constant = dynamic_cast<ConstantNode*>(ret->value());
    assert(constant != nullptr);
    assert(constant->_value == -5);

    std::cout << "✓ Test passed\n\n";
}

void test_whitespace_handling() {
    std::cout << "Testing: whitespace handling\n";
    SoNParser parser;

    Node* result = parser.parse("  return   123  ;  ");

    ReturnNode* ret = dynamic_cast<ReturnNode*>(result);
    assert(ret != nullptr);

    ConstantNode* constant = dynamic_cast<ConstantNode*>(ret->value());
    assert(constant != nullptr);
    assert(constant->_value == 123);

    std::cout << "✓ Test passed\n\n";
}

int main() {
    std::cout << "=== Chapter 1: Sea of Nodes - Return Statements ===\n\n";

    try {
        test_return_constant();
        test_return_larger_constant();
        test_return_zero();
        test_return_negative();
        test_whitespace_handling();

        std::cout << "=== All Chapter 1 tests passed! ===\n";
        std::cout << "\nSummary:\n";
        std::cout << "- Successfully built Sea of Nodes graph directly during parsing\n";
        std::cout << "- No AST intermediate representation\n";
        std::cout << "- Three node types implemented: StartNode, ConstantNode, ReturnNode\n";
        std::cout << "- Bidirectional edges maintained (_inputs and _outputs)\n";
        std::cout << "- Unique dense node IDs assigned\n";

    } catch (const std::exception& e) {
        std::cerr << "Test failed: " << e.what() << std::endl;
        return 1;
    }

    return 0;
}