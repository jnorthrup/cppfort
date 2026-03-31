// Sea of Nodes Chapter 1 Tests
// Tests the base Node class, StartNode, ReturnNode, ConstantNode, and Parser

#include <iostream>
#include <cassert>
#include "son/chapter01.cpp"

using namespace son;

void test_node_creation() {
    std::cout << "Test: Node creation... ";
    
    Node::reset();
    
    // Create a start node
    StartNode* start = StartNode::New();
    assert(start->nid() == 1);
    assert(start->nIns() == 0);
    assert(start->isCFG() == true);
    
    // Create a constant node
    Node::reset();
    ConstantNode* cn = ConstantNode::New(start, 42);
    assert(cn->nid() == 1);  // After reset
    assert(cn->value() == 42);
    assert(cn->ctrl() == start);
    
    std::cout << "PASSED" << std::endl;
}

void test_return_node() {
    std::cout << "Test: Return node... ";
    
    Node::reset();
    
    StartNode* start = StartNode::New();
    ConstantNode* cn = ConstantNode::New(start, 100);
    ReturnNode* ret = ReturnNode::New(start, cn);
    
    assert(ret->nid() == 3);
    assert(ret->expr() == cn);
    assert(ret->ctrl() == start);
    
    std::cout << "PASSED" << std::endl;
}

void test_parser_simple() {
    std::cout << "Test: Parser simple return... ";
    
    Parser p("return 42;");
    ReturnNode* ret = p.parse();
    
    assert(ret != nullptr);
    // The constant value should be accessible through the return's expr
    
    std::cout << "PASSED" << std::endl;
}

void test_parser_zero() {
    std::cout << "Test: Parser return zero... ";
    
    Node::reset();
    Parser p("return 0;");
    ReturnNode* ret = p.parse();
    
    assert(ret != nullptr);
    
    std::cout << "PASSED" << std::endl;
}

void test_parser_multi_digit() {
    std::cout << "Test: Parser multi-digit... ";
    
    Node::reset();
    Parser p("return 12345;");
    ReturnNode* ret = p.parse();
    
    assert(ret != nullptr);
    
    std::cout << "PASSED" << std::endl;
}

int main() {
    std::cout << "=== Sea of Nodes Chapter 1 Tests ===" << std::endl;
    
    test_node_creation();
    test_return_node();
    test_parser_simple();
    test_parser_zero();
    test_parser_multi_digit();
    
    std::cout << "=== All Chapter 1 Tests Passed ===" << std::endl;
    return 0;
}
