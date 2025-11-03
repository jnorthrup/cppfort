#include "graph_nodes.h"
#include <cassert>
#include <iostream>
#include <string>

using cppfort::stage0::GraphNode;
using cppfort::stage0::GraphNodeType;

void test_basic_node_creation() {
    std::cout << "Testing basic node creation...\n";
    
    // Test creating a basic function declaration node
    GraphNode func_node(GraphNodeType::FUNCTION_DECLARATION);
    func_node.setProperty("name", "main");
    func_node.setProperty("return_type", "int");
    
    assert(func_node.getType() == GraphNodeType::FUNCTION_DECLARATION);
    assert(func_node.getProperty("name") == "main");
    assert(func_node.getProperty("return_type") == "int");
    
    std::cout << "Basic node creation test passed.\n";
}

void test_parameter_node() {
    std::cout << "Testing parameter node...\n";
    
    // Test creating a parameter node with kind information
    GraphNode param_node(GraphNodeType::PARAMETER);
    param_node.setProperty("name", "x");
    param_node.setProperty("type", "int");
    param_node.setProperty("kind", "in");  // in, inout, out, copy, move, forward
    
    assert(param_node.getType() == GraphNodeType::PARAMETER);
    assert(param_node.getProperty("name") == "x");
    assert(param_node.getProperty("type") == "int");
    assert(param_node.getProperty("kind") == "in");
    
    std::cout << "Parameter node test passed.\n";
}

void test_main_function_node() {
    std::cout << "Testing main function node...\n";
    
    // Test creating a proper main function node
    GraphNode main_node(GraphNodeType::MAIN_FUNCTION);
    main_node.setProperty("signature", "int main()");
    
    assert(main_node.getType() == GraphNodeType::MAIN_FUNCTION);
    assert(main_node.getProperty("signature") == "int main()");
    
    std::cout << "Main function node test passed.\n";
}

void test_generic_parameter_node() {
    std::cout << "Testing generic parameter node...\n";
    
    // Test creating a generic parameter node
    GraphNode generic_node(GraphNodeType::GENERIC_PARAMETER);
    generic_node.setProperty("placeholder", "_");
    generic_node.setProperty("constraint", "type");
    
    assert(generic_node.getType() == GraphNodeType::GENERIC_PARAMETER);
    assert(generic_node.getProperty("placeholder") == "_");
    assert(generic_node.getProperty("constraint") == "type");
    
    std::cout << "Generic parameter node test passed.\n";
}

void test_node_relationships() {
    std::cout << "Testing node relationships...\n";
    
    // Test creating nodes with parent-child relationships
    GraphNode func_node(GraphNodeType::FUNCTION_DECLARATION);
    func_node.setProperty("name", "process");
    
    GraphNode param_node(GraphNodeType::PARAMETER);
    param_node.setProperty("name", "data");
    param_node.setProperty("type", "std::string");
    param_node.setProperty("kind", "in");
    
    // Add child relationship
    func_node.addChild(&param_node);
    
    assert(func_node.getChildren().size() == 1);
    assert(func_node.getChildren()[0] == &param_node);
    
    std::cout << "Node relationships test passed.\n";
}

int main() {
    std::cout << "Running graph node tests...\n\n";
    
    test_basic_node_creation();
    std::cout << "\n";
    
    test_parameter_node();
    std::cout << "\n";
    
    test_main_function_node();
    std::cout << "\n";
    
    test_generic_parameter_node();
    std::cout << "\n";
    
    test_node_relationships();
    std::cout << "\n";
    
    std::cout << "All graph node tests passed!\n";
    return 0;
}