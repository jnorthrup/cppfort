#include "simple_graph_nodes.h"
#include <cassert>
#include <iostream>
#include <string>

using cppfort::stage0::GraphNode;
using cppfort::stage0::GraphNodeType;
using namespace cppfort::stage0::GraphNodeFactory;

void test_basic_node_creation() {
    std::cout << "Testing basic node creation...\n";
    
    // Test creating a basic function declaration node
    auto func_node = createFunctionDeclaration("main", "int");
    
    assert(func_node->getType() == GraphNodeType::FUNCTION_DECLARATION);
    assert(func_node->getProperty("name") == "main");
    assert(func_node->getProperty("return_type") == "int");
    
    std::cout << "Basic node creation test passed.\n";
}

void test_parameter_node() {
    std::cout << "Testing parameter node...\n";
    
    // Test creating a parameter node with kind information
    auto param_node = createParameter("x", "int", "in");
    
    assert(param_node->getType() == GraphNodeType::PARAMETER);
    assert(param_node->getProperty("name") == "x");
    assert(param_node->getProperty("type") == "int");
    assert(param_node->getProperty("kind") == "in");
    
    std::cout << "Parameter node test passed.\n";
}

void test_main_function_node() {
    std::cout << "Testing main function node...\n";
    
    // Test creating a proper main function node
    auto main_node = createMainFunction();
    
    assert(main_node->getType() == GraphNodeType::MAIN_FUNCTION);
    assert(main_node->getProperty("signature") == "int main()");
    
    std::cout << "Main function node test passed.\n";
}

void test_generic_parameter_node() {
    std::cout << "Testing generic parameter node...\n";
    
    // Test creating a generic parameter node
    auto generic_node = createGenericParameter("_", "type");
    
    assert(generic_node->getType() == GraphNodeType::GENERIC_PARAMETER);
    assert(generic_node->getProperty("placeholder") == "_");
    assert(generic_node->getProperty("constraint") == "type");
    
    std::cout << "Generic parameter node test passed.\n";
}

void test_node_relationships() {
    std::cout << "Testing node relationships...\n";
    
    // Test creating nodes with parent-child relationships
    auto func_node = createFunctionDeclaration("process", "void");
    auto param_node = createParameter("data", "std::string", "in");
    
    // Add child relationship
    func_node->addChild(param_node.get());
    
    assert(func_node->getChildren().size() == 1);
    assert(func_node->getChildren()[0] == param_node.get());
    
    std::cout << "Node relationships test passed.\n";
}

void test_to_string() {
    std::cout << "Testing node to_string...\n";
    
    auto node = createFunctionDeclaration("test", "void");
    node->setProperty("test_prop", "test_value");
    
    std::string str = node->toString();
    std::cout << "Node string representation: " << str << "\n";
    
    // Basic check that the string contains expected elements
    assert(str.find("FUNCTION_DECLARATION") != std::string::npos);
    assert(str.find("name") != std::string::npos);
    assert(str.find("test") != std::string::npos);
    
    std::cout << "Node to_string test passed.\n";
}

int main() {
    std::cout << "Running simple graph node tests...\n\n";
    
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
    
    test_to_string();
    std::cout << "\n";
    
    std::cout << "All simple graph node tests passed!\n";
    return 0;
}