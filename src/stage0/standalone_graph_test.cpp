#include <iostream>
#include <string>
#include <unordered_map>
#include <vector>
#include <memory>
#include <cassert>

// Simple GraphNode implementation for testing
enum class GraphNodeType {
    UNKNOWN,
    FUNCTION_DECLARATION,
    MAIN_FUNCTION,
    PARAMETER,
    GENERIC_PARAMETER,
    RETURN_VALUE,
    TYPE_DECLARATION,
    EXPRESSION,
    STATEMENT,
    BLOCK,
    TEMPLATE_PARAMETER,
    CONTRACT,
    CAPTURE
};

class GraphNode {
private:
    GraphNodeType type_;
    std::unordered_map<std::string, std::string> properties_;
    std::vector<GraphNode*> children_;

public:
    // Constructor
    explicit GraphNode(GraphNodeType type = GraphNodeType::UNKNOWN) 
        : type_(type) {}
    
    // Getters and setters
    GraphNodeType getType() const { return type_; }
    void setType(GraphNodeType type) { type_ = type; }
    
    const std::string& getProperty(const std::string& key) const {
        static const std::string empty;
        auto it = properties_.find(key);
        return (it != properties_.end()) ? it->second : empty;
    }
    
    void setProperty(const std::string& key, const std::string& value) {
        properties_[key] = value;
    }
    
    bool hasProperty(const std::string& key) const {
        return properties_.find(key) != properties_.end();
    }
    
    const std::unordered_map<std::string, std::string>& getAllProperties() const {
        return properties_;
    }
    
    // Parent-child relationships
    void addChild(GraphNode* child) {
        if (child) {
            children_.push_back(child);
        }
    }
    
    const std::vector<GraphNode*>& getChildren() const {
        return children_;
    }
    
    // Utility methods
    std::string toString() const;
};

std::string GraphNode::toString() const {
    std::string typeStr;
    switch (type_) {
        case GraphNodeType::FUNCTION_DECLARATION: typeStr = "FUNCTION_DECLARATION"; break;
        case GraphNodeType::MAIN_FUNCTION: typeStr = "MAIN_FUNCTION"; break;
        case GraphNodeType::PARAMETER: typeStr = "PARAMETER"; break;
        case GraphNodeType::GENERIC_PARAMETER: typeStr = "GENERIC_PARAMETER"; break;
        case GraphNodeType::RETURN_VALUE: typeStr = "RETURN_VALUE"; break;
        case GraphNodeType::TYPE_DECLARATION: typeStr = "TYPE_DECLARATION"; break;
        case GraphNodeType::EXPRESSION: typeStr = "EXPRESSION"; break;
        case GraphNodeType::STATEMENT: typeStr = "STATEMENT"; break;
        case GraphNodeType::BLOCK: typeStr = "BLOCK"; break;
        case GraphNodeType::TEMPLATE_PARAMETER: typeStr = "TEMPLATE_PARAMETER"; break;
        case GraphNodeType::CONTRACT: typeStr = "CONTRACT"; break;
        case GraphNodeType::CAPTURE: typeStr = "CAPTURE"; break;
        default: typeStr = "UNKNOWN"; break;
    }
    
    return typeStr;
}

// Factory functions
std::unique_ptr<GraphNode> createFunctionDeclaration(const std::string& name, const std::string& returnType) {
    auto node = std::make_unique<GraphNode>(GraphNodeType::FUNCTION_DECLARATION);
    node->setProperty("name", name);
    node->setProperty("return_type", returnType);
    return node;
}

std::unique_ptr<GraphNode> createMainFunction() {
    auto node = std::make_unique<GraphNode>(GraphNodeType::MAIN_FUNCTION);
    node->setProperty("signature", "int main()");
    return node;
}

std::unique_ptr<GraphNode> createParameter(const std::string& name, const std::string& type, const std::string& kind = "in") {
    auto node = std::make_unique<GraphNode>(GraphNodeType::PARAMETER);
    node->setProperty("name", name);
    node->setProperty("type", type);
    node->setProperty("kind", kind);  // in, inout, out, copy, move, forward
    return node;
}

std::unique_ptr<GraphNode> createGenericParameter(const std::string& placeholder = "_", const std::string& constraint = "type") {
    auto node = std::make_unique<GraphNode>(GraphNodeType::GENERIC_PARAMETER);
    node->setProperty("placeholder", placeholder);
    node->setProperty("constraint", constraint);
    return node;
}

std::unique_ptr<GraphNode> createReturnValue(const std::string& type) {
    auto node = std::make_unique<GraphNode>(GraphNodeType::RETURN_VALUE);
    node->setProperty("type", type);
    return node;
}

// Test functions
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
    assert(str == "FUNCTION_DECLARATION");
    
    std::cout << "Node to_string test passed.\n";
}

int main() {
    std::cout << "Running standalone graph node tests...\n\n";
    
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
    
    std::cout << "All standalone graph node tests passed!\n";
    return 0;
}