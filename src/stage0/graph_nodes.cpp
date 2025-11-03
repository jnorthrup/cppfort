#include "graph_nodes.h"
#include <sstream>
#include <iomanip>

namespace cppfort {
namespace stage0 {

std::string GraphNode::toString() const {
    std::ostringstream oss;
    
    // Convert node type to string
    switch (type_) {
        case GraphNodeType::FUNCTION_DECLARATION: oss << "FUNCTION_DECLARATION"; break;
        case GraphNodeType::MAIN_FUNCTION: oss << "MAIN_FUNCTION"; break;
        case GraphNodeType::PARAMETER: oss << "PARAMETER"; break;
        case GraphNodeType::GENERIC_PARAMETER: oss << "GENERIC_PARAMETER"; break;
        case GraphNodeType::RETURN_VALUE: oss << "RETURN_VALUE"; break;
        case GraphNodeType::TYPE_DECLARATION: oss << "TYPE_DECLARATION"; break;
        case GraphNodeType::EXPRESSION: oss << "EXPRESSION"; break;
        case GraphNodeType::STATEMENT: oss << "STATEMENT"; break;
        case GraphNodeType::BLOCK: oss << "BLOCK"; break;
        case GraphNodeType::TEMPLATE_PARAMETER: oss << "TEMPLATE_PARAMETER"; break;
        case GraphNodeType::CONTRACT: oss << "CONTRACT"; break;
        case GraphNodeType::CAPTURE: oss << "CAPTURE"; break;
        default: oss << "UNKNOWN"; break;
    }
    
    // Add properties
    if (!properties_.empty()) {
        oss << " {";
        bool first = true;
        for (const auto& pair : properties_) {
            if (!first) oss << ", ";
            oss << pair.first << ": \"" << pair.second << "\"";
            first = false;
        }
        oss << "}";
    }
    
    return oss.str();
}

// Factory implementations
namespace GraphNodeFactory {
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
    
    std::unique_ptr<GraphNode> createParameter(const std::string& name, const std::string& type, const std::string& kind) {
        auto node = std::make_unique<GraphNode>(GraphNodeType::PARAMETER);
        node->setProperty("name", name);
        node->setProperty("type", type);
        node->setProperty("kind", kind);  // in, inout, out, copy, move, forward
        return node;
    }
    
    std::unique_ptr<GraphNode> createGenericParameter(const std::string& placeholder, const std::string& constraint) {
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
}

} // namespace stage0
} // namespace cppfort