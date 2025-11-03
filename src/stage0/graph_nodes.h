#ifndef GRAPH_NODES_H
#define GRAPH_NODES_H

#include <string>
#include <unordered_map>
#include <vector>
#include <memory>
#include <algorithm>

namespace cppfort {
namespace stage0 {

// Enum defining different types of graph nodes for Cpp2 constructs
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

// GraphNode represents a node in our semantic graph for Cpp2 constructs
class GraphNode {
private:
    GraphNodeType type_;
    std::unordered_map<std::string, std::string> properties_;
    std::vector<GraphNode*> children_;
    GraphNode* parent_;

public:
    // Constructor
    explicit GraphNode(GraphNodeType type = GraphNodeType::UNKNOWN) 
        : type_(type), parent_(nullptr) {}
    
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
            // Note: We're not setting the parent here to avoid circular references
            // In a real implementation, we might want to use weak_ptr or similar
        }
    }
    
    const std::vector<GraphNode*>& getChildren() const {
        return children_;
    }
    
    GraphNode* getParent() const {
        return parent_;
    }
    
    // Utility methods
    std::string toString() const;
};

// Factory functions for creating specific node types
namespace GraphNodeFactory {
    std::unique_ptr<GraphNode> createFunctionDeclaration(const std::string& name, const std::string& returnType);
    std::unique_ptr<GraphNode> createMainFunction();
    std::unique_ptr<GraphNode> createParameter(const std::string& name, const std::string& type, const std::string& kind = "in");
    std::unique_ptr<GraphNode> createGenericParameter(const std::string& placeholder = "_", const std::string& constraint = "type");
    std::unique_ptr<GraphNode> createReturnValue(const std::string& type);
}

} // namespace stage0
} // namespace cppfort

#endif // GRAPH_NODES_H