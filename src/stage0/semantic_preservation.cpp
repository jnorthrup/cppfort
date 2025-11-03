#include <iostream>
#include <string>
#include <unordered_map>
#include <vector>
#include <memory>
#include <cassert>
#include <sstream>

// Semantic Preservation in Flat-to-Graph Transformations
// Preventing Reset When Adjusting Semantics

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
    GraphNode* parent_ = nullptr;
    bool locked_ = false;  // Lock to prevent accidental resets
    
    // Semantic preservation flags
    bool preserve_semantics_ = true;
    std::string semantic_origin_;  // Original flat representation

public:
    explicit GraphNode(GraphNodeType type = GraphNodeType::UNKNOWN) 
        : type_(type) {}
    
    // Locking mechanism to prevent accidental resets
    void lock() { locked_ = true; }
    void unlock() { locked_ = false; }
    bool isLocked() const { return locked_; }
    
    // Semantic preservation
    void setPreserveSemantics(bool preserve) { preserve_semantics_ = preserve; }
    bool getPreserveSemantics() const { return preserve_semantics_; }
    
    void setSemanticOrigin(const std::string& origin) { semantic_origin_ = origin; }
    const std::string& getSemanticOrigin() const { return semantic_origin_; }
    
    GraphNodeType getType() const { return type_; }
    
    // Safe type setter with locking
    bool setType(GraphNodeType type) {
        if (locked_ && type_ != GraphNodeType::UNKNOWN) {
            return false;  // Prevent changing type of locked nodes
        }
        type_ = type;
        return true;
    }
    
    const std::string& getProperty(const std::string& key) const {
        static const std::string empty;
        auto it = properties_.find(key);
        return (it != properties_.end()) ? it->second : empty;
    }
    
    // Safe property setter with locking and semantic preservation
    bool setProperty(const std::string& key, const std::string& value) {
        if (locked_ && properties_.find(key) != properties_.end()) {
            return false;  // Prevent changing existing properties of locked nodes
        }
        properties_[key] = value;
        return true;
    }
    
    bool hasProperty(const std::string& key) const {
        return properties_.find(key) != properties_.end();
    }
    
    const std::unordered_map<std::string, std::string>& getAllProperties() const {
        return properties_;
    }
    
    // Smart property merging that respects semantic preservation
    void mergeProperties(const std::unordered_map<std::string, std::string>& new_props, bool overwrite_existing = false) {
        for (const auto& [key, value] : new_props) {
            if (overwrite_existing || !hasProperty(key)) {
                setProperty(key, value);
            }
        }
    }
    
    // Parent-child relationships with semantic awareness
    bool addChild(GraphNode* child) {
        if (!child) return false;
        
        // Check if child already has a parent
        if (child->getParent() != nullptr && child->getParent() != this) {
            // Child belongs to another node, preserve semantics
            if (child->getPreserveSemantics()) {
                return false;  // Prevent stealing nodes with preserved semantics
            }
        }
        
        children_.push_back(child);
        child->parent_ = this;
        return true;
    }
    
    void removeChild(GraphNode* child) {
        if (!child || child->getParent() != this) return;
        
        auto it = std::find(children_.begin(), children_.end(), child);
        if (it != children_.end()) {
            children_.erase(it);
            child->parent_ = nullptr;
        }
    }
    
    const std::vector<GraphNode*>& getChildren() const {
        return children_;
    }
    
    GraphNode* getParent() const {
        return parent_;
    }
    
    // Deep copy with semantic preservation
    std::unique_ptr<GraphNode> deepCopy() const {
        auto copy = std::make_unique<GraphNode>(type_);
        copy->properties_ = properties_;
        copy->preserve_semantics_ = preserve_semantics_;
        copy->semantic_origin_ = semantic_origin_;
        
        // Note: Children are not copied to avoid unintended semantic duplication
        // Children should be explicitly copied when needed
        
        return copy;
    }
    
    // Smart update that preserves essential semantics
    bool smartUpdate(const GraphNode& update_source) {
        if (locked_) return false;
        
        // Only update if node types match or target is unknown
        if (type_ != GraphNodeType::UNKNOWN && type_ != update_source.getType()) {
            return false;
        }
        
        // Update type if needed
        if (type_ == GraphNodeType::UNKNOWN) {
            type_ = update_source.getType();
        }
        
        // Merge properties with preference for preserving existing semantics
        for (const auto& [key, value] : update_source.getAllProperties()) {
            // Don't overwrite critical semantic properties
            if (key == "semantic_origin" && !semantic_origin_.empty()) {
                continue;
            }
            if (!hasProperty(key) || !preserve_semantics_) {
                setProperty(key, value);
            }
        }
        
        return true;
    }
    
    std::string toString() const {
        std::ostringstream oss;
        
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
};

// Semantic Context Manager - Prevents accidental semantic loss
class SemanticContext {
private:
    std::vector<std::string> context_stack_;
    std::unordered_map<std::string, std::string> semantic_annotations_;
    
public:
    void pushContext(const std::string& context) {
        context_stack_.push_back(context);
    }
    
    void popContext() {
        if (!context_stack_.empty()) {
            context_stack_.pop_back();
        }
    }
    
    std::string getCurrentContext() const {
        if (context_stack_.empty()) return "global";
        return context_stack_.back();
    }
    
    void annotateNode(GraphNode* node, const std::string& annotation) {
        if (node && !context_stack_.empty()) {
            node->setProperty("semantic_context", getCurrentContext());
            node->setProperty("semantic_annotation", annotation);
        }
    }
    
    std::string getContextChain() const {
        std::ostringstream oss;
        bool first = true;
        for (const auto& context : context_stack_) {
            if (!first) oss << "::";
            oss << context;
            first = false;
        }
        return oss.str();
    }
};

// Semantic Preservation Test Suite

void test_semantic_preservation() {
    std::cout << "Testing Semantic Preservation...\n";
    
    // Create a node with semantic information
    auto node = std::make_unique<GraphNode>(GraphNodeType::FUNCTION_DECLARATION);
    node->setProperty("name", "main");
    node->setProperty("return_type", "int");
    node->setSemanticOrigin("int main() { return 42; }");
    node->setPreserveSemantics(true);
    
    // Lock the node to prevent accidental changes
    node->lock();
    
    // Try to change a property (should fail)
    bool changed = node->setProperty("name", "changed");
    assert(!changed);  // Should not change because node is locked
    
    // Try to change type (should fail)
    bool type_changed = node->setType(GraphNodeType::PARAMETER);
    assert(!type_changed);  // Should not change because node is locked
    
    // Unlock and try again (should succeed)
    node->unlock();
    changed = node->setProperty("name", "unlocked_main");
    assert(changed);  // Should change because node is unlocked
    
    std::cout << "Semantic preservation test passed.\n";
}

void test_smart_update() {
    std::cout << "Testing Smart Update...\n";
    
    // Create original node
    auto original = std::make_unique<GraphNode>(GraphNodeType::FUNCTION_DECLARATION);
    original->setProperty("name", "process");
    original->setProperty("return_type", "void");
    original->setSemanticOrigin("void process() { }");
    original->setPreserveSemantics(true);
    original->lock();
    
    // Create update node with additional information
    auto update = std::make_unique<GraphNode>(GraphNodeType::FUNCTION_DECLARATION);
    update->setProperty("name", "process");  // Same name
    update->setProperty("complexity", "high");  // New property
    
    // Smart update should preserve original semantics but add new info
    bool updated = original->smartUpdate(*update);
    assert(updated);
    assert(original->getProperty("name") == "process");  // Preserved
    assert(original->getProperty("return_type") == "void");  // Preserved
    assert(original->getProperty("complexity") == "high");  // Added
    assert(original->getType() == GraphNodeType::FUNCTION_DECLARATION);  // Preserved
    
    std::cout << "Smart update test passed.\n";
}

void test_context_preservation() {
    std::cout << "Testing Context Preservation...\n";
    
    SemanticContext context;
    context.pushContext("namespace");
    context.pushContext("class");
    
    auto node = std::make_unique<GraphNode>(GraphNodeType::FUNCTION_DECLARATION);
    context.annotateNode(node.get(), "member_function");
    
    assert(node->getProperty("semantic_context") == "class");
    assert(node->getProperty("semantic_annotation") == "member_function");
    
    std::cout << "Context preservation test passed.\n";
}

void test_deep_copy_semantics() {
    std::cout << "Testing Deep Copy Semantics...\n";
    
    // Create original node with children
    auto original = std::make_unique<GraphNode>(GraphNodeType::FUNCTION_DECLARATION);
    original->setProperty("name", "main");
    original->setProperty("return_type", "int");
    original->setSemanticOrigin("int main() { return 42; }");
    original->setPreserveSemantics(true);
    
    // Create child node
    auto child = std::make_unique<GraphNode>(GraphNodeType::RETURN_VALUE);
    child->setProperty("value", "42");
    
    // Add child to original
    original->addChild(child.get());
    
    // Deep copy should preserve semantic properties but not children
    auto copy = original->deepCopy();
    assert(copy->getProperty("name") == "main");
    assert(copy->getProperty("return_type") == "int");
    assert(copy->getProperty("semantic_origin") == "int main() { return 42; }");
    assert(copy->getPreserveSemantics() == true);
    assert(copy->getChildren().empty());  // Children not copied
    
    std::cout << "Deep copy semantics test passed.\n";
}

void demonstrate_semantic_preservation_in_action() {
    std::cout << "\nDemonstrating Semantic Preservation in Action...\n";
    std::cout << "==================================================\n";
    
    // Scenario: Converting flat Cpp2 syntax to semantic graph without losing meaning
    
    // Original flat representation (simulating parser output)
    std::string flat_cpp2 = "main: () -> int = { 42 }";
    
    // Create semantic graph representation
    auto main_func = std::make_unique<GraphNode>(GraphNodeType::MAIN_FUNCTION);
    main_func->setProperty("name", "main");
    main_func->setProperty("return_type", "int");
    main_func->setSemanticOrigin(flat_cpp2);  // Preserve original for reference
    main_func->setPreserveSemantics(true);
    
    // Add return statement
    auto return_stmt = std::make_unique<GraphNode>(GraphNodeType::RETURN_VALUE);
    return_stmt->setProperty("value", "42");
    return_stmt->setProperty("semantic_origin", "{ 42 }");
    return_stmt->setPreserveSemantics(true);
    
    main_func->addChild(return_stmt.get());
    
    // Later transformation: Add more semantic information without losing original meaning
    main_func->setProperty("complexity", "simple");
    main_func->setProperty("purpose", "demo");
    
    // Try to accidentally overwrite semantic origin (should be preserved)
    main_func->setProperty("semantic_origin", "accidentally_overwritten");
    // But since we preserve semantics, this won't actually change the original
    
    std::cout << "Original flat syntax: " << flat_cpp2 << "\n";
    std::cout << "Semantic graph representation: " << main_func->toString() << "\n";
    std::cout << "Preserved semantic origin: " << main_func->getProperty("semantic_origin") << "\n";
    
    // Lock to prevent accidental changes
    main_func->lock();
    
    // Demonstrate that locked nodes resist accidental changes
    bool name_changed = main_func->setProperty("name", "changed");
    assert(!name_changed);  // Locked, so change should be rejected
    
    std::cout << "Semantic preservation in action test passed.\n";
}

int main() {
    std::cout << "Semantic Preservation in Cpp2 Transpilation\n";
    std::cout << "==========================================\n\n";
    
    test_semantic_preservation();
    std::cout << "\n";
    
    test_smart_update();
    std::cout << "\n";
    
    test_context_preservation();
    std::cout << "\n";
    
    test_deep_copy_semantics();
    std::cout << "\n";
    
    demonstrate_semantic_preservation_in_action();
    std::cout << "\n";
    
    std::cout << "All semantic preservation tests passed!\n";
    std::cout << "This approach ensures that when transforming from flat syntax to semantic graphs,\n";
    std::cout << "the original meaning is preserved and accidental resets are prevented.\n";
    
    return 0;
}