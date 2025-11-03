#ifndef SEMANTIC_PRESERVATION_H
#define SEMANTIC_PRESERVATION_H

#include <string>
#include <unordered_map>
#include <vector>
#include <memory>
#include <set>
#include <functional>

namespace cppfort {
namespace stage0 {

// Semantic preservation system to prevent reset when adjusting semantics from flat representation

// Forward declarations
class SemanticGraphNode;
class FlatRepresentation;

// Semantic preservation flags
enum class PreservationFlag {
    NONE = 0,
    PRESERVE_STRUCTURE = 1 << 0,
    PRESERVE_RELATIONSHIPS = 1 << 1,
    PRESERVE_PROPERTIES = 1 << 2,
    PRESERVE_CONTEXT = 1 << 3,
    PRESERVE_ALL = (1 << 4) - 1
};

// Enable bitwise operations for PreservationFlag enum
inline PreservationFlag operator|(PreservationFlag a, PreservationFlag b) {
    return static_cast<PreservationFlag>(static_cast<int>(a) | static_cast<int>(b));
}

inline PreservationFlag operator&(PreservationFlag a, PreservationFlag b) {
    return static_cast<PreservationFlag>(static_cast<int>(a) & static_cast<int>(b));
}

inline PreservationFlag& operator|=(PreservationFlag& a, PreservationFlag b) {
    a = a | b;
    return a;
}

// Semantic context that preserves information during transformations
class SemanticContext {
private:
    std::unordered_map<std::string, std::string> context_data_;
    std::set<std::string> preserved_keys_;
    
public:
    SemanticContext() = default;
    
    // Copy constructor that preserves context
    SemanticContext(const SemanticContext& other) 
        : context_data_(other.context_data_)
        , preserved_keys_(other.preserved_keys_) {}
    
    // Assignment operator that preserves context
    SemanticContext& operator=(const SemanticContext& other) {
        if (this != &other) {
            context_data_ = other.context_data_;
            preserved_keys_ = other.preserved_keys_;
        }
        return *this;
    }
    
    // Add context data
    void addContext(const std::string& key, const std::string& value) {
        context_data_[key] = value;
        preserved_keys_.insert(key);
    }
    
    // Get context data
    const std::string& getContext(const std::string& key) const {
        static const std::string empty;
        auto it = context_data_.find(key);
        return (it != context_data_.end()) ? it->second : empty;
    }
    
    // Check if context key exists
    bool hasContext(const std::string& key) const {
        return context_data_.find(key) != context_data_.end();
    }
    
    // Get all preserved keys
    const std::set<std::string>& getPreservedKeys() const {
        return preserved_keys_;
    }
    
    // Merge context from another context
    void mergeContext(const SemanticContext& other) {
        for (const auto& pair : other.context_data_) {
            context_data_[pair.first] = pair.second;
            preserved_keys_.insert(pair.first);
        }
    }
};

// Enhanced GraphNode with semantic preservation capabilities
class SemanticGraphNode {
public:
    enum class NodeType {
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

private:
    NodeType type_;
    std::unordered_map<std::string, std::string> properties_;
    std::vector<std::shared_ptr<SemanticGraphNode>> children_;
    std::weak_ptr<SemanticGraphNode> parent_;
    SemanticContext semantic_context_;
    bool is_flat_representation_;
    
    // Semantic preservation metadata
    mutable bool semantic_dirty_ = false;
    mutable std::unordered_map<std::string, std::string> semantic_cache_;

public:
    // Constructor
    explicit SemanticGraphNode(NodeType type = NodeType::UNKNOWN, bool is_flat = false)
        : type_(type)
        , is_flat_representation_(is_flat) {}
    
    // Destructor
    virtual ~SemanticGraphNode() = default;
    
    // Getters and setters
    NodeType getType() const { return type_; }
    void setType(NodeType type) { 
        type_ = type; 
        markSemanticDirty();
    }
    
    const std::string& getProperty(const std::string& key) const {
        static const std::string empty;
        auto it = properties_.find(key);
        return (it != properties_.end()) ? it->second : empty;
    }
    
    void setProperty(const std::string& key, const std::string& value) {
        properties_[key] = value;
        markSemanticDirty();
    }
    
    bool hasProperty(const std::string& key) const {
        return properties_.find(key) != properties_.end();
    }
    
    const std::unordered_map<std::string, std::string>& getAllProperties() const {
        return properties_;
    }
    
    // Parent-child relationships with shared_ptr for automatic memory management
    void addChild(std::shared_ptr<SemanticGraphNode> child) {
        if (child) {
            children_.push_back(child);
            child->parent_ = shared_from_this();
            markSemanticDirty();
        }
    }
    
    void removeChild(const std::shared_ptr<SemanticGraphNode>& child) {
        if (child) {
            auto it = std::find(children_.begin(), children_.end(), child);
            if (it != children_.end()) {
                children_.erase(it);
                child->parent_.reset();
                markSemanticDirty();
            }
        }
    }
    
    const std::vector<std::shared_ptr<SemanticGraphNode>>& getChildren() const {
        return children_;
    }
    
    std::shared_ptr<SemanticGraphNode> getParent() const {
        return parent_.lock();
    }
    
    // Semantic context management
    SemanticContext& getSemanticContext() { 
        return semantic_context_; 
    }
    
    const SemanticContext& getSemanticContext() const { 
        return semantic_context_; 
    }
    
    void setSemanticContext(const SemanticContext& context) {
        semantic_context_ = context;
        markSemanticDirty();
    }
    
    // Flat representation detection
    bool isFlatRepresentation() const { 
        return is_flat_representation_; 
    }
    
    void setFlatRepresentation(bool is_flat) { 
        is_flat_representation_ = is_flat;
        markSemanticDirty();
    }
    
    // Semantic preservation methods
    void preserveSemanticState() const {
        // Cache current semantic state
        semantic_cache_.clear();
        for (const auto& pair : properties_) {
            semantic_cache_[pair.first] = pair.second;
        }
    }
    
    void restoreSemanticState() {
        // Restore cached semantic state
        properties_ = semantic_cache_;
        semantic_dirty_ = false;
    }
    
    bool isSemanticDirty() const {
        return semantic_dirty_;
    }
    
    void markSemanticDirty() const {
        semantic_dirty_ = true;
    }
    
    // Prevent reset during semantic adjustment
    class SemanticAdjustmentGuard {
    private:
        std::shared_ptr<SemanticGraphNode> node_;
        bool was_dirty_;
        
    public:
        explicit SemanticAdjustmentGuard(std::shared_ptr<SemanticGraphNode> node)
            : node_(node)
            , was_dirty_(node->isSemanticDirty()) {
            // Preserve current state before adjustment
            node_->preserveSemanticState();
        }
        
        ~SemanticAdjustmentGuard() {
            // Restore state if it wasn't dirty before
            if (!was_dirty_) {
                node_->restoreSemanticState();
            }
        }
        
        // Prevent copying
        SemanticAdjustmentGuard(const SemanticAdjustmentGuard&) = delete;
        SemanticAdjustmentGuard& operator=(const SemanticAdjustmentGuard&) = delete;
    };
    
    // Smart semantic adjustment that prevents unnecessary resets
    template<typename AdjustmentFunc>
    void adjustSemantics(AdjustmentFunc&& func, PreservationFlag flags = PreservationFlag::PRESERVE_ALL) {
        // Create guard to prevent reset
        SemanticAdjustmentGuard guard(shared_from_this());
        
        // Apply the adjustment function
        func();
        
        // Only mark dirty if semantic content actually changed
        if (flags != PreservationFlag::NONE) {
            markSemanticDirty();
        }
    }
    
    // Utility methods
    std::string toString() const;
    
    // Deep copy with semantic preservation
    std::shared_ptr<SemanticGraphNode> deepCopy(PreservationFlag flags = PreservationFlag::PRESERVE_ALL) const;
    
    // Merge with another node while preserving semantics
    void mergeWith(const std::shared_ptr<SemanticGraphNode>& other, PreservationFlag flags = PreservationFlag::PRESERVE_ALL);
    
    // Check semantic equivalence
    bool isSemanticallyEquivalent(const std::shared_ptr<SemanticGraphNode>& other) const;
};

// Enable shared_from_this for SemanticGraphNode
class SemanticGraphNode : public std::enable_shared_from_this<SemanticGraphNode> {
    // ... existing implementation ...
};

// Factory functions for creating specific node types with semantic preservation
namespace SemanticNodeFactory {
    std::shared_ptr<SemanticGraphNode> createFunctionDeclaration(const std::string& name, const std::string& returnType);
    std::shared_ptr<SemanticGraphNode> createMainFunction();
    std::shared_ptr<SemanticGraphNode> createParameter(const std::string& name, const std::string& type, const std::string& kind = "in");
    std::shared_ptr<SemanticGraphNode> createGenericParameter(const std::string& placeholder = "_", const std::string& constraint = "type");
    std::shared_ptr<SemanticGraphNode> createReturnValue(const std::string& type);
    std::shared_ptr<SemanticGraphNode> createFromFlatRepresentation(const FlatRepresentation& flat);
}

// Semantic preservation manager for complex transformations
class SemanticPreservationManager {
private:
    std::unordered_map<std::string, std::shared_ptr<SemanticGraphNode>> node_registry_;
    std::unordered_map<std::string, SemanticContext> context_registry_;
    std::set<std::string> preserved_nodes_;
    
public:
    SemanticPreservationManager() = default;
    
    // Register a node for semantic preservation
    void registerNode(const std::string& id, std::shared_ptr<SemanticGraphNode> node);
    
    // Get a preserved node
    std::shared_ptr<SemanticGraphNode> getNode(const std::string& id) const;
    
    // Preserve current state of all registered nodes
    void preserveAllStates();
    
    // Restore states of all registered nodes
    void restoreAllStates();
    
    // Mark nodes as preserved during transformation
    void markNodesPreserved(const std::set<std::string>& node_ids);
    
    // Check if nodes are preserved
    bool areNodesPreserved(const std::set<std::string>& node_ids) const;
    
    // Get preservation statistics
    size_t getPreservedNodeCount() const { return preserved_nodes_.size(); }
    size_t getTotalNodeCount() const { return node_registry_.size(); }
};

// Flat to semantic transformation with preservation
class FlatToSemanticTransformer {
private:
    SemanticPreservationManager preservation_manager_;
    PreservationFlag default_flags_;
    
public:
    explicit FlatToSemanticTransformer(PreservationFlag flags = PreservationFlag::PRESERVE_ALL)
        : default_flags_(flags) {}
    
    // Transform flat representation to semantic graph while preserving state
    std::shared_ptr<SemanticGraphNode> transform(
        const FlatRepresentation& flat,
        PreservationFlag flags = PreservationFlag::PRESERVE_ALL);
    
    // Batch transform multiple flat representations
    std::vector<std::shared_ptr<SemanticGraphNode>> transformBatch(
        const std::vector<FlatRepresentation>& flats,
        PreservationFlag flags = PreservationFlag::PRESERVE_ALL);
    
    // Incremental adjustment without full reset
    bool adjustSemanticsIncrementally(
        std::shared_ptr<SemanticGraphNode> node,
        const std::function<void()>& adjustment);
    
    // Get preservation manager
    SemanticPreservationManager& getPreservationManager() { 
        return preservation_manager_; 
    }
    
    const SemanticPreservationManager& getPreservationManager() const { 
        return preservation_manager_; 
    }
};

} // namespace stage0
} // namespace cppfort

#endif // SEMANTIC_PRESERVATION_H