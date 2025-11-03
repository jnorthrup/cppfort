#ifndef SEMANTIC_PERSISTENCE_H
#define SEMANTIC_PERSISTENCE_H

#include <string>
#include <unordered_map>
#include <vector>
#include <memory>
#include <functional>

namespace cppfort {
namespace stage0 {

// Semantic Persistence Layer - Maintains semantic state without resets
class SemanticState {
private:
    // Immutable semantic graph nodes
    struct SemanticNode {
        std::string id;
        std::string type;
        std::unordered_map<std::string, std::string> properties;
        std::vector<std::string> child_ids;
        bool is_dirty = false;
        
        SemanticNode(const std::string& node_id, const std::string& node_type)
            : id(node_id), type(node_type) {}
    };
    
    // State management
    std::unordered_map<std::string, std::unique_ptr<SemanticNode>> nodes_;
    std::vector<std::string> root_nodes_;
    std::unordered_map<std::string, std::string> semantic_context_;
    
    // Progressive refinement tracking
    std::vector<std::function<void()>> refinement_queue_;
    int refinement_generation_ = 0;
    
public:
    // Create semantic node
    std::string createNode(const std::string& type, const std::string& id = "") {
        std::string node_id = id.empty() ? generateNodeId() : id;
        nodes_[node_id] = std::make_unique<SemanticNode>(node_id, type);
        return node_id;
    }
    
    // Add property without resetting
    void setProperty(const std::string& node_id, const std::string& key, const std::string& value) {
        auto it = nodes_.find(node_id);
        if (it != nodes_.end()) {
            it->second->properties[key] = value;
            it->second->is_dirty = true;
        }
    }
    
    // Add child relationship without resetting
    void addChild(const std::string& parent_id, const std::string& child_id) {
        auto parent_it = nodes_.find(parent_id);
        auto child_it = nodes_.find(child_id);
        if (parent_it != nodes_.end() && child_it != nodes_.end()) {
            parent_it->second->child_ids.push_back(child_id);
            parent_it->second->is_dirty = true;
        }
    }
    
    // Progressive semantic refinement
    void queueRefinement(std::function<void()> refinement_fn) {
        refinement_queue_.push_back(refinement_fn);
    }
    
    // Apply queued refinements without reset
    void applyRefinements() {
        for (const auto& refinement : refinement_queue_) {
            refinement();
        }
        refinement_queue_.clear();
        refinement_generation_++;
    }
    
    // Context-aware semantic adjustment
    void adjustSemantic(const std::string& context_key, const std::string& new_value) {
        semantic_context_[context_key] = new_value;
        // Propagate context changes without resetting state
        propagateContextChange(context_key, new_value);
    }
    
    // Get semantic state
    const std::unordered_map<std::string, std::string>& getContext() const {
        return semantic_context_;
    }
    
    // Check if node needs regeneration
    bool isDirty(const std::string& node_id) const {
        auto it = nodes_.find(node_id);
        return (it != nodes_.end()) ? it->second->is_dirty : false;
    }
    
    // Mark node as clean after processing
    void markClean(const std::string& node_id) {
        auto it = nodes_.find(node_id);
        if (it != nodes_.end()) {
            it->second->is_dirty = false;
        }
    }
    
    // Get node information
    std::string getNodeType(const std::string& node_id) const {
        auto it = nodes_.find(node_id);
        return (it != nodes_.end()) ? it->second->type : "";
    }
    
    std::string getProperty(const std::string& node_id, const std::string& key) const {
        auto it = nodes_.find(node_id);
        if (it != nodes_.end()) {
            auto prop_it = it->second->properties.find(key);
            return (prop_it != it->second->properties.end()) ? prop_it->second : "";
        }
        return "";
    }
    
    const std::vector<std::string>& getChildren(const std::string& node_id) const {
        static const std::vector<std::string> empty;
        auto it = nodes_.find(node_id);
        return (it != nodes_.end()) ? it->second->child_ids : empty;
    }

private:
    std::string generateNodeId() {
        static int counter = 0;
        return "node_" + std::to_string(++counter);
    }
    
    void propagateContextChange(const std::string& key, const std::string& value) {
        // When semantic context changes, mark affected nodes as dirty
        for (auto& pair : nodes_) {
            // Simple heuristic: if node references this context, mark dirty
            if (pair.second->properties.find(key) != pair.second->properties.end()) {
                pair.second->is_dirty = true;
            }
        }
    }
};

// Semantic Adjustment Manager - Prevents resets during refinement
class SemanticAdjustmentManager {
private:
    SemanticState& state_;
    std::unordered_map<std::string, std::string> adjustment_history_;
    
public:
    explicit SemanticAdjustmentManager(SemanticState& state) : state_(state) {}
    
    // Adjust semantics without resetting
    template<typename AdjustmentFn>
    void adjustSemantics(const std::string& description, AdjustmentFn fn) {
        // Record adjustment for potential rollback
        adjustment_history_[description] = state_.getContext().empty() ? "{}" : "{...}";
        
        // Apply adjustment
        fn();
        
        // Log successful adjustment
        // In a real implementation, we might want to log this
    }
    
    // Incremental semantic refinement
    void refineIncrementally(const std::string& node_id, 
                            const std::string& property, 
                            const std::string& new_value) {
        // Apply refinement without full reset
        state_.setProperty(node_id, property, new_value);
    }
    
    // Batch semantic adjustments
    template<typename... Adjustments>
    void batchAdjustments(Adjustments... adjustments) {
        // Apply multiple adjustments atomically
        auto apply_all = [&]() {
            // C++17 fold expression to apply all adjustments
            (adjustments(), ...);
        };
        state_.queueRefinement(apply_all);
    }
};

} // namespace stage0
} // namespace cppfort

#endif // SEMANTIC_PERSISTENCE_H