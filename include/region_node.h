// Compatibility header: re-export MLIR RegionNode under a stable name
#pragma once

#include "mlir_region_node.h"

namespace cppfort {
namespace ir {
    // Expose MLIR RegionNode as MlirRegionNode to avoid collision with stage0 CFG RegionNode.
    // Do NOT alias this to `RegionNode` to avoid colliding with CFG RegionNode definitions.
    using MlirRegionNode = mlir::RegionNode;
}
}
#pragma once
// Compatibility header: expose the MLIR RegionNode as MlirRegionNode
#include "mlir_region_node.h"

namespace cppfort {
namespace ir {
    namespace mlir {
        using MlirRegionNode = RegionNode;
    }
}
}
    
    /**
     * Type and identity
     */
    RegionType getType() const { return type_; }
    void setType(RegionType type) { type_ = type; }
    
    const std::string& getName() const { return name_; }
    void setName(const std::string& name) { name_ = name; }
    
    /**
     * Hierarchical structure - terraced field architecture
     */
    void addChild(std::unique_ptr<RegionNode> child) {
        if (child) {
            child->parent_ = this;
            child->nesting_level_ = nesting_level_ + 1;
            children_.push_back(std::move(child));
        }
    }
    
    const std::vector<std::unique_ptr<RegionNode>>& getChildren() const {
        return children_;
    }
    
    RegionNode* getChild(size_t index) const {
        return (index < children_.size()) ? children_[index].get() : nullptr;
    }
    
    RegionNode* getParent() const { return parent_; }
    size_t getNestingLevel() const { return nesting_level_; }
    
    /**
     * Operations and values - SSA form preparation
     */
    size_t addOperation(Operation op) {
        operations_.push_back(std::move(op));
        return operations_.size() - 1;
    }
    
    size_t addValue(Value val) {
        values_.push_back(std::move(val));
        return values_.size() - 1;
    }
    
    const std::vector<Operation>& getOperations() const { return operations_; }
    const std::vector<Value>& getValues() const { return values_; }
    
    // Get operation by index
    const Operation* getOperation(size_t index) const {
        return (index < operations_.size()) ? &operations_[index] : nullptr;
    }
    
    // Get value by index
    const Value* getValue(size_t index) const {
        return (index < values_.size()) ? &values_[index] : nullptr;
    }
    
    /**
     * Block arguments
     */
    void addArgument(const std::string& arg) {
        arguments_.push_back(arg);
    }
    
    // The above APIs come from mlir_region_node.h; nothing more to declare here.
    
    /**
     * MLIR mapping metadata
     */
    const std::string& getMlirDialect() const { return mlir_dialect_; }
    void setMlirDialect(const std::string& dialect) { mlir_dialect_ = dialect; }
    
    void setMlirAttribute(const std::string& key, const std::string& value) {
        mlir_attributes_[key] = value;
    }
    
    const std::unordered_map<std::string, std::string>& getMlirAttributes() const {
        return mlir_attributes_;
    }
    
    /**
     * Source location mapping
     */
    void setSourceLocation(size_t start, size_t end) {
        source_start_ = start;
        source_end_ = end;
    }
    
    size_t getSourceStart() const { return source_start_; }
    size_t getSourceEnd() const { return source_end_; }
    size_t getSourceLength() const { return source_end_ - source_start_; }
    
    /**
     * Orbit evidence for structural validation
     */
    double getOrbitConfidence() const { return orbit_confidence_; }
    void setOrbitConfidence(double confidence) { orbit_confidence_ = confidence; }
    
    void addOrbitPosition(size_t pos) {
        orbit_positions_.push_back(pos);
    }
    
    const std::vector<size_t>& getOrbitPositions() const {
        return orbit_positions_;
    }
    
    /**
     * MLIR generation helpers
     */
    // Check if this region represents a function (has func.func op)
    bool isFunctionRegion() const {
        return type_ == RegionType::FUNCTION || 
               std::any_of(operations_.begin(), operations_.end(),
                          [](const Operation& op) { 
                              return op.name == "func.func" || op.name.find("func.") == 0; 
                          });
    }
    
    // Check if this region represents a block
    bool isBlockRegion() const {
        return type_ == RegionType::BLOCK || (!arguments_.empty());
    }
    
    /**
     * Find child regions by type
     */
    std::vector<const RegionNode*> findChildrenByType(RegionType type) const {
        std::vector<const RegionNode*> results;
        for (const auto& child : children_) {
            if (child->getType() == type) {
                results.push_back(child.get());
            }
            // Recursively check grandchildren
            auto grandchildren = child->findChildrenByType(type);
            results.insert(results.end(), grandchildren.begin(), grandchildren.end());
        }
        return results;
    }
    
    /**
     * Validation
     */
    bool validate() const {
        // Basic validation rules
        if (type_ == RegionType::UNKNOWN && !children_.empty()) {
            return false; // Unknown type should not have children
        }
        
        if (isFunctionRegion() && !children_.empty()) {
            // Functions should have exactly one body block
            auto body_regions = findChildrenByType(RegionType::BLOCK);
            if (body_regions.size() != 1) {
                return false;
            }
        }
        
        return true;
    }
    
    /**
     * Debug and diagnostics
     */
    std::string toString(size_t indent = 0) const;
    void printTree(size_t indent = 0) const;
    
    /**
     * Memory estimation
     */
    size_t estimateMemoryUsage() const {
        size_t total = sizeof(RegionNode);
        total += name_.capacity();
        total += mlir_dialect_.capacity();
        
        // Estimate for children
        for (const auto& child : children_) {
            total += child->estimateMemoryUsage();
        }
        
        // Estimate for operations and values
        total += operations_.capacity() * sizeof(Operation);
        total += values_.capacity() * sizeof(Value);
        
        // Estimate for strings in attributes
        for (const auto& [k, v] : mlir_attributes_) {
            total += k.capacity() + v.capacity();
        }
        
        return total;
    }
};

/**
 * Forward declarations for stubs
 * Lightweight representation of operations and values before MLIR generation
 */
struct OpStub {
    std::string name;
    std::vector<size_t> operands;  // Indices into ValueStub list
    std::vector<size_t> results;   // Indices into ValueStub list
    std::unordered_map<std::string, std::string> attributes;
    
    OpStub(std::string op_name = "")
        : name(std::move(op_name)) {}
};

struct ValueStub {
    std::string type;
    std::string name;
    std::optional<size_t> defining_op;  // Optional defining operation index
    std::vector<size_t> uses;           // Operations that use this value
    
    ValueStub(std::string val_type = "", std::string val_name = "")
        : type(std::move(val_type)), name(std::move(val_name)) {}
};

} // namespace ir
} // namespace cppfort
