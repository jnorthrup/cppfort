// Thin wrapper to expose MLIR region node to stage0
#pragma once

// Don't include the system MLIR header; provide a lightweight shim through
// `mlir_region_node.h` which aliases our RegionNode into the `mlir` namespace
// when the real MLIR headers are not available.

#pragma once

#include <string>
#include <vector>
#include <memory>
#include <unordered_map>
#include <optional>
#include <algorithm>

namespace cppfort {
namespace ir {
namespace mlir {

/**
 * Forward declarations for MLIR mapping stubs
 */
struct OpStub;
struct ValueStub;

/**
 * RegionNode: Evolved from GraphNode to directly map to MLIR Region/Block structure
 * 
 * This implements the "terraced field" design where regions and blocks are 
 * first-class citizens. Each RegionNode maps directly to an mlir::Region or 
 * mlir::Block in the final MLIR output.
 */
class RegionNode {
public:
    enum class RegionType {
        UNKNOWN,
        FUNCTION,      // Maps to mlir::func::FuncOp
        BLOCK,         // Maps to mlir::Block
        NAMED_REGION,  // Named region (e.g., if/else, loop body)
        CONDITIONAL,   // Conditional region (if/then/else)
        LOOP,          // Loop body region
        INITIALIZER,   // Variable initializer region
        RETURN_REGION  // Return expression region
    };
    
    struct Operation {
        std::string name;                          // Operation name (e.g., "arith.addi")
        std::vector<size_t> operand_indices;       // Indices of operands (pointers to ValueStub)
        size_t result_index;                       // Index of result value
        std::unordered_map<std::string, std::string> attributes; // Operation attributes
        
        Operation(std::string op_name = "")
            : name(std::move(op_name)), result_index(SIZE_MAX) {}
    };
    
    struct Value {
        std::string type;                          // Value type (e.g., "i32", "f64")
        std::string name;                          // Value name or identifier
        size_t defining_op;                        // Index of defining operation
        std::vector<size_t> use_ops;               // Indices of operations that use this value
        
        Value(std::string val_type = "", std::string val_name = "")
            : type(std::move(val_type)), name(std::move(val_name)), defining_op(SIZE_MAX) {}
    };
    
private:
    RegionType type_;
    std::string name_;                           // Region name (e.g., function name)
    std::vector<std::unique_ptr<RegionNode>> children_;  // Child regions/blocks
    std::vector<Operation> operations_;          // Operations within this region/block
    std::vector<Value> values_;                  // Values defined in this region
    std::vector<std::string> arguments_;         // Block arguments
    RegionNode* parent_ = nullptr;              // Parent region
    size_t nesting_level_ = 0;                  // Nesting depth
    
    // MLIR-ready metadata
    std::string mlir_dialect_;                   // Target MLIR dialect (e.g., "func", "arith")
    std::unordered_map<std::string, std::string> mlir_attributes_;  // MLIR attributes
    
    // Source location mapping
    size_t source_start_ = 0;
    size_t source_end_ = 0;
    
    // Evidence for structural validation
    double orbit_confidence_ = 0.0;
    std::vector<size_t> orbit_positions_;       // Orbit anchor positions
    
public:
    /**
     * Constructors
     */
    explicit RegionNode(RegionType type = RegionType::UNKNOWN) 
        : type_(type) {}
    
    RegionNode(RegionType type, std::string name)
        : type_(type), name_(std::move(name)) {}
    
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
        // Add operation and return its index; additionally update SSA links
        operations_.push_back(std::move(op));
        size_t idx = operations_.size() - 1;
        // Wire operand uses: ensure each referenced value records this use
        for (size_t operand_idx : operations_[idx].operand_indices) {
            if (operand_idx < values_.size()) {
                values_[operand_idx].use_ops.push_back(idx);
            }
        }
        // If the operation declares a result index which points to an existing
        // value, set the value's defining_op
        if (operations_[idx].result_index != SIZE_MAX &&
            operations_[idx].result_index < values_.size()) {
            values_[operations_[idx].result_index].defining_op = idx;
        }
        return idx;
    }
    
    size_t addValue(Value val) {
        // Add a value and wire SSA links between the value and any referenced op
        values_.push_back(std::move(val));
        size_t idx = values_.size() - 1;
        // If this value claims to be defined by an existing operation, set that
        // operation's result index to point back to this value
        if (values_[idx].defining_op != SIZE_MAX && values_[idx].defining_op < operations_.size()) {
            operations_[values_[idx].defining_op].result_index = idx;
        }
        // If this value has stated uses, for each use operation ensure the op
        // lists this value as an operand
        for (size_t use_op : values_[idx].use_ops) {
            if (use_op < operations_.size()) {
                operations_[use_op].operand_indices.push_back(idx);
            }
        }
        return idx;
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
    
    const std::vector<std::string>& getArguments() const {
        return arguments_;
    }
    
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

} // namespace mlir
} // namespace ir
} // namespace cppfort
