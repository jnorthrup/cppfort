// MLIR-style RegionNode definition — moved to its own header for namespacing
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

struct OpStub;
struct ValueStub;

class RegionNode {
public:
    enum class RegionType {
        UNKNOWN,
        FUNCTION,
        BLOCK,
        NAMED_REGION,
        CONDITIONAL,
        LOOP,
        INITIALIZER,
        RETURN_REGION
    };

    struct Operation { /* same as old MLIR region operation */
        std::string name;
        std::vector<size_t> operand_indices;
        size_t result_index;
        std::unordered_map<std::string, std::string> attributes;
        Operation(std::string op_name = "") : name(std::move(op_name)), result_index(SIZE_MAX) {}
    };

    struct Value {
        std::string type;
        std::string name;
        size_t defining_op;
        std::vector<size_t> use_ops;
        Value(std::string val_type = "", std::string val_name = "") : type(std::move(val_type)), name(std::move(val_name)), defining_op(SIZE_MAX) {}
    };

private:
    RegionType type_ = RegionType::UNKNOWN;
    std::string name_;
    std::vector<std::unique_ptr<RegionNode>> children_;
    std::vector<Operation> operations_;
    std::vector<Value> values_;
    std::vector<std::string> arguments_;
    RegionNode* parent_ = nullptr;
    size_t nesting_level_ = 0;
    std::string mlir_dialect_;
    std::unordered_map<std::string, std::string> mlir_attributes_;
    size_t source_start_ = 0;
    size_t source_end_ = 0;
    double orbit_confidence_ = 0.0;
    std::vector<size_t> orbit_positions_;

public:
    RegionNode() = default;
    explicit RegionNode(RegionType type) : type_(type) {}
    RegionNode(RegionType type, std::string name) : type_(type), name_(std::move(name)) {}

    // Type and identity
    RegionType getType() const { return type_; }
    void setType(RegionType t) { type_ = t; }
    const std::string& getName() const { return name_; }
    void setName(const std::string& name) { name_ = name; }

    // Hierarchical structure
    void addChild(std::unique_ptr<RegionNode> child) {
        if (child) { child->parent_ = this; child->nesting_level_ = nesting_level_ + 1; children_.push_back(std::move(child)); }
    }
    const std::vector<std::unique_ptr<RegionNode>>& getChildren() const { return children_; }
    RegionNode* getChild(size_t i) const { return (i < children_.size()) ? children_[i].get() : nullptr; }
    RegionNode* getParent() const { return parent_; }
    size_t getNestingLevel() const { return nesting_level_; }

    // Operations and values
    size_t addOperation(Operation op) { operations_.push_back(std::move(op)); return operations_.size() - 1; }
    size_t addValue(Value val) { values_.push_back(std::move(val)); return values_.size() - 1; }
    const std::vector<Operation>& getOperations() const { return operations_; }
    const std::vector<Value>& getValues() const { return values_; }
    const Operation* getOperation(size_t idx) const { return (idx < operations_.size()) ? &operations_[idx] : nullptr; }
    const Value* getValue(size_t idx) const { return (idx < values_.size()) ? &values_[idx] : nullptr; }

    void addArgument(const std::string& arg) { arguments_.push_back(arg); }
    const std::vector<std::string>& getArguments() const { return arguments_; }

    // MLIR mapping metadata
    const std::string& getMlirDialect() const { return mlir_dialect_; }
    void setMlirDialect(const std::string& d) { mlir_dialect_ = d; }
    void setMlirAttribute(const std::string& key, const std::string& value) { mlir_attributes_[key] = value; }
    const std::unordered_map<std::string, std::string>& getMlirAttributes() const { return mlir_attributes_; }

    // Source mapping
    void setSourceLocation(size_t start, size_t end) { source_start_ = start; source_end_ = end; }
    size_t getSourceStart() const { return source_start_; }
    size_t getSourceEnd() const { return source_end_; }
    size_t getSourceLength() const { return source_end_ - source_start_; }

    // Orbit evidence
    void setOrbitConfidence(double c) { orbit_confidence_ = c; }
    double getOrbitConfidence() const { return orbit_confidence_; }
    void addOrbitPosition(size_t pos) { orbit_positions_.push_back(pos); }
    const std::vector<size_t>& getOrbitPositions() const { return orbit_positions_; }

    // MLIR helpers
    bool isFunctionRegion() const {
        return type_ == RegionType::FUNCTION || std::any_of(operations_.begin(), operations_.end(), [](const Operation& op){ return op.name == "func.func" || op.name.rfind("func.", 0) == 0; });
    }
    bool isBlockRegion() const { return type_ == RegionType::BLOCK || !arguments_.empty(); }

    std::vector<const RegionNode*> findChildrenByType(RegionType t) const {
        std::vector<const RegionNode*> res;
        for (const auto& c : children_) {
            if (c->getType() == t) res.push_back(c.get());
            auto grandchildren = c->findChildrenByType(t);
            res.insert(res.end(), grandchildren.begin(), grandchildren.end());
        }
        return res;
    }

    bool validate() const {
        if (type_ == RegionType::UNKNOWN && !children_.empty()) return false;
        if (isFunctionRegion() && !children_.empty()) {
            auto body = findChildrenByType(RegionType::BLOCK);
            if (body.size() != 1) return false;
        }
        return true;
    }

    std::string toString(size_t indent = 0) const;
    void printTree(size_t indent = 0) const;
    size_t estimateMemoryUsage() const;
};

// Lightweight stubs
struct OpStub { std::string name; std::vector<size_t> operands; std::vector<size_t> results; std::unordered_map<std::string, std::string> attributes; OpStub(std::string op_name = "") : name(std::move(op_name)) {} };
struct ValueStub { std::string type; std::string name; std::optional<size_t> defining_op; std::vector<size_t> uses; ValueStub(std::string t = "", std::string n = "") : type(std::move(t)), name(std::move(n)) {} };

} // namespace mlir
} // namespace ir
} // namespace cppfort
