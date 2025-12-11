#pragma once

#include "mlir/IR/Operation.h"
#include "mlir/IR/Value.h"
#include "mlir/IR/Block.h"
#include "mlir/IR/Region.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/Types.h"
#include "Cpp2Dialect.h"
#include <unordered_map>
#include <unordered_set>
#include <vector>
#include <cstdint>

namespace cppfort::mlir_son {

// Unique identifier for MLIR operations in the CRDT
using OpID = uint64_t;

// Pijul-style patch for MLIR operations
struct MLIRPatch {
    OpID target;
    enum class Op {
        CreateOp,      // Create new MLIR operation
        RemoveOp,      // Remove operation
        SetOperand,    // Update operation operand
        SetAttribute,  // Update operation attribute
        SetSuccessor,  // Update block successor
        InsertRegion,  // Add region to operation
        RemoveRegion   // Remove region from operation
    } operation;

    // Patch data (operation-specific)
    std::variant<
        mlir::OperationName,           // For CreateOp
        std::pair<unsigned, OpID>,     // For SetOperand (index, value)
        std::pair<std::string, mlir::Attribute>, // For SetAttribute
        std::pair<unsigned, OpID>      // For SetSuccessor (index, block)
    > data;

    uint64_t timestamp;
    std::unordered_set<OpID> dependencies;
};

// Semantic operation node (not textual)
struct SemanticOp {
    OpID id;
    mlir::OperationName op_name;

    // Operands reference other operations by ID
    std::vector<OpID> operands;

    // Results (this op produces values used by others)
    std::vector<mlir::Type> result_types;

    // Attributes (semantic properties, not strings)
    mlir::DictionaryAttr attributes;

    // Successors for control flow
    std::vector<OpID> successors;

    // Nested regions
    std::vector<Region> regions;

    // CRDT metadata
    uint64_t timestamp;
    std::unordered_set<OpID> depends_on;

    struct Region {
        std::vector<Block> blocks;
    };

    struct Block {
        std::vector<OpID> operations;
        std::vector<OpID> arguments;  // Block arguments
    };
};

// MLIR-native CRDT graph
class MLIRSemanticGraph {
private:
    std::unordered_map<OpID, SemanticOp> operations;
    std::unordered_map<OpID, std::unordered_set<OpID>> uses; // op -> ops that use it
    uint64_t next_id = 1;
    uint64_t timestamp = 0;
    mlir::MLIRContext* ctx;

public:
    explicit MLIRSemanticGraph(mlir::MLIRContext* context) : ctx(context) {}

    // Create semantic operations (no text involved)
    OpID create_constant(mlir::Attribute value, mlir::Type type) {
        SemanticOp op;
        op.id = next_id++;
        op.op_name = mlir::OperationName("cpp2.constant", ctx);
        op.result_types = {type};

        auto attrs = mlir::NamedAttrList();
        attrs.set("value", value);
        op.attributes = attrs.getDictionary(ctx);
        op.timestamp = ++timestamp;

        MLIRPatch patch;
        patch.target = op.id;
        patch.operation = MLIRPatch::Op::CreateOp;
        patch.data = op.op_name;
        patch.timestamp = op.timestamp;

        operations[op.id] = op;
        return op.id;
    }

    OpID create_add(OpID lhs, OpID rhs, mlir::Type result_type) {
        SemanticOp op;
        op.id = next_id++;
        op.op_name = mlir::OperationName("cpp2.add", ctx);
        op.operands = {lhs, rhs};
        op.result_types = {result_type};
        op.timestamp = ++timestamp;
        op.depends_on = {lhs, rhs};

        operations[op.id] = op;
        uses[lhs].insert(op.id);
        uses[rhs].insert(op.id);

        return op.id;
    }

    OpID create_sub(OpID lhs, OpID rhs, mlir::Type result_type) {
        SemanticOp op;
        op.id = next_id++;
        op.op_name = mlir::OperationName("cpp2.sub", ctx);
        op.operands = {lhs, rhs};
        op.result_types = {result_type};
        op.timestamp = ++timestamp;
        op.depends_on = {lhs, rhs};

        operations[op.id] = op;
        uses[lhs].insert(op.id);
        uses[rhs].insert(op.id);

        return op.id;
    }

    OpID create_mul(OpID lhs, OpID rhs, mlir::Type result_type) {
        SemanticOp op;
        op.id = next_id++;
        op.op_name = mlir::OperationName("cpp2.mul", ctx);
        op.operands = {lhs, rhs};
        op.result_types = {result_type};
        op.timestamp = ++timestamp;
        op.depends_on = {lhs, rhs};

        operations[op.id] = op;
        uses[lhs].insert(op.id);
        uses[rhs].insert(op.id);

        return op.id;
    }

    OpID create_div(OpID lhs, OpID rhs, mlir::Type result_type) {
        SemanticOp op;
        op.id = next_id++;
        op.op_name = mlir::OperationName("cpp2.div", ctx);
        op.operands = {lhs, rhs};
        op.result_types = {result_type};
        op.timestamp = ++timestamp;
        op.depends_on = {lhs, rhs};

        operations[op.id] = op;
        uses[lhs].insert(op.id);
        uses[rhs].insert(op.id);

        return op.id;
    }

    OpID create_phi(const std::vector<OpID>& values, mlir::Type result_type) {
        SemanticOp op;
        op.id = next_id++;
        op.op_name = mlir::OperationName("cpp2.phi", ctx);
        op.operands = values;
        op.result_types = {result_type};
        op.timestamp = ++timestamp;

        for (OpID val : values) {
            op.depends_on.insert(val);
            uses[val].insert(op.id);
        }

        operations[op.id] = op;
        return op.id;
    }

    OpID create_if(OpID condition) {
        SemanticOp op;
        op.id = next_id++;
        op.op_name = mlir::OperationName("cpp2.if", ctx);
        op.operands = {condition};
        op.timestamp = ++timestamp;
        op.depends_on = {condition};

        // Create two regions for then/else
        op.regions = {{}, {}};

        operations[op.id] = op;
        uses[condition].insert(op.id);

        return op.id;
    }

    OpID create_region() {
        SemanticOp op;
        op.id = next_id++;
        op.op_name = mlir::OperationName("cpp2.region", ctx);
        op.timestamp = ++timestamp;

        operations[op.id] = op;
        return op.id;
    }

    OpID create_loop() {
        SemanticOp op;
        op.id = next_id++;
        op.op_name = mlir::OperationName("cpp2.loop", ctx);
        op.timestamp = ++timestamp;

        // Create region for loop body
        op.regions = {{}};

        operations[op.id] = op;
        return op.id;
    }

    OpID create_new(OpID ctrl, mlir::TypeAttr struct_type) {
        SemanticOp op;
        op.id = next_id++;
        op.op_name = mlir::OperationName("cpp2.new", ctx);
        op.operands = {ctrl};
        op.timestamp = ++timestamp;
        op.depends_on = {ctrl};

        auto attrs = mlir::NamedAttrList();
        attrs.set("structType", struct_type);
        op.attributes = attrs.getDictionary(ctx);

        // Result is pointer type
        auto struct_ty = struct_type.getValue();
        auto ptr_type = mlir::cpp2::PtrType::get(struct_ty, false);
        op.result_types = {ptr_type};

        operations[op.id] = op;
        uses[ctrl].insert(op.id);

        return op.id;
    }

    OpID create_load(OpID mem, OpID ptr, mlir::StringAttr field,
                     mlir::IntegerAttr alias_class, mlir::Type result_type) {
        SemanticOp op;
        op.id = next_id++;
        op.op_name = mlir::OperationName("cpp2.load", ctx);
        op.operands = {mem, ptr};
        op.timestamp = ++timestamp;
        op.depends_on = {mem, ptr};

        auto attrs = mlir::NamedAttrList();
        attrs.set("field", field);
        attrs.set("aliasClass", alias_class);
        op.attributes = attrs.getDictionary(ctx);

        // Load returns value and updated memory
        auto mem_type = mlir::cpp2::MemType::get(ctx, alias_class.getInt());
        op.result_types = {result_type, mem_type};

        operations[op.id] = op;
        uses[mem].insert(op.id);
        uses[ptr].insert(op.id);

        return op.id;
    }

    OpID create_store(OpID mem, OpID ptr, mlir::StringAttr field,
                      OpID value, mlir::IntegerAttr alias_class) {
        SemanticOp op;
        op.id = next_id++;
        op.op_name = mlir::OperationName("cpp2.store", ctx);
        op.operands = {mem, ptr, value};
        op.timestamp = ++timestamp;
        op.depends_on = {mem, ptr, value};

        auto attrs = mlir::NamedAttrList();
        attrs.set("field", field);
        attrs.set("aliasClass", alias_class);
        op.attributes = attrs.getDictionary(ctx);

        // Store returns updated memory
        auto mem_type = mlir::cpp2::MemType::get(ctx, alias_class.getInt());
        op.result_types = {mem_type};

        operations[op.id] = op;
        uses[mem].insert(op.id);
        uses[ptr].insert(op.id);
        uses[value].insert(op.id);

        return op.id;
    }

    OpID create_ufcs_call(mlir::StringAttr callee, const std::vector<OpID>& args,
                          const std::vector<mlir::Type>& result_types) {
        SemanticOp op;
        op.id = next_id++;
        op.op_name = mlir::OperationName("cpp2.ufcs_call", ctx);
        op.operands = args;
        op.result_types = result_types;
        op.timestamp = ++timestamp;

        auto attrs = mlir::NamedAttrList();
        attrs.set("callee", callee);
        op.attributes = attrs.getDictionary(ctx);

        for (OpID arg : args) {
            op.depends_on.insert(arg);
            uses[arg].insert(op.id);
        }

        operations[op.id] = op;
        return op.id;
    }

    OpID create_contract(OpID condition, mlir::StringAttr kind,
                         mlir::StringAttr message) {
        SemanticOp op;
        op.id = next_id++;
        op.op_name = mlir::OperationName("cpp2.contract", ctx);
        op.operands = {condition};
        op.timestamp = ++timestamp;
        op.depends_on = {condition};

        auto attrs = mlir::NamedAttrList();
        attrs.set("kind", kind);
        attrs.set("message", message);
        op.attributes = attrs.getDictionary(ctx);

        operations[op.id] = op;
        uses[condition].insert(op.id);

        return op.id;
    }

    OpID create_return(const std::vector<OpID>& operands) {
        SemanticOp op;
        op.id = next_id++;
        op.op_name = mlir::OperationName("cpp2.return", ctx);
        op.operands = operands;
        op.timestamp = ++timestamp;

        for (OpID operand : operands) {
            op.depends_on.insert(operand);
            uses[operand].insert(op.id);
        }

        operations[op.id] = op;
        return op.id;
    }

    // CRDT operations
    bool apply_patch(const MLIRPatch& patch) {
        switch (patch.operation) {
            case MLIRPatch::Op::CreateOp:
                // Already handled by create_* methods
                return true;

            case MLIRPatch::Op::RemoveOp: {
                auto it = operations.find(patch.target);
                if (it == operations.end()) return false;

                // Remove from uses
                for (OpID dep : it->second.depends_on) {
                    uses[dep].erase(patch.target);
                }

                operations.erase(it);
                return true;
            }

            case MLIRPatch::Op::SetOperand: {
                auto [index, value] = std::get<std::pair<unsigned, OpID>>(patch.data);
                auto it = operations.find(patch.target);
                if (it == operations.end()) return false;

                if (index >= it->second.operands.size()) {
                    it->second.operands.resize(index + 1);
                }

                OpID old_value = it->second.operands[index];
                it->second.operands[index] = value;

                // Update uses
                if (old_value != 0) {
                    uses[old_value].erase(patch.target);
                    it->second.depends_on.erase(old_value);
                }
                uses[value].insert(patch.target);
                it->second.depends_on.insert(value);

                return true;
            }

            case MLIRPatch::Op::SetAttribute: {
                auto [name, attr] = std::get<std::pair<std::string, mlir::Attribute>>(patch.data);
                auto it = operations.find(patch.target);
                if (it == operations.end()) return false;

                auto attrs = mlir::NamedAttrList(it->second.attributes);
                attrs.set(name, attr);
                it->second.attributes = attrs.getDictionary(ctx);

                return true;
            }

            default:
                return false;
        }
    }

    // Merge graphs with LWW resolution
    void merge(const MLIRSemanticGraph& other) {
        for (const auto& [id, other_op] : other.operations) {
            auto it = operations.find(id);
            if (it == operations.end() || it->second.timestamp < other_op.timestamp) {
                operations[id] = other_op;

                // Rebuild uses
                for (OpID dep : other_op.depends_on) {
                    uses[dep].insert(id);
                }
            }
        }
    }

    // Materialize to MLIR IR
    mlir::ModuleOp materialize(mlir::OpBuilder& builder) {
        auto module = builder.create<mlir::ModuleOp>(builder.getUnknownLoc());
        builder.setInsertionPointToEnd(module.getBody());

        std::unordered_map<OpID, mlir::Value> id_to_value;

        // Topologically sort operations by dependencies
        auto sorted = topological_sort();

        for (OpID id : sorted) {
            const SemanticOp& sem_op = operations[id];

            // Collect operand values
            llvm::SmallVector<mlir::Value> operand_values;
            for (OpID operand_id : sem_op.operands) {
                auto it = id_to_value.find(operand_id);
                if (it != id_to_value.end()) {
                    operand_values.push_back(it->second);
                }
            }

            // Create MLIR operation
            mlir::OperationState state(builder.getUnknownLoc(), sem_op.op_name);
            state.addOperands(operand_values);
            state.addTypes(sem_op.result_types);
            state.addAttributes(sem_op.attributes);

            mlir::Operation* op = builder.create(state);

            // Store result values
            if (op->getNumResults() > 0) {
                id_to_value[id] = op->getResult(0);
            }
        }

        return module;
    }

    const SemanticOp* get_operation(OpID id) const {
        auto it = operations.find(id);
        return it != operations.end() ? &it->second : nullptr;
    }

    const std::unordered_set<OpID>* get_uses(OpID id) const {
        auto it = uses.find(id);
        return it != uses.end() ? &it->second : nullptr;
    }

private:
    std::vector<OpID> topological_sort() const {
        std::vector<OpID> result;
        std::unordered_set<OpID> visited;
        std::unordered_set<OpID> in_stack;

        for (const auto& [id, op] : operations) {
            if (!visited.contains(id)) {
                dfs_topo(id, visited, in_stack, result);
            }
        }

        std::reverse(result.begin(), result.end());
        return result;
    }

    void dfs_topo(OpID id, std::unordered_set<OpID>& visited,
                  std::unordered_set<OpID>& in_stack,
                  std::vector<OpID>& result) const {
        if (in_stack.contains(id)) {
            // Cycle detected - shouldn't happen in valid IR
            return;
        }
        if (visited.contains(id)) {
            return;
        }

        in_stack.insert(id);
        visited.insert(id);

        const SemanticOp* op = get_operation(id);
        if (op) {
            for (OpID dep : op->depends_on) {
                dfs_topo(dep, visited, in_stack, result);
            }
        }

        in_stack.erase(id);
        result.push_back(id);
    }
};

} // namespace cppfort::mlir_son