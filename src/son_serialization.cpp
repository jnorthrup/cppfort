#include "Cpp2SONDialect.h"
#include "mlir_cpp2_dialect.hpp"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "llvm/ADT/DenseMap.h"

using namespace mlir;
using namespace mlir::sond;
using namespace cppfort::mlir_son;

// Serialize SON dialect MLIR to CRDT graph
CRDTGraph serializeSONDialect(mlir::ModuleOp module) {
    CRDTGraph graph;

    // Map MLIR values to CRDT node IDs (using LLVM DenseMap for MLIR types)
    llvm::DenseMap<Value, NodeID> valueToNode;

    // Timestamp for all nodes
    uint64_t timestamp = 100;

    // Walk all operations in the module
    module.walk([&](Operation* op) {
        // Skip module and function ops
        if (isa<ModuleOp>(op) || isa<func::FuncOp>(op) || isa<func::ReturnOp>(op)) {
            return;
        }

        NodeID node_id = graph.generate_id();
        Node node;
        node.id = node_id;
        node.timestamp = timestamp++;

        // Map operation types to node kinds
        if (auto constOp = dyn_cast<sond::ConstantOp>(op)) {
            node.kind = Node::Kind::Constant;

            // Extract constant value
            auto attr = constOp.getValue();
            if (auto intAttr = dyn_cast<IntegerAttr>(attr)) {
                node.value = intAttr.getInt();
            }

            // Map result value to this node
            valueToNode[constOp.getResult()] = node_id;

        } else if (auto addOp = dyn_cast<sond::AddOp>(op)) {
            node.kind = Node::Kind::Add;

            // Get input node IDs
            Value lhs = addOp.getLhs();
            Value rhs = addOp.getRhs();

            if (valueToNode.count(lhs)) {
                node.inputs.push_back(valueToNode[lhs]);
            }
            if (valueToNode.count(rhs)) {
                node.inputs.push_back(valueToNode[rhs]);
            }

            // Map result to this node
            valueToNode[addOp.getResult()] = node_id;

        } else {
            // Skip unknown operations
            return;
        }

        // Add node to graph
        Patch add_node_patch;
        add_node_patch.operation = Patch::Op::AddNode;
        add_node_patch.data = node;
        add_node_patch.target = node_id;
        graph.apply_patch(add_node_patch);

        // Create edges for inputs
        for (NodeID input_id : node.inputs) {
            Patch add_edge_patch;
            add_edge_patch.operation = Patch::Op::AddEdge;
            add_edge_patch.data = std::make_pair(input_id, node_id);
            graph.apply_patch(add_edge_patch);
        }
    });

    return graph;
}

// Deserialize CRDT graph back to SON dialect MLIR
mlir::ModuleOp deserializeSONDialect(const CRDTGraph& graph, mlir::MLIRContext* context) {
    OpBuilder builder(context);
    auto loc = builder.getUnknownLoc();
    auto module = ModuleOp::create(loc);
    builder.setInsertionPointToEnd(module.getBody());

    // Create a function to hold the operations
    auto i32Type = builder.getI32Type();
    auto funcType = builder.getFunctionType({}, {i32Type});
    auto funcOp = builder.create<func::FuncOp>(loc, "deserialized_func", funcType);

    Block* entry = funcOp.addEntryBlock();
    builder.setInsertionPointToStart(entry);

    // Map CRDT node IDs to MLIR values
    llvm::DenseMap<NodeID, Value> nodeToValue;

    // Topological sort: process nodes in dependency order
    // Collect all node IDs
    std::vector<NodeID> node_ids;
    for (const auto& [id, node] : graph.get_nodes()) {
        node_ids.push_back(id);
    }

    // Sort by timestamp (rough topological order since earlier nodes have lower timestamps)
    std::sort(node_ids.begin(), node_ids.end(), [&](NodeID a, NodeID b) {
        const Node* nodeA = graph.get_node(a);
        const Node* nodeB = graph.get_node(b);
        return nodeA->timestamp < nodeB->timestamp;
    });

    // Create MLIR operations for each node
    for (NodeID node_id : node_ids) {
        const Node* node = graph.get_node(node_id);
        if (!node) continue;

        Value result;

        switch (node->kind) {
            case Node::Kind::Constant: {
                // Extract constant value
                int64_t val = 0;
                if (std::holds_alternative<int64_t>(node->value)) {
                    val = std::get<int64_t>(node->value);
                }

                auto constOp = builder.create<sond::ConstantOp>(
                    loc, i32Type, builder.getI32IntegerAttr(val));
                result = constOp.getResult();
                break;
            }

            case Node::Kind::Add: {
                // Get operands from node inputs
                if (node->inputs.size() >= 2) {
                    Value lhs = nodeToValue[node->inputs[0]];
                    Value rhs = nodeToValue[node->inputs[1]];

                    auto addOp = builder.create<sond::AddOp>(loc, lhs, rhs);
                    result = addOp.getResult();
                }
                break;
            }

            default:
                // Skip unknown node kinds
                continue;
        }

        // Map this node ID to its MLIR value
        if (result) {
            nodeToValue[node_id] = result;
        }
    }

    // Create return operation (return the last value if any)
    if (!nodeToValue.empty()) {
        // Find the node with highest ID (likely the final result)
        NodeID max_id = 0;
        for (const auto& [id, val] : nodeToValue) {
            if (id > max_id) max_id = id;
        }

        if (nodeToValue.count(max_id)) {
            builder.create<func::ReturnOp>(loc, ValueRange{nodeToValue[max_id]});
        } else {
            // No value to return, create dummy constant
            auto dummy = builder.create<sond::ConstantOp>(
                loc, i32Type, builder.getI32IntegerAttr(0));
            builder.create<func::ReturnOp>(loc, ValueRange{dummy.getResult()});
        }
    } else {
        // Empty function, return dummy value
        auto dummy = builder.create<sond::ConstantOp>(
            loc, i32Type, builder.getI32IntegerAttr(0));
        builder.create<func::ReturnOp>(loc, ValueRange{dummy.getResult()});
    }

    return module;
}
