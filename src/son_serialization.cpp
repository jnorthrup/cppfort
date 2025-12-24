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
