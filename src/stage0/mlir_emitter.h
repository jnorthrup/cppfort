#pragma once

#include "node.h"
#include "machine.h"
#include "instruction_selection.h"
#include <memory>
#include <unordered_map>

// MLIR headers for actual implementation
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Module.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/Location.h"
#include "mlir/IR/Types.h"
#include "mlir/IR/Values.h"
#include "mlir/IR/Block.h"
#include "mlir/IR/Region.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/OpDefinition.h"

// Forward declarations for MLIR types - now using actual includes above
// namespace mlir {
//     // Types now properly included
// }
}

namespace cppfort::ir {

/**
 * MLIR Emitter for Sea of Nodes IR
 *
 * Converts unscheduled Sea of Nodes graph to scheduled MLIR operations.
 * Implements scheduling algorithm to linearize the graph while preserving
 * dependencies.
 */
class MLIREmitter {
private:
    mlir::MLIRContext* context;
    mlir::ModuleOp* module;
    mlir::OpBuilder* builder;

    // Map from SoN nodes to MLIR values
    ::std::unordered_map<Node*, mlir::Value*> valueMap;

    // Map from SoN nodes to MLIR blocks (for control nodes)
    ::std::unordered_map<Node*, mlir::Block*> blockMap;

    // Track visited nodes during scheduling
    ::std::unordered_set<Node*> visited;
    ::std::unordered_set<Node*> scheduled;

public:
    MLIREmitter();
    ~MLIREmitter();

    /**
     * Convert a Sea of Nodes graph to MLIR module.
     * Entry point is typically a StartNode.
     */
    mlir::ModuleOp* emit(Node* entry);

private:
    /**
     * Schedule nodes in a region between control points.
     * Returns nodes in execution order.
     */
    ::std::vector<Node*> scheduleRegion(Node* from, Node* to);

    /**
     * Emit MLIR operation for a single node.
     */
    mlir::Value* emitNode(Node* node);

    /**
     * Emit control flow nodes (If, Region, etc).
     */
    mlir::Block* emitControlNode(Node* node);

    /**
     * Emit function nodes (Fun, Parm, Call, CallEnd).
     */
    mlir::Value* emitFunctionNode(Node* node);

    /**
     * Convert SoN Region+Phi pattern to MLIR block with arguments.
     */
    mlir::Block* emitRegionWithPhis(RegionNode* region);

    /**
     * Get MLIR type for a SoN node.
     */
    mlir::Type getMLIRType(Node* node);

    /**
     * Get MLIR location for debugging.
     */
    mlir::Location getLocation(Node* node);
};

/**
 * Scheduler for Sea of Nodes.
 * Converts unscheduled graph to linear sequence while preserving dependencies.
 */
class SoNScheduler {
private:
    ::std::unordered_set<Node*> visited;
    ::std::vector<Node*> schedule;

public:
    /**
     * Schedule nodes between two control points.
     * Uses reverse postorder traversal of data dependencies.
     */
    ::std::vector<Node*> scheduleBlock(Node* entry, Node* exit);

private:
    void visitNode(Node* node);
    bool isSchedulable(Node* node);
    void scheduleInputs(Node* node);
};

} // namespace cppfort::ir