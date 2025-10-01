#include "mlir_emitter.h"
#include <queue>
#include <stdexcept>

// This is a sketch implementation showing the conversion strategy
// Real implementation would include actual MLIR headers

namespace cppfort::ir {

MLIREmitter::MLIREmitter() {
    // Initialize MLIR context
    // context = new mlir::MLIRContext();
    // module = mlir::ModuleOp::create(mlir::UnknownLoc::get(context));
    // builder = new mlir::OpBuilder(module->getBodyRegion());
}

MLIREmitter::~MLIREmitter() {
    // Cleanup
}

// Main entry point: convert SoN graph to MLIR
mlir::ModuleOp* MLIREmitter::emit(Node* entry) {
    // Create main function
    // auto funcOp = builder->create<mlir::func::FuncOp>(...);

    // Schedule and emit the entry block
    auto* startBlock = emitControlNode(entry);

    // Process control flow graph
    std::queue<Node*> worklist;
    worklist.push(entry);

    while (!worklist.empty()) {
        Node* ctrl = worklist.front();
        worklist.pop();

        if (visited.count(ctrl)) continue;
        visited.insert(ctrl);

        if (auto* region = dynamic_cast<RegionNode*>(ctrl)) {
            emitRegionWithPhis(region);
        } else if (auto* ifNode = dynamic_cast<IfNode*>(ctrl)) {
            // Create conditional branch
            auto* pred = emitNode(ifNode->in(1));  // Predicate

            // Find projections (true/false branches)
            ProjNode* trueProj = nullptr;
            ProjNode* falseProj = nullptr;
            for (Node* use : ifNode->_outputs) {
                if (auto* proj = dynamic_cast<ProjNode*>(use)) {
                    // proj->idx(): 0=true, 1=false
                    if (proj->idx() == 0) trueProj = proj;
                    else if (proj->idx() == 1) falseProj = proj;
                }
            }

            // Schedule operations in each branch
            if (trueProj) {
                auto trueOps = scheduleRegion(trueProj, nullptr);
                // Emit true branch block
            }
            if (falseProj) {
                auto falseOps = scheduleRegion(falseProj, nullptr);
                // Emit false branch block
            }

            // builder->create<mlir::cf::CondBranchOp>(loc, pred, trueBlock, falseBlock);
        }

        // Add successors to worklist
        for (Node* out : ctrl->_outputs) {
            if (out->isCFG()) {
                worklist.push(out);
            }
        }
    }

    return module;
}

// Critical: Convert Region+Phi pattern to MLIR block arguments
mlir::Block* MLIREmitter::emitRegionWithPhis(RegionNode* region) {
    // Create new MLIR block
    // auto* block = new mlir::Block();

    // Collect all Phi nodes controlled by this region
    std::vector<PhiNode*> phis;
    for (Node* use : region->_outputs) {
        if (auto* phi = dynamic_cast<PhiNode*>(use)) {
            if (phi->region() == region) {
                phis.push_back(phi);
            }
        }
    }

    // Add block arguments for each Phi
    for (PhiNode* phi : phis) {
        // block->addArgument(getMLIRType(phi), getLocation(phi));
        // Map phi to block argument
        // valueMap[phi] = block->getArgument(index);
    }

    // Update predecessor terminators to pass Phi values
    Node* pred1 = region->in(0);  // First predecessor
    Node* pred2 = region->in(1);  // Second predecessor

    // For pred1's terminator: add phi->in(1) as operand
    // For pred2's terminator: add phi->in(2) as operand

    return nullptr; // return block;
}

// Schedule nodes between control points
std::vector<Node*> MLIREmitter::scheduleRegion(Node* from, Node* to) {
    SoNScheduler scheduler;
    return scheduler.scheduleBlock(from, to);
}

// Emit single node as MLIR operation
mlir::Value* MLIREmitter::emitNode(Node* node) {
    // Check cache
    if (valueMap.count(node)) {
        return valueMap[node];
    }

    // Emit based on node type
    if (auto* constant = dynamic_cast<ConstantNode*>(node)) {
        // auto op = builder->create<mlir::arith::ConstantIntOp>(
        //     getLocation(node), constant->_value, 32);
        // valueMap[node] = op.getResult();
    } else if (auto* add = dynamic_cast<AddNode*>(node)) {
        // auto lhs = emitNode(add->in(0));
        // auto rhs = emitNode(add->in(1));
        // auto op = builder->create<mlir::arith::AddIOp>(
        //     getLocation(node), lhs, rhs);
        // valueMap[node] = op.getResult();
    } else if (auto* eq = dynamic_cast<EQNode*>(node)) {
        // auto lhs = emitNode(eq->in(0));
        // auto rhs = emitNode(eq->in(1));
        // auto op = builder->create<mlir::arith::CmpIOp>(
        //     getLocation(node), mlir::arith::CmpIPredicate::eq, lhs, rhs);
        // valueMap[node] = op.getResult();
    }
    // ... other node types

    return nullptr; // return valueMap[node];
}

// Scheduler implementation
std::vector<Node*> SoNScheduler::scheduleBlock(Node* entry, Node* exit) {
    schedule.clear();
    visited.clear();

    // Start from entry and follow data dependencies
    visitNode(entry);

    // Process data nodes reachable from entry
    std::queue<Node*> worklist;
    for (Node* out : entry->_outputs) {
        if (!out->isCFG()) {
            worklist.push(out);
        }
    }

    while (!worklist.empty()) {
        Node* node = worklist.front();
        worklist.pop();

        if (visited.count(node)) continue;

        // Check if all inputs are scheduled
        if (isSchedulable(node)) {
            visitNode(node);

            // Add users to worklist
            for (Node* use : node->_outputs) {
                if (!use->isCFG() && use != exit) {
                    worklist.push(use);
                }
            }
        } else {
            // Re-queue for later
            worklist.push(node);
        }
    }

    return schedule;
}

void SoNScheduler::visitNode(Node* node) {
    if (visited.count(node)) return;

    // Schedule inputs first (reverse postorder)
    scheduleInputs(node);

    // Then schedule this node
    visited.insert(node);
    schedule.push_back(node);
}

bool SoNScheduler::isSchedulable(Node* node) {
    // Check if all inputs are scheduled
    for (size_t i = 0; i < node->nIns(); i++) {
        Node* input = node->in(i);
        if (input && !input->isCFG() && !visited.count(input)) {
            return false;
        }
    }
    return true;
}

void SoNScheduler::scheduleInputs(Node* node) {
    for (size_t i = 0; i < node->nIns(); i++) {
        Node* input = node->in(i);
        if (input && !input->isCFG()) {
            visitNode(input);
        }
    }
}

} // namespace cppfort::ir