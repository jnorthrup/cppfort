3#include "mlir_emitter.h"
#include <queue>
#include <stdexcept>

// This is a sketch implementation showing the conversion strategy
// Real implementation would include actual MLIR headers

namespace cppfort::ir {

MLIREmitter::MLIREmitter() {
    // Initialize MLIR context
    context = new mlir::MLIRContext();
    module = new mlir::ModuleOp(mlir::ModuleOp::create(mlir::UnknownLoc::get(*context)));
    builder = new mlir::OpBuilder(module->getBodyRegion());
}

MLIREmitter::~MLIREmitter() {
    delete builder;
    delete module;
    delete context;
}

 // Main entry point: convert SoN graph to MLIR
mlir::ModuleOp* MLIREmitter::emit(Node* entry) {
    if (!entry) return nullptr;

    // Chapter 19: Run instruction selection before emitting
    InstructionSelection isel({"mlir-arith", "mlir-func", "mlir-memref", "mlir-cf"});
    Node* selectedEntry = isel.selectInstructions(entry);

    // Create main function with void() signature
    auto loc = mlir::UnknownLoc::get(context);
    auto funcType = builder->getFunctionType({}, {});
    auto funcOp = builder->create<mlir::func::FuncOp>(loc, "main", funcType);

    // Set insertion point to function body
    auto* entryBlock = funcOp.addEntryBlock();
    builder->setInsertionPointToStart(entryBlock);

    // Schedule and emit the entry block
    auto* startBlock = emitControlNode(selectedEntry);

    // Process control flow graph
    std::queue<Node*> worklist;
    worklist.push(selectedEntry);

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

    // Add return operation
    builder->create<mlir::func::ReturnOp>(loc);

    return module;
}

    // Add return operation
    builder->create<mlir::func::ReturnOp>(loc);

    // Emit control flow nodes (Start, Region, If, etc.)
    mlir::Block* MLIREmitter::emitControlNode(Node* node) {
        if (auto* start = dynamic_cast<StartNode*>(node)) {
            // StartNode represents function entry - return current block
            return builder->getBlock();
        } else if (auto* region = dynamic_cast<RegionNode*>(node)) {
            return emitRegionWithPhis(region);
        } else if (auto* ifNode = dynamic_cast<IfNode*>(node)) {
            // IfNode will be handled in emit() method
            return nullptr;
        }
        return nullptr;
    }
        }
    }

    // Add return operation
    builder->create<mlir::func::ReturnOp>(loc);

    return module;
}

    return module;
}

// Critical: Convert Region+Phi pattern to MLIR block arguments
mlir::Block* MLIREmitter::emitRegionWithPhis(RegionNode* region) {
    // Create new MLIR block
    auto* block = new mlir::Block();

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
    for (size_t i = 0; i < phis.size(); ++i) {
        PhiNode* phi = phis[i];
        block->addArgument(getMLIRType(phi), getLocation(phi));
        // Map phi to block argument
        valueMap[phi] = block->getArgument(i);
    }

    // Store block mapping
    blockMap[region] = block;

    return block; 
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

    auto loc = getLocation(node);

    // Emit based on node type
    if (auto* constant = dynamic_cast<ConstantNode*>(node)) {
        if (auto* intType = dynamic_cast<TypeInteger*>(constant->_type)) {
            auto op = builder->create<mlir::arith::ConstantIntOp>(
                loc, constant->_value, 32);  // Assume 32-bit integers for now
            valueMap[node] = op.getResult();
            return op.getResult();
        }
    }
    } else if (auto* add = dynamic_cast<AddNode*>(node)) {
        auto lhs = emitNode(add->in(0));
        auto rhs = emitNode(add->in(1));
        if (lhs && rhs) {
            auto op = builder->create<mlir::arith::AddIOp>(loc, lhs, rhs);
            valueMap[node] = op.getResult();
            return op.getResult();
        }
    } else if (auto* eq = dynamic_cast<EQNode*>(node)) {
        auto lhs = emitNode(eq->in(0));
        auto rhs = emitNode(eq->in(1));
        if (lhs && rhs) {
            auto op = builder->create<mlir::arith::CmpIOp>(
                loc, mlir::arith::CmpIPredicate::eq, lhs, rhs);
            valueMap[node] = op.getResult();
            return op.getResult();
        }

        // Emit function nodes (Fun, Parm, Call, CallEnd).
        mlir::Value* MLIREmitter::emitFunctionNode(Node* node) {
            if (auto* fun = dynamic_cast<FunNode*>(node)) {
                // Function definition - create MLIR function
                // For now, return a placeholder
                return nullptr;
            } else if (auto* parm = dynamic_cast<ParmNode*>(node)) {
                // Parameter - should be mapped to function argument
                // For now, return a placeholder
                return nullptr;
            } else if (auto* call = dynamic_cast<CallNode*>(node)) {
                // Function call - create MLIR call operation
                auto loc = getLocation(node);
                auto fptr = emitNode(call->fptr());

                // Collect arguments
                std::vector<mlir::Value*> args;
                for (int i = 0; i < call->nArgs(); ++i) {
                    if (auto* arg = emitNode(call->arg(i))) {
                        args.push_back(arg);
                    }
                }

                // Create call operation (simplified - assumes void return)
                // auto callOp = builder->create<mlir::func::CallOp>(loc, fptr, args);
                // For now, return placeholder
                return nullptr;
            } else if (auto* cend = dynamic_cast<CallEndNode*>(node)) {
                // Call end - represents return from call
                // For now, return placeholder
                return nullptr;
            }

            return nullptr;
        }
    }
    // ... other node types

    return nullptr;

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

// Get MLIR type for a SoN node.
mlir::Type MLIREmitter::getMLIRType(Node* node) {
    if (!node || !node->_type) {
        return mlir::Type();
    }

    // Convert SoN types to MLIR types
    if (auto* intType = dynamic_cast<TypeInteger*>(node->_type)) {
        return builder->getIntegerType(intType->_size * 8);
    } else if (auto* boolType = dynamic_cast<TypeBool*>(node->_type)) {
        return builder->getIntegerType(1);
    } else if (auto* ptrType = dynamic_cast<TypePtr*>(node->_type)) {
        return mlir::IntegerType::get(builder->getContext(), 64); // Pointer as 64-bit int for now
    } else if (auto* funType = dynamic_cast<TypeFunPtr*>(node->_type)) {
        // Function pointer - return opaque pointer type for now
        return mlir::IntegerType::get(builder->getContext(), 64);
    }

    // Default to void type
    return mlir::Type();
}

// Get MLIR location for debugging.
mlir::Location MLIREmitter::getLocation(Node* node) {
    // For now, return unknown location
    // In a real implementation, this would map to source locations
    return mlir::UnknownLoc::get(builder->getContext());
}

} // namespace cppfort::ir