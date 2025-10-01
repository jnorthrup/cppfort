#include "machine.h"
#include "pattern_matcher.h"
#include <stdexcept>

namespace cppfort::ir {

// ============================================================================
// MLIR Arith Machine Implementation
// ============================================================================

void MLIRArithMachine::registerPatterns(PatternMatcher& matcher) {
    // AddNode → arith.addi
    matcher.registerPattern(
        NodeKind::ADD,
        TargetLanguage::MLIR_ARITH,
        [](Node* node) -> std::string {
            auto* add = static_cast<AddNode*>(node);
            return "arith.addi";
        },
        100
    );

    // SubNode → arith.subi
    matcher.registerPattern(
        NodeKind::SUB,
        TargetLanguage::MLIR_ARITH,
        [](Node* node) -> std::string {
            auto* sub = static_cast<SubNode*>(node);
            return "arith.subi";
        },
        100
    );

    // MulNode → arith.muli
    matcher.registerPattern(
        NodeKind::MUL,
        TargetLanguage::MLIR_ARITH,
        [](Node* node) -> std::string {
            auto* mul = static_cast<MulNode*>(node);
            return "arith.muli";
        },
        100
    );

    // DivNode → arith.divsi (signed division)
    matcher.registerPattern(
        NodeKind::DIV,
        TargetLanguage::MLIR_ARITH,
        [](Node* node) -> std::string {
            auto* div = static_cast<DivNode*>(node);
            return "arith.divsi";
        },
        100
    );

    // EQNode → arith.cmpi eq
    matcher.registerPattern(
        NodeKind::EQ,
        TargetLanguage::MLIR_ARITH,
        [](Node* node) -> std::string {
            auto* eq = static_cast<EQNode*>(node);
            return "arith.cmpi eq";
        },
        100
    );

    // LTNode → arith.cmpi slt
    matcher.registerPattern(
        NodeKind::LT,
        TargetLanguage::MLIR_ARITH,
        [](Node* node) -> std::string {
            auto* lt = static_cast<LTNode*>(node);
            return "arith.cmpi slt";
        },
        100
    );

    // ConstantNode → arith.constant
    matcher.registerPattern(
        NodeKind::CONSTANT,
        TargetLanguage::MLIR_ARITH,
        [](Node* node) -> std::string {
            auto* constant = static_cast<ConstantNode*>(node);
            return "arith.constant " + std::to_string(constant->_value);
        },
        100
    );
}

bool MLIRArithMachine::canHandle(NodeKind kind) const {
    return kind == NodeKind::ADD || kind == NodeKind::SUB ||
           kind == NodeKind::MUL || kind == NodeKind::DIV ||
           kind == NodeKind::EQ || kind == NodeKind::LT ||
           kind == NodeKind::CONSTANT;
}

// ============================================================================
// MLIR CF Machine Implementation
// ============================================================================

void MLIRCFMachine::registerPatterns(PatternMatcher& matcher) {
    // IfNode → cf.cond_br
    matcher.registerPattern(
        NodeKind::IF,
        TargetLanguage::MLIR_CF,
        [](Node* node) -> std::string {
            auto* ifNode = static_cast<IfNode*>(node);
            return "cf.cond_br";
        },
        100
    );

    // ProjNode (true branch) → cf.br
    matcher.registerPattern(
        NodeKind::PROJ,
        TargetLanguage::MLIR_CF,
        [](Node* node) -> std::string {
            auto* proj = static_cast<ProjNode*>(node);
            if (proj->idx() == 0) {
                return "cf.br true_branch";
            } else {
                return "cf.br false_branch";
            }
        },
        100,
        [](Node* node) -> bool {  // Type constraint for control flow projections
            auto* proj = static_cast<ProjNode*>(node);
            return proj->in(0) && proj->in(0)->isCFG();
        }
    );
}

bool MLIRCFMachine::canHandle(NodeKind kind) const {
    return kind == NodeKind::IF || kind == NodeKind::PROJ;
}

// ============================================================================
// MLIR SCF Machine Implementation
// ============================================================================

void MLIRSCFMachine::registerPatterns(PatternMatcher& matcher) {
    // IfNode → scf.if
    matcher.registerPattern(
        NodeKind::IF,
        TargetLanguage::MLIR_SCF,
        [](Node* node) -> std::string {
            auto* ifNode = static_cast<IfNode*>(node);
            return "scf.if";
        },
        90  // Lower priority than CF
    );

    // RegionNode → scf.execute_region
    matcher.registerPattern(
        NodeKind::REGION,
        TargetLanguage::MLIR_SCF,
        [](Node* node) -> std::string {
            auto* region = static_cast<RegionNode*>(node);
            return "scf.execute_region";
        },
        100
    );
}

bool MLIRSCFMachine::canHandle(NodeKind kind) const {
    return kind == NodeKind::IF || kind == NodeKind::REGION;
}

// ============================================================================
// MLIR MemRef Machine Implementation
// ============================================================================

void MLIRMemRefMachine::registerPatterns(PatternMatcher& matcher) {
    // LoadNode → memref.load
    matcher.registerPattern(
        NodeKind::LOAD,
        TargetLanguage::MLIR_MEMREF,
        [](Node* node) -> std::string {
            auto* load = static_cast<LoadNode*>(node);
            return "memref.load";
        },
        100
    );

    // StoreNode → memref.store
    matcher.registerPattern(
        NodeKind::STORE,
        TargetLanguage::MLIR_MEMREF,
        [](Node* node) -> std::string {
            auto* store = static_cast<StoreNode*>(node);
            return "memref.store";
        },
        100
    );

    // NewNode → memref.alloc
    matcher.registerPattern(
        NodeKind::NEW,
        TargetLanguage::MLIR_MEMREF,
        [](Node* node) -> std::string {
            auto* newNode = static_cast<NewNode*>(node);
            return "memref.alloc";
        },
        100
    );
}

bool MLIRMemRefMachine::canHandle(NodeKind kind) const {
    return kind == NodeKind::LOAD || kind == NodeKind::STORE || kind == NodeKind::NEW;
}

// ============================================================================
// MLIR Func Machine Implementation
// ============================================================================

void MLIRFuncMachine::registerPatterns(PatternMatcher& matcher) {
    // FunNode → func.func
    matcher.registerPattern(
        NodeKind::FUN,
        TargetLanguage::MLIR_FUNC,
        [](Node* node) -> std::string {
            auto* fun = static_cast<FunNode*>(node);
            return "func.func";
        },
        100
    );

    // CallNode → func.call
    matcher.registerPattern(
        NodeKind::CALL,
        TargetLanguage::MLIR_FUNC,
        [](Node* node) -> std::string {
            auto* call = static_cast<CallNode*>(node);
            return "func.call";
        },
        100
    );

    // ReturnNode → func.return
    matcher.registerPattern(
        NodeKind::RETURN,
        TargetLanguage::MLIR_FUNC,
        [](Node* node) -> std::string {
            auto* ret = static_cast<ReturnNode*>(node);
            return "func.return";
        },
        100
    );

    // ParmNode → function parameter
    matcher.registerPattern(
        NodeKind::PARM,
        TargetLanguage::MLIR_FUNC,
        [](Node* node) -> std::string {
            auto* parm = static_cast<ParmNode*>(node);
            return "func.parameter";
        },
        100
    );
}

bool MLIRFuncMachine::canHandle(NodeKind kind) const {
    return kind == NodeKind::FUN || kind == NodeKind::CALL ||
           kind == NodeKind::RETURN || kind == NodeKind::PARM;
}

// ============================================================================
// Machine Registry Implementation
// ============================================================================

MachineRegistry::MachineRegistry() {
    // Register all available machines
    registerMachine(std::make_unique<MLIRArithMachine>());
    registerMachine(std::make_unique<MLIRCFMachine>());
    registerMachine(std::make_unique<MLIRSCFMachine>());
    registerMachine(std::make_unique<MLIRMemRefMachine>());
    registerMachine(std::make_unique<MLIRFuncMachine>());
}

Machine* MachineRegistry::getMachine(const std::string& name) const {
    auto it = _machines.find(name);
    if (it == _machines.end()) {
        return nullptr;
    }
    return it->second.get();
}

void MachineRegistry::registerMachine(std::unique_ptr<Machine> machine) {
    _machines[machine->name()] = std::move(machine);
}

std::vector<std::string> MachineRegistry::availableMachines() const {
    std::vector<std::string> names;
    for (const auto& pair : _machines) {
        names.push_back(pair.first);
    }
    return names;
}

} // namespace cppfort::ir