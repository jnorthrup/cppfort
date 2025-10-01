#include "instruction_selection.h"
#include "pattern_matcher.h"
#include <unordered_set>

namespace cppfort::ir {

InstructionSelection::InstructionSelection(const std::vector<std::string>& targetMachines)
    : _targetMachines(targetMachines) {
}

Node* InstructionSelection::selectInstructions(Node* start) {
    if (!start) return nullptr;

    // Create pattern matcher and register patterns for all target machines
    PatternMatcher matcher;

    for (const std::string& machineName : _targetMachines) {
        Machine* machine = _registry.getMachine(machineName);
        if (machine) {
            machine->registerPatterns(matcher);
        }
    }

    // Walk the graph and apply transformations
    std::unordered_set<Node*> visited;
    walkGraph(start, matcher);

    return start;  // Graph is transformed in-place
}

void InstructionSelection::walkGraph(Node* node, PatternMatcher& matcher) {
    if (!node) return;

    // Avoid processing the same node multiple times
    static std::unordered_set<Node*> visited;
    if (visited.count(node)) return;
    visited.insert(node);

    // Process inputs first (post-order traversal)
    for (int i = 0; i < node->nIns(); ++i) {
        walkGraph(node->in(i), matcher);
    }

    // Apply instruction selection to this node
    if (shouldTransform(node)) {
        TargetLanguage target = getTargetLanguage(node);
        const Pattern* pattern = matcher.findBestMatch(node, target);

        if (pattern) {
            // In a full implementation, this would create a new machine-specific node
            // For now, we just mark that the pattern was found
            // TODO: Actually transform the node
        }
    }

    // Process outputs
    for (Node* output : node->_outputs) {
        walkGraph(output, matcher);
    }
}

bool InstructionSelection::shouldTransform(Node* node) const {
    if (!node) return false;

    // Don't transform control flow nodes (handled separately)
    if (node->isCFG()) return false;

    // Transform computation nodes
    return node->getKind() == NodeKind::ADD ||
           node->getKind() == NodeKind::SUB ||
           node->getKind() == NodeKind::MUL ||
           node->getKind() == NodeKind::DIV ||
           node->getKind() == NodeKind::EQ ||
           node->getKind() == NodeKind::LT ||
           node->getKind() == NodeKind::CONSTANT ||
           node->getKind() == NodeKind::LOAD ||
           node->getKind() == NodeKind::STORE ||
           node->getKind() == NodeKind::FUN ||
           node->getKind() == NodeKind::CALL ||
           node->getKind() == NodeKind::RETURN ||
           node->getKind() == NodeKind::PARM;
}

TargetLanguage InstructionSelection::getTargetLanguage(Node* node) const {
    if (!node) return TargetLanguage::UNKNOWN;

    switch (node->getKind()) {
        case NodeKind::ADD:
        case NodeKind::SUB:
        case NodeKind::MUL:
        case NodeKind::DIV:
        case NodeKind::EQ:
        case NodeKind::LT:
        case NodeKind::CONSTANT:
            return TargetLanguage::MLIR_ARITH;

        case NodeKind::LOAD:
        case NodeKind::STORE:
            return TargetLanguage::MLIR_MEMREF;

        case NodeKind::FUN:
        case NodeKind::CALL:
        case NodeKind::RETURN:
        case NodeKind::PARM:
            return TargetLanguage::MLIR_FUNC;

        case NodeKind::IF:
            // Could be CF or SCF depending on context
            return TargetLanguage::MLIR_CF;

        default:
            return TargetLanguage::UNKNOWN;
    }
}

} // namespace cppfort::ir