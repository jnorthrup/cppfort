#pragma once

#include "node.h"
#include "machine.h"
#include <memory>
#include <string>
#include <vector>

namespace cppfort::ir {

/**
 * Chapter 19: Instruction Selection Phase
 *
 * Converts ideal Sea of Nodes to machine-specific operations using pattern matching.
 * Similar to Simple compiler's instruction selection, but targets MLIR dialects
 * instead of machine instructions.
 *
 * Runs BEFORE Global Code Motion in the compilation pipeline.
 */
class InstructionSelection {
private:
    MachineRegistry _registry;
    ::std::vector<::std::string> _targetMachines;

public:
    /**
     * Create instruction selection for specific target machines.
     */
    InstructionSelection(const ::std::vector<::std::string>& targetMachines = {"mlir-arith", "mlir-func"});

    /**
     * Run instruction selection on the given start node.
     * Returns the transformed graph with machine-specific nodes.
     */
    Node* selectInstructions(Node* start);

private:
    /**
     * Walk the graph and apply instruction selection patterns.
     */
    void walkGraph(Node* node, PatternMatcher& matcher);

    /**
     * Check if a node should be transformed by instruction selection.
     */
    bool shouldTransform(Node* node) const;

    /**
     * Get the best target language for a given node.
     */
    TargetLanguage getTargetLanguage(Node* node) const;
};

} // namespace cppfort::ir