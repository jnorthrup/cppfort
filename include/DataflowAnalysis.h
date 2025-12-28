//===- DataflowAnalysis.h - SCCP Dataflow Analysis Engine -------------------===//
///
/// Dataflow analysis engine for Sparse Conditional Constant Propagation.
/// Manages lattice values for program values and coordinates the worklist
/// for iterative analysis.
///
//===----------------------------------------------------------------------===//

#ifndef CPPFORT_DATAFLOW_ANALYSIS_H
#define CPPFORT_DATAFLOW_ANALYSIS_H

#include "LatticeValue.h"
#include "SCCPWorklist.h"
#include <unordered_map>
#include <cstddef>
#include <functional>
#include <vector>

namespace cppfort::sccp {

/// Dataflow analysis engine for SCCP.
///
/// Tracks lattice values for program values (represented as opaque pointers)
/// and manages the worklist for iterative dataflow analysis. Values default
/// to Top (unknown) when not explicitly set.
class DataflowAnalysis {
private:
    /// Map from program value to its lattice value
    std::unordered_map<void*, LatticeValue, VoidPtrHash> latticeValues;

    /// Worklist for tracking operations that need processing
    SCCPWorklist worklist;

    /// Set of reachable basic blocks (for control flow analysis)
    std::unordered_set<void*, VoidPtrHash> reachableBlocks;

public:
    /// Create an empty dataflow analysis
    DataflowAnalysis() = default;

    /// Get the lattice value for a program value.
    /// Returns Top if the value has not been set.
    LatticeValue getLatticeValue(void* key) const {
        auto it = latticeValues.find(key);
        if (it == latticeValues.end()) {
            return LatticeValue::getTop();
        }
        return it->second;
    }

    /// Set the lattice value for a program value.
    void setLatticeValue(void* key, const LatticeValue& value) {
        latticeValues[key] = value;
    }

    /// Update the lattice value for a program value.
    /// Returns true if the value changed, false otherwise.
    bool updateLatticeValue(void* key, const LatticeValue& value) {
        auto it = latticeValues.find(key);
        if (it == latticeValues.end()) {
            // New value - only store if not Top (Top is default)
            if (!value.isTop()) {
                latticeValues[key] = value;
                return true;
            }
            return false;
        }

        // Check if value is different
        // For simplicity, we compare the kind and stored values
        LatticeValue& current = it->second;

        // Quick check: if kinds are different, value changed
        if (current.getKind() != value.getKind()) {
            it->second = value;
            return true;
        }

        // Same kind - check stored values
        bool changed = false;

        // Check integer values
        if (current.getAsInteger().has_value() != value.getAsInteger().has_value()) {
            changed = true;
        } else if (current.getAsInteger().has_value() &&
                   current.getAsInteger().value() != value.getAsInteger().value()) {
            changed = true;
        }

        // Check boolean values
        if (!changed && current.getAsBoolean().has_value() != value.getAsBoolean().has_value()) {
            changed = true;
        } else if (!changed && current.getAsBoolean().has_value() &&
                   current.getAsBoolean().value() != value.getAsBoolean().value()) {
            changed = true;
        }

        // Check float special values
        if (!changed && current.getAsFloatSpecial().has_value() != value.getAsFloatSpecial().has_value()) {
            changed = true;
        } else if (!changed && current.getAsFloatSpecial().has_value() &&
                   current.getAsFloatSpecial().value() != value.getAsFloatSpecial().value()) {
            changed = true;
        }

        // Check range values
        if (!changed && current.getMin().has_value() != value.getMin().has_value()) {
            changed = true;
        } else if (!changed && current.getMin().has_value() &&
                   current.getMin().value() != value.getMin().value()) {
            changed = true;
        }

        if (!changed && current.getMax().has_value() != value.getMax().has_value()) {
            changed = true;
        } else if (!changed && current.getMax().has_value() &&
                   current.getMax().value() != value.getMax().value()) {
            changed = true;
        }

        if (changed) {
            it->second = value;
        }

        return changed;
    }

    /// Get the worklist for this analysis
    SCCPWorklist& getWorklist() {
        return worklist;
    }

    /// Get the number of values being tracked
    size_t getValueCount() const {
        return latticeValues.size();
    }

    /// Clear all lattice values and the worklist
    void clear() {
        latticeValues.clear();
        worklist.clear();
    }

    /// Merge phi node inputs using the meet operation.
    /// Phi nodes represent values from different control flow paths merging.
    /// This function combines all input values using the lattice meet operation.
    static LatticeValue mergePhiInputs(const std::vector<LatticeValue>& inputs) {
        if (inputs.empty()) {
            return LatticeValue::getTop();
        }

        // Check for any Top inputs (unknown paths) - Top "wins"
        for (const auto& input : inputs) {
            if (input.isTop()) {
                return LatticeValue::getTop();
            }
        }

        // Check for any Bottom inputs (unreachable) - Bottom "wins"
        for (const auto& input : inputs) {
            if (input.isBottom()) {
                return LatticeValue::getBottom();
            }
        }

        // No Top or Bottom - use standard meet for remaining values
        LatticeValue result = inputs[0];

        for (size_t i = 1; i < inputs.size(); ++i) {
            result = LatticeValue::meet(result, inputs[i]);
        }

        return result;
    }

    /// Check if a basic block is reachable
    bool isBlockReachable(void* block) const {
        return reachableBlocks.find(block) != reachableBlocks.end();
    }

    /// Mark a basic block as reachable
    void markBlockReachable(void* block) {
        reachableBlocks.insert(block);
    }

    /// Get the number of reachable blocks
    size_t getReachableBlockCount() const {
        return reachableBlocks.size();
    }

    /// Clear all reachable blocks
    void clearReachableBlocks() {
        reachableBlocks.clear();
    }

    /// Evaluate a conditional branch.
    /// If the condition is a constant true, marks trueBlock as reachable.
    /// If the condition is a constant false, marks falseBlock as reachable.
    /// If the condition is Top (unknown), neither block is marked.
    void evaluateBranch(void* condition, void* trueBlock, void* falseBlock) {
        LatticeValue condValue = getLatticeValue(condition);

        if (condValue.isConstant() && condValue.getAsBoolean().has_value()) {
            if (condValue.getAsBoolean().value()) {
                // Condition is true - mark true block reachable
                markBlockReachable(trueBlock);
            } else {
                // Condition is false - mark false block reachable
                markBlockReachable(falseBlock);
            }
        }
        // If condition is Top, we don't mark either block (sparse analysis)
    }
};

} // namespace cppfort::sccp

#endif // CPPFORT_DATAFLOW_ANALYSIS_H
