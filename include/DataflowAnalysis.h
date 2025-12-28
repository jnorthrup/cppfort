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
};

} // namespace cppfort::sccp

#endif // CPPFORT_DATAFLOW_ANALYSIS_H
