//===- LatticeValue.h - SCCP Lattice Value Analysis ------------------------===//
///
/// Lattice value representation for Sparse Conditional Constant Propagation.
/// Supports TOP (unknown), Constant, BOTTOM (unreachable/conflict), and
/// range analysis for integer types.
///
//===----------------------------------------------------------------------===//

#ifndef CPPFORT_LATTICE_VALUE_H
#define CPPFORT_LATTICE_VALUE_H

#include <algorithm>
#include <optional>
#include <variant>

namespace cppfort::sccp {

/// Lattice value for SCCP analysis.
/// Represents values in the lattice: Top (unknown), Constant (known value),
/// Bottom (unreachable/conflict), IntegerRange (bounded integer), FloatSpecial.
class LatticeValue {
public:
    enum Kind {
        Top,        /// Unknown value, needs analysis
        Constant,   /// Known constant value
        Bottom,     /// Unreachable or conflicting values
        IntegerRange, /// Integer with known min/max range
        FloatSpecial /// Special float value (NaN, Infinity)
    };

private:
    Kind kind;

    // Internal storage for constant values
    std::optional<int64_t> intValue;
    std::optional<bool> boolValue;

    // Range tracking for integers
    std::optional<int64_t> min;
    std::optional<int64_t> max;

    // Private constructor for Kind-based creation
    LatticeValue(Kind kind) : kind(kind) {}

public:
    /// Get the kind of this lattice value
    Kind getKind() const { return kind; }

    /// Check if this is a constant value
    bool isConstant() const { return kind == Constant; }

    /// Check if this is Top (unknown)
    bool isTop() const { return kind == Top; }

    /// Check if this is Bottom (unreachable/conflict)
    bool isBottom() const { return kind == Bottom; }

    /// Get Top value
    static LatticeValue getTop() {
        return LatticeValue(Top);
    }

    /// Get Bottom value
    static LatticeValue getBottom() {
        return LatticeValue(Bottom);
    }

    /// Get integer range value
    static LatticeValue getIntegerRange(int64_t minVal, int64_t maxVal) {
        LatticeValue result(IntegerRange);
        result.min = minVal;
        result.max = maxVal;
        return result;
    }

    /// Get minimum value of range
    std::optional<int64_t> getMin() const {
        return min;
    }

    /// Get maximum value of range
    std::optional<int64_t> getMax() const {
        return max;
    }

    /// Get constant value from integer
    static LatticeValue getConstant(int64_t i) {
        LatticeValue result(Constant);
        result.intValue = i;
        return result;
    }

    /// Get constant value from boolean
    static LatticeValue getConstant(bool b) {
        LatticeValue result(Constant);
        result.boolValue = b;
        return result;
    }

    /// Get integer constant value if present
    std::optional<int64_t> getAsInteger() const {
        return intValue;
    }

    /// Get boolean constant value if present
    std::optional<bool> getAsBoolean() const {
        return boolValue;
    }

    /// Meet operation for lattice values
    /// Returns the greatest lower bound of two lattice values
    static LatticeValue meet(const LatticeValue& a, const LatticeValue& b) {
        // meet(anything, Bottom) = Bottom
        if (a.isBottom() || b.isBottom()) {
            return getBottom();
        }

        // meet(Top, x) = x
        if (a.isTop()) return b;
        if (b.isTop()) return a;

        // meet(IntegerRange, IntegerRange) = intersection
        if (a.kind == IntegerRange && b.kind == IntegerRange) {
            int64_t newMin = std::max(a.min.value_or(INT64_MIN), b.min.value_or(INT64_MIN));
            int64_t newMax = std::min(a.max.value_or(INT64_MAX), b.max.value_or(INT64_MAX));

            if (newMin > newMax) {
                // Disjoint ranges -> Bottom
                return getBottom();
            }
            return getIntegerRange(newMin, newMax);
        }

        // meet(IntegerRange, Constant) = Constant if in range, else Bottom
        if (a.kind == IntegerRange && b.isConstant()) {
            if (b.intValue.has_value() &&
                b.intValue >= a.min.value_or(INT64_MIN) &&
                b.intValue <= a.max.value_or(INT64_MAX)) {
                return b; // Constant is within range
            }
            return getBottom(); // Constant outside range
        }
        if (a.isConstant() && b.kind == IntegerRange) {
            return meet(b, a); // Commutative
        }

        // meet(Constant, Constant) with same value = Constant
        if (a.isConstant() && b.isConstant()) {
            // Check integer values
            if (a.intValue.has_value() && b.intValue.has_value()) {
                if (a.intValue == b.intValue) {
                    return a; // Values are the same
                }
                // Different constant values -> conflict
                return getBottom();
            }
            // Check boolean values
            if (a.boolValue.has_value() && b.boolValue.has_value()) {
                if (a.boolValue == b.boolValue) {
                    return a; // Values are the same
                }
                // Different constant values -> conflict
                return getBottom();
            }
            // If types don't match or can't compare -> conflict
            return getBottom();
        }

        // Default: conservatively return Bottom for unhandled cases
        return getBottom();
    }
};

} // namespace cppfort::sccp

#endif // CPPFORT_LATTICE_VALUE_H
