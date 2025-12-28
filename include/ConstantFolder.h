//===- ConstantFolder.h - SCCP Constant Folding ---------------------------===//
///
/// Constant folding utilities for SCCP analysis.
/// Provides operations to fold arithmetic, logical, and comparison operations
/// when their operands are constant values or ranges.
///
//===----------------------------------------------------------------------===//

#ifndef CPPFORT_CONSTANT_FOLDER_H
#define CPPFORT_CONSTANT_FOLDER_H

#include "LatticeValue.h"
#include <algorithm>

namespace cppfort::sccp {

/// Utility class for constant folding operations in SCCP analysis.
class ConstantFolder {
public:
    /// Fold addition: a + b
    /// Handles constant + constant = constant
    /// Handles range + constant = shifted range
    /// Handles range + range = combined range
    /// Handles Top + anything = Top
    static LatticeValue foldAdd(const LatticeValue& a, const LatticeValue& b) {
        // Anything + Top = Top
        if (a.isTop() || b.isTop()) {
            return LatticeValue::getTop();
        }

        // Anything + Bottom = Bottom
        if (a.isBottom() || b.isBottom()) {
            return LatticeValue::getBottom();
        }

        // Constant + Constant = Constant
        if (a.isConstant() && b.isConstant()) {
            if (a.intValue.has_value() && b.intValue.has_value()) {
                return LatticeValue::getConstant(a.intValue.value() + b.intValue.value());
            }
            // Boolean + Boolean not supported, return Bottom
            return LatticeValue::getBottom();
        }

        // Range + Constant
        if (a.kind == LatticeValue::IntegerRange && b.isConstant()) {
            if (b.intValue.has_value() && a.min.has_value() && a.max.has_value()) {
                int64_t newMin = a.min.value() + b.intValue.value();
                int64_t newMax = a.max.value() + b.intValue.value();
                // Check for overflow
                if (wouldOverflowAdd(a.min.value(), b.intValue.value()) ||
                    wouldOverflowAdd(a.max.value(), b.intValue.value())) {
                    return LatticeValue::getTop(); // Overflow -> undefined
                }
                return LatticeValue::getIntegerRange(newMin, newMax);
            }
        }
        if (a.isConstant() && b.kind == LatticeValue::IntegerRange) {
            return foldAdd(b, a); // Commutative
        }

        // Range + Range = combined range
        if (a.kind == LatticeValue::IntegerRange && b.kind == LatticeValue::IntegerRange) {
            if (a.min.has_value() && a.max.has_value() &&
                b.min.has_value() && b.max.has_value()) {
                int64_t newMin = a.min.value() + b.min.value();
                int64_t newMax = a.max.value() + b.max.value();
                // Check for overflow
                if (wouldOverflowAdd(a.min.value(), b.min.value()) ||
                    wouldOverflowAdd(a.max.value(), b.max.value())) {
                    return LatticeValue::getTop(); // Overflow -> undefined
                }
                return LatticeValue::getIntegerRange(newMin, newMax);
            }
        }

        // Default: Top for unhandled cases
        return LatticeValue::getTop();
    }

    /// Fold subtraction: a - b
    static LatticeValue foldSub(const LatticeValue& a, const LatticeValue& b) {
        // Anything - Top = Top
        if (a.isTop() || b.isTop()) {
            return LatticeValue::getTop();
        }

        // Anything - Bottom = Bottom
        if (a.isBottom() || b.isBottom()) {
            return LatticeValue::getBottom();
        }

        // Constant - Constant = Constant
        if (a.isConstant() && b.isConstant()) {
            if (a.intValue.has_value() && b.intValue.has_value()) {
                // Check for divide by zero
                if (b.intValue.value() == 0) {
                    return LatticeValue::getTop(); // Undefined behavior
                }
                return LatticeValue::getConstant(a.intValue.value() - b.intValue.value());
            }
            return LatticeValue::getBottom();
        }

        // Range - Constant
        if (a.kind == LatticeValue::IntegerRange && b.isConstant()) {
            if (b.intValue.has_value() && a.min.has_value() && a.max.has_value()) {
                int64_t newMin = a.min.value() - b.intValue.value();
                int64_t newMax = a.max.value() - b.intValue.value();
                // Check for underflow/overflow
                if (wouldOverflowSub(a.min.value(), b.intValue.value()) ||
                    wouldOverflowSub(a.max.value(), b.intValue.value())) {
                    return LatticeValue::getTop();
                }
                return LatticeValue::getIntegerRange(newMin, newMax);
            }
        }

        // Default: Top for unhandled cases
        return LatticeValue::getTop();
    }

    /// Fold multiplication: a * b
    static LatticeValue foldMul(const LatticeValue& a, const LatticeValue& b) {
        // Anything * Top = Top
        if (a.isTop() || b.isTop()) {
            return LatticeValue::getTop();
        }

        // Anything * Bottom = Bottom
        if (a.isBottom() || b.isBottom()) {
            return LatticeValue::getBottom();
        }

        // Constant * Constant = Constant
        if (a.isConstant() && b.isConstant()) {
            if (a.intValue.has_value() && b.intValue.has_value()) {
                int64_t result = a.intValue.value() * b.intValue.value();
                // Check for overflow
                if (b.intValue.value() != 0 &&
                    a.intValue.value() > INT64_MAX / b.intValue.value()) {
                    return LatticeValue::getTop(); // Overflow
                }
                return LatticeValue::getConstant(result);
            }
            return LatticeValue::getBottom();
        }

        // Default: Top for unhandled cases (range * range not yet implemented)
        return LatticeValue::getTop();
    }

    /// Fold division: a / b
    static LatticeValue foldDiv(const LatticeValue& a, const LatticeValue& b) {
        // Anything / Top = Top
        if (a.isTop() || b.isTop()) {
            return LatticeValue::getTop();
        }

        // Anything / Bottom = Bottom
        if (a.isBottom() || b.isBottom()) {
            return LatticeValue::getBottom();
        }

        // Constant / Constant = Constant
        if (a.isConstant() && b.isConstant()) {
            if (a.intValue.has_value() && b.intValue.has_value()) {
                if (b.intValue.value() == 0) {
                    return LatticeValue::getTop(); // Divide by zero -> undefined
                }
                return LatticeValue::getConstant(a.intValue.value() / b.intValue.value());
            }
            return LatticeValue::getBottom();
        }

        // Default: Top for unhandled cases
        return LatticeValue::getTop();
    }

private:
    // Check if addition would overflow
    static bool wouldOverflowAdd(int64_t a, int64_t b) {
        if (b >= 0) {
            return a > INT64_MAX - b;
        } else {
            return a < INT64_MIN - b;
        }
    }

    // Check if subtraction would underflow/overflow
    static bool wouldOverflowSub(int64_t a, int64_t b) {
        if (b >= 0) {
            return a < INT64_MIN + b;
        } else {
            return a > INT64_MAX + b;
        }
    }
};

} // namespace cppfort::sccp

#endif // CPPFORT_CONSTANT_FOLDER_H
