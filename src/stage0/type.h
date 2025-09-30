#ifndef CPPFORT_TYPE_H
#define CPPFORT_TYPE_H

#include <memory>
#include <string>

namespace cppfort::ir {

/**
 * Base class for types in the Sea of Nodes type system.
 * Following Simple compiler Chapter 2 approach.
 *
 * Types form a lattice with:
 * - TOP (⊤): value may or may not be a compile time constant
 * - Constants: all constant values in the middle
 * - BOTTOM (⊥): value is NOT a compile time constant
 */
class Type {
public:
    virtual ~Type() = default;

    /**
     * Check if this type represents a constant value.
     */
    virtual bool isConstant() const { return false; }

    /**
     * Get string representation for debugging.
     */
    virtual std::string toString() const = 0;

    /**
     * Check if this is bottom type (not constant).
     */
    virtual bool isBottom() const { return false; }

    /**
     * Check if this is top type (unknown).
     */
    virtual bool isTop() const { return false; }

    // Singleton BOTTOM instance
    static Type* BOTTOM;

    // Singleton TOP instance
    static Type* TOP;
};

/**
 * Bottom type - represents non-constant values.
 */
class TypeBottom : public Type {
public:
    bool isBottom() const override { return true; }
    std::string toString() const override { return "⊥"; }
};

/**
 * Top type - represents unknown values.
 */
class TypeTop : public Type {
public:
    bool isTop() const override { return true; }
    std::string toString() const override { return "⊤"; }
};

/**
 * Integer type in the type lattice.
 * Can represent either a constant integer value or bottom (non-constant).
 */
class TypeInteger : public Type {
private:
    const bool _is_constant;
    const long _value;  // Only valid if _is_constant is true

    TypeInteger(bool is_constant, long value)
        : _is_constant(is_constant), _value(value) {}

public:
    /**
     * Create a constant integer type with a specific value.
     */
    static TypeInteger* constant(long value);

    /**
     * Create a bottom (non-constant) integer type.
     */
    static TypeInteger* bottom();

    bool isConstant() const override { return _is_constant; }
    bool isBottom() const override { return !_is_constant; }

    /**
     * Get the constant value. Only valid if isConstant() is true.
     */
    long value() const {
        return _value;
    }

    std::string toString() const override {
        if (_is_constant) {
            return std::to_string(_value);
        } else {
            return "int⊥";
        }
    }
};

} // namespace cppfort::ir

#endif // CPPFORT_TYPE_H