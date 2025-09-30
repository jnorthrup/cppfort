#ifndef CPPFORT_TYPE_H
#define CPPFORT_TYPE_H

#include <memory>
#include <string>
#include <limits>

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
protected:
    Type() : _gen(0), _meet_cache(nullptr) {}

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

    /**
     * Compute the meet (greatest lower bound) against another type.
     */
    virtual Type* meet(Type* t) = 0;

    // Singleton BOTTOM instance
    static Type* BOTTOM;

    // Singleton TOP instance
    static Type* TOP;

    // Generation counter used for cycle detection during meet
    static int GENERATION;

protected:
    mutable int _gen;
    mutable Type* _meet_cache;
};

/**
 * Bottom type - represents non-constant values.
 */
class TypeBottom : public Type {
public:
    bool isBottom() const override { return true; }
    std::string toString() const override { return "⊥"; }
    Type* meet(Type* t) override { (void)t; return this; }
};

/**
 * Top type - represents unknown values.
 */
class TypeTop : public Type {
public:
    bool isTop() const override { return true; }
    std::string toString() const override { return "⊤"; }
    Type* meet(Type* t) override { return t ? t : this; }
};

/**
 * Integer type in the type lattice.
 * Can represent either a constant integer value or bottom (non-constant).
 */
class TypeInteger : public Type {
private:
    long _lo;
    long _hi;

    TypeInteger(long lo, long hi)
        : _lo(lo), _hi(hi) {}

public:
    /**
     * Create a constant integer type with a specific value.
     */
    static TypeInteger* constant(long value);

    /**
     * Create a bottom (non-constant) integer type.
     */
    static TypeInteger* bottom();

    /**
     * Create a boolean range type (0-1 values).
     */
    static TypeInteger* boolean();

    // Convenience: boolean constants 0/1 use same integer domain
    static TypeInteger* boolTrue() { return constant(1); }
    static TypeInteger* boolFalse() { return constant(0); }

    bool isConstant() const override { return _lo == _hi; }
    bool isBottom() const override {
        return _lo == std::numeric_limits<long>::min() &&
               _hi == std::numeric_limits<long>::max();
    }

    /**
     * Get the constant value. Only valid if isConstant() is true.
     */
    long value() const {
        return _lo;
    }

    std::string toString() const override {
        if (isConstant()) {
            return std::to_string(_lo);
        }
        if (isBottom()) {
            return "int⊥";
        }
        if (_lo == 0 && _hi == 1) {
            return "bool";
        }
        return "[" + std::to_string(_lo) + "," + std::to_string(_hi) + "]";
    }

    long lo() const { return _lo; }
    long hi() const { return _hi; }

    Type* meet(Type* t) override;
};

} // namespace cppfort::ir

#endif // CPPFORT_TYPE_H
