#ifndef CPPFORT_TYPE_H
#define CPPFORT_TYPE_H

#include <memory>
#include <string>
#include <limits>
#include <vector>
#include <unordered_map>

namespace cppfort::ir {

// Forward declarations
class Node;

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

// ============================================================================
// Band 4: Type System Expansion (Chapters 12-15)
// ============================================================================

/**
 * Chapter 12: Floating Point Types
 *
 * Floating point types follow similar lattice structure as integers:
 * - TOP: unknown float
 * - Constants: specific float values
 * - BOTTOM: non-constant float
 */
class TypeFloat : public Type {
public:
    enum Precision {
        F32,  // 32-bit IEEE 754 single precision
        F64   // 64-bit IEEE 754 double precision
    };

private:
    Precision _precision;
    double _value;
    bool _is_constant;

    TypeFloat(Precision prec, double val, bool is_const)
        : _precision(prec), _value(val), _is_constant(is_const) {}

public:
    /**
     * Create a constant float type.
     */
    static TypeFloat* constant(double value, Precision prec = F64);

    /**
     * Create a bottom (non-constant) float type.
     */
    static TypeFloat* bottom(Precision prec = F64);

    bool isConstant() const override { return _is_constant; }
    bool isBottom() const override { return !_is_constant; }

    double value() const { return _value; }
    Precision precision() const { return _precision; }

    std::string toString() const override {
        if (isConstant()) {
            return std::to_string(_value) + (_precision == F32 ? "f" : "d");
        }
        return _precision == F32 ? "f32⊥" : "f64⊥";
    }

    Type* meet(Type* t) override;
};

/**
 * Chapter 13: Reference Types
 *
 * References to structs/objects with nullable distinction.
 * Follows Simple's struct pointer semantics.
 */
class TypePointer : public Type {
private:
    std::string _target_name;  // Name of the struct/type being referenced
    bool _nullable;            // Can be null?
    bool _is_null;             // Is definitely null?

    TypePointer(const std::string& name, bool nullable, bool is_null)
        : _target_name(name), _nullable(nullable), _is_null(is_null) {}

public:
    /**
     * Create a nullable reference type (Struct?)
     */
    static TypePointer* nullable(const std::string& struct_name);

    /**
     * Create a non-nullable reference type (Struct)
     */
    static TypePointer* nonNullable(const std::string& struct_name);

    /**
     * Create a null constant reference
     */
    static TypePointer* nullType();

    bool isConstant() const override { return _is_null; }
    bool isNullable() const { return _nullable; }
    bool isNull() const { return _is_null; }

    const std::string& targetName() const { return _target_name; }

    std::string toString() const override {
        if (_is_null) return "null";
        return _target_name + (_nullable ? "?" : "");
    }

    Type* meet(Type* t) override;
};

/**
 * Chapter 14: Narrow Integer Types
 *
 * Sub-word integer types with sign and width.
 * Requires explicit coercion/widening operations.
 */
class TypeNarrow : public Type {
public:
    enum Width {
        I8,   // 8-bit signed
        I16,  // 16-bit signed
        I32,  // 32-bit signed
        I64,  // 64-bit signed
        U8,   // 8-bit unsigned
        U16,  // 16-bit unsigned
        U32   // 32-bit unsigned
    };

private:
    Width _width;
    long _lo;
    long _hi;

    TypeNarrow(Width w, long lo, long hi)
        : _width(w), _lo(lo), _hi(hi) {}

public:
    /**
     * Create a constant narrow integer.
     */
    static TypeNarrow* constant(long value, Width width);

    /**
     * Create a bottom narrow integer.
     */
    static TypeNarrow* bottom(Width width);

    bool isConstant() const override { return _lo == _hi; }
    bool isBottom() const override;

    long value() const { return _lo; }
    Width width() const { return _width; }
    bool isSigned() const { return _width <= I64; }

    int bitWidth() const {
        switch(_width) {
            case I8: case U8: return 8;
            case I16: case U16: return 16;
            case I32: case U32: return 32;
            case I64: return 64;
        }
        return 32;
    }

    std::string toString() const override;

    Type* meet(Type* t) override;
};

/**
 * Chapter 15: Array Types
 *
 * Fixed-size and dynamic arrays with element type.
 * Tracks length for bounds checking.
 */
class TypeArray : public Type {
private:
    Type* _element_type;  // Element type
    long _length;         // -1 for dynamic, >= 0 for fixed size
    bool _nullable;       // Can the array reference be null?

    TypeArray(Type* elem, long len, bool nullable)
        : _element_type(elem), _length(len), _nullable(nullable) {}

public:
    /**
     * Create a fixed-size array type.
     */
    static TypeArray* fixedSize(Type* element_type, long length);

    /**
     * Create a dynamic array type (length unknown at compile time).
     */
    static TypeArray* dynamic(Type* element_type);

    /**
     * Create a nullable array type.
     */
    static TypeArray* nullable(Type* element_type);

    Type* elementType() const { return _element_type; }
    long length() const { return _length; }
    bool isDynamic() const { return _length < 0; }
    bool isNullable() const { return _nullable; }

    std::string toString() const override {
        std::string result = _element_type->toString();
        if (_nullable) result += "?";
        result += "[]";
        if (!isDynamic()) {
            result += "[" + std::to_string(_length) + "]";
        }
        return result;
    }

    Type* meet(Type* t) override;
};

/**
 * Chapter 16: Field Metadata
 *
 * Represents a field within a struct, tracking:
 * - Field name
 * - Field type
 * - Whether the field is final (immutable after initialization)
 * - Initial value expression (if any)
 * - Offset within struct layout
 */
struct Field {
    std::string name;       // Field name
    Type* type;             // Field type
    bool isFinal;           // Is this field final (immutable)?
    Node* initialValue;     // Initial value expression (may be null)
    int offset;             // Byte offset in struct layout

    Field(const std::string& n, Type* t, bool final = false, Node* init = nullptr, int off = 0)
        : name(n), type(t), isFinal(final), initialValue(init), offset(off) {}
};

/**
 * Chapter 16: Struct Types
 *
 * Represents a struct type with named fields.
 * Tracks field metadata including finality, initial values, and layout.
 */
class TypeStruct : public Type {
private:
    std::string _name;                  // Struct name
    std::vector<Field> _fields;         // Ordered list of fields
    std::unordered_map<std::string, int> _fieldMap;  // Name -> index mapping
    bool _nullable;                     // Can struct reference be null?
    int _totalSize;                     // Total size in bytes

    TypeStruct(const std::string& name, bool nullable = false)
        : _name(name), _nullable(nullable), _totalSize(0) {}

public:
    /**
     * Create a new struct type.
     */
    static TypeStruct* create(const std::string& name, bool nullable = false);

    /**
     * Add a field to this struct.
     * Returns the field's index.
     */
    int addField(const std::string& name, Type* type, bool isFinal = false, Node* initVal = nullptr);

    /**
     * Lookup a field by name.
     * Returns nullptr if not found.
     */
    const Field* getField(const std::string& name) const;

    /**
     * Get field by index.
     */
    const Field& getField(int index) const { return _fields[index]; }

    /**
     * Get all fields.
     */
    const std::vector<Field>& fields() const { return _fields; }

    /**
     * Check if a field exists.
     */
    bool hasField(const std::string& name) const {
        return _fieldMap.find(name) != _fieldMap.end();
    }

    /**
     * Get the number of fields.
     */
    int fieldCount() const { return _fields.size(); }

    /**
     * Check if all required fields (final and non-nullable) have initial values.
     */
    bool isFullyInitialized() const;

    /**
     * Get struct name.
     */
    const std::string& name() const { return _name; }

    /**
     * Check if nullable.
     */
    bool isNullable() const { return _nullable; }

    /**
     * Get total size.
     */
    int totalSize() const { return _totalSize; }

    std::string toString() const override {
        return _name + (_nullable ? "?" : "");
    }

    Type* meet(Type* t) override;
};

} // namespace cppfort::ir

#endif // CPPFORT_TYPE_H
