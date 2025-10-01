#ifndef CPPFORT_TYPE_H
#define CPPFORT_TYPE_H

#include <memory>
#include <string>
#include <limits>
#include <vector>
#include <unordered_map>
#include <algorithm>

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
    virtual ::std::string toString() const = 0;

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

    /**
     * Chapter 17: Compute the GLB (Greatest Lower Bound) type for var/val inference.
     * This is similar to meet but follows specific rules:
     * - Narrow types widen to int/flt
     * - References always infer as nullable
     * - Primitives remain the same
     */
    virtual Type* glb() const;

    /**
     * Chapter 19: Emit the MLIR type representation.
     *
     * @return A string representing the MLIR type for this object.
     * Default implementation returns "i32" as a fallback.
     */
    virtual ::std::string emitMLIR() const { return "i32"; }

    // Singleton BOTTOM instance
    static Type* BOTTOM;

    // Singleton TOP instance
    static Type* TOP;

    // Singleton CONTROL instance for control flow
    // static Type* CONTROL;

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
    ::std::string toString() const override { return "⊥"; }
    Type* meet(Type* t) override { (void)t; return this; }
};

/**
 * Top type - represents unknown values.
 */
class TypeTop : public Type {
public:
    bool isTop() const override { return true; }
    ::std::string toString() const override { return "⊤"; }
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
        return _lo == ::std::numeric_limits<long>::min() &&
               _hi == ::std::numeric_limits<long>::max();
    }

    /**
     * Get the constant value. Only valid if isConstant() is true.
     */
    long value() const {
        return _lo;
    }

    ::std::string toString() const override {
        if (isConstant()) {
            return ::std::to_string(_lo);
        }
        if (isBottom()) {
            return "int⊥";
        }
        if (_lo == 0 && _hi == 1) {
            return "bool";
        }
        return "[" + ::std::to_string(_lo) + "," + ::std::to_string(_hi) + "]";
    }

    /**
     * Chapter 19: MLIR type emission for integers.
     * - Constants use specific bitwidth
     * - Bottom type defaults to i32
     * - Booleans use i1
     */
    ::std::string emitMLIR() const override {
        if (isBottom()) return "i32";
        if (_lo == 0 && _hi == 1) return "i1";

        // Otherwise, choose appropriate bit width
        long absMax = ::std::max(::std::abs(_lo), ::std::abs(_hi));
        if (absMax <= 127) return "i8";
        if (absMax <= 32767) return "i16";
        return "i32";
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

    ::std::string toString() const override {
        if (isConstant()) {
            return ::std::to_string(_value) + (_precision == F32 ? "f" : "d");
        }
        return _precision == F32 ? "f32⊥" : "f64⊥";
    }

    /**
     * Chapter 19: MLIR type emission for float.
     * - Maps float types to MLIR corresponding types
     * - Bottom type defaults to f64 (double precision)
     */
    ::std::string emitMLIR() const override {
        if (isBottom()) return _precision == F32 ? "f32" : "f64";
        return _precision == F32 ? "f32" : "f64";
    }

    Type* meet(Type* t) override;
};

/**
 * Chapter 13-17: Reference Types
 *
 * References to structs/objects with nullable distinction.
 * Follows Simple's struct pointer semantics.
 * Chapter 17: Adds mutability tracking for deep immutability.
 */
class TypePointer : public Type {
private:
    ::std::string _target_name;  // Name of the struct/type being referenced
    bool _nullable;            // Can be null?
    bool _is_null;             // Is definitely null?
    bool _mutable;             // Chapter 17: Is this reference mutable?

    TypePointer(const ::std::string& name, bool nullable, bool is_null, bool mutable_ref = true)
        : _target_name(name), _nullable(nullable), _is_null(is_null), _mutable(mutable_ref) {}

public:
    /**
     * Create a nullable reference type (Struct?)
     */
    static TypePointer* nullable(const ::std::string& struct_name, bool mutable_ref = true);

    /**
     * Create a non-nullable reference type (Struct)
     */
    static TypePointer* nonNullable(const ::std::string& struct_name, bool mutable_ref = true);

    /**
     * Create a null constant reference
     */
    static TypePointer* nullType();

    /**
     * Chapter 17: Create an immutable reference (val semantics)
     */
    static TypePointer* immutable(const ::std::string& struct_name, bool nullable = false);

    /**
     * Chapter 17: Create a mutable reference (var or ! semantics)
     */
    static TypePointer* mutable_(const ::std::string& struct_name, bool nullable = false);

    bool isConstant() const override { return _is_null; }
    bool isNullable() const { return _nullable; }
    bool isNull() const { return _is_null; }
    bool isMutable() const { return _mutable; }

    const ::std::string& targetName() const { return _target_name; }

    ::std::string toString() const override {
        if (_is_null) return "null";
        ::std::string result;
        if (!_mutable) result = "val ";
        result += _target_name + (_nullable ? "?" : "");
        return result;
    }

    /**
     * Chapter 19: MLIR type emission for pointers.
     * - Emit as memref type for structs and arrays
     * - Nullable types get wrapped in optional
     */
    ::std::string emitMLIR() const override {
        ::std::string baseType = "memref<" + _target_name + ">";
        if (_nullable) return "!llvm.ptr<" + baseType + ">";
        return baseType;
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

    ::std::string toString() const override;

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

    ::std::string toString() const override {
        ::std::string result = _element_type->toString();
        if (_nullable) result += "?";
        result += "[]";
        if (!isDynamic()) {
            result += "[" + ::std::to_string(_length) + "]";
        }
        return result;
    }

    Type* meet(Type* t) override;
};

/**
 * Chapter 16-17: Field Metadata
 *
 * Represents a field within a struct, tracking:
 * - Field name
 * - Field type
 * - Whether the field is final (immutable after initialization)
 * - Initial value expression (if any)
 * - Offset within struct layout
 * - Mutability qualifier (Chapter 17)
 */
struct Field {
    enum MutabilityQualifier {
        MUTABLE,       // Default for primitives, or explicitly marked with '!'
        IMMUTABLE,     // References with initializers, or marked with 'val'
        VAR_INFERRED,  // Marked with 'var' keyword (always mutable)
        VAL_INFERRED   // Marked with 'val' keyword (always immutable)
    };

    ::std::string name;                // Field name
    Type* type;                      // Field type
    bool isFinal;                    // Is this field final (immutable)?
    Node* initialValue;              // Initial value expression (may be null)
    int offset;                      // Byte offset in struct layout
    MutabilityQualifier mutability;  // Chapter 17: Mutability qualifier

    Field(const ::std::string& n, Type* t, bool final = false, Node* init = nullptr, int off = 0,
          MutabilityQualifier mut = MUTABLE)
        : name(n), type(t), isFinal(final), initialValue(init), offset(off), mutability(mut) {}

    /**
     * Check if this field is mutable through a given reference.
     * Chapter 17: Deep immutability rules.
     */
    bool isMutableThrough(bool refIsMutable) const {
        // If the reference itself is immutable, field cannot be mutated
        if (!refIsMutable) return false;

        // VAR is always mutable if reference allows
        if (mutability == VAR_INFERRED) return true;

        // VAL is always immutable
        if (mutability == VAL_INFERRED) return false;

        // IMMUTABLE fields cannot be mutated
        if (mutability == IMMUTABLE) return false;

        // MUTABLE fields can be mutated if reference allows
        return mutability == MUTABLE;
    }
};

/**
 * Chapter 16: Struct Types
 *
 * Represents a struct type with named fields.
 * Tracks field metadata including finality, initial values, and layout.
 */
class TypeStruct : public Type {
private:
    ::std::string _name;                  // Struct name
    ::std::vector<Field> _fields;         // Ordered list of fields
    ::std::unordered_map<::std::string, int> _fieldMap;  // Name -> index mapping
    bool _nullable;                     // Can struct reference be null?
    int _totalSize;                     // Total size in bytes

    TypeStruct(const ::std::string& name, bool nullable = false)
        : _name(name), _nullable(nullable), _totalSize(0) {}

public:
    /**
     * Create a new struct type.
     */
    static TypeStruct* create(const ::std::string& name, bool nullable = false);

    /**
     * Add a field to this struct.
     * Returns the field's index.
     */
    int addField(const ::std::string& name, Type* type, bool isFinal = false, Node* initVal = nullptr,
                 Field::MutabilityQualifier mutability = Field::MUTABLE);

    /**
     * Lookup a field by name.
     * Returns nullptr if not found.
     */
    const Field* getField(const ::std::string& name) const;

    /**
     * Get field by index.
     */
    const Field& getField(int index) const { return _fields[index]; }

    /**
     * Get all fields.
     */
    const ::std::vector<Field>& fields() const { return _fields; }

    /**
     * Check if a field exists.
     */
    bool hasField(const ::std::string& name) const {
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
    const ::std::string& name() const { return _name; }

    /**
     * Check if nullable.
     */
    bool isNullable() const { return _nullable; }

    /**
     * Get total size.
     */
    int totalSize() const { return _totalSize; }

    ::std::string toString() const override {
        return _name + (_nullable ? "?" : "");
    }

    /**
     * Chapter 19: MLIR type emission for structs.
     * - Emit struct name as memref type
     * - Always use memref for consistency with other types
     * - Track nullable and field information
     */
    ::std::string emitMLIR() const override {
        if (_nullable) return "!llvm.ptr<memref<" + _name + ">>";
        return "memref<" + _name + ">";
    }

    Type* meet(Type* t) override;
};

/**
 * Chapter 18: Tuple Types
 *
 * Represents a tuple of types used for function arguments.
 * Tuples are immutable ordered collections of types.
 */
class TypeTuple : public Type {
private:
    ::std::vector<Type*> _types;

    TypeTuple(const ::std::vector<Type*>& types) : _types(types) {}

public:
    /**
     * Create a tuple type from a list of types.
     */
    static TypeTuple* create(const ::std::vector<Type*>& types);

    /**
     * Get the types in this tuple.
     */
    const ::std::vector<Type*>& types() const { return _types; }

    /**
     * Get the number of types in this tuple.
     */
    size_t size() const { return _types.size(); }

    /**
     * Get a specific type from the tuple.
     */
    Type* get(size_t index) const {
        return index < _types.size() ? _types[index] : nullptr;
    }

    ::std::string toString() const override {
        ::std::string result = "(";
        for (size_t i = 0; i < _types.size(); ++i) {
            if (i > 0) result += ",";
            result += _types[i]->toString();
        }
        result += ")";
        return result;
    }

    bool isConstant() const override {
        // A tuple is constant if all its elements are constant
        for (Type* t : _types) {
            if (!t->isConstant()) return false;
        }
        return true;
    }

    /**
     * Chapter 19: MLIR type emission for tuples.
     * - Translate type tuple to MLIR tuple/function type
     * - Empty tuples map to unit type
     * - Single element tuples map to their contained type
     * - Composite tuples use MLIR tuple type: tuple<...>
     */
    ::std::string emitMLIR() const override {
        if (_types.empty()) return "()";  // Unit type
        if (_types.size() == 1) return _types[0]->emitMLIR();

        // Create a tuple type by emitting each type
        ::std::string result = "tuple<";
        for (size_t i = 0; i < _types.size(); ++i) {
            if (i > 0) result += ", ";
            result += _types[i]->emitMLIR();
        }
        result += ">";
        return result;
    }

    Type* meet(Type* t) override;
};

/**
 * Chapter 18: Function Pointer Types
 *
 * Represents a function pointer type with signature.
 * Tracks function index (fidx) for code generation.
 * Can track multiple functions with same signature via bitset.
 *
 * Mutability: Function pointers themselves can be var/val
 */
class TypeFunPtr : public Type {
private:
    TypeTuple* _args;       // Argument types
    Type* _ret;             // Return type
    int _fidx;              // Function index (-1 for multiple/unknown)
    bool _nullable;         // Can be null?
    bool _mutable;          // Is this reference mutable?
    ::std::vector<int> _fidxs; // Bit set of possible function indices

    TypeFunPtr(TypeTuple* args, Type* ret, int fidx = -1, bool nullable = false, bool mutable_ref = true)
        : _args(args), _ret(ret), _fidx(fidx), _nullable(nullable), _mutable(mutable_ref) {
        if (fidx >= 0) _fidxs.push_back(fidx);
    }

public:
    /**
     * Create a function pointer type.
     */
    static TypeFunPtr* create(TypeTuple* args, Type* ret, int fidx = -1);

    /**
     * Create a nullable function pointer type.
     */
    static TypeFunPtr* nullable(TypeTuple* args, Type* ret);

    /**
     * Create a mutable function pointer type.
     */
    static TypeFunPtr* mutable_(TypeTuple* args, Type* ret, bool nullable = false);

    /**
     * Create an immutable function pointer type.
     */
    static TypeFunPtr* immutable(TypeTuple* args, Type* ret, bool nullable = false);

    /**
     * Create a null function pointer constant.
     */
    static TypeFunPtr* nullType();

    TypeTuple* args() const { return _args; }
    Type* ret() const { return _ret; }
    int fidx() const { return _fidx; }
    bool isNullable() const { return _nullable; }
    bool isMutable() const { return _mutable; }
    const ::std::vector<int>& fidxs() const { return _fidxs; }

    /**
     * Add a function index to the set.
     */
    void addFidx(int fidx) {
        if (::std::find(_fidxs.begin(), _fidxs.end(), fidx) == _fidxs.end()) {
            _fidxs.push_back(fidx);
        }
    }

    bool isConstant() const override {
        // A function pointer is constant if it's null or refers to a single known function
        return _fidx >= 0 || (_nullable && _fidxs.empty());
    }

    ::std::string toString() const override {
        if (_fidx < 0 && _fidxs.empty() && _nullable) return "null";

        ::std::string result;
        if (!_mutable) result = "val ";
        result += "{" + _args->toString() + "->" + _ret->toString() + "}";
        if (_nullable) result += "?";
        if (_fidx >= 0) result += "[fidx=" + ::std::to_string(_fidx) + "]";
        return result;
    }

    /**
     * Chapter 19: MLIR type emission for function pointers.
     * - Emit as function type with argument and return type
     * - Nullable types wrapped in !llvm.ptr
     * - Map argument and return types to their MLIR representation
     */
    ::std::string emitMLIR() const override {
        // Construct the function signature
        ::std::string signature = "(" +
            (_args ? _args->emitMLIR() : "()") +
            " -> " +
            (_ret ? _ret->emitMLIR() : "void") +
            ")";

        // Wrap with nullability if needed
        return _nullable
            ? "!llvm.ptr<" + signature + ">"
            : signature;
    }

    Type* meet(Type* t) override;
};

/**
 * Chapter 18: Return Program Counter Types
 *
 * Represents a return program counter for function calls.
 * Each call site gets a unique RPC value.
 * Similar to TypeFunPtr but signature doesn't matter.
 *
 * Used by evaluator to return from functions without relying on
 * the implementation language's stack.
 */
class TypeRPC : public Type {
private:
    int _rpc;              // Return program counter ID (-1 for unknown)
    bool _nullable;        // Can be null?

    TypeRPC(int rpc = -1, bool nullable = false)
        : _rpc(rpc), _nullable(nullable) {}

public:
    /**
     * Create an RPC type with specific ID.
     */
    static TypeRPC* create(int rpc);

    /**
     * Create a nullable RPC type.
     */
    static TypeRPC* nullable();

    /**
     * Create a null RPC constant.
     */
    static TypeRPC* nullType();

    int rpc() const { return _rpc; }
    bool isNullable() const { return _nullable; }

    bool isConstant() const override {
        return _rpc >= 0 || (_nullable && _rpc < 0);
    }

    ::std::string toString() const override {
        if (_rpc < 0 && _nullable) return "null";
        ::std::string result = "RPC";
        if (_rpc >= 0) result += "[" + ::std::to_string(_rpc) + "]";
        if (_nullable) result += "?";
        return result;
    }

    Type* meet(Type* t) override;
};

} // namespace cppfort::ir

#endif // CPPFORT_TYPE_H
