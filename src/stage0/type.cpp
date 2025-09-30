#include "type.h"

#include <algorithm>
#include <unordered_map>

namespace cppfort::ir {

// Singleton instances
Type* Type::BOTTOM = new TypeBottom();
Type* Type::TOP = new TypeTop();
int Type::GENERATION = 1;

// Cache for constant integer types to avoid duplicates
static std::unordered_map<long, TypeInteger*> constant_cache;

// Singleton for bottom integer
static TypeInteger* BOTTOM_INTEGER = nullptr;

// Singleton for boolean range
static TypeInteger* BOOL_RANGE = nullptr;

TypeInteger* TypeInteger::constant(long value) {
    auto it = constant_cache.find(value);
    if (it != constant_cache.end()) {
        return it->second;
    }

    auto* type = new TypeInteger(value, value);
    constant_cache[value] = type;
    return type;
}

TypeInteger* TypeInteger::bottom() {
    if (BOTTOM_INTEGER == nullptr) {
        BOTTOM_INTEGER = new TypeInteger(std::numeric_limits<long>::min(),
                                         std::numeric_limits<long>::max());
    }
    return BOTTOM_INTEGER;
}

TypeInteger* TypeInteger::boolean() {
    if (BOOL_RANGE == nullptr) {
        BOOL_RANGE = new TypeInteger(0, 1);
    }
    return BOOL_RANGE;
}

Type* TypeInteger::meet(Type* t) {
    if (!t) {
        _meet_cache = Type::BOTTOM;
        return _meet_cache;
    }

    if (_gen == Type::GENERATION) {
        return _meet_cache;
    }
    _gen = Type::GENERATION++;

    auto ti = dynamic_cast<TypeInteger*>(t);
    if (!ti) {
        _meet_cache = Type::BOTTOM;
        return _meet_cache;
    }

    long lo = std::max(_lo, ti->_lo);
    long hi = std::min(_hi, ti->_hi);

    if (lo > hi) {
        _meet_cache = Type::BOTTOM;
    } else if (lo == hi) {
        _meet_cache = TypeInteger::constant(lo);
    } else if (lo == 0 && hi == 1) {
        _meet_cache = TypeInteger::boolean();
    } else if (lo == std::numeric_limits<long>::min() &&
               hi == std::numeric_limits<long>::max()) {
        _meet_cache = TypeInteger::bottom();
    } else {
        _meet_cache = new TypeInteger(lo, hi);
    }

    return _meet_cache;
}

// ============================================================================
// Band 4: Type System Expansion Implementations
// ============================================================================

// Chapter 12: Floating Point Types
// ============================================================================

static std::unordered_map<double, TypeFloat*> float_constant_cache_f32;
static std::unordered_map<double, TypeFloat*> float_constant_cache_f64;
static TypeFloat* BOTTOM_FLOAT_F32 = nullptr;
static TypeFloat* BOTTOM_FLOAT_F64 = nullptr;

TypeFloat* TypeFloat::constant(double value, Precision prec) {
    auto& cache = (prec == F32) ? float_constant_cache_f32 : float_constant_cache_f64;
    auto it = cache.find(value);
    if (it != cache.end()) {
        return it->second;
    }

    auto* type = new TypeFloat(prec, value, true);
    cache[value] = type;
    return type;
}

TypeFloat* TypeFloat::bottom(Precision prec) {
    TypeFloat*& bottom = (prec == F32) ? BOTTOM_FLOAT_F32 : BOTTOM_FLOAT_F64;
    if (bottom == nullptr) {
        bottom = new TypeFloat(prec, 0.0, false);
    }
    return bottom;
}

Type* TypeFloat::meet(Type* t) {
    if (!t) return Type::BOTTOM;
    if (t->isTop()) return this;
    if (t->isBottom()) return t;

    TypeFloat* other = dynamic_cast<TypeFloat*>(t);
    if (!other) return Type::BOTTOM;

    // Precision mismatch
    if (_precision != other->_precision) return Type::BOTTOM;

    // Meet of two constants
    if (isConstant() && other->isConstant()) {
        return _value == other->_value ? this : Type::BOTTOM;
    }

    // One is bottom
    return bottom(_precision);
}

// Chapter 13: Reference Types
// ============================================================================

static std::unordered_map<std::string, TypePointer*> nullable_ptr_cache;
static std::unordered_map<std::string, TypePointer*> non_nullable_ptr_cache;
static TypePointer* NULL_TYPE = nullptr;

TypePointer* TypePointer::nullable(const std::string& struct_name) {
    auto it = nullable_ptr_cache.find(struct_name);
    if (it != nullable_ptr_cache.end()) {
        return it->second;
    }

    auto* type = new TypePointer(struct_name, true, false);
    nullable_ptr_cache[struct_name] = type;
    return type;
}

TypePointer* TypePointer::nonNullable(const std::string& struct_name) {
    auto it = non_nullable_ptr_cache.find(struct_name);
    if (it != non_nullable_ptr_cache.end()) {
        return it->second;
    }

    auto* type = new TypePointer(struct_name, false, false);
    non_nullable_ptr_cache[struct_name] = type;
    return type;
}

TypePointer* TypePointer::nullType() {
    if (NULL_TYPE == nullptr) {
        NULL_TYPE = new TypePointer("", true, true);
    }
    return NULL_TYPE;
}

Type* TypePointer::meet(Type* t) {
    if (!t) return Type::BOTTOM;
    if (t->isTop()) return this;
    if (t->isBottom()) return t;

    TypePointer* other = dynamic_cast<TypePointer*>(t);
    if (!other) return Type::BOTTOM;

    // Meet with null
    if (_is_null) return other->_nullable ? this : Type::BOTTOM;
    if (other->_is_null) return _nullable ? other : Type::BOTTOM;

    // Struct type mismatch
    if (_target_name != other->_target_name) return Type::BOTTOM;

    // Nullability: meet is the more permissive (nullable)
    if (_nullable || other->_nullable) {
        return nullable(_target_name);
    }

    return this;
}

// Chapter 14: Narrow Integer Types
// ============================================================================

struct NarrowKeyHash {
    std::size_t operator()(const std::pair<long, TypeNarrow::Width>& p) const {
        return std::hash<long>()(p.first) ^ (std::hash<int>()(static_cast<int>(p.second)) << 1);
    }
};

static std::unordered_map<std::pair<long, TypeNarrow::Width>, TypeNarrow*, NarrowKeyHash> narrow_constant_cache;

static std::unordered_map<TypeNarrow::Width, TypeNarrow*> narrow_bottom_cache;

TypeNarrow* TypeNarrow::constant(long value, Width width) {
    auto key = std::make_pair(value, width);
    auto it = narrow_constant_cache.find(key);
    if (it != narrow_constant_cache.end()) {
        return it->second;
    }

    auto* type = new TypeNarrow(width, value, value);
    narrow_constant_cache[key] = type;
    return type;
}

TypeNarrow* TypeNarrow::bottom(Width width) {
    auto it = narrow_bottom_cache.find(width);
    if (it != narrow_bottom_cache.end()) {
        return it->second;
    }

    // Compute range based on width
    long lo, hi;
    switch(width) {
        case I8:  lo = -128; hi = 127; break;
        case I16: lo = -32768; hi = 32767; break;
        case I32: lo = -(1L << 31); hi = (1L << 31) - 1; break;
        case I64: lo = std::numeric_limits<long>::min(); hi = std::numeric_limits<long>::max(); break;
        case U8:  lo = 0; hi = 255; break;
        case U16: lo = 0; hi = 65535; break;
        case U32: lo = 0; hi = (1L << 32) - 1; break;
    }

    auto* type = new TypeNarrow(width, lo, hi);
    narrow_bottom_cache[width] = type;
    return type;
}

bool TypeNarrow::isBottom() const {
    TypeNarrow* bot = bottom(_width);
    return _lo == bot->_lo && _hi == bot->_hi;
}

std::string TypeNarrow::toString() const {
    if (isConstant()) {
        std::string suffix;
        switch(_width) {
            case I8:  suffix = "i8"; break;
            case I16: suffix = "i16"; break;
            case I32: suffix = "i32"; break;
            case I64: suffix = "i64"; break;
            case U8:  suffix = "u8"; break;
            case U16: suffix = "u16"; break;
            case U32: suffix = "u32"; break;
        }
        return std::to_string(_lo) + suffix;
    }

    if (isBottom()) {
        switch(_width) {
            case I8:  return "i8⊥";
            case I16: return "i16⊥";
            case I32: return "i32⊥";
            case I64: return "i64⊥";
            case U8:  return "u8⊥";
            case U16: return "u16⊥";
            case U32: return "u32⊥";
        }
    }

    return "[" + std::to_string(_lo) + "," + std::to_string(_hi) + "]";
}

Type* TypeNarrow::meet(Type* t) {
    if (!t) return Type::BOTTOM;
    if (t->isTop()) return this;
    if (t->isBottom()) return t;

    TypeNarrow* other = dynamic_cast<TypeNarrow*>(t);
    if (!other) return Type::BOTTOM;

    // Width mismatch
    if (_width != other->_width) return Type::BOTTOM;

    // Meet of two constants
    if (isConstant() && other->isConstant()) {
        return _lo == other->_lo ? this : Type::BOTTOM;
    }

    // Compute range intersection
    long lo = std::max(_lo, other->_lo);
    long hi = std::min(_hi, other->_hi);

    if (lo > hi) return Type::BOTTOM;
    if (lo == hi) return constant(lo, _width);

    return bottom(_width);
}

// Chapter 15: Array Types
// ============================================================================

// Array types are created dynamically, no caching for now
// In a production compiler, you'd cache based on (element_type, length, nullable) tuple

TypeArray* TypeArray::fixedSize(Type* element_type, long length) {
    return new TypeArray(element_type, length, false);
}

TypeArray* TypeArray::dynamic(Type* element_type) {
    return new TypeArray(element_type, -1, false);
}

TypeArray* TypeArray::nullable(Type* element_type) {
    return new TypeArray(element_type, -1, true);
}

Type* TypeArray::meet(Type* t) {
    if (!t) return Type::BOTTOM;
    if (t->isTop()) return this;
    if (t->isBottom()) return t;

    TypeArray* other = dynamic_cast<TypeArray*>(t);
    if (!other) return Type::BOTTOM;

    // Element type must meet
    Type* elem_meet = _element_type->meet(other->_element_type);
    if (elem_meet->isBottom()) return Type::BOTTOM;

    // Nullability: meet is more permissive (nullable)
    bool nullable = _nullable || other->_nullable;

    // Length: -1 (dynamic) meets with any length
    long length = -1;
    if (_length >= 0 && other->_length >= 0) {
        if (_length != other->_length) return Type::BOTTOM;
        length = _length;
    }

    return new TypeArray(elem_meet, length, nullable);
}

} // namespace cppfort::ir
