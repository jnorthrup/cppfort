#include "type.h"

#include <algorithm>
#include <unordered_map>

namespace cppfort::ir {

// Singleton instances
Type* Type::BOTTOM = new TypeBottom();
Type* Type::TOP = new TypeTop();
int Type::GENERATION = 1;

// Chapter 17: Default GLB implementation
Type* Type::glb() const {
    // Default: return bottom (widest type)
    return Type::BOTTOM;
}

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

// Chapter 17: Cache key includes mutability
struct PtrCacheKey {
    std::string name;
    bool nullable;
    bool mutable_ref;

    bool operator==(const PtrCacheKey& other) const {
        return name == other.name && nullable == other.nullable && mutable_ref == other.mutable_ref;
    }
};

struct PtrCacheKeyHash {
    std::size_t operator()(const PtrCacheKey& k) const {
        std::size_t h1 = std::hash<std::string>()(k.name);
        std::size_t h2 = std::hash<bool>()(k.nullable);
        std::size_t h3 = std::hash<bool>()(k.mutable_ref);
        return h1 ^ (h2 << 1) ^ (h3 << 2);
    }
};

static std::unordered_map<PtrCacheKey, TypePointer*, PtrCacheKeyHash> ptr_cache;
static TypePointer* NULL_TYPE = nullptr;

TypePointer* TypePointer::nullable(const std::string& struct_name, bool mutable_ref) {
    PtrCacheKey key{struct_name, true, mutable_ref};
    auto it = ptr_cache.find(key);
    if (it != ptr_cache.end()) {
        return it->second;
    }

    auto* type = new TypePointer(struct_name, true, false, mutable_ref);
    ptr_cache[key] = type;
    return type;
}

TypePointer* TypePointer::nonNullable(const std::string& struct_name, bool mutable_ref) {
    PtrCacheKey key{struct_name, false, mutable_ref};
    auto it = ptr_cache.find(key);
    if (it != ptr_cache.end()) {
        return it->second;
    }

    auto* type = new TypePointer(struct_name, false, false, mutable_ref);
    ptr_cache[key] = type;
    return type;
}

TypePointer* TypePointer::nullType() {
    if (NULL_TYPE == nullptr) {
        NULL_TYPE = new TypePointer("", true, true, false);
    }
    return NULL_TYPE;
}

TypePointer* TypePointer::immutable(const std::string& struct_name, bool nullable) {
    return nullable ? TypePointer::nullable(struct_name, false)
                    : TypePointer::nonNullable(struct_name, false);
}

TypePointer* TypePointer::mutable_(const std::string& struct_name, bool nullable) {
    return nullable ? TypePointer::nullable(struct_name, true)
                    : TypePointer::nonNullable(struct_name, true);
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
    bool result_nullable = _nullable || other->_nullable;

    // Chapter 17: Mutability: meet is the less permissive (immutable)
    // If either reference is immutable, the result is immutable
    bool result_mutable = _mutable && other->_mutable;

    if (result_nullable) {
        return nullable(_target_name, result_mutable);
    }

    return nonNullable(_target_name, result_mutable);
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

// ============================================================================
// Chapter 16: Struct Types
// ============================================================================

// Struct type cache - indexed by name
static std::unordered_map<std::string, TypeStruct*> struct_type_cache;

TypeStruct* TypeStruct::create(const std::string& name, bool nullable) {
    // Check cache for non-nullable version
    std::string cache_key = name + (nullable ? "?" : "");
    auto it = struct_type_cache.find(cache_key);
    if (it != struct_type_cache.end()) {
        return it->second;
    }

    auto* type = new TypeStruct(name, nullable);
    struct_type_cache[cache_key] = type;
    return type;
}

int TypeStruct::addField(const std::string& name, Type* type, bool isFinal, Node* initVal,
                         Field::MutabilityQualifier mutability) {
    // Check for duplicate field names
    if (_fieldMap.find(name) != _fieldMap.end()) {
        // Field already exists - error
        return -1;
    }

    int offset = _totalSize;
    int index = _fields.size();

    // Estimate field size (simplified - in real compiler would use proper layout)
    // For now, just use 8 bytes per field
    _totalSize += 8;

    _fields.emplace_back(name, type, isFinal, initVal, offset, mutability);
    _fieldMap[name] = index;

    return index;
}

const Field* TypeStruct::getField(const std::string& name) const {
    auto it = _fieldMap.find(name);
    if (it == _fieldMap.end()) {
        return nullptr;
    }
    return &_fields[it->second];
}

bool TypeStruct::isFullyInitialized() const {
    for (const Field& field : _fields) {
        // Final fields must have an initial value
        if (field.isFinal && field.initialValue == nullptr) {
            return false;
        }

        // Non-nullable pointer types must have an initial value
        // (simplified - would check TypePointer->isNullable())
        if (field.initialValue == nullptr) {
            // Check if type requires initialization
            TypePointer* ptrType = dynamic_cast<TypePointer*>(field.type);
            if (ptrType && !ptrType->isNullable()) {
                return false;
            }
        }
    }
    return true;
}

Type* TypeStruct::meet(Type* t) {
    if (!t) return Type::BOTTOM;
    if (t->isTop()) return this;
    if (t->isBottom()) return t;

    TypeStruct* other = dynamic_cast<TypeStruct*>(t);
    if (!other) return Type::BOTTOM;

    // Struct types must have the same name to meet
    if (_name != other->_name) return Type::BOTTOM;

    // Meet of nullable and non-nullable is nullable
    bool nullable = _nullable || other->_nullable;

    return create(_name, nullable);
}

// ============================================================================
// Chapter 18: Function Types
// ============================================================================

// TypeTuple cache - avoid duplicate tuples with same signature
static std::vector<TypeTuple*> tuple_cache;

TypeTuple* TypeTuple::create(const std::vector<Type*>& types) {
    // Check cache for existing tuple with same types
    for (TypeTuple* cached : tuple_cache) {
        if (cached->types().size() != types.size()) continue;

        bool match = true;
        for (size_t i = 0; i < types.size(); ++i) {
            if (cached->get(i) != types[i]) {
                match = false;
                break;
            }
        }

        if (match) return cached;
    }

    // Create new tuple and cache it
    auto* tuple = new TypeTuple(types);
    tuple_cache.push_back(tuple);
    return tuple;
}

Type* TypeTuple::meet(Type* t) {
    if (!t) return Type::BOTTOM;
    if (t->isTop()) return this;
    if (t == Type::BOTTOM) return t;  // Check for universal BOTTOM

    TypeTuple* other = dynamic_cast<TypeTuple*>(t);
    if (!other) return Type::BOTTOM;

    // Tuples must have same size
    if (_types.size() != other->_types.size()) return Type::BOTTOM;

    // Meet each element
    std::vector<Type*> result_types;
    for (size_t i = 0; i < _types.size(); ++i) {
        Type* elem_meet = _types[i]->meet(other->_types[i]);
        // Check for universal BOTTOM (meet failed), not type-specific bottom
        if (elem_meet == Type::BOTTOM) return Type::BOTTOM;
        result_types.push_back(elem_meet);
    }

    return create(result_types);
}

// TypeFunPtr cache
struct FunPtrCacheKey {
    TypeTuple* args;
    Type* ret;
    int fidx;
    bool nullable;
    bool mutable_ref;

    bool operator==(const FunPtrCacheKey& other) const {
        return args == other.args && ret == other.ret && fidx == other.fidx &&
               nullable == other.nullable && mutable_ref == other.mutable_ref;
    }
};

struct FunPtrCacheKeyHash {
    std::size_t operator()(const FunPtrCacheKey& k) const {
        std::size_t h1 = std::hash<void*>()(k.args);
        std::size_t h2 = std::hash<void*>()(k.ret);
        std::size_t h3 = std::hash<int>()(k.fidx);
        std::size_t h4 = std::hash<bool>()(k.nullable);
        std::size_t h5 = std::hash<bool>()(k.mutable_ref);
        return h1 ^ (h2 << 1) ^ (h3 << 2) ^ (h4 << 3) ^ (h5 << 4);
    }
};

static std::unordered_map<FunPtrCacheKey, TypeFunPtr*, FunPtrCacheKeyHash> funptr_cache;
static TypeFunPtr* NULL_FUNPTR = nullptr;

TypeFunPtr* TypeFunPtr::create(TypeTuple* args, Type* ret, int fidx) {
    FunPtrCacheKey key{args, ret, fidx, false, true};
    auto it = funptr_cache.find(key);
    if (it != funptr_cache.end()) {
        return it->second;
    }

    auto* type = new TypeFunPtr(args, ret, fidx, false, true);
    funptr_cache[key] = type;
    return type;
}

TypeFunPtr* TypeFunPtr::nullable(TypeTuple* args, Type* ret) {
    FunPtrCacheKey key{args, ret, -1, true, true};
    auto it = funptr_cache.find(key);
    if (it != funptr_cache.end()) {
        return it->second;
    }

    auto* type = new TypeFunPtr(args, ret, -1, true, true);
    funptr_cache[key] = type;
    return type;
}

TypeFunPtr* TypeFunPtr::mutable_(TypeTuple* args, Type* ret, bool nullable) {
    FunPtrCacheKey key{args, ret, -1, nullable, true};
    auto it = funptr_cache.find(key);
    if (it != funptr_cache.end()) {
        return it->second;
    }

    auto* type = new TypeFunPtr(args, ret, -1, nullable, true);
    funptr_cache[key] = type;
    return type;
}

TypeFunPtr* TypeFunPtr::immutable(TypeTuple* args, Type* ret, bool nullable) {
    FunPtrCacheKey key{args, ret, -1, nullable, false};
    auto it = funptr_cache.find(key);
    if (it != funptr_cache.end()) {
        return it->second;
    }

    auto* type = new TypeFunPtr(args, ret, -1, nullable, false);
    funptr_cache[key] = type;
    return type;
}

TypeFunPtr* TypeFunPtr::nullType() {
    if (NULL_FUNPTR == nullptr) {
        // Create an empty tuple for null function
        TypeTuple* empty = TypeTuple::create({});
        NULL_FUNPTR = new TypeFunPtr(empty, Type::BOTTOM, -1, true, false);
    }
    return NULL_FUNPTR;
}

Type* TypeFunPtr::meet(Type* t) {
    if (!t) return Type::BOTTOM;
    if (t->isTop()) return this;
    if (t == Type::BOTTOM) return t;  // Check for universal BOTTOM

    TypeFunPtr* other = dynamic_cast<TypeFunPtr*>(t);
    if (!other) return Type::BOTTOM;

    // Meet argument types
    Type* args_meet = _args->meet(other->_args);
    if (args_meet == Type::BOTTOM) return Type::BOTTOM;  // Check for universal BOTTOM
    TypeTuple* args_tuple = dynamic_cast<TypeTuple*>(args_meet);
    if (!args_tuple) return Type::BOTTOM;

    // Meet return type
    Type* ret_meet = _ret->meet(other->_ret);
    if (ret_meet == Type::BOTTOM) return Type::BOTTOM;  // Check for universal BOTTOM

    // Nullability: meet is more permissive (nullable)
    bool result_nullable = _nullable || other->_nullable;

    // Mutability: meet is less permissive (immutable)
    bool result_mutable = _mutable && other->_mutable;

    // Function index: if both have same fidx, keep it; otherwise -1
    int result_fidx = (_fidx >= 0 && _fidx == other->_fidx) ? _fidx : -1;

    // Create result function type
    FunPtrCacheKey key{args_tuple, ret_meet, result_fidx, result_nullable, result_mutable};
    auto it = funptr_cache.find(key);
    if (it != funptr_cache.end()) {
        return it->second;
    }

    auto* result = new TypeFunPtr(args_tuple, ret_meet, result_fidx, result_nullable, result_mutable);
    funptr_cache[key] = result;
    return result;
}

// TypeRPC implementation
static std::unordered_map<int, TypeRPC*> rpc_cache;
static TypeRPC* NULL_RPC = nullptr;
static TypeRPC* NULLABLE_RPC = nullptr;

TypeRPC* TypeRPC::create(int rpc) {
    auto it = rpc_cache.find(rpc);
    if (it != rpc_cache.end()) {
        return it->second;
    }

    auto* type = new TypeRPC(rpc, false);
    rpc_cache[rpc] = type;
    return type;
}

TypeRPC* TypeRPC::nullable() {
    if (NULLABLE_RPC == nullptr) {
        NULLABLE_RPC = new TypeRPC(-1, true);
    }
    return NULLABLE_RPC;
}

TypeRPC* TypeRPC::nullType() {
    if (NULL_RPC == nullptr) {
        NULL_RPC = new TypeRPC(-1, true);
    }
    return NULL_RPC;
}

Type* TypeRPC::meet(Type* t) {
    if (!t) return Type::BOTTOM;
    if (t->isTop()) return this;
    if (t->isBottom()) return t;

    TypeRPC* other = dynamic_cast<TypeRPC*>(t);
    if (!other) return Type::BOTTOM;

    // Nullability: meet is more permissive (nullable)
    bool result_nullable = _nullable || other->_nullable;

    // RPC: if both have same RPC, keep it; otherwise -1
    int result_rpc = (_rpc >= 0 && _rpc == other->_rpc) ? _rpc : -1;

    if (result_nullable && result_rpc < 0) {
        return nullable();
    }

    if (result_rpc >= 0) {
        auto it = rpc_cache.find(result_rpc);
        if (it != rpc_cache.end()) {
            return it->second;
        }
        auto* result = new TypeRPC(result_rpc, result_nullable);
        rpc_cache[result_rpc] = result;
        return result;
    }

    return nullable();
}

} // namespace cppfort::ir
