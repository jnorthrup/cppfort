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

} // namespace cppfort::ir
