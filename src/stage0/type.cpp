#include "type.h"
#include <unordered_map>

namespace cppfort::ir {

// Singleton instances
Type* Type::BOTTOM = new TypeBottom();
Type* Type::TOP = new TypeTop();

// Cache for constant integer types to avoid duplicates
static std::unordered_map<long, TypeInteger*> constant_cache;

// Singleton for bottom integer
static TypeInteger* BOTTOM_INTEGER = nullptr;

TypeInteger* TypeInteger::constant(long value) {
    auto it = constant_cache.find(value);
    if (it != constant_cache.end()) {
        return it->second;
    }

    auto* type = new TypeInteger(true, value);
    constant_cache[value] = type;
    return type;
}

TypeInteger* TypeInteger::bottom() {
    if (BOTTOM_INTEGER == nullptr) {
        BOTTOM_INTEGER = new TypeInteger(false, 0);
    }
    return BOTTOM_INTEGER;
}

} // namespace cppfort::ir