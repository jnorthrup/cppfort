#pragma once

#include <cstddef>
#include <ostream>
#include <string>

namespace cppfort::stage0 {

struct HeapLimitResult {
    bool attempted = false;
    bool success = false;
    bool required = true;
    std::size_t limit_bytes = 0;
    std::string source;
    std::string detail;
};

// Returns the cached heap limit status, installing the limit on first use.
[[nodiscard]] const HeapLimitResult& heap_limit_status();

inline bool ensure_heap_limit(std::ostream& os) {
    const auto& guard = heap_limit_status();
    if (guard.required && !guard.success) {
        os << "Error: unable to enforce stage0 heap limit: " << guard.detail << '\n';
        return false;
    }
    return true;
}

} // namespace cppfort::stage0
