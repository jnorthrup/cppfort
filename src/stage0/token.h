#pragma once

#include <cstddef>

namespace cppfort::stage0 {

struct SourceLocation {
    size_t line;
    size_t column;
};

} // namespace cppfort::stage0
