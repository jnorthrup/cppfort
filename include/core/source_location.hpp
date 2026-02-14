// include/core/source_location.hpp - Source location tracking types
// Part of modular build restructuring
#pragma once

#include <cstddef>
#include <string_view>
#include <string>

namespace cpp2_transpiler {

/// Source location for diagnostics
struct SourceLocation {
    std::size_t line{1};
    std::size_t column{1};
    std::size_t offset{0};
    std::string_view filename;

    constexpr SourceLocation() = default;
    constexpr SourceLocation(std::size_t l, std::size_t c, std::size_t o = 0)
        : line(l), column(c), offset(o) {}
    
    constexpr SourceLocation(std::size_t l, std::size_t c, std::size_t o, std::string_view f)
        : line(l), column(c), offset(o), filename(f) {}

    /// Format as "filename:line:column" or "line:column" if no filename
    std::string to_string() const {
        std::string result;
        if (!filename.empty()) {
            result += filename;
            result += ':';
        }
        result += std::to_string(line);
        result += ':';
        result += std::to_string(column);
        return result;
    }
};

/// Source range spanning from start to end location
struct SourceRange {
    SourceLocation start;
    SourceLocation end;

    constexpr SourceRange() = default;
    constexpr SourceRange(SourceLocation s, SourceLocation e) : start(s), end(e) {}
    
    /// Create single-position range
    constexpr explicit SourceRange(SourceLocation loc) : start(loc), end(loc) {}
};

} // namespace cpp2_transpiler
