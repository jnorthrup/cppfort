// Minimal GraphMatcher placeholder
#pragma once

#include <string>
#include <string_view>
#include "pattern_loader.h"

namespace cppfort {
namespace stage0 {

// Basic GraphMatcher interface placeholder. Implementations should use
// `pijul_parameter_graph` and `pijul::Graph` primitives to perform robust
// matches. For now, this is a simple signature/graph matcher where available.
class GraphMatcher {
public:
    GraphMatcher() = default;
    ~GraphMatcher() = default;

    // Match the `pattern` against the `text` (signature/shallow matching)
    bool match(std::string_view pattern, std::string_view text) const;

    // Match using a full PatternData instance (preferred when available)
    bool matchPattern(const PatternData& pattern, std::string_view text) const;
};

} // namespace stage0
} // namespace cppfort
