#pragma once

#include <iostream>
#include <string>
#include <string_view>

#include "orbit_iterator.h"
#include "orbit_ring.h"
#include "confix_orbit.h"

namespace cppfort::stage0 {

struct PatternData; // Forward declaration

// Pattern-driven emitter (uses anchor-based substitution from YAML patterns)
class CPP2Emitter {
public:
    // Emit transformed output from orbit iterator
    void emit(OrbitIterator& iterator, std::string_view source, std::ostream& out, const std::vector<PatternData>& patterns) const;

    // Emit a single fragment (deprecated - use emit_orbit)
    void emit_fragment(const OrbitFragment& fragment, std::string_view source, std::ostream& out) const;

    // Emit orbit using pattern-driven transformation
    void emit_orbit(const ConfixOrbit& orbit, std::string_view source, std::ostream& out, const PatternData* pattern) const;

private:
    // Extract text for fragment from source
    std::string_view extract_fragment_text(const OrbitFragment& fragment, std::string_view source) const;
};

} // namespace cppfort::stage0