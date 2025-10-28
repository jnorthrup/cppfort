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
    // Emit transformed output from orbit iterator (legacy)
    void emit(OrbitIterator& iterator, std::string_view source, std::ostream& out, const std::vector<PatternData>& patterns) const;

    // Emit using depth-based pattern matching (deterministic)
    void emit_depth_based(std::string_view source, std::ostream& out, const std::vector<PatternData>& patterns) const;

    // Emit a single fragment (deprecated - use emit_orbit)
    void emit_fragment(const OrbitFragment& fragment, std::string_view source, std::ostream& out) const;

    // Emit orbit using pattern-driven transformation
    void emit_orbit(const ConfixOrbit& orbit, std::string_view source, std::ostream& out, const PatternData* pattern, const std::vector<PatternData>& all_patterns) const;

    // Extract segments for alternating anchor/evidence patterns (public for recursive use)
    std::vector<std::string> extract_alternating_segments(std::string_view text, const PatternData& pattern) const;

private:
    // Extract text for fragment from source
    std::string_view extract_fragment_text(const OrbitFragment& fragment, std::string_view source) const;
};

namespace testing {

// Expose parameter canonicalization for focused unit tests
std::string transform_parameter_for_testing(std::string_view param);

} // namespace testing

} // namespace cppfort::stage0
