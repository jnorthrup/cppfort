#include "orbit_emitter.h"
#include "confix_orbit.h"
#include <algorithm>
#include <sstream>

namespace cppfort::stage0 {

std::vector<OrbitEmitter::Token> OrbitEmitter::emit_tokens(Orbit* orbit, std::string_view source) const {
    std::vector<Token> tokens;

    if (!orbit) return tokens;

    // Extract token boundaries from orbit
    auto boundaries = compute_boundaries(orbit);

    for (const auto& [start, end] : boundaries) {
        Token tok;
        tok.start_pos = start;
        tok.end_pos = end;
        tok.confidence = orbit->confidence;

        // Extract text from source or evidence
        tok.text = extract_text(orbit, source.substr(start, end - start));

        // Tag orbit type for debugging
        if (auto* confix = dynamic_cast<ConfixOrbit*>(orbit)) {
            tok.orbit_type = "confix[";
            tok.orbit_type += confix->open_symbol();
            tok.orbit_type += confix->close_symbol();
            tok.orbit_type += "]";
        } else {
            tok.orbit_type = "base";
        }

        tokens.push_back(std::move(tok));
    }

    return tokens;
}

std::string OrbitEmitter::reconstruct_source(OrbitIterator& iterator, std::string_view source) const {
    std::ostringstream output;
    size_t last_pos = 0;

    iterator.reset();
    while (Orbit* orbit = iterator.next()) {
        // Emit any gap between orbits as-is
        if (orbit->start_pos > last_pos) {
            output << source.substr(last_pos, orbit->start_pos - last_pos);
        }

        // Emit tokens from this orbit
        auto tokens = emit_tokens(orbit, source);
        for (const auto& token : tokens) {
            output << token.text;
        }

        last_pos = orbit->end_pos;
    }

    // Emit any trailing content
    if (last_pos < source.size()) {
        output << source.substr(last_pos);
    }

    return output.str();
}

std::string OrbitEmitter::extract_text(Orbit* orbit, std::string_view source) const {
    // For reconstruction, just use the source text
    return std::string(source);
}

std::vector<std::pair<size_t, size_t>> OrbitEmitter::compute_boundaries(Orbit* orbit) const {
    std::vector<std::pair<size_t, size_t>> boundaries;

    // For now, treat the whole orbit as one token
    // Later we can split based on internal structure
    boundaries.emplace_back(orbit->start_pos, orbit->end_pos);

    // TODO: For compound orbits, recurse into children
    // TODO: For confix orbits, potentially emit open/close separately

    return boundaries;
}

} // namespace cppfort::stage0