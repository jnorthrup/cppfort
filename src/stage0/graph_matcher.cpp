#include "graph_matcher.h"
#include "pijul_graph_matcher.h"

namespace cppfort {
namespace stage0 {

bool GraphMatcher::match(std::string_view pattern, std::string_view text) const {
    // Basic fallback: substring match for signature-style patterns
    return text.find(pattern) != std::string_view::npos;
}

bool GraphMatcher::matchPattern(const PatternData& pattern, std::string_view text) const {
    // Prefer PijulGraphMatcher, which offers more sophisticated matching
    PijulGraphMatcher gm(pattern);
    auto matches = gm.find_matches(text);
    return !matches.empty();
}

} // namespace stage0
} // namespace cppfort
