#pragma once

#include <string_view>
#include <vector>
#include <optional>

#include "pattern_loader.h"
#include "unified_pattern_matcher.h"
#include "pijul_parameter_graph.h"

namespace cppfort::stage0 {

class PijulGraphMatcher {
public:
    explicit PijulGraphMatcher(const PatternData& pattern);

    // Find matches for this pattern within text (minimal Graph-based approach)
    std::vector<UnifiedPatternMatch> find_matches(std::string_view text) const;
    // Scoped region search - limits matches to [regionStart, regionEnd) and annotates metadata
    std::vector<UnifiedPatternMatch> find_matches_in_region(std::string_view text, size_t regionStart, size_t regionEnd) const;

private:
    PatternData pattern_;
};

} // namespace cppfort::stage0
