#pragma once

#include <string_view>
#include <vector>
#include <optional>
#include "pattern_loader.h"

namespace cppfort::stage0 {

// Depth-based deterministic pattern matching
// No speculation, no confidence - just: "Does this pattern fit here?"
struct PatternMatch {
    size_t start_pos;
    size_t end_pos;
    const PatternData* pattern;
    std::vector<std::string> segments;
    int depth;
};

class DepthPatternMatcher {
public:
    // Find all pattern matches in text, ordered by depth (innermost first)
    static std::vector<PatternMatch> find_matches(
        std::string_view text,
        const std::vector<PatternData>& patterns
    );

    // Find all pattern matches with recursive application to segments
    static std::vector<PatternMatch> find_matches_recursive(
        std::string_view text,
        const std::vector<PatternData>& patterns,
        int max_depth = 5
    );

    // Check if a specific pattern matches at a given position
    static std::optional<PatternMatch> try_match(
        std::string_view text,
        size_t start_pos,
        const PatternData& pattern,
        const std::vector<int>& depth_map
    );

    // Extract segments for a pattern match, return (segments, actual_end_pos)
    static std::pair<std::vector<std::string>, size_t> extract_segments(
        std::string_view text,
        const PatternData& pattern,
        size_t anchor_pos
    );

    // Apply recursive transformations to segments with depth-limited recursion
    static std::vector<std::string> apply_recursive_segment_transformations(
        const std::vector<std::string>& segments,
        const std::vector<PatternData>& patterns,
        int current_depth = 0,
        int max_depth = 5
    );

private:
    // Check if pattern's anchors exist at this position
    static bool anchors_present(
        std::string_view text,
        size_t pos,
        const PatternData& pattern
    );

    // Validate evidence doesn't cross confix boundaries
    static bool evidence_valid(
        std::string_view text,
        size_t start,
        size_t end,
        const std::vector<int>& depth_map
    );
};

} // namespace cppfort::stage0
