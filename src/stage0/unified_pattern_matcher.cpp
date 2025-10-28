#include "unified_pattern_matcher.h"
#include <algorithm>
#include <cctype>
#include <iostream>

namespace cppfort::stage0 {

std::vector<UnifiedPatternMatch> UnifiedPatternMatcher::find_matches(
    std::string_view text,
    const std::vector<PatternData>& patterns,
    bool track_depth,
    int max_depth
) {
    std::vector<UnifiedPatternMatch> matches;

    // Build depth map if requested
    std::vector<int> depth_map;
    if (track_depth) {
        depth_map.resize(text.size(), 0);
        int current_depth = 0;
        for (size_t i = 0; i < text.size(); ++i) {
            if (text[i] == '(' || text[i] == '{' || text[i] == '[') {
                ++current_depth;
            } else if (text[i] == ')' || text[i] == '}' || text[i] == ']') {
                --current_depth;
            }
            depth_map[i] = current_depth;
        }
    }

    // Try each pattern at each position
    for (const auto& pattern : patterns) {
        if (pattern.alternating_anchors.empty()) continue;

        const std::string& first_anchor = pattern.alternating_anchors[0];
        size_t pos = 0;

        while ((pos = text.find(first_anchor, pos)) != std::string::npos) {
            auto match = try_match_at(text, pos, pattern, track_depth ? &depth_map : nullptr);
            if (match) {
                if (!track_depth || match->depth <= max_depth) {
                    matches.push_back(*match);
                    pos = match->end_pos; // Skip past this match
                } else {
                    ++pos;
                }
            } else {
                ++pos;
            }
        }
    }

    // Sort by position and filter overlaps
    std::sort(matches.begin(), matches.end(),
        [](const auto& a, const auto& b) { return a.start_pos < b.start_pos; });

    std::vector<UnifiedPatternMatch> filtered;
    for (const auto& match : matches) {
        bool overlaps = false;
        for (const auto& existing : filtered) {
            if (!(match.end_pos <= existing.start_pos || match.start_pos >= existing.end_pos)) {
                overlaps = true;
                break;
            }
        }
        if (!overlaps) {
            filtered.push_back(match);
        }
    }

    return filtered;
}

std::optional<std::vector<std::string>> UnifiedPatternMatcher::extract_segments(
    const std::string& pattern,
    const std::string& input
) {
    // Extract anchors and segment indices from pattern
    std::vector<std::string> anchors;
    std::vector<int> segment_indices;

    size_t pos = 0;
    std::string current_anchor;

    while (pos < pattern.size()) {
        if (pattern[pos] == '$' && pos + 1 < pattern.size() && std::isdigit(pattern[pos + 1])) {
            if (!current_anchor.empty()) {
                anchors.push_back(current_anchor);
                current_anchor.clear();
            }
            segment_indices.push_back(pattern[pos + 1] - '0');
            pos += 2;
        } else {
            current_anchor += pattern[pos];
            ++pos;
        }
    }
    if (!current_anchor.empty()) {
        anchors.push_back(current_anchor);
    }

    // Extract segments from input
    int max_idx = segment_indices.empty() ? 0 :
        *std::max_element(segment_indices.begin(), segment_indices.end());
    std::vector<std::string> segments(max_idx + 1);

    size_t input_pos = 0;
    for (size_t i = 0; i < segment_indices.size(); ++i) {
        int seg_idx = segment_indices[i];

        size_t next_anchor_pos = input.size();
        if (i < anchors.size()) {
            next_anchor_pos = input.find(anchors[i], input_pos);
            if (next_anchor_pos == std::string::npos) {
                return std::nullopt;
            }
        }

        std::string segment = input.substr(input_pos, next_anchor_pos - input_pos);

        // Trim whitespace
        auto start = segment.find_first_not_of(" \t\n\r");
        auto end = segment.find_last_not_of(" \t\n\r");
        if (start != std::string::npos && end != std::string::npos) {
            segment = segment.substr(start, end - start + 1);
        } else {
            segment.clear();
        }

        segments[seg_idx] = segment;

        if (i < anchors.size()) {
            input_pos = next_anchor_pos + anchors[i].size();
        }
    }

    return segments;
}

void UnifiedPatternMatcher::register_rewrite(
    const std::string& pattern_name,
    int target_language,
    RewriteFunc rewrite,
    int priority
) {
    rewrites_[pattern_name][target_language] = {rewrite, priority};
}

std::optional<std::string> UnifiedPatternMatcher::apply_rewrite(
    const UnifiedPatternMatch& match,
    int target_language
) const {
    if (!match.pattern) return std::nullopt;

    auto it = rewrites_.find(match.pattern->name);
    if (it == rewrites_.end()) return std::nullopt;

    auto target_it = it->second.find(target_language);
    if (target_it == it->second.end()) return std::nullopt;

    return target_it->second.func(match);
}

std::optional<UnifiedPatternMatch> UnifiedPatternMatcher::try_match_at(
    std::string_view text,
    size_t pos,
    const PatternData& pattern,
    const std::vector<int>* depth_map
) {
    UnifiedPatternMatch match;
    match.pattern = &pattern;
    match.start_pos = pos;

    // Extract segments based on alternating anchors
    size_t current_pos = pos;
    for (size_t i = 0; i < pattern.alternating_anchors.size(); ++i) {
        const std::string& anchor = pattern.alternating_anchors[i];

        size_t anchor_pos = text.find(anchor, current_pos);
        if (anchor_pos == std::string::npos) {
            return std::nullopt;
        }

        // Extract segment before this anchor
        if (i < pattern.evidence_types.size()) {
            std::string segment(text.substr(current_pos, anchor_pos - current_pos));

            // Trim
            auto start = segment.find_first_not_of(" \t\n\r");
            auto end = segment.find_last_not_of(" \t\n\r");
            if (start != std::string::npos && end != std::string::npos) {
                segment = segment.substr(start, end - start + 1);
            } else {
                segment.clear();
            }

            match.segments.push_back(segment);
        }

        current_pos = anchor_pos + anchor.size();
    }

    // Extract final segment after last anchor
    if (pattern.evidence_types.size() > pattern.alternating_anchors.size()) {
        // Find end of statement (semicolon, newline, or closing brace)
        size_t end_pos = current_pos;
        while (end_pos < text.size() &&
               text[end_pos] != ';' &&
               text[end_pos] != '\n' &&
               text[end_pos] != '}') {
            ++end_pos;
        }

        std::string segment(text.substr(current_pos, end_pos - current_pos));

        // Trim
        auto start = segment.find_first_not_of(" \t\n\r");
        auto end = segment.find_last_not_of(" \t\n\r");
        if (start != std::string::npos && end != std::string::npos) {
            segment = segment.substr(start, end - start + 1);
        } else {
            segment.clear();
        }

        match.segments.push_back(segment);
        match.end_pos = end_pos;
    } else {
        match.end_pos = current_pos;
    }

    // Set depth if tracking
    if (depth_map && !depth_map->empty()) {
        match.depth = (*depth_map)[pos];
    }

    // Validate segment count
    if (match.segments.size() != pattern.evidence_types.size()) {
        return std::nullopt;
    }

    return match;
}

} // namespace cppfort::stage0