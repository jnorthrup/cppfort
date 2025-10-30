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

    constexpr bool kPatternDebug = false;

    // Try each pattern at each position
    for (const auto& pattern : patterns) {
        if (pattern.alternating_anchors.empty()) continue;

        const std::string& first_anchor = pattern.alternating_anchors[0];
        size_t pos = 0;

        while ((pos = text.find(first_anchor, pos)) != std::string::npos) {
            auto match = try_match_at(text, pos, pattern, track_depth ? &depth_map : nullptr);
            if (match) {
                if (!track_depth || match->depth <= max_depth) {
                    if (kPatternDebug) {
                        std::cerr << "DEBUG find_matches: matched pattern='" << pattern.name
                                  << "' start=" << match->start_pos
                                  << " end=" << match->end_pos
                                  << " depth=" << match->depth << "\n";
                    }
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
        [](const auto& a, const auto& b) {
            if (a.start_pos != b.start_pos) {
                return a.start_pos < b.start_pos;
            }

            int priority_a = a.pattern ? a.pattern->priority : 0;
            int priority_b = b.pattern ? b.pattern->priority : 0;
            if (priority_a != priority_b) {
                return priority_a > priority_b;
            }

            size_t span_a = (a.end_pos > a.start_pos) ? (a.end_pos - a.start_pos) : 0;
            size_t span_b = (b.end_pos > b.start_pos) ? (b.end_pos - b.start_pos) : 0;
            if (span_a != span_b) {
                return span_a > span_b;
            }

            if (a.pattern && b.pattern) {
                return a.pattern->name < b.pattern->name;
            }
            return false;
        });

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

    // Determine logical start position so we capture the segment that precedes
    // the first anchor (e.g. identifier before ':' in cpp2 declarations).
    size_t logical_start = pos;
    if (!pattern.evidence_types.empty()) {
        while (logical_start > 0) {
            char prev = text[logical_start - 1];
            bool is_delimiter = prev == ';' || prev == '\n' || prev == '\r' ||
                                prev == '{' || prev == '}' || prev == '(' ||
                                prev == ')' || prev == ',';
            if (is_delimiter) {
                break;
            }
            --logical_start;
        }
    }

    match.start_pos = logical_start;

    // Locate all anchors in order starting from logical_start
    std::vector<size_t> anchor_positions;
    anchor_positions.reserve(pattern.alternating_anchors.size());

    size_t search_pos = logical_start;
    for (const auto& anchor : pattern.alternating_anchors) {
        size_t anchor_pos = text.find(anchor, search_pos);
        if (anchor_pos == std::string::npos) {
            return std::nullopt;
        }
        anchor_positions.push_back(anchor_pos);
        search_pos = anchor_pos + anchor.size();
    }

    // Extract any evidence that precedes the first anchor (identifier, etc.)
    size_t evidence_index = 0;
    auto allows_empty_segment = [](std::string_view evidence_type) {
        return evidence_type == "parameters";
    };

    if (pattern.evidence_types.size() > pattern.alternating_anchors.size()) {
        size_t first_anchor_pos = anchor_positions.front();
        std::string leading(text.substr(logical_start, first_anchor_pos - logical_start));

        auto start = leading.find_first_not_of(" \t\n\r");
        auto end = leading.find_last_not_of(" \t\n\r");
        if (start != std::string::npos && end != std::string::npos) {
            leading = leading.substr(start, end - start + 1);
        } else {
            leading.clear();
        }

        const std::string& evidence_type = pattern.evidence_types[evidence_index];
        if (leading.empty() && !allows_empty_segment(evidence_type)) {
            return std::nullopt;
        }

        match.segments.push_back(leading);
        ++evidence_index;
    }

    // Extract evidence segments that follow each anchor
    for (size_t anchor_idx = 0; anchor_idx < pattern.alternating_anchors.size() && evidence_index < pattern.evidence_types.size(); ++anchor_idx) {
        size_t segment_start = anchor_positions[anchor_idx] + pattern.alternating_anchors[anchor_idx].size();
        size_t segment_end = text.size();

        if (anchor_idx + 1 < anchor_positions.size()) {
            segment_end = anchor_positions[anchor_idx + 1];
        } else {
            // Find natural termination for the final segment
            size_t end_pos = segment_start;
            int brace_depth = 0;
            int paren_depth = 0;
            int bracket_depth = 0;

            while (end_pos < text.size()) {
                char ch = text[end_pos];
                if (ch == '{') {
                    ++brace_depth;
                } else if (ch == '}') {
                    if (brace_depth > 0) {
                        --brace_depth;
                    }
                    if (brace_depth == 0 && paren_depth == 0 && bracket_depth == 0) {
                        ++end_pos; // Include closing brace in the span
                        break;
                    }
                    ++end_pos;
                    continue;
                } else if (ch == '(') {
                    ++paren_depth;
                } else if (ch == ')') {
                    if (paren_depth > 0) {
                        --paren_depth;
                    }
                } else if (ch == '[') {
                    ++bracket_depth;
                } else if (ch == ']') {
                    if (bracket_depth > 0) {
                        --bracket_depth;
                    }
                }

                if ((ch == ';' || ch == '\n') && brace_depth == 0 && paren_depth == 0 && bracket_depth == 0) {
                    break;
                }

                ++end_pos;
            }

            segment_end = end_pos;
            match.end_pos = end_pos;
        }

        std::string segment(text.substr(segment_start, segment_end - segment_start));

        auto start = segment.find_first_not_of(" \t\n\r");
        auto end = segment.find_last_not_of(" \t\n\r");
        if (start != std::string::npos && end != std::string::npos) {
            segment = segment.substr(start, end - start + 1);
        } else {
            segment.clear();
        }

        const std::string& evidence_type = pattern.evidence_types[evidence_index];
        if (segment.empty() && !allows_empty_segment(evidence_type)) {
            return std::nullopt;
        }

        match.segments.push_back(segment);
        ++evidence_index;
    }

    if (match.segments.size() != pattern.evidence_types.size()) {
        return std::nullopt;
    }

    if (match.end_pos == 0) {
        match.end_pos = anchor_positions.back() + pattern.alternating_anchors.back().size();
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