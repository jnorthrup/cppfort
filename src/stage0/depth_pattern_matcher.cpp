#include "depth_pattern_matcher.h"
#include "confix_tracker.h"
#include "cpp2_emitter.h"
#include <algorithm>
#include <iostream>
#include <sstream>

namespace cppfort::stage0 {

std::vector<PatternMatch> DepthPatternMatcher::find_matches(
    std::string_view text,
    const std::vector<PatternData>& patterns
) {
    std::vector<PatternMatch> matches;

    // Build confix depth map
    auto depth_map = build_depth_map(text);

    // Sort patterns by first anchor length (longest first) to avoid prefix collisions
    std::vector<const PatternData*> sorted_patterns;
    for (const auto& pattern : patterns) {
        if (pattern.use_alternating && !pattern.alternating_anchors.empty()) {
            // Skip parameter patterns - they should only be applied inside function parameters
            if (pattern.name.find("parameter") != std::string::npos) {
                continue;
            }
            // Skip nested function patterns - they should only be applied inside function bodies recursively
            if (pattern.name.find("nested") != std::string::npos) {
                continue;
            }
            sorted_patterns.push_back(&pattern);
        }
    }
    std::sort(sorted_patterns.begin(), sorted_patterns.end(), [](const PatternData* a, const PatternData* b) {
        // First, prefer patterns with more anchors (more specific)
        if (a->alternating_anchors.size() != b->alternating_anchors.size()) {
            return a->alternating_anchors.size() > b->alternating_anchors.size();
        }
        // Then prefer longer first anchors
        return a->alternating_anchors[0].length() > b->alternating_anchors[0].length();
    });

    // Scan text for each pattern's anchors (already filtered for alternating)
    for (const auto* pattern_ptr : sorted_patterns) {
        const auto& pattern = *pattern_ptr;
        const std::string& first_anchor = pattern.alternating_anchors[0];

        std::cerr << "DEBUG find_matches: Searching for pattern '" << pattern.name << "' with anchor '" << first_anchor << "'\n";

        // Find all occurrences of the first anchor
        size_t pos = 0;
        while ((pos = text.find(first_anchor, pos)) != std::string::npos) {
            std::cerr << "DEBUG find_matches: Found anchor at pos " << pos << "\n";
            // Try to match the pattern at this anchor position
            auto match = try_match(text, pos, pattern, depth_map);
            if (match) {
                std::cerr << "DEBUG find_matches: MATCH! start=" << match->start_pos << " end=" << match->end_pos << "\n";
                matches.push_back(*match);
                pos = match->end_pos; // Skip past this match
            } else {
                pos += first_anchor.length(); // Move past this anchor and keep searching
            }
        }
    }

    // Filter overlapping matches - keep only non-overlapping longest matches
    std::vector<PatternMatch> filtered;
    for (const auto& match : matches) {
        bool overlaps = false;
        for (const auto& existing : filtered) {
            // Check if this match overlaps with an existing match
            if (!(match.end_pos <= existing.start_pos || match.start_pos >= existing.end_pos)) {
                // Overlaps - skip this match (keep first/longer one)
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

std::optional<PatternMatch> DepthPatternMatcher::try_match(
    std::string_view text,
    size_t start_pos,
    const PatternData& pattern,
    const std::vector<int>& depth_map
) {
    // Check if pattern uses alternating anchors
    if (pattern.use_alternating) {
        if (pattern.alternating_anchors.empty()) {
            return std::nullopt;
        }

        const std::string& first_anchor = pattern.alternating_anchors[0];

        // Check if first anchor is at this position
        if (text.substr(start_pos).find(first_anchor) != 0) {
            // Anchor not at start position
            return std::nullopt;
        }

        // Extract segments and get actual end position
        auto [segments, actual_end_pos] = extract_segments(text, pattern, start_pos);
        std::cerr << "DEBUG try_match: pattern=" << pattern.name << ", segments.size()=" << segments.size() << ", expected=" << pattern.evidence_types.size() << "\n";
        if (segments.empty() || segments.size() != pattern.evidence_types.size()) {
            std::cerr << "DEBUG try_match: FAILED (size mismatch)\n";
            return std::nullopt;
        }

        // Determine start and end positions for the match
        // If evidence comes before first anchor, match starts at beginning of that evidence
        bool has_evidence_before = (pattern.evidence_types.size() > pattern.alternating_anchors.size());
        bool is_parameter = (pattern.name.find("parameter") != std::string::npos);
        bool is_nested_function = (pattern.name.find("nested_function") != std::string::npos);

        size_t match_start = start_pos;
        if (has_evidence_before && !is_parameter && !is_nested_function) {
            match_start = 0;
        } else if (has_evidence_before && (is_parameter || is_nested_function) && !segments.empty()) {
            // For parameters and nested functions, match starts where the identifier begins
            // Find the start of the identifier (first segment) before the anchor
            size_t identifier_len = segments[0].length();
            if (identifier_len > 0 && start_pos >= identifier_len) {
                match_start = start_pos - identifier_len;
                // Skip leading whitespace before identifier
                while (match_start > 0 && std::isspace(text[match_start - 1])) match_start--;
            }
        }

        size_t end_pos = actual_end_pos;

        // Validate evidence doesn't cross boundaries
        if (!evidence_valid(text, match_start, end_pos, depth_map)) {
            return std::nullopt;
        }

        // Create match
        PatternMatch match;
        match.start_pos = match_start;
        match.end_pos = std::min(end_pos, text.size());
        match.pattern = &pattern;
        match.segments = segments;
        match.depth = (match_start < depth_map.size()) ? depth_map[match_start] : 0;

        return match;
    }

    // Segment-based pattern (legacy)
    if (!pattern.signature_patterns.empty()) {
        for (const auto& sig : pattern.signature_patterns) {
            if (text.substr(start_pos).find(sig) == 0) {
                // Signature matches - extract segments would go here
                // For now, just indicate signature present
                return std::nullopt; // TODO: implement segment extraction
            }
        }
    }

    return std::nullopt;
}

std::pair<std::vector<std::string>, size_t> DepthPatternMatcher::extract_segments(
    std::string_view text,
    const PatternData& pattern,
    size_t anchor_pos
) {
    std::vector<std::string> segments;

    if (!pattern.use_alternating || pattern.alternating_anchors.empty()) {
        return {segments, anchor_pos};
    }

    const std::string& first_anchor = pattern.alternating_anchors[0];
    std::cerr << "DEBUG extract_segments: pattern=" << pattern.name << " anchor='" << first_anchor << "' text='" << text << "' anchor_pos=" << anchor_pos << "\n";

    // Special case for parameter patterns: look backward from anchor for evidence
    bool is_parameter_pattern = (pattern.name.find("parameter") != std::string::npos);

    // Check if we need evidence BEFORE first anchor
    size_t evidence_start_idx = 0;
    if (pattern.evidence_types.size() > pattern.alternating_anchors.size()) {
        // Extract evidence before first anchor
        size_t before_start = 0;
        size_t before_end = anchor_pos;

        // For parameter patterns inside function params, search backward within parens
        if (is_parameter_pattern) {
            // Find start by looking backward for '(' or ','
            before_start = anchor_pos;
            while (before_start > 0) {
                char ch = text[before_start - 1];
                if (ch == '(' || ch == ',') break;
                before_start--;
            }
        } else {
            // For variable patterns, search backward for statement boundary
            // Stop at {, ;, or , to avoid capturing previous statements or function signature
            before_start = anchor_pos;
            while (before_start > 0) {
                char ch = text[before_start - 1];
                if (ch == '{' || ch == ';' || ch == ',') break;
                before_start--;
            }
        }

        std::string before = std::string(text.substr(before_start, before_end - before_start));
        // Trim whitespace
        size_t start = 0;
        while (start < before.size() && std::isspace(static_cast<unsigned char>(before[start]))) ++start;
        size_t end = before.size();
        while (end > start && std::isspace(static_cast<unsigned char>(before[end - 1]))) --end;
        before = before.substr(start, end - start);

        std::cerr << "DEBUG extract_segments: evidence_before='" << before << "' (from " << before_start << " to " << before_end << ")\n";

        segments.push_back(before);
        evidence_start_idx = 1;
    }

    size_t current_pos = anchor_pos + first_anchor.length();

    // Extract evidence spans between anchors
    for (size_t i = evidence_start_idx; i < pattern.evidence_types.size(); ++i) {
        size_t next_anchor_pos = std::string::npos;
        size_t anchor_idx = i - evidence_start_idx + 1;

        if (anchor_idx < pattern.alternating_anchors.size()) {
            const std::string& next_anchor = pattern.alternating_anchors[anchor_idx];
            next_anchor_pos = text.find(next_anchor, current_pos);

            // If this anchor is required but not found, the pattern doesn't match
            if (next_anchor_pos == std::string::npos) {
                return {{}, anchor_pos}; // Return empty segments to signal no match
            }
        }

        size_t evidence_end = (next_anchor_pos != std::string::npos) ? next_anchor_pos : text.length();

        // For patterns with no next anchor, scan forward to find statement boundary
        // Stop at semicolon or closing brace to avoid crossing statement boundaries
        if (next_anchor_pos == std::string::npos && !is_parameter_pattern) {
            evidence_end = current_pos;
            int depth = 0;
            while (evidence_end < text.length()) {
                char ch = text[evidence_end];
                if (ch == '(' || ch == '[' || ch == '{') depth++;
                else if (ch == ')' || ch == ']' || ch == '}') {
                    if (depth == 0) break; // Hit closing delimiter at same scope - stop
                    depth--;
                }
                else if (ch == ';' && depth == 0) {
                    // Hit semicolon at statement level - include it and stop
                    evidence_end++;
                    break;
                }
                evidence_end++;
            }
        }

        std::string evidence = std::string(text.substr(current_pos, evidence_end - current_pos));

        // Trim whitespace
        size_t start = 0;
        while (start < evidence.size() && std::isspace(static_cast<unsigned char>(evidence[start]))) ++start;
        size_t end = evidence.size();
        while (end > start && std::isspace(static_cast<unsigned char>(evidence[end - 1]))) --end;
        evidence = evidence.substr(start, end - start);

        // Strip trailing semicolon from segment, but track actual position
        if (!evidence.empty() && evidence.back() == ';') {
            evidence.pop_back();
        }

        std::cerr << "DEBUG extract_segments: evidence_after='" << evidence << "' (from " << current_pos << " to " << evidence_end << ")\n";

        // Validate confix balance for this evidence segment
        // For body segments (which include braces), the balance should be checked correctly
        const std::string& evidence_type = pattern.evidence_types[i];
        bool is_body = (evidence_type == "body");

        // Only check balance for non-body segments or check full balance for body
        if (!is_body) {
            int balance = 0;
            for (char ch : evidence) {
                if (ch == '(' || ch == '[' || ch == '{' || ch == '<') balance++;
                if (ch == ')' || ch == ']' || ch == '}' || ch == '>') {
                    balance--;
                    if (balance < 0) {
                        std::cerr << "DEBUG extract_segments: REJECTED - negative balance (crossed boundary)\n";
                        return {{}, anchor_pos}; // Crossed scope boundary - invalid span
                    }
                }
            }
        } else {
            // For body segments, just check overall balance (should be 0 at end)
            int balance = 0;
            for (char ch : evidence) {
                if (ch == '(' || ch == '[' || ch == '{') balance++;
                if (ch == ')' || ch == ']' || ch == '}') balance--;
            }
            if (balance != 0) {
                std::cerr << "DEBUG extract_segments: REJECTED - unbalanced body (balance=" << balance << ")\n";
                return {{}, anchor_pos};
            }
        }

        segments.push_back(evidence);
        current_pos = evidence_end;

        if (next_anchor_pos != std::string::npos && anchor_idx < pattern.alternating_anchors.size()) {
            current_pos += pattern.alternating_anchors[anchor_idx].length();
        }
    }

    // current_pos now points to the actual end of extraction in the original text
    // Scan forward to include any trailing semicolon/whitespace
    while (current_pos < text.size() && (text[current_pos] == ';' || std::isspace(text[current_pos]))) {
        current_pos++;
        if (text[current_pos - 1] == ';') break; // Stop after semicolon
    }

    return {segments, current_pos};
}

bool DepthPatternMatcher::anchors_present(
    std::string_view text,
    size_t pos,
    const PatternData& pattern
) {
    if (pattern.use_alternating && !pattern.alternating_anchors.empty()) {
        return text.substr(pos).find(pattern.alternating_anchors[0]) == 0;
    }

    if (!pattern.signature_patterns.empty()) {
        for (const auto& sig : pattern.signature_patterns) {
            if (text.substr(pos).find(sig) == 0) {
                return true;
            }
        }
    }

    return false;
}

bool DepthPatternMatcher::evidence_valid(
    std::string_view text,
    size_t start,
    size_t end,
    const std::vector<int>& depth_map
) {
    if (start >= depth_map.size() || end > depth_map.size()) {
        return false;
    }

    // Evidence is valid if it doesn't cross to a different depth level
    int start_depth = depth_map[start];

    // For top-level patterns (depth 0), allow any depth variation
    // This is necessary for matching functions that contain nested structures
    if (start_depth == 0) {
        return true;
    }

    for (size_t i = start; i < end && i < depth_map.size(); ++i) {
        // Allow depth to vary within reasonable bounds (nested confixes)
        // But not completely different scope
        if (std::abs(depth_map[i] - start_depth) > 2) {
            return false;
        }
    }

    return true;
}

std::vector<PatternMatch> DepthPatternMatcher::find_matches_recursive(
    std::string_view text,
    const std::vector<PatternData>& patterns,
    int max_depth
) {
    // Base case: if max depth reached, return normal matches
    if (max_depth <= 0) {
        return find_matches(text, patterns);
    }

    // First, get the basic matches
    auto matches = find_matches(text, patterns);

    // Sort matches by depth (deepest first) for inside-out transformation
    std::sort(matches.begin(), matches.end(), [](const PatternMatch& a, const PatternMatch& b) {
        // Sort by depth descending (deepest matches first)
        // If depths are equal, sort by length descending to process larger nested patterns first
        if (a.depth != b.depth) {
            return a.depth > b.depth;  // Deeper matches first
        }
        return (a.end_pos - a.start_pos) > (b.end_pos - b.start_pos);  // Longer matches first if same depth
    });

    // Apply recursive transformations to segments
    for (auto& match : matches) {
        if (match.segments.size() > 0) {
            match.segments = apply_recursive_segment_transformations(
                match.segments, patterns, 0, max_depth - 1
            );
        }
    }

    return matches;
}

std::vector<std::string> DepthPatternMatcher::apply_recursive_segment_transformations(
    const std::vector<std::string>& segments,
    const std::vector<PatternData>& patterns,
    int current_depth,
    int max_depth
) {
    if (current_depth >= max_depth) {
        // Maximum depth reached, return segments as-is
        return segments;
    }

    std::vector<std::string> result;
    
    for (const auto& segment : segments) {
        // For each segment, recursively apply pattern matching and transformation
        std::string transformed = segment;
        
        // Only proceed with transformation if segment is not empty
        if (!segment.empty()) {
            CPP2Emitter emitter;
            std::ostringstream temp_stream;
            
            // Apply depth-based emission recursively to this segment with limited depth
            emitter.emit_depth_based(segment, temp_stream, patterns);
            transformed = temp_stream.str();
            
            // Apply nested transformations within this segment if we haven't reached max depth
            if (transformed != segment && current_depth + 1 < max_depth) {
                // For nested transformations, recursively apply to the transformed segment
                std::vector<std::string> nested_segments = {transformed};
                auto nested_result = apply_recursive_segment_transformations(
                    nested_segments, patterns, current_depth + 1, max_depth
                );
                if (!nested_result.empty()) {
                    transformed = nested_result[0];
                }
            }
        }
        
        result.push_back(transformed);
    }

    return result;
}

} // namespace cppfort::stage0
