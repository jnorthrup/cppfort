#include "tblgen_pattern_matcher.h"
#include <algorithm>
#include <cctype>
#include <iostream>

namespace cppfort::stage0 {

// CONFIX ORBIT TRACKING: Extract segments between anchors without regex
std::string TblgenPatternMatcher::pattern_to_regex(const std::string& pattern) {
    return ""; // Not used
}

std::optional<std::vector<std::string>> TblgenPatternMatcher::match(
    const std::string& pattern,
    const std::string& input
) {
    // Extract anchor literals from pattern (everything that's not $N)
    std::vector<std::string> anchors;
    std::vector<int> segment_indices;

    size_t pos = 0;
    std::string current_anchor;

    while (pos < pattern.size()) {
        if (pattern[pos] == '$' && pos + 1 < pattern.size() && std::isdigit(pattern[pos + 1])) {
            // Found segment placeholder
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

    // Extract segments from input using anchors
    std::vector<std::string> segments(segment_indices.empty() ? 0 : *std::max_element(segment_indices.begin(), segment_indices.end()) + 1);

    size_t input_pos = 0;
    for (size_t i = 0; i < segment_indices.size(); ++i) {
        int seg_idx = segment_indices[i];

        // Find the next anchor (or end of input)
        size_t next_anchor_pos = input.size();
        if (i < anchors.size()) {
            next_anchor_pos = input.find(anchors[i], input_pos);
            if (next_anchor_pos == std::string::npos) {
                return std::nullopt; // Anchor not found
            }
        }

        // Extract segment between current position and anchor
        std::string segment = input.substr(input_pos, next_anchor_pos - input_pos);

        // Trim whitespace
        size_t start = 0;
        while (start < segment.size() && std::isspace(static_cast<unsigned char>(segment[start]))) ++start;
        size_t end = segment.size();
        while (end > start && std::isspace(static_cast<unsigned char>(segment[end - 1]))) --end;
        segment = segment.substr(start, end - start);

        segments[seg_idx] = segment;

        // Advance past the anchor
        if (i < anchors.size()) {
            input_pos = next_anchor_pos + anchors[i].size();
        }
    }

    return segments;
}

} // namespace cppfort::stage0
