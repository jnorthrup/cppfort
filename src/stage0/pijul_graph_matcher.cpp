#include "pijul_graph_matcher.h"

#include <algorithm>
#include <cstring>
#include <iostream>

namespace cppfort::stage0 {

PijulGraphMatcher::PijulGraphMatcher(const PatternData& pattern)
  : pattern_(pattern) {
}

std::vector<UnifiedPatternMatch> PijulGraphMatcher::find_matches(std::string_view text) const {
    std::vector<UnifiedPatternMatch> out;
    if (text.empty()) return out;

    // Support alternating anchors first
    if (!pattern_.use_alternating || pattern_.alternating_anchors.empty()) {
        // For now, fall back to signature pattern matching (first signature) as a minimal graph match
        if (!pattern_.signature_patterns.empty()) {
            const auto& sig = pattern_.signature_patterns.front();
            size_t pos = text.find(sig);
            if (pos != std::string_view::npos) {
                UnifiedPatternMatch match;
                match.start_pos = pos;
                match.end_pos = pos + sig.size();
                match.pattern = &pattern_;
                match.semantic_label = pattern_.name; // Pattern name as semantic label
                match.orbit_label = std::to_string(pattern_.orbit_id);
                out.push_back(match);
            }
        }
        return out;
    }

    // Alternating anchors: find occurrences of the first anchor then validate subsequent anchors and basic evidence
    const std::string& first_anchor = pattern_.alternating_anchors.front();
    size_t pos = 0;
    while (true) {
        size_t found = text.find(first_anchor, pos);
        if (found == std::string::npos) break;

        size_t cursor = found + first_anchor.size();
        bool ok = true;
        // For each subsequent anchor, find the next occurrence after the cursor
        for (size_t i = 1; i < pattern_.alternating_anchors.size(); ++i) {
            const auto& anchor = pattern_.alternating_anchors[i];
            size_t next = text.find(anchor, cursor);
            if (next == std::string::npos) {
                ok = false;
                break;
            }
            // Basic evidence trimming and simple validation (non-empty)
            size_t evidence_start = cursor;
            size_t evidence_end = next;
            if (evidence_end <= evidence_start) {
                ok = false;
                break;
            }
            // Accept any non-empty evidence for now
            cursor = next + anchor.size();
        }
        if (ok) {
            UnifiedPatternMatch match;
            match.start_pos = found;
            match.end_pos = cursor;
            match.pattern = &pattern_;
            match.semantic_label = pattern_.name;
            match.orbit_label = std::to_string(pattern_.orbit_id);
            out.push_back(match);
        }
        pos = found + 1;
    }

    return out;
}

std::vector<UnifiedPatternMatch> PijulGraphMatcher::find_matches_in_region(std::string_view text, size_t regionStart, size_t regionEnd) const {
    std::vector<UnifiedPatternMatch> out;
    if (regionStart >= text.size() || regionStart >= regionEnd) return out;

    std::string_view regionText = text.substr(regionStart, std::min(regionEnd, text.size()) - regionStart);
    auto matches = find_matches(regionText);
    for (auto& m : matches) {
        // Adjust positions to the original text space
        m.start_pos += regionStart;
        m.end_pos += regionStart;
        // Orbit label: inherit or augment
        if (m.orbit_label.empty()) m.orbit_label = std::to_string(pattern_.orbit_id);
        out.push_back(m);
    }
    return out;
}

} // namespace cppfort::stage0
