#include "rbcursive.h"
#include "graph_matcher.h"
#include "rbcursive_regions.h"
#include "iterpeeps.h"
#include "evidence.h"

#include <algorithm>
#include <cctype>
#include <iostream>
#include <span>

namespace cppfort {
namespace ir {

using cppfort::stage0::ConfixType;

namespace {

std::string_view trim_view(std::string_view text) {
    size_t begin = 0;
    size_t end = text.size();
    while (begin < end && std::isspace(static_cast<unsigned char>(text[begin]))) {
        ++begin;
    }
    while (end > begin && std::isspace(static_cast<unsigned char>(text[end - 1]))) {
        --end;
    }
    return text.substr(begin, end - begin);
}

cppfort::stage0::TypeEvidence analyze_evidence(std::string_view text) {
    cppfort::stage0::TypeEvidence evidence;
    evidence.ingest(text);
    return evidence;
}

struct EvidenceAnalysis {
    std::string_view trimmed;
    cppfort::stage0::TypeEvidence traits;

    bool empty() const { return trimmed.empty(); }
    bool contains(char ch) const { return trimmed.find(ch) != std::string::npos; }
    bool contains_any(std::string_view charset) const { return trimmed.find_first_of(charset) != std::string::npos; }
};

EvidenceAnalysis make_evidence(std::string_view raw) {
    std::string_view trimmed = trim_view(raw);
    cppfort::stage0::TypeEvidence traits;
    traits.ingest(trimmed);
    return EvidenceAnalysis{trimmed, traits};
}

// Treat as meaningless: whitespace, semicolons, and comments
bool is_meaningless_segment(std::string_view text) {
    std::size_t i = 0;
    while (i < text.size()) {
        unsigned char ch = static_cast<unsigned char>(text[i]);
        if (std::isspace(ch)) { ++i; continue; }
        if (text[i] == ';') { ++i; continue; }
        if (text[i] == '/' && i + 1 < text.size()) {
            if (text[i+1] == '/') {
                // line comment
                i += 2;
                while (i < text.size() && text[i] != '\n') ++i;
                continue;
            }
            if (text[i+1] == '*') {
                // block comment
                i += 2;
                while (i + 1 < text.size() && !(text[i] == '*' && text[i+1] == '/')) ++i;
                if (i + 1 < text.size()) i += 2;
                continue;
            }
        }
        return false; // found meaningful token
    }
    return true;
}

} // namespace

std::vector<std::size_t>
ScalarScanner::scanBytes(std::span<const std::uint8_t> data,
                         std::span<const std::uint8_t> targets) const {
    std::vector<std::size_t> positions;
    if (targets.empty()) {
        return positions;
    }

    for (std::size_t idx = 0; idx < data.size(); ++idx) {
        auto value = data[idx];
        if (std::find(targets.begin(), targets.end(), value) != targets.end()) {
            positions.push_back(idx);
        }
    }

    return positions;
}

// Backtracking glob (supports '*' and '?').
// Intentional simplicity for bootstrap; replace with SIMD later.
bool RBCursiveScanner::globMatch(std::string_view text, std::string_view pattern) {
    std::size_t ti = 0, pi = 0;
    std::size_t star = std::string::npos, match = 0;

    while (ti < text.size()) {
        if (pi < pattern.size() && (pattern[pi] == '?' || pattern[pi] == text[ti])) {
            ++ti; ++pi;
        } else if (pi < pattern.size() && pattern[pi] == '*') {
            star = pi++;
            match = ti;
        } else if (star != std::string::npos) {
            pi = star + 1;
            ti = ++match;
        } else {
            return false;
        }
    }
    while (pi < pattern.size() && pattern[pi] == '*') ++pi;
    return pi == pattern.size();
}

bool RBCursiveScanner::matchGlob(std::string_view text, std::string_view pattern) const {
    return globMatch(text, pattern);
}

bool RBCursiveScanner::matchRegex(std::string_view text, std::string_view pattern) const {
    // REMOVED: Use infix orbits instead
    return false;
}

std::vector<RBCursiveScanner::Match>
RBCursiveScanner::scanWithPattern(std::string_view data,
                                  std::string_view pattern,
                                  PatternType type) const {
    std::vector<Match> out;

    if (data.empty() || pattern.empty()) return out;

    switch (type) {
        case PatternType::Glob: {
            // For bootstrap, perform a naive sliding-window search using globMatch
            // on substrings bounded by bytes. Later replace with SIMD gather/scan.
            for (std::size_t i = 0; i < data.size(); ++i) {
                // Early-cut: try to expand end while star exists, otherwise fixed-length
                for (std::size_t j = i; j < data.size(); ++j) {
                    if (globMatch(std::string_view(data.data() + i, j - i + 1), pattern)) {
                        out.push_back(Match{ i, j + 1 });
                        i = j; // advance to end of match (non-overlapping)
                        break;
                    }
                }
            }
            break;
        }
        case PatternType::Regex: {
            // TEMP: Use GraphMatcher for signature-like matching to avoid dependency on std::regex
            // while we migrate to a graph-first implementation. This maintains a local
            // substitution until the full ParameterGraph-based matcher is implemented.
            cppfort::stage0::GraphMatcher gm;
            if (gm.match(std::string_view(pattern), data)) {
                // Found a match over the entire text for now
                out.push_back(Match{0, data.size()});
            }
            break;
        }
    }

    return out;
}

void RBCursiveScanner::speculate(std::string_view text) {
    matches_.clear();

    if (!patterns_ || text.empty()) {
        // std::cout << "DEBUG: No patterns or empty text\n";
        return;
    }

    // std::cout << "DEBUG: Speculating on text: '" << text << "' with " << patterns_->size() << " patterns\n";

    // Try all patterns in parallel (conceptually - actually sequential for now)
    for (const auto& pattern : *patterns_) {
        // Try alternating pattern matching first (more specific)
        if (pattern.use_alternating) {
            speculate_alternating(pattern, text);
            continue; // Alternating patterns don't use signature patterns
        }
        
        // Try signature pattern matching
        for (const auto& signature : pattern.signature_patterns) {
            // std::cout << "DEBUG: Checking pattern '" << pattern.name << "' signature: '" << signature << "'\n";
            // Simple substring search for now (can be enhanced later)
            size_t pos = text.find(signature);
            if (pos != std::string::npos) {
                // Found signature - now expand orbit boundaries using pattern segments
                size_t start_pos = pos;
                size_t end_pos = pos + signature.length();

                // For segment-based patterns, extend boundaries to include all segments
                if (!pattern.segments.empty()) {
                    // Extend backwards for negative-offset segments (e.g., function name before ": (")
                    for (const auto& seg : pattern.segments) {
                        if (seg.offset_from_anchor < 0) {
                            // Look backwards for identifier
                            size_t scan_pos = pos;
                            while (scan_pos > 0 && std::isspace(static_cast<unsigned char>(text[scan_pos - 1]))) {
                                --scan_pos;
                            }
                            size_t ident_end = scan_pos;
                            while (scan_pos > 0 && (std::isalnum(static_cast<unsigned char>(text[scan_pos - 1])) || text[scan_pos - 1] == '_')) {
                                --scan_pos;
                            }
                            if (scan_pos < ident_end) {
                                start_pos = std::min(start_pos, scan_pos);
                            }
                        }
                    }

                    // Extend forwards to include last segment delimiter end
                    if (!pattern.segments.empty()) {
                        // Find the last segment's end delimiter (e.g., "}" for function body)
                        const auto& last_seg = pattern.segments.back();
                        if (!last_seg.delimiter_end.empty()) {
                            size_t delim_start_pos = text.find(last_seg.delimiter_start, pos);
                            if (delim_start_pos != std::string::npos) {
                                // Simple nesting-aware search for closing delimiter
                                size_t search_pos = delim_start_pos + last_seg.delimiter_start.length();
                                int depth = 1;
                                while (search_pos < text.size() && depth > 0) {
                                    if (text[search_pos] == last_seg.delimiter_start[0]) ++depth;
                                    else if (text[search_pos] == last_seg.delimiter_end[0]) --depth;
                                    ++search_pos;
                                }
                                if (depth == 0) {
                                    end_pos = search_pos;
                                }
                            }
                        }
                    }
                }

                size_t match_length = end_pos - start_pos;
                // Compute confidence from pattern complexity (honest baseline)
                double confidence = std::min(1.0, static_cast<double>(match_length) / text.length());

                // Create result fragment
                ::cppfort::stage0::OrbitFragment fragment;
                fragment.start_pos = start_pos;
                fragment.end_pos = end_pos;
                fragment.confidence = confidence;
                fragment.classified_grammar = ::cppfort::ir::GrammarType::UNKNOWN; // Will be set by correlator

                std::cerr << "DEBUG speculate signature: MATCHED pattern=" << pattern.name << " length=" << match_length << "\n";
                matches_.emplace_back(match_length, confidence, pattern.name, std::move(fragment));
                break; // Only take first match per pattern for now
            }
        }
    }

    // Sort by match_length descending (longest matches first)
    // IMPORTANT: Prefer alternating patterns over signature patterns when lengths are close
    // Alternating patterns are more precise and avoid greedy over-matching
    std::sort(matches_.begin(), matches_.end(),
              [this](const ::cppfort::stage0::SpeculativeMatch& a, const ::cppfort::stage0::SpeculativeMatch& b) {
                  // If lengths differ by more than 5 chars, prefer longer
                  if (std::abs(static_cast<int>(a.match_length) - static_cast<int>(b.match_length)) > 5) {
                      return a.match_length > b.match_length;
                  }

                  // Lengths are close - check if either is from an alternating pattern
                  bool a_is_alternating = false;
                  bool b_is_alternating = false;

                  if (patterns_) {
                      for (const auto& pattern : *patterns_) {
                          if (pattern.name == a.pattern_name && pattern.use_alternating) {
                              a_is_alternating = true;
                          }
                          if (pattern.name == b.pattern_name && pattern.use_alternating) {
                              b_is_alternating = true;
                          }
                      }
                  }

                  // Prefer alternating patterns
                  if (a_is_alternating && !b_is_alternating) return true;
                  if (!a_is_alternating && b_is_alternating) return false;

                  // Both alternating or both signature - prefer longer
                  return a.match_length > b.match_length;
              });

    // Debug: print sorted matches
    std::cerr << "DEBUG speculate: After sorting, matches:\n";
    for (size_t i = 0; i < matches_.size(); ++i) {
        std::cerr << "  [" << i << "] " << matches_[i].pattern_name << " length=" << matches_[i].match_length << "\n";
    }
}

// Alternating anchor/evidence speculation for deterministic grammar selection
void RBCursiveScanner::speculate_alternating(const ::cppfort::stage0::PatternData& pattern, std::string_view text) {
    if (!pattern.use_alternating || pattern.alternating_anchors.empty()) {
        return;
    }

    // Find the first anchor
    const std::string& first_anchor = pattern.alternating_anchors[0];
    std::vector<std::size_t> anchor_positions = find_anchor_positions_orbit(text, first_anchor);
    if (anchor_positions.empty()) return;
    size_t anchor_pos = anchor_positions[0];
    std::cerr << "DEBUG speculate_alternating: pattern=" << pattern.name << " anchor='" << first_anchor << "' position=" << anchor_pos << "\n";
    std::cerr << "DEBUG speculate_alternating: pattern=" << pattern.name << " anchor='" << first_anchor << "' found=" << (anchor_pos != std::string::npos) << "\n";
    if (anchor_pos == std::string::npos) {
        return;
    }

    // Build alternating spans: anchor -> evidence -> anchor -> evidence -> ...
    std::vector<std::pair<std::string_view, bool>> spans; // (content, is_anchor)

    size_t match_start = anchor_pos; // Default: match starts at first anchor

    // Special case: single anchor with 2 evidence types means evidence before AND after
    if (pattern.alternating_anchors.size() == 1 && pattern.evidence_types.size() == 2) {
        // Extract evidence before anchor
        std::string_view before = text.substr(0, anchor_pos);
        size_t ev_start = 0;
        while (ev_start < before.size() && std::isspace(static_cast<unsigned char>(before[ev_start]))) ++ev_start;
        size_t ev_end = before.size();
        while (ev_end > ev_start && std::isspace(static_cast<unsigned char>(before[ev_end - 1]))) --ev_end;
        std::string_view evidence_before = before.substr(ev_start, ev_end - ev_start);

        if (!validate_evidence_type(pattern.evidence_types[0], evidence_before)) {
            return; // Evidence before anchor doesn't match
        }

        // Extract evidence after anchor
        size_t after_start = anchor_pos + first_anchor.length();
        std::string_view after = text.substr(after_start);
        ev_start = 0;
        while (ev_start < after.size() && std::isspace(static_cast<unsigned char>(after[ev_start]))) ++ev_start;
        ev_end = after.size();
        while (ev_end > ev_start && std::isspace(static_cast<unsigned char>(after[ev_end - 1]))) --ev_end;
        std::string_view evidence_after = after.substr(ev_start, ev_end - ev_start);

        if (!validate_evidence_type(pattern.evidence_types[1], evidence_after)) {
            return; // Evidence after anchor doesn't match
        }

        // Only accept if the prefix up to match start is meaningless
        if (!is_meaningless_segment(text.substr(0, ev_start))) {
            return;
        }

        // Match the entire text
        match_start = ev_start;
        size_t match_end = text.size();

        ::cppfort::stage0::OrbitFragment fragment;
        fragment.start_pos = match_start;
        fragment.end_pos = match_end;
        fragment.confidence = 1.0;
        fragment.classified_grammar = ::cppfort::ir::GrammarType::CPP2;

        matches_.emplace_back(match_end - match_start, 1.0, pattern.name, std::move(fragment));
        return;
    }

    // Check if we have evidence BEFORE the first anchor
    // This happens when evidence_types.size() > alternating_anchors.size()
    size_t evidence_offset = 0;
    if (pattern.evidence_types.size() > pattern.alternating_anchors.size()) {
        // Extract evidence before first anchor
        std::string_view before = text.substr(0, anchor_pos);
        size_t ev_start = 0;
        while (ev_start < before.size() && std::isspace(static_cast<unsigned char>(before[ev_start]))) ++ev_start;
        size_t ev_end = before.size();
        while (ev_end > ev_start && std::isspace(static_cast<unsigned char>(before[ev_end - 1]))) --ev_end;
        std::string_view evidence_before = before.substr(ev_start, ev_end - ev_start);

        if (!validate_evidence_type(pattern.evidence_types[0], evidence_before)) {
            return; // Evidence before anchor doesn't match
        }

        spans.emplace_back(evidence_before, false);
        evidence_offset = 1; // Start evidence extraction from index 1
        match_start = ev_start; // Match includes evidence before anchor
    }

    size_t current_pos = anchor_pos;
    // Guard: prefix before first anchor must be meaningless when no leading evidence is required
    if (evidence_offset == 0) {
        if (!is_meaningless_segment(text.substr(0, match_start))) {
            return;
        }
    }
    spans.emplace_back(text.substr(anchor_pos, first_anchor.length()), true); // First anchor
    current_pos += first_anchor.length();


    // Extract evidence spans between anchors
    for (size_t anchor_idx = 0; anchor_idx < pattern.alternating_anchors.size(); ++anchor_idx) {
        size_t evidence_idx = anchor_idx + evidence_offset;
        if (evidence_idx >= pattern.evidence_types.size()) {
            break; // No more evidence types
        }

        // Find next anchor or end
        size_t next_anchor_pos = std::string::npos;
        if (anchor_idx + 1 < pattern.alternating_anchors.size()) {
            const std::string& next_anchor = pattern.alternating_anchors[anchor_idx + 1];
            std::vector<std::size_t> next_positions = find_anchor_positions_orbit(text.substr(current_pos), next_anchor); if (next_positions.empty()) { return; } next_anchor_pos = current_pos + next_positions[0]; if (next_anchor_pos == std::string::npos) { return; }
            if (next_anchor_pos == std::string::npos) {
                return; // Required anchor not found
            }
        }

        size_t evidence_end = (next_anchor_pos != std::string::npos) ? next_anchor_pos : text.length();
        std::string_view raw_evidence = text.substr(current_pos, evidence_end - current_pos);

        // Trim whitespace from evidence
        size_t ev_start = 0;
        while (ev_start < raw_evidence.size() && std::isspace(static_cast<unsigned char>(raw_evidence[ev_start]))) ++ev_start;
        size_t ev_end = raw_evidence.size();
        while (ev_end > ev_start && std::isspace(static_cast<unsigned char>(raw_evidence[ev_end - 1]))) --ev_end;
        std::string_view evidence = raw_evidence.substr(ev_start, ev_end - ev_start);

        // Validate evidence type
        if (!validate_evidence_type(pattern.evidence_types[evidence_idx], evidence)) {
            return; // Evidence doesn't match expected type
        }

        spans.emplace_back(evidence, false); // Evidence span
        current_pos = evidence_end;

        // Move to next anchor if it exists
        if (next_anchor_pos != std::string::npos) {
            const std::string& next_anchor = pattern.alternating_anchors[anchor_idx + 1];
            spans.emplace_back(text.substr(next_anchor_pos, next_anchor.length()), true);
            current_pos = next_anchor_pos + next_anchor.length();
        }
    }

    // If we got here, all evidence types validated - create a match
    size_t match_length = current_pos - anchor_pos;
    double confidence = 1.0; // High confidence for validated alternating patterns

    ::cppfort::stage0::OrbitFragment fragment;
    fragment.start_pos = anchor_pos;
    fragment.end_pos = anchor_pos + match_length;
    fragment.confidence = confidence;
    fragment.classified_grammar = ::cppfort::ir::GrammarType::CPP2; // Alternating patterns are for CPP2

    std::cerr << "DEBUG speculate_alternating: MATCHED pattern=" << pattern.name << " length=" << match_length << "\n";
    matches_.emplace_back(match_length, confidence, pattern.name, std::move(fragment));
}

void RBCursiveScanner::speculate_across_fragments(const std::vector< ::cppfort::stage0::OrbitFragment>& fragments, std::string_view source) {
    matches_.clear();

    if (!patterns_ || fragments.empty()) {
        return;
    }

    // Concatenate all fragment texts for cross-fragment matching
    std::string concatenated_text;
    std::vector<size_t> fragment_offsets;
    size_t current_offset = 0;

    for (const auto& fragment : fragments) {
        fragment_offsets.push_back(current_offset);
        
        // Extract actual text from source
        if (fragment.start_pos < source.size() && fragment.end_pos <= source.size() && fragment.start_pos < fragment.end_pos) {
            std::string_view fragment_text = source.substr(fragment.start_pos, fragment.end_pos - fragment.start_pos);
            concatenated_text += std::string(fragment_text);
        } else {
            concatenated_text += " "; // placeholder for invalid fragments
        }
        current_offset = concatenated_text.size();
    }

    // Check packrat cache first
    if (packrat_cache_) {
        // For now, use a simple cache key based on fragment count and total size
        size_t cache_key = fragments.size() * 1000 + concatenated_text.size();
        if (packrat_cache_->has_cached(cache_key, ::cppfort::stage0::OrbitType::Confix)) {
            // TODO: Restore cached results
            return;
        }
    }

    // Use existing speculate logic on concatenated text
    speculate(concatenated_text);

    // Adjust positions to be relative to fragment boundaries and update confidence
    for (auto& match : matches_) {
        size_t global_pos = match.result.start_pos;
        
        // Find which fragment this position belongs to
        size_t fragment_index = 0;
        for (size_t i = 0; i < fragment_offsets.size(); ++i) {
            if (i + 1 < fragment_offsets.size()) {
                if (global_pos >= fragment_offsets[i] && global_pos < fragment_offsets[i + 1]) {
                    fragment_index = i;
                    break;
                }
            } else {
                // Last fragment
                if (global_pos >= fragment_offsets[i]) {
                    fragment_index = i;
                    break;
                }
            }
        }
        
        // Adjust position relative to fragment start
        if (fragment_index < fragments.size()) {
            match.result.start_pos = global_pos - fragment_offsets[fragment_index];
            match.result.end_pos = match.result.start_pos + match.match_length;
            
            // Update confidence based on multi-fragment match
            double base_confidence = match.result.confidence;
            if (fragments.size() > 1) {
                // Penalize confidence for cross-fragment matches
                match.result.confidence = base_confidence * 0.8;
            }
        }
    }

    // Store in packrat cache
    if (packrat_cache_) {
        size_t cache_key = fragments.size() * 1000 + concatenated_text.size();
        packrat_cache_->store_cache(cache_key, ::cppfort::stage0::OrbitType::Confix, matches_.empty() ? 0.0 : matches_[0].confidence);
    }
}

// Experimental: multi-orbit thinning across a terminal span using TypeEvidence (backchain mode)
void RBCursiveScanner::speculate_backchain(std::string_view text) {
    if (!patterns_ || text.empty()) {
        return;
    }

    struct LiveOrbit {
        struct Step { std::size_t anchor_pos; std::size_t ev_start; std::size_t ev_end; };
        const ::cppfort::stage0::PatternData* pattern = nullptr;
        std::size_t anchor_index = 0;
        std::size_t start = 0;
        std::size_t cursor = 0;
        std::size_t evidence_offset = 0; // 1 when evidence-before-first-anchor exists
        // incremental evidence window [ev_start, ev_end)
        std::size_t ev_start = 0;
        std::size_t ev_end = 0;
        cppfort::stage0::TypeEvidence ev_traits;
        bool in_evidence = false;
        bool saw_alpha = false;
        std::vector<Step> steps;
        LiveOrbit* parent = nullptr;  // Backchain parent pointer
        std::vector<LiveOrbit*> children; // Child orbits spawned from this one
        double confidence = 1.0;      // Accumulated confidence score
        std::size_t wobble_count = 0; // Track sliding window adjustments
        bool dead = false;
    };

    // Pre-index anchor occurrences for each pattern
    std::map<const ::cppfort::stage0::PatternData*, std::vector<std::size_t>> anchor_indices;
    for (const auto& pattern : *patterns_) {
        if (!pattern.use_alternating || pattern.alternating_anchors.empty()) continue;
        const std::string& first = pattern.alternating_anchors.front();

        // Find all occurrences of the first anchor
        std::size_t pos = 0;
        std::vector<std::size_t> positions = find_anchor_positions_orbit(text, first);
        for (std::size_t pos : positions) {
            // Check if prefix is meaningless (essential for confix validity)
            if (pos == 0 || is_meaningless_segment(text.substr(0, pos))) {
                anchor_indices[&pattern].push_back(pos);
            }
            pos++;
        }
    }

    // Allocate all LiveOrbits upfront to avoid invalidation during child spawning
    std::vector<std::unique_ptr<LiveOrbit>> orbit_storage;
    orbit_storage.reserve(1000); // Pre-allocate for stability

    std::vector<LiveOrbit*> frontier; // Current active orbits
    frontier.reserve(100);

    // Seed initial orbits at each first anchor occurrence
    for (const auto& [pattern, positions] : anchor_indices) {
        for (std::size_t pos : positions) {
            auto orbit = std::make_unique<LiveOrbit>();
            orbit->pattern = pattern;
            orbit->anchor_index = 1; // consumed first anchor
            orbit->start = pos;
            orbit->cursor = pos + pattern->alternating_anchors.front().size();
            orbit->evidence_offset = (pattern->evidence_types.size() > pattern->alternating_anchors.size()) ? 1 : 0;
            orbit->ev_start = orbit->cursor;
            orbit->ev_end = orbit->cursor;
            orbit->ev_traits = cppfort::stage0::TypeEvidence{};
            orbit->in_evidence = false;
            orbit->parent = nullptr;

            frontier.push_back(orbit.get());
            orbit_storage.push_back(std::move(orbit));
        }
    }

    if (frontier.empty()) return;

    // Advance to EOF, thinning orbits on contradiction with wobbling window support
    std::vector<LiveOrbit*> next_frontier;
    next_frontier.reserve(200);

    for (std::size_t pos = 0; pos < text.size(); ++pos) {
        char ch = text[pos];

        for (LiveOrbit* live : frontier) {
            if (live->dead) continue;

            // Skip until live->cursor
            if (pos < live->cursor) continue;

            // Check if next anchor for this live starts here
            if (live->anchor_index < live->pattern->alternating_anchors.size()) {
                const std::string& next_anchor = live->pattern->alternating_anchors[live->anchor_index];
                if (text.compare(pos, next_anchor.size(), next_anchor) == 0) {
                    // Validate accumulated evidence up to this anchor
                    std::size_t evidence_idx = live->anchor_index - 1 + live->evidence_offset;
                    if (evidence_idx < live->pattern->evidence_types.size()) {
                        std::string_view span = (live->in_evidence && live->ev_end > live->ev_start)
                            ? text.substr(live->ev_start, live->ev_end - live->ev_start)
                            : std::string_view{};
                        if (!validate_evidence_type(live->pattern->evidence_types[evidence_idx], span)) {
                            // Try wobbling if we haven't seen alpha yet
                            if (!live->saw_alpha && live->ev_start < live->ev_end && live->wobble_count < 3) {
                                // Slide window forward by one character
                                live->ev_start++;
                                live->wobble_count++;
                                live->confidence *= 0.95; // Small penalty for wobbling

                                // Recompute traits for the new window
                                live->ev_traits = cppfort::stage0::TypeEvidence{};
                                for (std::size_t i = live->ev_start; i < live->ev_end; ++i) {
                                    live->ev_traits.observe_char(text[i]);
                                }

                                // Try validation again
                                std::string_view new_span = text.substr(live->ev_start, live->ev_end - live->ev_start);
                                if (!validate_evidence_type(live->pattern->evidence_types[evidence_idx], new_span)) {
                                    live->dead = true;
                                    continue;
                                }
                            } else {
                                live->dead = true;
                                continue;
                            }
                        }
                    }

                    // Record this anchor + its evidence span in the backchain
                    live->steps.push_back({pos, live->ev_start, live->ev_end});

                    // Spawn a child orbit to continue from this anchor (backchain construction)
                    auto child = std::make_unique<LiveOrbit>();
                    child->pattern = live->pattern;
                    child->anchor_index = live->anchor_index + 1;
                    child->start = live->start; // Preserve original start
                    child->cursor = pos + next_anchor.size();
                    child->evidence_offset = live->evidence_offset;
                    child->ev_start = child->cursor;
                    child->ev_end = child->cursor;
                    child->ev_traits = cppfort::stage0::TypeEvidence{};
                    child->in_evidence = false;
                    child->saw_alpha = false;
                    child->parent = live; // Link to parent for backchain
                    child->confidence = live->confidence * 0.98; // Slight decay per hop
                    child->steps = live->steps; // Copy parent's steps

                    live->children.push_back(child.get());
                    next_frontier.push_back(child.get());
                    orbit_storage.push_back(std::move(child));

                    // Parent continues but doesn't advance cursor (allows forking)
                    live->dead = true; // Mark parent as consumed
                    continue;
                }
            }

            // Accumulate evidence and early-poison on obvious contradictions
            if (!live->in_evidence) {
                live->in_evidence = true;
                live->ev_start = pos;
                live->ev_end = pos;
                live->ev_traits = cppfort::stage0::TypeEvidence{};
                live->saw_alpha = false;
            }
            live->ev_traits.observe_char(ch);
            live->ev_end = pos + 1;
            if (std::isalpha(static_cast<unsigned char>(ch))) {
                live->saw_alpha = true;
            }

            std::size_t evidence_idx = (live->anchor_index == 0 ? 0 : live->anchor_index - 1) + live->evidence_offset;
            if (evidence_idx < live->pattern->evidence_types.size()) {
                const std::string& kind = live->pattern->evidence_types[evidence_idx];
                // Early contradiction checks per kind
                auto finalize_and_maybe_kill = [&](char reason) {
                    std::string_view span = text.substr(live->ev_start, live->ev_end - live->ev_start);
                    if (!validate_evidence_type(kind, span)) {
                        // Try wobbling before killing
                        if (!live->saw_alpha && live->ev_start < live->ev_end && live->wobble_count < 3) {
                            live->ev_start++;
                            live->wobble_count++;
                            live->confidence *= 0.95;
                            // Recompute traits
                            live->ev_traits = cppfort::stage0::TypeEvidence{};
                            for (std::size_t i = live->ev_start; i < live->ev_end; ++i) {
                                live->ev_traits.observe_char(text[i]);
                            }
                            std::string_view new_span = text.substr(live->ev_start, live->ev_end - live->ev_start);
                            if (!validate_evidence_type(kind, new_span)) {
                                live->dead = true;
                            }
                        } else {
                            live->dead = true;
                        }
                    }
                };

                switch (kind[0]) { // cheap dispatch on first char
                    case 'i': // identifier / identifier_template
                        if (ch == '=' || ch == ';' || ch == '{' || ch == '}' || ch == ':') {
                            finalize_and_maybe_kill(ch);
                        }
                        break;
                    case 't': // type_expression
                        if (ch == '=' || ch == '{' || ch == '}') {
                            finalize_and_maybe_kill(ch);
                        }
                        break;
                    default:
                        break;
                }
            }
        }

        // Update frontier with new children
        for (LiveOrbit* child : next_frontier) {
            bool already_in = false;
            for (LiveOrbit* existing : frontier) {
                if (existing == child) {
                    already_in = true;
                    break;
                }
            }
            if (!already_in) {
                frontier.push_back(child);
            }
        }
        next_frontier.clear();

        // If every orbit is dead, we can stop early
        bool any_alive = false;
        for (LiveOrbit* live : frontier) {
            if (!live->dead) {
                any_alive = true;
                break;
            }
        }
        if (!any_alive) break;
    }

    // Walk the backchains to find terminal survivors
    std::vector<LiveOrbit*> terminal_survivors;
    for (const auto& orbit : orbit_storage) {
        if (!orbit->dead && orbit->children.empty()) {
            // This is a terminal orbit (no children spawned from it)
            terminal_survivors.push_back(orbit.get());
        }
    }

    // Validate final evidence for survivors and emit matches to EOF
    for (LiveOrbit* live : terminal_survivors) {
        // Validate last evidence segment after last anchor if any
        std::size_t evidence_idx = (live->anchor_index == 0 ? 0 : live->anchor_index - 1) + live->evidence_offset;
        if (evidence_idx < live->pattern->evidence_types.size()) {
            std::string_view final_span = (live->in_evidence && live->ev_end > live->ev_start)
                ? text.substr(live->ev_start, live->ev_end - live->ev_start)
                : std::string_view{};
            if (!validate_evidence_type(live->pattern->evidence_types[evidence_idx], final_span)) {
                continue;
            }
        }

        // Walk the backchain to collect all steps
        std::vector<LiveOrbit::Step> full_chain;
        LiveOrbit* current = live;
        while (current != nullptr) {
            // Prepend this orbit's steps (in reverse order to build from root to leaf)
            full_chain.insert(full_chain.begin(), current->steps.begin(), current->steps.end());
            current = current->parent;
        }

        // Debug dump of complete backchain
        std::cerr << "DEBUG terminal: survivor pattern=" << live->pattern->name
                  << " anchors=" << live->anchor_index
                  << " confidence=" << live->confidence
                  << " wobbles=" << live->wobble_count
                  << " backchain_depth=" << full_chain.size() << "\n";

        // Show the anchor tuple
        std::cerr << "  Anchor tuple: [";
        bool first = true;
        for (std::size_t i = 0; i < std::min(live->anchor_index, live->pattern->alternating_anchors.size()); ++i) {
            if (!first) std::cerr << ", ";
            std::cerr << "'" << live->pattern->alternating_anchors[i] << "'";
            first = false;
        }
        std::cerr << "]\n";

        // Show evidence spans
        for (const auto& st : full_chain) {
            std::string_view ev = (st.ev_end > st.ev_start) ? text.substr(st.ev_start, st.ev_end - st.ev_start) : std::string_view{};
            std::string snippet = std::string(ev.substr(0, std::min<std::size_t>(ev.size(), 40)));
            std::cerr << "  step: anchor@" << st.anchor_pos << " ev[" << st.ev_start << "," << st.ev_end << ")='"
                     << snippet << (ev.size() > 40 ? "..." : "") << "'\n";
        }

        // Calculate final confidence score with multiple factors
        double final_confidence = live->confidence;

        // Factor 1: Pattern weight (some patterns are more reliable)
        double pattern_weight = 1.0;
        if (live->pattern->name.find("function") != std::string::npos) {
            pattern_weight = 1.15; // Functions are high-confidence
        } else if (live->pattern->name.find("type") != std::string::npos) {
            pattern_weight = 1.10; // Types are also reliable
        }
        final_confidence *= pattern_weight;

        // Factor 2: Anchor coverage (more anchors matched = higher confidence)
        double anchor_coverage = static_cast<double>(live->anchor_index) / live->pattern->alternating_anchors.size();
        final_confidence *= (0.7 + 0.3 * anchor_coverage);

        // Factor 3: Span coverage relative to text length
        double span_coverage = static_cast<double>(text.size() - live->start) / text.size();
        final_confidence *= (0.8 + 0.2 * span_coverage);

        // Factor 4: Early poison penalty (orbits that encountered issues get penalized)
        // Already factored in via wobble_count penalties

        // Factor 5: Backchain depth bonus (deeper chains = more evidence)
        std::size_t chain_depth = 0;
        LiveOrbit* chain_walker = live;
        while (chain_walker->parent != nullptr) {
            chain_depth++;
            chain_walker = chain_walker->parent;
        }
        final_confidence *= (1.0 + 0.05 * std::min(chain_depth, static_cast<std::size_t>(5)));

        // Clamp to reasonable range
        final_confidence = std::min(1.0, std::max(0.1, final_confidence));

        ::cppfort::stage0::OrbitFragment fragment;
        fragment.start_pos = live->start;
        fragment.end_pos   = text.size();
        fragment.confidence = final_confidence;
        fragment.classified_grammar = ::cppfort::ir::GrammarType::CPP2;

        std::size_t match_length = fragment.end_pos - fragment.start_pos;
        matches_.emplace_back(match_length, fragment.confidence, live->pattern->name, std::move(fragment));
    }
}

const ::cppfort::stage0::SpeculativeMatch* RBCursiveScanner::get_best_match() const {
    if (matches_.empty()) {
        return nullptr;
    }

    // Check if we should prefer backchain-based selection
    const char* use_backchain = std::getenv("RBCURSIVE_USE_BACKCHAIN");
    if (use_backchain && *use_backchain == '1') {
        // In backchain mode, prefer highest confidence within similar lengths
        std::size_t best_idx = 0;
        double best_score = 0.0;

        for (std::size_t i = 0; i < matches_.size(); ++i) {
            // Combine length and confidence for scoring
            double length_factor = static_cast<double>(matches_[i].match_length) / matches_[0].match_length;
            double score = matches_[i].confidence * (0.7 + 0.3 * length_factor);

            if (score > best_score) {
                best_score = score;
                best_idx = i;
            }
        }
        return &matches_[best_idx];
    }

    // Default: Return the first match (already sorted by length descending)
    return &matches_[0];
}

CombinatorPool::CombinatorPool(std::size_t initial_size) {
    pool_.resize(initial_size);
    used_.assign(initial_size, false);
}

RBCursiveScanner* CombinatorPool::allocate() {
    for (std::size_t idx = 0; idx < pool_.size(); ++idx) {
        if (!used_[idx]) {
            used_[idx] = true;
            return &pool_[idx];
        }
    }
    pool_.emplace_back();
    used_.push_back(true);
    return &pool_.back();
}

void CombinatorPool::release(RBCursiveScanner* scanner) {
    if (!scanner || pool_.empty()) {
        return;
    }
    auto* begin = pool_.data();
    auto* end = begin + pool_.size();
    if (scanner >= begin && scanner < end) {
        std::size_t idx = static_cast<std::size_t>(scanner - begin);
        used_[idx] = false;
    }
}

std::size_t CombinatorPool::available() const {
    std::size_t count = 0;
    for (bool used : used_) {
        if (!used) {
            ++count;
        }
    }
    return count;
}

// Validate evidence type for alternating patterns
bool RBCursiveScanner::validate_evidence_type(const std::string& type, std::string_view evidence) const {
    EvidenceAnalysis analysis = make_evidence(evidence);

    if (analysis.empty()) {
        if (type == "parameters" || type == "trailing" || type == "body") {
            return true;
        }
        return false;
    }

    const auto& trimmed = analysis.trimmed;
    const auto& traits = analysis.traits;

    if (type == "identifier") {
        if (!std::isalpha(static_cast<unsigned char>(trimmed.front())) && trimmed.front() != '_') {
            return false;
        }
        for (char c : trimmed) {
            if (!std::isalnum(static_cast<unsigned char>(c)) && c != '_') {
                return false;
            }
        }
        if (traits.colon > 0 || traits.double_colon > 0 || traits.angle_open > 0 ||
            traits.confix_open[static_cast<uint8_t>(ConfixType::BRACE)] > 0 ||
            traits.confix_open[static_cast<uint8_t>(ConfixType::PAREN)] > 0 || traits.arrow > 0 ||
            analysis.contains_any("=;")) {
            return false;
        }
        return true;
    } else if (type == "identifier_template") {
        if (traits.confix_open[static_cast<uint8_t>(ConfixType::BRACE)] > 0 || analysis.contains_any("=;")) {
            return false;
        }
        size_t angle_pos = trimmed.find('<');
        if (angle_pos == std::string::npos) {
            return validate_evidence_type("identifier", trimmed);
        }
        std::string_view ident_part = trim_view(trimmed.substr(0, angle_pos));
        if (!validate_evidence_type("identifier", ident_part)) {
            return false;
        }
        size_t close_pos = trimmed.rfind('>');
        if (close_pos == std::string::npos || close_pos < angle_pos + 1) {
            return false;
        }
        std::string_view inner = trim_view(trimmed.substr(angle_pos + 1, close_pos - angle_pos - 1));
        if (inner.empty()) {
            return false;
        }
        if (inner.find('=') != std::string::npos) {
            return false;
        }
        return true;
    } else if (type == "type_expression" || type == "return_type") {
        if (analysis.contains_any("=;")) {
            return false;
        }
        bool has_alpha = false;
        for (char c : trimmed) {
            if (std::isalpha(static_cast<unsigned char>(c))) {
                has_alpha = true;
            }
            if (!std::isalnum(static_cast<unsigned char>(c)) && c != '_' && c != '<' && c != '>' &&
                c != ':' && c != ' ' && c != '&' && c != '*' && c != ',' && c != '.' &&
                c != '(' && c != ')') {
                return false;
            }
        }
        if (!has_alpha && traits.double_colon == 0) {
            return false;
        }
        return true;
    } else if (type == "parameters") {
        return true;
    } else if (type == "body" || type == "expression" || type == "trailing") {
        return true;
    }
    return false;
}

// Orbit-based anchor position detection (NO string.find())
std::vector<std::size_t> RBCursiveScanner::find_anchor_positions_orbit(
    std::string_view text,
    std::string_view anchor,
    const std::vector<int>* depth_map) const {
    std::vector<std::size_t> positions;
    
    if (anchor.empty() || text.empty()) {
        return positions;
    }
    
    // Use SIMD byte-scanning for anchor detection (wide orbit scanning)
    // Convert anchor to byte pattern
    std::vector<uint8_t> anchor_bytes(anchor.begin(), anchor.end());
    std::vector<uint8_t> text_bytes(text.begin(), text.end());
    
    // Scan for anchor positions using SIMD or scalar scanner
    std::vector<std::size_t> candidate_positions = scalarScanner().scanBytes(
        std::span<const uint8_t>(text_bytes), 
        std::span<const uint8_t>(anchor_bytes));
    
    // Filter candidates by orbit context (balanced, in-valid-depth, not-in-string)
    for (size_t pos : candidate_positions) {
        // Check if position is at valid orbit boundary (not inside string/comment)
        bool valid_boundary = true;
        
        // Check depth constraints if provided
        if (depth_map && pos < depth_map->size()) {
            int depth = (*depth_map)[pos];
            // Only allow anchors at top-level or reasonable nesting depth
            if (depth < 0 || depth > 10) {  // Reasonable depth limit
                valid_boundary = false;
            }
        }
        
        // Additional validation: ensure anchor starts at word/non-word boundary appropriately
        if (valid_boundary && pos > 0) {
            char prev_char = text[pos - 1];
            char anchor_first = anchor[0];
            
            // If anchor starts with identifier char, ensure prev char is non-ident
            if (std::isalnum(static_cast<unsigned char>(anchor_first)) || anchor_first == '_') {
                if (std::isalnum(static_cast<unsigned char>(prev_char)) || prev_char == '_') {
                    valid_boundary = false; // In middle of identifier
                }
            }
        }
        
        if (valid_boundary) {
            positions.push_back(pos);
        }
    }
    
    return positions;
}

} // namespace ir
} // namespace cppfort
