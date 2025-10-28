#include "rbcursive.h"

#include "evidence.h"

#include <algorithm>
#include <cctype>
#include <iostream>
#include <map>
#include <memory>
#include <span>
#include <vector>
#include <deque>

namespace cppfort {
namespace stage0 {

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
            // REMOVED: Use infix orbits instead
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
    size_t anchor_pos = text.find(first_anchor);
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
            next_anchor_pos = text.find(next_anchor, current_pos);
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

// Experimental: alternating-anchor backchain thinning driven by TypeEvidence
void RBCursiveScanner::speculate_backchain(std::string_view text) {
    matches_.clear();
    semantic_traces_.clear();
    if (!patterns_ || text.empty()) {
        return;
    }

    struct LiveOrbit {
        struct Step { std::size_t anchor_pos; std::size_t ev_start; std::size_t ev_end; };
        const ::cppfort::stage0::PatternData* pattern = nullptr;
        std::size_t anchor_index = 0;
        std::size_t start = 0;
        std::size_t cursor = 0;
        std::size_t evidence_offset = 0;
        std::size_t ev_start = 0;
        std::size_t ev_end = 0;
        bool in_evidence = false;
        double confidence = 1.0;
        std::size_t wobble_count = 0;
        std::vector<Step> steps;
        LiveOrbit* parent = nullptr;
        std::vector<LiveOrbit*> children;
        bool dead = false;
        std::vector<::cppfort::stage0::SemanticTrace> semantic_traces;  // NEW: record validation steps
    };

    auto validate_with_wobble = [&](LiveOrbit& live, const std::string& kind, std::string_view text_span) -> bool {
        if (kind.empty()) return true;
        std::size_t start = live.ev_start;
        std::size_t end = live.ev_end;
        if (end <= start) {
            return kind == "trailing" || kind == "body" || kind == "expression" || kind == "parameters";
        }
        std::size_t local_wobbles = 0;
        while (true) {
            std::string_view span = text_span.substr(start, end - start);
            bool valid = validate_evidence_type(kind, span);
            
            // Record semantic trace for this validation step
            ::cppfort::stage0::SemanticTrace trace;
            trace.pattern_name = live.pattern->name;
            trace.anchor_index = live.anchor_index;
            trace.evidence_start = start;
            trace.evidence_end = end;
            trace.evidence_content = std::string(span);
            trace.traits = analyze_evidence(span);
            trace.expected_type = kind;
            trace.verdict = valid;
            
            if (!valid) {
                // Determine failure reason
                if (span.empty()) {
                    trace.failure_reason = "empty evidence span";
                } else if (kind == "identifier" && 
                          (trace.traits.colon > 0 || trace.traits.double_colon > 0 || 
                           trace.traits.angle_open > 0 || trace.traits.brace_open > 0 || 
                           trace.traits.paren_open > 0 || trace.traits.arrow > 0)) {
                    trace.failure_reason = "identifier contains structural punctuation";
                } else if (kind == "type_expression" && 
                          (std::string_view(span).find('=') != std::string::npos || 
                           std::string_view(span).find(';') != std::string::npos)) {
                    trace.failure_reason = "type expression contains assignment or semicolon";
                } else {
                    trace.failure_reason = "evidence type validation failed";
                }
                // CRITICAL: Mark orbit dead immediately on contradiction
                live.dead = true;
            }
            
            live.semantic_traces.push_back(std::move(trace));
            
            if (valid) {
                if (local_wobbles > 0) {
                    live.ev_start = start;
                    live.wobble_count += local_wobbles;
                    for (std::size_t i = 0; i < local_wobbles; ++i) {
                        live.confidence *= 0.95;
                    }
                }
                return true;
            }
            if (live.wobble_count + local_wobbles >= 3 || start + 1 >= end) {
                return false;
            }
            ++start;
            ++local_wobbles;
        }
        return false;
    };

    // Seed initial orbits
    std::deque<std::unique_ptr<LiveOrbit>> storage;
    std::vector<LiveOrbit*> frontier;

    for (const auto& pattern : *patterns_) {
        if (!pattern.use_alternating || pattern.alternating_anchors.empty()) {
            continue;
        }
        const std::string& first_anchor = pattern.alternating_anchors.front();
        const bool has_leading_evidence = pattern.evidence_types.size() > pattern.alternating_anchors.size();

        std::size_t search_pos = 0;
        while ((search_pos = text.find(first_anchor, search_pos)) != std::string::npos) {
            // If there is no leading evidence requirement, enforce meaningless prefix
            if (!has_leading_evidence && search_pos > 0 && !is_meaningless_segment(text.substr(0, search_pos))) {
                ++search_pos;
                continue;
            }

            auto live = std::make_unique<LiveOrbit>();
            live->pattern = &pattern;
            live->anchor_index = 1; // consumed the first anchor
            live->start = search_pos;
            live->cursor = search_pos + first_anchor.size();
            live->evidence_offset = has_leading_evidence ? 1 : 0;
            live->ev_start = live->cursor;
            live->ev_end = live->cursor;
            live->confidence = 1.0;

            // Validate leading evidence if required
            if (has_leading_evidence && !pattern.evidence_types.empty()) {
                std::string_view prefix = text.substr(0, search_pos);
                if (!validate_evidence_type(pattern.evidence_types[0], prefix)) {
                    ++search_pos;
                    continue;
                }
                EvidenceAnalysis lead_analysis = make_evidence(prefix);
                std::size_t lead_start = lead_analysis.trimmed.empty() ? search_pos
                    : static_cast<std::size_t>(lead_analysis.trimmed.data() - text.data());
                std::size_t lead_end = lead_analysis.trimmed.empty() ? search_pos
                    : lead_start + lead_analysis.trimmed.size();
                live->steps.push_back({search_pos, lead_start, lead_end});
            }

            frontier.push_back(live.get());
            storage.push_back(std::move(live));
            ++search_pos;
        }
    }

    if (frontier.empty()) {
        return;
    }

    std::vector<LiveOrbit*> next_frontier;
    next_frontier.reserve(frontier.size() * 2);

    for (std::size_t pos = 0; pos < text.size() && !frontier.empty(); ++pos) {
        char ch = text[pos];
        next_frontier.clear();

        for (LiveOrbit* live : frontier) {
            if (live->dead) {
                continue;
            }
            if (pos < live->cursor) {
                continue;
            }

            // Anchor opportunity
            if (live->anchor_index < live->pattern->alternating_anchors.size()) {
                const std::string& next_anchor = live->pattern->alternating_anchors[live->anchor_index];
                if (text.compare(pos, next_anchor.size(), next_anchor) == 0) {
                    std::size_t evidence_idx = live->anchor_index - 1 + live->evidence_offset;
                    if (evidence_idx < live->pattern->evidence_types.size()) {
                        if (!validate_with_wobble(*live, live->pattern->evidence_types[evidence_idx], text)) {
                            live->dead = true;
                            continue;
                        }
                    }

                    std::size_t evidence_start = live->ev_start;
                    std::size_t evidence_end = live->ev_end;
                    live->steps.push_back({pos, evidence_start, evidence_end});

                    auto child = std::make_unique<LiveOrbit>();
                    child->pattern = live->pattern;
                    child->anchor_index = live->anchor_index + 1;
                    child->start = live->start;
                    child->cursor = pos + next_anchor.size();
                    child->evidence_offset = live->evidence_offset;
                    child->ev_start = child->cursor;
                    child->ev_end = child->cursor;
                    child->confidence = live->confidence * 0.98;
                    child->wobble_count = live->wobble_count;
                    child->steps = live->steps;
                    child->parent = live;
                    live->children.push_back(child.get());

                    next_frontier.push_back(child.get());
                    storage.push_back(std::move(child));

                    // Parent no longer participates further down the chain
                    live->dead = true;
                    continue;
                }
            }

            // Evidence accumulation
            if (!live->in_evidence) {
                live->in_evidence = true;
                live->ev_start = pos;
                live->ev_end = pos;
            }
            live->ev_end = pos + 1;

            std::size_t evidence_idx = (live->anchor_index == 0 ? 0 : live->anchor_index - 1) + live->evidence_offset;
            if (evidence_idx < live->pattern->evidence_types.size()) {
                const std::string& kind = live->pattern->evidence_types[evidence_idx];
                bool trigger_check = false;
                switch (!kind.empty() ? kind[0] : '\0') {
                    case 'i': // identifier / identifier_template
                        trigger_check = (ch == '=' || ch == ';' || ch == '{' || ch == '}' || ch == ':');
                        break;
                    case 't': // type_expression
                        trigger_check = (ch == '=' || ch == '{' || ch == '}');
                        break;
                    default:
                        break;
                }
                if (trigger_check) {
                    if (!validate_with_wobble(*live, kind, text)) {
                        live->dead = true;
                        continue;
                    }
                }
            }
        }

        // Rebuild frontier with surviving parents plus newly spawned children
        std::vector<LiveOrbit*> updated;
        updated.reserve(frontier.size() + next_frontier.size());
        for (LiveOrbit* live : frontier) {
            if (!live->dead) {
                updated.push_back(live);
            }
        }
        updated.insert(updated.end(), next_frontier.begin(), next_frontier.end());
        frontier.swap(updated);
    }

    // Collect terminal survivors (leaves that remain alive)
    std::vector<LiveOrbit*> survivors;
    for (const auto& orbit : storage) {
        if (!orbit->dead && orbit->children.empty()) {
            survivors.push_back(orbit.get());
        }
    }

    if (survivors.empty()) {
        return;
    }

    // Emit matches for survivors
    for (LiveOrbit* live : survivors) {
        if (live->pattern && live->anchor_index < live->pattern->alternating_anchors.size()) {
            continue;
        }

        std::size_t evidence_idx = (live->anchor_index == 0 ? 0 : live->anchor_index - 1) + live->evidence_offset;
        if (evidence_idx < live->pattern->evidence_types.size()) {
            if (!validate_with_wobble(*live, live->pattern->evidence_types[evidence_idx], text)) {
                continue;
            }
        }

        // Debug output to understand the chain
        std::cerr << "DEBUG backchain: pattern=" << live->pattern->name
                  << " anchors=" << live->anchor_index
                  << " steps=" << live->steps.size()
                  << " conf=" << live->confidence
                  << " wobbles=" << live->wobble_count << "\n";
        for (const auto& step : live->steps) {
            std::string_view span = (step.ev_end > step.ev_start)
                ? text.substr(step.ev_start, step.ev_end - step.ev_start)
                : std::string_view{};
            std::string snippet = std::string(span.substr(0, std::min<std::size_t>(span.size(), 32)));
            std::cerr << "  step: anchor@" << step.anchor_pos
                      << " span[" << step.ev_start << "," << step.ev_end << ")='"
                      << snippet << (span.size() > snippet.size() ? "..." : "") << "'\n";
        }

        double final_conf = live->confidence;

        if (live->pattern->name.find("function") != std::string::npos) {
            final_conf *= 1.15;
        } else if (live->pattern->name.find("type") != std::string::npos) {
            final_conf *= 1.10;
        }

        double anchor_coverage = live->pattern->alternating_anchors.empty()
            ? 1.0
            : std::min<double>(live->anchor_index, live->pattern->alternating_anchors.size()) /
              static_cast<double>(live->pattern->alternating_anchors.size());
        final_conf *= (0.7 + 0.3 * anchor_coverage);

        double span_coverage = text.empty() ? 1.0
            : static_cast<double>(text.size() - live->start) / static_cast<double>(text.size());
        final_conf *= (0.8 + 0.2 * span_coverage);

        std::size_t chain_depth = live->steps.size();
        final_conf *= (1.0 + 0.05 * std::min<std::size_t>(chain_depth, 5));

        final_conf = std::clamp(final_conf, 0.1, 1.0);

        ::cppfort::stage0::OrbitFragment fragment;
        fragment.start_pos = live->start;
        fragment.end_pos = text.size();
        fragment.confidence = final_conf;
        fragment.classified_grammar = ::cppfort::ir::GrammarType::CPP2;

        std::size_t match_length = fragment.end_pos - fragment.start_pos;
        matches_.emplace_back(match_length, fragment.confidence, live->pattern->name, std::move(fragment));
        
        // Collect semantic traces if capture mode is enabled
        if (capture_traces_) {
            for (const auto& trace : live->semantic_traces) {
                semantic_traces_.push_back(trace);
            }
        }
    }

    std::sort(matches_.begin(), matches_.end(), [](const auto& a, const auto& b) {
        if (a.match_length != b.match_length) {
            return a.match_length > b.match_length;
        }
        return a.confidence > b.confidence;
    });
}

const ::cppfort::stage0::SpeculativeMatch* RBCursiveScanner::get_best_match() const {
    if (matches_.empty()) {
        return nullptr;
    }

    // Always use backchain-based selection: prefer highest confidence within similar lengths
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
            traits.brace_open > 0 || traits.paren_open > 0 || traits.arrow > 0 ||
            analysis.contains_any("=;")) {
            return false;
        }
        return true;
    } else if (type == "identifier_template") {
        if (traits.brace_open > 0 || analysis.contains_any("=;")) {
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

} // namespace stage0
} // namespace cppfort
