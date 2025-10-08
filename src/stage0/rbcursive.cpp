#include "rbcursive.h"

#include <algorithm>
#include <iostream>
#include <span>

namespace cppfort {
namespace ir {

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
    try {
        std::regex re{std::string(pattern)};
        std::string s{text};
        return std::regex_search(s, re);
    } catch (...) {
        return false;
    }
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
            try {
                std::regex re{std::string(pattern)};
                std::string s{data};
                std::smatch m;
                auto begin = s.cbegin();
                auto end = s.cend();

                while (begin != end && std::regex_search(begin, end, m, re)) {
                    std::size_t sidx = static_cast<std::size_t>(m.position(0) + (begin - s.cbegin()));
                    std::size_t eidx = sidx + static_cast<std::size_t>(m.length(0));
                    out.push_back(Match{ sidx, eidx });
                    begin = m.suffix().first; // advance past this match
                }
            } catch (...) {
                // Invalid regex -> no matches
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
                // Found signature
                // CHEAT REMOVED: Was hardcoded CPP2 scope expansion (lines 135-179)
                // Should be pattern-driven via orbit recursion, not string matching
                size_t start_pos = pos;
                size_t end_pos = pos + signature.length();

                size_t match_length = end_pos - start_pos;
                // Compute confidence from pattern complexity (honest baseline)
                double confidence = std::min(1.0, static_cast<double>(match_length) / text.length());

                // Create result fragment
                ::cppfort::stage0::OrbitFragment fragment;
                fragment.start_pos = start_pos;
                fragment.end_pos = end_pos;
                fragment.confidence = confidence;
                fragment.classified_grammar = ::cppfort::ir::GrammarType::UNKNOWN; // Will be set by correlator

                matches_.emplace_back(match_length, confidence, pattern.name, std::move(fragment));
                // std::cout << "DEBUG: Found match for pattern '" << pattern.name << "' at pos " << start_pos << " with length " << match_length << "\n";
                break; // Only take first match per pattern for now
            }
        }
    }

    // Sort by match_length descending (longest matches first)
    std::sort(matches_.begin(), matches_.end(), 
              [](const ::cppfort::stage0::SpeculativeMatch& a, const ::cppfort::stage0::SpeculativeMatch& b) {
                  return a.match_length > b.match_length;
              });
}

// Alternating anchor/evidence speculation for deterministic grammar selection
void RBCursiveScanner::speculate_alternating(const ::cppfort::stage0::PatternData& pattern, std::string_view text) {
    if (!pattern.use_alternating || pattern.alternating_anchors.empty()) {
        return;
    }

    // Find the first anchor
    const std::string& first_anchor = pattern.alternating_anchors[0];
    size_t anchor_pos = text.find(first_anchor);
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

        // Match the entire text
        match_start = 0;
        size_t match_end = text.size();

        ::cppfort::stage0::OrbitFragment fragment;
        fragment.start_pos = match_start;
        fragment.end_pos = match_end;
        fragment.confidence = 1.0;
        fragment.classified_grammar = ::cppfort::ir::GrammarType::CPP2;

        matches_.emplace_back(match_end - match_start, 1.0, pattern.name, std::move(fragment));
        return;
    }

    size_t current_pos = anchor_pos;
    spans.emplace_back(text.substr(anchor_pos, first_anchor.length()), true); // First anchor
    current_pos += first_anchor.length();

    // Alternate between evidence and anchors
    for (size_t i = 0; i < pattern.alternating_anchors.size() + pattern.evidence_types.size(); ++i) {
        if (i % 2 == 0) {
            // Expect evidence span
            size_t evidence_idx = i / 2;
            if (evidence_idx >= pattern.evidence_types.size()) {
                break; // No more evidence types
            }
            
            // Find next anchor or end
            size_t next_anchor_pos = std::string::npos;
            if (evidence_idx + 1 < pattern.alternating_anchors.size()) {
                const std::string& next_anchor = pattern.alternating_anchors[evidence_idx + 1];
                next_anchor_pos = text.find(next_anchor, current_pos);
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
            
            if (next_anchor_pos != std::string::npos) {
                spans.emplace_back(text.substr(next_anchor_pos, pattern.alternating_anchors[evidence_idx + 1].length()), true);
                current_pos += pattern.alternating_anchors[evidence_idx + 1].length();
            }
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

const ::cppfort::stage0::SpeculativeMatch* RBCursiveScanner::get_best_match() const {
    if (matches_.empty()) {
        return nullptr;
    }
    
    // Return the first match (already sorted by length descending)
    // If there are ties in length, the first one wins (could be improved to consider confidence)
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
    if (type == "identifier") {
        // Simple identifier: starts with letter/underscore, contains letters/digits/underscores
        if (evidence.empty()) return false;
        if (!std::isalpha(evidence[0]) && evidence[0] != '_') return false;
        for (char c : evidence) {
            if (!std::isalnum(c) && c != '_') return false;
        }
        return true;
    } else if (type == "identifier_template") {
        // Template identifier: identifier followed by optional <...>
        if (evidence.empty()) return false;
        size_t angle_pos = evidence.find('<');
        if (angle_pos == std::string::npos) {
            // No template parameters, validate as regular identifier
            return validate_evidence_type("identifier", evidence);
        }
        // Validate identifier part
        std::string_view ident_part = evidence.substr(0, angle_pos);
        if (!validate_evidence_type("identifier", ident_part)) return false;
        // Check for matching angle brackets (simplified)
        size_t close_pos = evidence.find('>', angle_pos);
        return close_pos != std::string::npos && close_pos == evidence.length() - 1;
    } else if (type == "type_expression") {
        // Type expression: can be complex, for now accept if not empty and contains valid chars
        if (evidence.empty()) return false;
        // Basic validation: contains letters, spaces, <>, ::, etc., and punctuation
        bool has_alpha = false;
        for (char c : evidence) {
            if (std::isalpha(c)) has_alpha = true;
            if (!std::isalnum(c) && c != '_' && c != '<' && c != '>' && c != ':' && c != ' ' && c != '&' && c != '*' && c != ';' && c != ',' && c != '.' && c != '(' && c != ')') {
                return false;
            }
        }
        return has_alpha;
    }
    return false; // Unknown type
}

} // namespace ir
} // namespace cppfort
