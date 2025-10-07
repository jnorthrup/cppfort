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

} // namespace ir
} // namespace cppfort
