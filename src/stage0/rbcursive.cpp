#include "rbcursive.h"

#include <algorithm>

namespace cppfort {
namespace ir {

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

} // namespace ir
} // namespace cppfort
