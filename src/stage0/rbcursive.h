#pragma once

#include <cstddef>
#include <optional>
#include <regex>
#include <string>
#include <string_view>
#include <vector>

namespace cppfort {
namespace ir {

// Minimal, internal scanning facade inspired by Litebike rbcursive.
// This is a private asset of the orbit scanner. It intentionally starts
// simple and can be extended with SIMD and protocol combinators later.
class RBCursiveScanner {
public:
    enum class PatternType { Glob, Regex };

    struct Match {
        std::size_t start = 0;
        std::size_t end = 0; // one-past-end
    };

    struct Capabilities {
        bool glob = true;
        bool regex = true;
        bool simd = false; // placeholder until SIMD paths are ported
    };

    RBCursiveScanner() = default;

    // Returns true if the whole text matches the glob pattern
    bool matchGlob(std::string_view text, std::string_view pattern) const;

    // Returns the first regex full-match if any
    bool matchRegex(std::string_view text, std::string_view pattern) const;

    // Scans and returns non-overlapping matches by pattern type
    std::vector<Match> scanWithPattern(std::string_view data,
                                       std::string_view pattern,
                                       PatternType type) const;

    Capabilities patternCapabilities() const { return {}; }

private:
    // Simple glob matcher supporting '*', '?'
    static bool globMatch(std::string_view text, std::string_view pattern);
};

} // namespace ir
} // namespace cppfort

