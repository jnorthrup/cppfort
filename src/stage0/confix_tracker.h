#pragma once

#include <string_view>
#include <vector>

namespace cppfort::stage0 {

// Track confix depth and balance for pattern matching priority
// Handles template angle bracket nesting: vector<vector<int>> vs a >> b
struct ConfixTracker {
    int paren_depth = 0;      // ()
    int brace_depth = 0;      // {}
    int bracket_depth = 0;    // []
    int angle_depth = 0;      // <>

    // Total nesting depth (sum of all confix types)
    int total_depth() const {
        return paren_depth + brace_depth + bracket_depth + angle_depth;
    }

    // Check if all confixes are balanced at this position
    bool is_balanced() const {
        return paren_depth == 0 && brace_depth == 0 &&
               bracket_depth == 0 && angle_depth == 0;
    }

    // Process character and update depths
    // Returns true if this is a confix character
    bool process_char(char ch, char prev_char, char next_char) {
        switch (ch) {
            case '(': paren_depth++; return true;
            case ')': paren_depth--; return true;
            case '{': brace_depth++; return true;
            case '}': brace_depth--; return true;
            case '[': bracket_depth++; return true;
            case ']': bracket_depth--; return true;

            case '<':
                // Only count as template open if preceded by identifier/type or comma
                // Not if it's a comparison operator
                if (prev_char == '_' || std::isalnum(prev_char) || prev_char == ',' || prev_char == ' ') {
                    angle_depth++;
                    return true;
                }
                return false;

            case '>':
                // Template close: could be one or two closes
                // Check for >> (two closes vs right-shift operator)
                if (angle_depth > 0) {
                    if (next_char == '>' && angle_depth >= 2) {
                        // This could be >> as two template closes
                        // Heuristic: if we're nested 2+ deep, treat as double close
                        return true; // Mark as confix, caller handles
                    }
                    angle_depth--;
                    return true;
                }
                return false;

            default:
                return false;
        }
    }

    // Check if a span crosses unbalanced confix boundaries
    // If it does, the pattern is illegal
    bool span_crosses_boundary(size_t start, size_t end, std::string_view text) const {
        ConfixTracker tracker;
        for (size_t i = start; i < end && i < text.size(); ++i) {
            char prev = (i > 0) ? text[i-1] : '\0';
            char next = (i + 1 < text.size()) ? text[i+1] : '\0';
            tracker.process_char(text[i], prev, next);
        }
        return !tracker.is_balanced();
    }
};

// Scan text and return confix depth at each position
inline std::vector<int> build_depth_map(std::string_view text) {
    std::vector<int> depth_map(text.size(), 0);
    ConfixTracker tracker;

    for (size_t i = 0; i < text.size(); ++i) {
        char prev = (i > 0) ? text[i-1] : '\0';
        char next = (i + 1 < text.size()) ? text[i+1] : '\0';
        tracker.process_char(text[i], prev, next);
        depth_map[i] = tracker.total_depth();
    }

    return depth_map;
}

} // namespace cppfort::stage0
