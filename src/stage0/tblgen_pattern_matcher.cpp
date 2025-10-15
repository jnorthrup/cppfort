#include "tblgen_pattern_matcher.h"
#include <regex>
#include <iostream>

namespace cppfort::stage0 {

std::string TblgenPatternMatcher::pattern_to_regex(const std::string& pattern) {
    std::string regex_str;
    regex_str.reserve(pattern.size() * 2);

    for (size_t i = 0; i < pattern.size(); ++i) {
        char c = pattern[i];

        if (c == '$' && i + 1 < pattern.size() && std::isdigit(pattern[i + 1])) {
            // Replace $N with capture group
            // Use (.+?) for non-greedy match or ([^delimiter]+) for delimited
            regex_str += "([^:(){}=]+?)";  // Capture until delimiter
            ++i; // Skip the digit
        } else {
            // Escape regex special characters
            if (c == '(' || c == ')' || c == '{' || c == '}' ||
                c == '[' || c == ']' || c == '.' || c == '*' ||
                c == '+' || c == '?' || c == '^' || c == '$' || c == '|') {
                regex_str += '\\';
            }
            regex_str += c;
        }
    }

    return regex_str;
}

std::optional<std::vector<std::string>> TblgenPatternMatcher::match(
    const std::string& pattern,
    const std::string& input
) {
    try {
        // Convert pattern to regex
        std::string regex_pattern = pattern_to_regex(pattern);

        // For CPP2 function pattern "$0: ($1) -> $2 = $3"
        // We need a more sophisticated approach

        // Hardcode for function pattern as POC
        if (pattern.find("$0: ($1) -> $2 = $3") != std::string::npos) {
            // Match: name: (params) -> return_type = { body }
            std::regex func_regex(R"((\w+)\s*:\s*\(([^)]*)\)\s*->\s*(\w+)\s*=\s*\{([^}]*)\})");
            std::smatch matches;

            if (std::regex_search(input, matches, func_regex)) {
                std::vector<std::string> segments;
                for (size_t i = 1; i < matches.size(); ++i) {
                    segments.push_back(matches[i].str());
                }
                return segments;
            }
        }

        return std::nullopt;
    } catch (const std::regex_error& e) {
        std::cerr << "Regex error: " << e.what() << "\n";
        return std::nullopt;
    }
}

} // namespace cppfort::stage0
