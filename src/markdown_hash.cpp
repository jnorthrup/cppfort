#include "markdown_hash.hpp"
#include "semantic_hash.hpp"
#include <sstream>
#include <algorithm>
#include <cctype>

namespace cpp2_transpiler {

// Helper: trim whitespace from both ends of a string
static std::string trim(std::string_view str) {
    // Find first non-whitespace
    auto start = std::find_if_not(str.begin(), str.end(),
        [](unsigned char ch) { return std::isspace(ch); });

    // Find last non-whitespace
    auto end = std::find_if_not(str.rbegin(), str.rend(),
        [](unsigned char ch) { return std::isspace(ch); }).base();

    if (start >= end) {
        return ""; // All whitespace
    }

    return std::string(start, end);
}

std::string compute_markdown_hash(std::string_view content) {
    std::ostringstream normalized;
    std::string content_str(content);
    std::istringstream lines(content_str);
    std::string line;
    bool first = true;

    // Process each line: trim and concatenate with \n
    while (std::getline(lines, line)) {
        std::string trimmed = trim(line);

        if (!first) {
            normalized << '\n';
        }
        normalized << trimmed;
        first = false;
    }

    // Compute SHA256 hash of normalized content
    std::string normalized_content = normalized.str();
    cppfort::crdt::SHA256Hash hash = cppfort::crdt::SHA256Hash::compute(normalized_content);

    // Return as lowercase hex string (64 characters)
    return hash.to_hex_string();
}

} // namespace cpp2_transpiler
