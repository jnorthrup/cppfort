#pragma once

#include <array>
#include <cstdint>
#include <string>
#include <vector>

namespace cppfort::ir {

// Confix pair types
enum class ConfixType : uint8_t {
    PARENTHESES,    // ()
    BRACKETS,       // []
    BRACES,         // {}
    ANGLE_BRACKETS, // <>
    STRINGS,        // ""
    CHAR_LITERALS,  // ''
    COMMENTS,       // /* */, //
    TEMPLATE_ARGS   // Special template handling
};

// Fishy pattern types
enum class FishyType : uint8_t {
    UNMATCHED_BRACKET,      // Unbalanced (), [], {}, <>
    TYPE_CONTRADICTION,      // Conflicting type evidence
    INDENTATION_JUMP,       // Suspicious indent changes
    KEYWORD_WRONG_CONTEXT,   // Keywords in wrong places
    SUSPICIOUS_DELIMITER,   // Odd delimiter usage
    EMPTY_BLOCK,           // Empty {} or () blocks
    NESTING_OVERFLOW,      // Excessive nesting depth
    MALFORMED_LITERAL      // Incorrect string/char literals
};

// Confix pair tracker
struct ConfixTracker {
    ConfixType confixType;
    char openChar;
    char closeChar;
    size_t openCount = 0;
    size_t closeCount = 0;
    size_t mismatchCount = 0;
    size_t maxDepth = 0;
    size_t currentDepth = 0;
};

// Fishy pattern detection result
struct FishyPattern {
    FishyType fishyType;
    size_t position;
    size_t line;
    size_t column;
    int severity;  // 1-10 scale
    ::std::string message;
    ::std::string context;
    double confidence; // How certain this is fishy (0.0-1.0)
};

// Confix context state
struct ConfixContext {
    ::std::array<ConfixTracker, 8> trackers; // All confix types
    ::std::vector<FishyPattern> fishyPatterns;
    size_t totalMismatches = 0;
    size_t maxNestingDepth = 0;
};

// Confix fishy detector - validates orbit parse results
class ConfixFishyDetector {
public:
    // Initialize confix trackers
    static ConfixContext initializeTrackers();

    // Track confix pairs in source code
    static void trackConfixPairs(ConfixContext& context, const ::std::string& source);

    // Detect fishy patterns based on type evidence contradictions
    static void detectTypeContradictions(ConfixContext& context);

    // Detect suspicious indentation jumps
    static void detectIndentationIssues(ConfixContext& context, const ::std::string& source);

    // Convert confix type to string
    static ::std::string confixTypeToString(ConfixType type);

private:
    // Get context around position
    static ::std::string getContext(const ::std::string& source, size_t pos, size_t window);

    // Helper functions
    static ::std::vector<::std::string> splitLines(const ::std::string& text);
    static size_t countLeadingSpaces(const ::std::string& line);
    static size_t findLineStart(const ::std::string& source, size_t lineNum);
};

} // namespace cppfort::ir
