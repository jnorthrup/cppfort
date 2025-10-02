#include "confix_fishy_detector.h"

#include <algorithm>
#include <cmath>

namespace cppfort::ir {

ConfixContext ConfixFishyDetector::initializeTrackers() {
    ConfixContext context;

    // Initialize all confix trackers
    context.trackers[0] = ConfixTracker{
        ConfixType::PARENTHESES, '(', ')'
    };

    context.trackers[1] = ConfixTracker{
        ConfixType::BRACKETS, '[', ']'
    };

    context.trackers[2] = ConfixTracker{
        ConfixType::BRACES, '{', '}'
    };

    context.trackers[3] = ConfixTracker{
        ConfixType::ANGLE_BRACKETS, '<', '>'
    };

    context.trackers[4] = ConfixTracker{
        ConfixType::STRINGS, '"', '"'
    };

    context.trackers[5] = ConfixTracker{
        ConfixType::CHAR_LITERALS, '\'', '\''
    };

    context.trackers[6] = ConfixTracker{
        ConfixType::COMMENTS, '/', '/'
    };

    context.trackers[7] = ConfixTracker{
        ConfixType::TEMPLATE_ARGS, '<', '>'
    };

    return context;
}

void ConfixFishyDetector::trackConfixPairs(ConfixContext& context, const ::std::string& source) {
    size_t line = 1;
    size_t column = 1;

    for (size_t i = 0; i < source.size(); ++i) {
        char ch = source[i];

        // Update line/column tracking
        if (ch == '\n') {
            line++;
            column = 1;
        } else {
            column++;
        }

        // Check each confix tracker
        for (auto& tracker : context.trackers) {
            // Special handling for different confix types
            switch (tracker.confixType) {
                case ConfixType::PARENTHESES:
                case ConfixType::BRACKETS:
                case ConfixType::BRACES:
                case ConfixType::ANGLE_BRACKETS:
                    if (ch == tracker.openChar) {
                        tracker.openCount++;
                        tracker.currentDepth++;
                        tracker.maxDepth = ::std::max(tracker.maxDepth, tracker.currentDepth);
                    } else if (ch == tracker.closeChar) {
                        tracker.closeCount++;
                        if (tracker.currentDepth > 0) {
                            tracker.currentDepth--;
                        } else {
                            // Mismatch - closing without opening
                            tracker.mismatchCount++;
                            context.totalMismatches++;

                            // Flag fishy pattern immediately
                            FishyPattern fishy{
                                FishyType::UNMATCHED_BRACKET,
                                i,
                                line,
                                column,
                                8,
                                "Unmatched closing bracket '" + ::std::string(1, ch) + "'",
                                getContext(source, i, 20),
                                0.95
                            };
                            context.fishyPatterns.push_back(fishy);
                        }
                    }
                    break;

                case ConfixType::STRINGS:
                    if (ch == '"' && (i == 0 || source[i-1] != '\\')) {
                        if (tracker.currentDepth == 0) {
                            // Opening string
                            tracker.currentDepth = 1;
                        } else {
                            // Closing string
                            tracker.currentDepth = 0;
                        }
                    }
                    break;

                case ConfixType::CHAR_LITERALS:
                    if (ch == '\'' && (i == 0 || source[i-1] != '\\')) {
                        if (tracker.currentDepth == 0) {
                            // Opening char literal
                            tracker.currentDepth = 1;
                        } else {
                            // Closing char literal
                            tracker.currentDepth = 0;
                        }
                    }
                    break;

                default:
                    break;
            }
        }
    }

    // Check for unclosed confix at end
    for (auto& tracker : context.trackers) {
        if (tracker.currentDepth > 0) {
            // Unclosed confix
            tracker.mismatchCount++;
            context.totalMismatches++;

            FishyPattern fishy{
                FishyType::UNMATCHED_BRACKET,
                source.size() - 1,
                line,
                column,
                7,
                "Unclosed " + confixTypeToString(tracker.confixType),
                "End of file",
                0.9
            };
            context.fishyPatterns.push_back(fishy);
        }
    }
}

void ConfixFishyDetector::detectTypeContradictions(ConfixContext& context) {
    // Look for conflicting type evidence
    // This would connect with the typevidence engine
    // For now, placeholder
    size_t contradictionCount = 0;

    if (contradictionCount > 0) {
        FishyPattern fishy{
            FishyType::TYPE_CONTRADICTION,
            0,
            1,
            1,
            6,
            "Detected " + ::std::to_string(contradictionCount) + " type contradictions",
            "Multiple conflicting type inferences",
            0.85
        };
        context.fishyPatterns.push_back(fishy);
    }
}

void ConfixFishyDetector::detectIndentationIssues(ConfixContext& context, const ::std::string& source) {
    auto lines = splitLines(source);
    size_t prevIndent = 0;

    for (size_t i = 0; i < lines.size(); ++i) {
        const auto& line = lines[i];
        size_t currentIndent = countLeadingSpaces(line);

        // Large indentation jump (> 8 spaces)
        if (currentIndent > prevIndent + 8 && prevIndent > 0) {
            FishyPattern fishy{
                FishyType::INDENTATION_JUMP,
                findLineStart(source, i),
                i + 1,
                currentIndent,
                3,
                "Suspicious indentation jump from " + ::std::to_string(prevIndent) +
                " to " + ::std::to_string(currentIndent) + " spaces",
                line.substr(0, ::std::min(line.size(), size_t(40))),
                0.7
            };
            context.fishyPatterns.push_back(fishy);
        }

        prevIndent = currentIndent;
    }
}

::std::string ConfixFishyDetector::confixTypeToString(ConfixType type) {
    switch (type) {
        case ConfixType::PARENTHESES: return "parentheses";
        case ConfixType::BRACKETS: return "brackets";
        case ConfixType::BRACES: return "braces";
        case ConfixType::ANGLE_BRACKETS: return "angle brackets";
        case ConfixType::STRINGS: return "string literal";
        case ConfixType::CHAR_LITERALS: return "character literal";
        case ConfixType::COMMENTS: return "comment";
        case ConfixType::TEMPLATE_ARGS: return "template arguments";
        default: return "unknown";
    }
}

::std::string ConfixFishyDetector::getContext(const ::std::string& source, size_t pos, size_t window) {
    size_t start = pos > window ? pos - window : 0;
    size_t end = pos + window < source.size() ? pos + window : source.size();
    return source.substr(start, end - start);
}

::std::vector<::std::string> ConfixFishyDetector::splitLines(const ::std::string& text) {
    ::std::vector<::std::string> lines;
    size_t start = 0;

    for (size_t i = 0; i < text.size(); ++i) {
        if (text[i] == '\n') {
            lines.push_back(text.substr(start, i - start));
            start = i + 1;
        }
    }

    // Add last line
    if (start < text.size()) {
        lines.push_back(text.substr(start));
    }

    return lines;
}

size_t ConfixFishyDetector::countLeadingSpaces(const ::std::string& line) {
    size_t count = 0;
    for (char ch : line) {
        if (ch == ' ') {
            count++;
        } else {
            break;
        }
    }
    return count;
}

size_t ConfixFishyDetector::findLineStart(const ::std::string& source, size_t lineNum) {
    size_t currentLine = 0;
    for (size_t i = 0; i < source.size(); ++i) {
        if (currentLine == lineNum) {
            return i;
        }
        if (source[i] == '\n') {
            currentLine++;
        }
    }
    return source.size();
}

} // namespace cppfort::ir
