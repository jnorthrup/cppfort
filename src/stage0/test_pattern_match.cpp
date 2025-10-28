#include <iostream>
#include <cassert>
#include "unified_pattern_matcher.h"

using namespace cppfort::stage0;

int main() {
    std::cout << "=== Pattern Matcher Test ===\n\n";

    std::string pattern = "$0: ($1) -> $2 = $3";
    std::string input = "main: () -> int = { s: std::string = \"world\"; }";

    auto result = UnifiedPatternMatcher::extract_segments(pattern, input);

    if (!result) {
        std::cerr << "FAILED: Pattern did not match\n";
        return 1;
    }

    auto& segments = *result;
    std::cout << "Matched " << segments.size() << " segments:\n";
    for (size_t i = 0; i < segments.size(); ++i) {
        std::cout << "  $" << i << " = \"" << segments[i] << "\"\n";
    }

    // Verify
    assert(segments.size() == 4);
    assert(segments[0] == "main");
    assert(segments[1] == "");  // Empty params
    assert(segments[2] == "int");
    // segments[3] should contain the body

    std::cout << "\n✓ Pattern matching works!\n";
    std::cout << "\nNow apply CPP template: $2 $0($1) $3\n";

    std::string cpp_output = segments[2] + " " + segments[0] + "(" + segments[1] + ") " + segments[3];
    std::cout << "Result: " << cpp_output << "\n";

    return 0;
}
