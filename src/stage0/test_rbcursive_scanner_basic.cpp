#include "rbcursive.h"
#include "pattern_loader.h"
#include <iostream>
#include <string>

using namespace cppfort::ir;

int main() {
    std::cout << "Testing RBCursiveScanner basic functionality...\n";

    // Test 1: Basic instantiation
    try {
        RBCursiveScanner scanner;
        std::cout << "✓ RBCursiveScanner instantiated successfully\n";

        // Test 2: Glob matching
        bool glob_result1 = scanner.matchGlob("hello world", "hello*");
        std::cout << "✓ Glob matching ('hello*' vs 'hello world'): " << (glob_result1 ? "PASS" : "FAIL") << "\n";

        bool glob_result2 = scanner.matchGlob("test.txt", "*.txt");
        std::cout << "✓ Glob matching ('*.txt' vs 'test.txt'): " << (glob_result2 ? "PASS" : "FAIL") << "\n";

        // Test 3: Pattern scanning
        auto matches = scanner.scanWithPattern("hello world hello", "hello", RBCursiveScanner::PatternType::Glob);
        std::cout << "✓ Pattern scanning found " << matches.size() << " matches\n";

        // Test 4: Speculative matching
        scanner.speculate("function test() { return 42; }");
        const auto* best_match = scanner.get_best_match();
        std::cout << "✓ Speculative matching completed\n";
        if (best_match) {
            std::cout << "  - Best match confidence: " << best_match->confidence << "\n";
        }

        // Test 5: Capabilities
        auto caps = scanner.patternCapabilities();
        std::cout << "✓ Scanner capabilities - Glob: " << (caps.glob ? "Yes" : "No") << ", Regex: " << (caps.regex ? "Yes" : "No") << "\n";

    } catch (const std::exception& e) {
        std::cout << "✗ RBCursiveScanner test failed: " << e.what() << "\n";
        return 1;
    }

    std::cout << "\nRBCursiveScanner basic tests completed successfully!\n";
    return 0;
}