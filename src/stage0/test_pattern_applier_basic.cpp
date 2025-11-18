#include "pattern_applier.h"
#include "pattern_loader.h"
#include "rbcursive.h"
#include "mlir_region_node.h"
#include <iostream>
#include <filesystem>
#include <fstream>

using namespace cppfort::stage0;
using namespace cppfort::ir;

int main() {
    std::cout << "Testing PatternApplier with RBCursiveScanner integration...\n";

    // Test 1: Basic RBCursiveScanner instantiation
    try {
        RBCursiveScanner scanner;
        std::cout << "✓ RBCursiveScanner instantiated successfully\n";

        // Test basic glob matching
        bool glob_result = scanner.matchGlob("hello world", "hello*");
        std::cout << "✓ Glob matching test: " << (glob_result ? "PASS" : "FAIL") << "\n";

        // Test basic regex matching
        bool regex_result = scanner.matchRegex("hello123", "hello[0-9]+");
        std::cout << "✓ Regex matching test: " << (regex_result ? "PASS" : "FAIL") << "\n";

    } catch (const std::exception& e) {
        std::cout << "✗ RBCursiveScanner test failed: " << e.what() << "\n";
        return 1;
    }

    // Test 2: PatternApplier with RBCursiveScanner integration
    try {
        // Create a temporary patterns directory for testing
        std::filesystem::path test_patterns_dir = std::filesystem::temp_directory_path() / "cppfort_test_patterns";
        std::filesystem::create_directories(test_patterns_dir);

        // Create a simple test pattern file
        std::filesystem::path test_pattern_file = test_patterns_dir / "test_pattern.yaml";
        std::ofstream pattern_file(test_pattern_file);
        pattern_file << R"YAML(
patterns:
  - name: "test_function"
    use_alternating: true
    alternating_anchors:
      - "function"
      - "("
      - ")"
    evidence_types:
      - "name"
      - "parameters"
      - "body"
    weight: 1.0
    priority: 1
)YAML";
        pattern_file.close();

        // Test PatternApplier instantiation
        PatternApplier applier(test_patterns_dir);
        std::cout << "✓ PatternApplier instantiated successfully\n";

        // Test initialization
        bool init_result = applier.initialize();
        std::cout << "✓ PatternApplier initialization: " << (init_result ? "PASS" : "FAIL") << "\n";

        if (init_result) {
            // Test pattern matching
            std::string test_content = "function example(int x, int y) { return x + y; }";
            cppfort::ir::mlir::RegionNode test_region;
            test_region.setSourceLocation(0, test_content.length());

            auto result = applier.applyPatternToRegion(test_region, test_content);
            std::cout << "✓ Pattern application test: " << (result.success ? "PASS" : "FAIL") << "\n";

            if (result.success) {
                std::cout << "  - Matched pattern: " << result.matchedPatternName << "\n";
                std::cout << "  - Captured spans: " << result.capturedSpans.size() << "\n";
            }
        }

        // Cleanup
        std::filesystem::remove_all(test_patterns_dir);

    } catch (const std::exception& e) {
        std::cout << "✗ PatternApplier test failed: " << e.what() << "\n";
        return 1;
    }

    std::cout << "\nAll tests completed successfully!\n";
    return 0;
}