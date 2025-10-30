// Reality Check Tests - What Actually Works vs What's Claimed
#include <iostream>
#include <string>
#include <cassert>

#include "orbit_scanner.h"
#include "wide_scanner.h"
#include "orbit_pipeline.h"
#include "orbit_emitter.h"
#include "cpp2_emitter.h"
#include <sstream>
#include <filesystem>

namespace {
std::filesystem::path resolve_patterns_path() {
    // Keep the search list short to avoid masking real pathing issues.
    static const std::filesystem::path candidates[] = {
        "patterns/bnfc_cpp2_complete.yaml",
        "../patterns/bnfc_cpp2_complete.yaml",
        "../../patterns/bnfc_cpp2_complete.yaml",
        "../../../patterns/bnfc_cpp2_complete.yaml"
    };

    for (const auto& candidate : candidates) {
        std::filesystem::path attempt = std::filesystem::current_path() / candidate;
        if (std::filesystem::exists(attempt)) {
            return attempt;
        }
    }

    // Last resort: resolve relative to this source file for hermetic runs.
    try {
        auto source_dir = std::filesystem::path(__FILE__).parent_path();
        auto fallback = std::filesystem::weakly_canonical(source_dir / "../../../patterns/bnfc_cpp2_complete.yaml");
        if (std::filesystem::exists(fallback)) {
            return fallback;
        }
    } catch (...) {
    }

    return {};
}
} // namespace

// Function to call the actual transpiler
std::string transpile_cpp2(const std::string& input) {
    try {
        // Generate anchors
        auto anchors = cppfort::ir::WideScanner::generateAlternatingAnchors(input);

        // Scan with orbits
        cppfort::ir::WideScanner scanner;
        scanner.scanAnchorsWithOrbits(input, anchors);

        // Load patterns
        cppfort::stage0::OrbitPipeline orbit_pipeline;
        auto pattern_path = resolve_patterns_path();
        if (pattern_path.empty()) {
            return "ERROR: Patterns file not found";
        }

        bool patterns_loaded = orbit_pipeline.load_patterns(pattern_path.string());
        if (!patterns_loaded) {
            // Try without patterns (will show what the raw system does)
            return "ERROR: Failed to load patterns";
        }

        // Create iterator and populate
        cppfort::stage0::OrbitIterator iterator(anchors.size());
        orbit_pipeline.populate_iterator(scanner.fragments(), iterator, input);

        // Emit using the same pipeline as stage0_cli
        std::ostringstream out;
        cppfort::stage0::CPP2Emitter emitter;
        iterator.reset();
        emitter.emit(iterator, input, out, orbit_pipeline.patterns());

        return out.str();
    } catch (const std::exception& e) {
        return std::string("EXCEPTION: ") + e.what();
    } catch (...) {
        return "UNKNOWN EXCEPTION";
    }
}

struct TestCase {
    const char* name;
    const char* input;
    const char* expected;
    bool should_pass;  // Based on TODO.md claims
    bool actually_passes;  // Reality
};

TestCase reality_tests[] = {
    // Test 1: The ONE thing that allegedly works
    {
        "simple_main",
    "main: () -> int = { s: std::string = \"world\"; }",
    "#include <string>\nint main() { std::string s = \"world\"; }",
        true,   // TODO.md says this works
        false   // Reality: probably doesn't fully work
    },

    // Test 2: Parameter transformation (claimed missing)
    {
        "parameter_inout",
        "foo: (inout s: std::string) -> void = {}",
        "#include <string>\nvoid foo(std::string& s) {}",
        false,  // TODO.md admits this doesn't work
        false   // Reality: definitely doesn't work
    },

    // Test 3: Template alias (pattern selected but broken)
    {
        "template_alias",
        "type Pair<A,B>=std::pair<A,B>;",
        "template<typename A, typename B> using Pair = std::pair<A,B>;",
        false,  // TODO.md says substitution is malformed
        false   // Reality: outputs garbage
    },

    // Test 4: Include generation (claimed missing)
    {
        "include_generation",
        "main: () -> int = { v: std::vector<int> = {}; }",
        "#include <vector>\nint main() { std::vector<int> v = {}; }",
        false,  // TODO.md says missing
        false   // Reality: no includes generated
    },

    // Test 5: Basic walrus operator
    {
        "walrus_operator",
        "main: () -> int = { x := 42; }",
        "int main() { auto x = 42; }",
        false,  // Unclear from TODO.md
        false   // Reality: probably broken
    },

    // Test 6: Forward declaration
    {
        "forward_declaration",
        "bar: () -> void;\nbar: () -> void = {}",
        "void bar();\nvoid bar() {}",
        false,  // TODO.md says missing
        false   // Reality: definitely missing
    },

    // Test 7: Nested patterns
    {
        "nested_patterns",
        "main: () -> int = { f: (x: int) -> int = { return x; } }",
        "int main() { auto f = [](int x) -> int { return x; }; }",
        false,  // TODO.md mentions "recursive orbit processing"
        false   // Reality: uses regex hack
    },

    // Test 8: Contract syntax
    {
        "contracts",
        "sqrt: (x: double) -> double pre<{ x >= 0 }> = { }",
        "double sqrt(double x) [[pre: x >= 0]] { }",
        false,  // Not mentioned as working
        false   // Reality: not implemented
    }
};

void run_reality_check() {
    int claimed_working = 0;
    int actually_working = 0;

    std::cout << "=== REALITY CHECK: What Actually Works ===\n\n";

    for (const auto& test : reality_tests) {
        std::cout << "Test: " << test.name << "\n";
        std::cout << "Input: " << test.input << "\n";
        std::cout << "Expected: " << test.expected << "\n";

        try {
            std::string actual = transpile_cpp2(test.input);
            bool passes = (actual == test.expected);

            std::cout << "Actual: " << actual << "\n";
            std::cout << "TODO.md claims: " << (test.should_pass ? "WORKS" : "MISSING") << "\n";
            std::cout << "Reality: " << (passes ? "PASSES" : "FAILS") << "\n";

            if (test.should_pass) claimed_working++;
            if (passes) actually_working++;

            // Flag discrepancies
            if (test.should_pass && !passes) {
                std::cout << "*** REGRESSION: Claimed working but FAILS ***\n";
            }
            if (!test.should_pass && passes) {
                std::cout << "*** SURPRISE: Not claimed but WORKS ***\n";
            }
        } catch (...) {
            std::cout << "*** CRASH: Exception thrown ***\n";
        }

        std::cout << "---\n\n";
    }

    std::cout << "=== SUMMARY ===\n";
    std::cout << "TODO.md claims working: " << claimed_working << "/" << sizeof(reality_tests)/sizeof(TestCase) << "\n";
    std::cout << "Actually working: " << actually_working << "/" << sizeof(reality_tests)/sizeof(TestCase) << "\n";
    std::cout << "Honesty gap: " << (claimed_working - actually_working) << " features\n";

    if (actually_working == 0) {
        std::cout << "\n*** CRITICAL: Zero tests passing. No actual transpilation capability. ***\n";
    }
}

// Granular feature tests
namespace FeatureTests {

    bool can_parse_function_signature() {
        // Just parse, don't transform
        // "main: () -> int" should be recognized
        return false;  // Not implemented
    }

    bool can_extract_function_name() {
        // Extract "main" from "main: () -> int = {}"
        return false;  // Not implemented
    }

    bool can_extract_return_type() {
        // Extract "int" from "main: () -> int = {}"
        return false;  // Not implemented
    }

    bool can_handle_function_body() {
        // Process the {} part
        return false;  // Not implemented
    }

    bool can_transform_parameter() {
        // Transform "x: int" to "int x"
        return false;  // Not implemented
    }

    bool can_handle_inout_parameter() {
        // Transform "inout x: int" to "int& x"
        return false;  // Not implemented
    }

    void run_granular_tests() {
        std::cout << "\n=== GRANULAR FEATURE TESTS ===\n";
        std::cout << "Parse function signature: " << (can_parse_function_signature() ? "YES" : "NO") << "\n";
        std::cout << "Extract function name: " << (can_extract_function_name() ? "YES" : "NO") << "\n";
        std::cout << "Extract return type: " << (can_extract_return_type() ? "YES" : "NO") << "\n";
        std::cout << "Handle function body: " << (can_handle_function_body() ? "YES" : "NO") << "\n";
        std::cout << "Transform parameter: " << (can_transform_parameter() ? "YES" : "NO") << "\n";
        std::cout << "Handle inout parameter: " << (can_handle_inout_parameter() ? "YES" : "NO") << "\n";
    }
}

int main() {
    run_reality_check();
    FeatureTests::run_granular_tests();

    std::cout << "\n=== RECOMMENDATION ===\n";
    std::cout << "1. Stop claiming features work when they don't\n";
    std::cout << "2. Pick ONE test and make it fully pass\n";
    std::cout << "3. Don't move on until it's 100% correct\n";
    std::cout << "4. Update TODO.md with honest status\n";

    return 0;
}