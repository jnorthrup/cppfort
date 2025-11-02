// Reality Check Tests - What Actually Works vs What's Claimed
#include <iostream>
#include <string>
#include <cassert>
#include <cctype>

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
    "patterns/cppfort_core_patterns.yaml",
    "../patterns/cppfort_core_patterns.yaml",
    "../../patterns/cppfort_core_patterns.yaml",
    "../../../patterns/cppfort_core_patterns.yaml"
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
    auto fallback = std::filesystem::weakly_canonical(source_dir / "../../../patterns/cppfort_core_patterns.yaml");
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
        true,   // TODO.md now claims inout lowering works
        true    // Reality: verified via stage0_cli
    },

    // Test 3: Template alias (pattern selected but broken)
    {
        "template_alias",
        "type Pair<A,B>=std::pair<A,B>;",
        "template<typename A, typename B> using Pair = std::pair<A,B>;",
        true,   // TODO.md now claims alias substitution works
        true    // Reality: stage0_cli emits correct alias
    },

    // Test 4: Include generation (claimed missing)
    {
        "include_generation",
        "main: () -> int = { v: std::vector<int> = {}; }",
        "#include <vector>\nint main() { std::vector<int> v = {}; }",
        true,   // TODO.md now claims include emission works
        true    // Reality: include + body emitted
    },

    // Test 5: Basic walrus operator
    {
        "walrus_operator",
        "main: () -> int = { x := 42; }",
        "int main() { auto x = 42; }",
        true,   // TODO.md now claims walrus lowering works
        true    // Reality: matches expectation
    },

    // Test 6: Forward declaration
    {
        "forward_declaration",
        "bar: () -> void;\nbar: () -> void = {}",
        "void bar();\nvoid bar() {}",
        true,   // TODO.md now claims forward declarations work
        true    // Reality: declaration + definition emitted
    },

    // Test 7: Nested patterns
    {
        "nested_patterns",
        "main: () -> int = { f: (x: int) -> int = { return x; } }",
        "int main() { auto f = [](int x) -> int { return x; }; }",
        true,   // TODO.md now claims nested lambda lowering works
        true    // Reality: lambda emitted correctly
    },

    // Test 8: Contract syntax
    {
        "contracts",
        "sqrt: (x: double) -> double pre<{ x >= 0 }> = { }",
        "double sqrt(double x) [[pre: x >= 0]] { }",
        true,   // TODO.md now claims contract emission works
        true    // Reality: pre-condition emitted
    }
};

bool run_reality_check() {
    int claimed_working = 0;
    int actually_working = 0;
    bool all_passed = true;

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
                all_passed = false;
            }
            if (!test.should_pass && passes) {
                std::cout << "*** SURPRISE: Not claimed but WORKS ***\n";
            }
        } catch (...) {
            std::cout << "*** CRASH: Exception thrown ***\n";
            all_passed = false;
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

    return all_passed;
}

// Granular feature tests
namespace FeatureTests {

    static std::string trim_text(std::string text) {
        size_t start = 0;
        while (start < text.size() && std::isspace(static_cast<unsigned char>(text[start]))) {
            ++start;
        }

        size_t end = text.size();
        while (end > start && std::isspace(static_cast<unsigned char>(text[end - 1]))) {
            --end;
        }

        return text.substr(start, end - start);
    }

    static bool matches(const std::string& input, const std::string& expected) {
        return trim_text(transpile_cpp2(input)) == trim_text(expected);
    }

    bool can_parse_function_signature() {
        // Just parse, don't transform
        // "main: () -> int" should be recognized
    return matches("main: () -> int = { }", "int main() {}");
    }

    bool can_extract_function_name() {
        // Extract "main" from "main: () -> int = {}"
        return matches("hello: () -> void = { }", "void hello() { }");
    }

    bool can_extract_return_type() {
        // Extract "int" from "main: () -> int = {}"
        return matches("foo: () -> double = { }", "double foo() { }");
    }

    bool can_handle_function_body() {
        // Process the {} part
        return matches("main: () -> int = { return 42; }", "int main() { return 42; }");
    }

    bool can_transform_parameter() {
        // Transform "x: int" to "int x"
        return matches("add: (x: int, y: int) -> int = { return x + y; }",
                       "int add(int x, int y) { return x + y; }");
    }

    bool can_handle_inout_parameter() {
        // Transform "inout x: int" to "int& x"
        return matches("use: (inout s: std::string) -> void = { }",
                       "#include <string>\nvoid use(std::string& s) { }");
    }

    bool deduplicates_includes() {
        // Avoid emitting duplicate includes when source already has them
        return matches("#include <iostream>\n\nmain: () -> int = { std::cout << \"hi\"; }",
                       "#include <iostream>\n\nint main() { std::cout << \"hi\"; }");
    }

    bool run_granular_tests() {
        auto report = [](const char* label, bool passed) {
            std::cout << label << ": " << (passed ? "YES" : "NO") << "\n";
            return passed;
        };

        std::cout << "\n=== GRANULAR FEATURE TESTS ===\n";

        bool all_passed = true;
        all_passed &= report("Parse function signature", can_parse_function_signature());
        all_passed &= report("Extract function name", can_extract_function_name());
        all_passed &= report("Extract return type", can_extract_return_type());
        all_passed &= report("Handle function body", can_handle_function_body());
        all_passed &= report("Transform parameter", can_transform_parameter());
        all_passed &= report("Handle inout parameter", can_handle_inout_parameter());
        all_passed &= report("Deduplicate includes", deduplicates_includes());

        return all_passed;
    }
}

int main() {
    bool reality_ok = run_reality_check();
    bool feature_ok = FeatureTests::run_granular_tests();

    std::cout << "\n=== RECOMMENDATION ===\n";
    std::cout << "1. Stop claiming features work when they don't\n";
    std::cout << "2. Pick ONE test and make it fully pass\n";
    std::cout << "3. Don't move on until it's 100% correct\n";
    std::cout << "4. Update TODO.md with honest status\n";

    return (reality_ok && feature_ok) ? 0 : 1;
}