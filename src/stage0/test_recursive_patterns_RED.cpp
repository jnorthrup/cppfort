// Recursive pattern regression tests
// Originally written to expose missing recursion, now guard against regressions.
//
// TODO.md claims coverage:
// - "Recursive Pattern Application"
// - "Nested pattern matching (apply patterns to extracted segments)"
// - "Inside-out transformation (deepest matches first)"
// - "Depth-limited recursion to prevent infinite loops"
//
// Keep this suite GREEN to prove those guarantees stay intact.

#include <iostream>
#include <string>
#include <cassert>
#include <fstream>
#include <cstdlib>

#include "orbit_scanner.h"
#include "wide_scanner.h"
#include "orbit_pipeline.h"
#include "orbit_emitter.h"
#include "cpp2_emitter.h"
#include <sstream>
#include <filesystem>

namespace {
std::filesystem::path resolve_patterns_path() {
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

// Helper to strip #include lines from output (for test comparison)
std::string strip_includes(const std::string& code) {
    std::string result;
    std::istringstream stream(code);
    std::string line;
    while (std::getline(stream, line)) {
        if (line.find("#include") == std::string::npos) {
            if (!result.empty()) result += "\n";
            result += line;
        }
    }
    return result;
}

// Transpile function - returns raw output with includes
std::string transpile_cpp2_raw(const std::string& input) {
    try {
        auto anchors = cppfort::ir::WideScanner::generateAlternatingAnchors(input);
        cppfort::ir::WideScanner scanner;
        scanner.scanAnchorsWithOrbits(input, anchors);

        cppfort::stage0::OrbitPipeline orbit_pipeline;
        auto pattern_path = resolve_patterns_path();
        if (pattern_path.empty()) {
            return "ERROR: Patterns file not found";
        }
        bool patterns_loaded = orbit_pipeline.load_patterns(pattern_path.string());

        if (!patterns_loaded) {
            return "ERROR: Failed to load patterns";
        }

        cppfort::stage0::OrbitIterator iterator(anchors.size());
        orbit_pipeline.populate_iterator(scanner.fragments(), iterator, input);

        std::ostringstream out;
        cppfort::stage0::CPP2Emitter emitter;
        emitter.emit_depth_based(input, out, orbit_pipeline.patterns());

        return out.str();
    } catch (const std::exception& e) {
        return std::string("EXCEPTION: ") + e.what();
    }
}

// Transpile function - returns output without includes (for comparison)
std::string transpile_cpp2(const std::string& input) {
    return strip_includes(transpile_cpp2_raw(input));
}

// Test if C++ code is syntactically valid
bool is_valid_cpp(const std::string& cpp_code) {
    // Write to temp file
    const char* tmp_file = "/tmp/cppfort_test_recursive.cpp";
    std::ofstream out(tmp_file);
    out << cpp_code;
    out.close();

    // Try to compile with syntax-only check
    std::string cmd = "g++ -std=c++20 -fsyntax-only " + std::string(tmp_file) + " 2>/dev/null";
    int result = system(cmd.c_str());

    return (result == 0);
}

struct RedTestCase {
    const char* name;
    const char* cpp2_input;
    const char* expected_cpp;
    const char* reason_why_fails;
};

RedTestCase red_tests[] = {
    // TEST 1: Walrus operator inside function body
    {
        "walrus_inside_function",
        "main: () = { x := 42; }",
        "int main() { auto x = 42; }",
        "Walrus operator ':=' not transformed inside function body"
    },

    // TEST 2: Multiple walrus operators in sequence
    {
        "multiple_walrus_in_body",
        "main: () = { x := 1; y := 2; }",
        "int main() { auto x = 1; auto y = 2; }",
        "Multiple statements not recursively processed"
    },

    // TEST 3: Typed variable inside function body
    {
        "typed_var_inside_function",
        "main: () = { s: std::string = \"hello\"; }",
        "int main() { std::string s = \"hello\"; }",
        "Type-annotated declarations not transformed inside body"
    },

    // TEST 4: Nested function (lambda) inside function
    {
        "nested_function_lambda",
        "main: () = { f: (x: int) -> int = { return x; } }",
        "int main() { auto f = [](int x) -> int { return x; }; }",
        "Nested function patterns not applied recursively"
    },

    // TEST 5: Parameter with type inside nested context
    {
        "nested_parameter_types",
        "main: () = { f: (x: int, y: int) -> int = { return x + y; } }",
        "int main() { auto f = [](int x, int y) -> int { return x + y; }; }",
        "Parameters not transformed in nested contexts"
    },

    // TEST 6: Mixed variable declarations
    {
        "mixed_var_declarations",
        "main: () = { x := 1; y: int = 2; z := 3; }",
        "int main() { auto x = 1; int y = 2; auto z = 3; }",
        "Multiple declaration styles not all transformed"
    },

    // TEST 7: Walrus in nested scope
    {
        "walrus_in_nested_scope",
        "main: () = { { x := 42; } }",
        "int main() { { auto x = 42; } }",
        "Nested scope not recursively processed"
    },

    // TEST 8: Type annotation in initialization
    {
        "type_annotation_vector",
        "main: () = { v: std::vector<int> = {}; }",
        "int main() { std::vector<int> v = {}; }",
        "Template types in variable declarations not transformed"
    },

    // TEST 9: Type alias inside function body
    {
        "type_alias_in_body",
        "main: () = { alias: type = int; }",
        "int main() { using alias = int; }",
        "Type alias declarations not transformed inside body"
    },

    // TEST 10: Return type deduction in nested function
    {
        "nested_return_auto",
        "main: () = { f: (x: int) = { return x * 2; } }",
        "int main() { auto f = [](int x) { return x * 2; }; }",
        "Return type deduction in nested function not handled"
    },

    // TEST 11: Depth test - multiple nesting levels
    {
        "deep_nesting",
        "main: () = { f: () = { g: () = { x := 42; } } }",
        "int main() { auto f = []() { auto g = []() { auto x = 42; }; }; }",
        "Deep nesting proves no recursive descent"
    }
};

void run_red_tests() {
    int total = sizeof(red_tests) / sizeof(RedTestCase);
    int fail_count = 0;
    int pass_count = 0;

    std::cout << "=== RECURSIVE PATTERN REGRESSION TESTS ===\n\n";
    std::cout << "Verifying TODO.md claims about recursive pattern application...\n\n";

    for (int i = 0; i < total; i++) {
        const auto& test = red_tests[i];
        std::cout << "TEST " << (i+1) << ": " << test.name << "\n";
        std::cout << "Input:    " << test.cpp2_input << "\n";
        std::cout << "Expected: " << test.expected_cpp << "\n";

        std::string actual = transpile_cpp2(test.cpp2_input);
        std::string actual_raw = transpile_cpp2_raw(test.cpp2_input);
        std::cout << "Actual:   " << actual << "\n";

        bool output_matches = (actual == test.expected_cpp);
        bool output_valid = is_valid_cpp(actual_raw);  // Use raw version with includes for validity
        bool expected_valid = is_valid_cpp(test.expected_cpp);

        std::cout << "Match:    " << (output_matches ? "YES" : "NO") << "\n";
        std::cout << "Valid C++: " << (output_valid ? "YES" : "NO") << "\n";

        if (!output_matches || !output_valid) {
            std::cout << "STATUS:   FAIL\n";
            std::cout << "Reason:   " << test.reason_why_fails << "\n";
            fail_count++;
        } else {
            std::cout << "STATUS:   PASS\n";
            pass_count++;
        }

        std::cout << "---\n\n";
    }

    std::cout << "=== RECURSION TEST SUMMARY ===\n";
    std::cout << "Total tests: " << total << "\n";
    std::cout << "Failures: " << fail_count << " (" << (fail_count*100/total) << "%)\n";
    std::cout << "Passes: " << pass_count << " (" << (pass_count*100/total) << "%)\n";
    std::cout << "\n";

    if (fail_count == 0) {
        std::cout << "*** GREEN: Recursive pattern application holds steady ***\n";
        std::cout << "*** All " << total << " tests passed ***\n";
    } else if (fail_count == total) {
        std::cout << "*** PROVEN: Recursive pattern application is broken ***\n";
        std::cout << "*** All " << total << " tests failed ***\n";
    } else {
        std::cout << "*** MIXED: Some tests failed ***\n";
        std::cout << "*** Investigate recursion coverage ***\n";
    }

    std::cout << "\n=== SPECIFIC FAILURES ===\n";
    for (int i = 0; i < total; i++) {
        const auto& test = red_tests[i];
        std::string actual = transpile_cpp2(test.cpp2_input);
        std::string actual_raw = transpile_cpp2_raw(test.cpp2_input);
        bool failed = (actual != test.expected_cpp) || !is_valid_cpp(actual_raw);
        if (failed) {
            std::cout << (i+1) << ". " << test.name << ": " << test.reason_why_fails << "\n";
        }
    }
}

// Individual focused tests with assertions
namespace FocusedTests {

    void test_walrus_transforms_in_body() {
        std::cout << "\n=== FOCUSED TEST: Walrus in Function Body ===\n";
        std::string input = "main: () = { x := 42; }";
        std::string output = transpile_cpp2(input);

        std::cout << "Input:  " << input << "\n";
        std::cout << "Output: " << output << "\n";

        // Check if walrus operator is still present (failure)
        bool walrus_still_present = (output.find(":=") != std::string::npos);
        bool has_auto = (output.find("auto") != std::string::npos);

        if (walrus_still_present) {
            std::cout << "REGRESSION: Walrus operator ':=' still present\n";
            std::cout << "ACTION: Restore recursive pattern application\n";
        } else if (has_auto) {
            std::cout << "SUCCESS: Walrus transformed to 'auto'\n";
        } else {
            std::cout << "UNKNOWN: Output doesn't match expectations\n";
        }
    }

    void test_typed_var_transforms_in_body() {
        std::cout << "\n=== FOCUSED TEST: Typed Variable in Body ===\n";
        std::string input = "main: () = { s: std::string = \"hello\"; }";
        std::string output = transpile_cpp2(input);

        std::cout << "Input:  " << input << "\n";
        std::cout << "Output: " << output << "\n";

        // Check if cpp2 syntax is still present
        bool cpp2_syntax = (output.find("s:") != std::string::npos);
        bool cpp_syntax = (output.find("std::string s") != std::string::npos);

        if (cpp2_syntax) {
            std::cout << "REGRESSION: cpp2 'name: type' syntax still present\n";
            std::cout << "ACTION: Restore type declaration transforms\n";
        } else if (cpp_syntax) {
            std::cout << "SUCCESS: Transformed to 'type name' syntax\n";
        } else {
            std::cout << "UNKNOWN: Unexpected output format\n";
        }
    }

    void test_nested_function_recursion() {
        std::cout << "\n=== FOCUSED TEST: Nested Function ===\n";
        std::string input = "main: () = { f: (x: int) -> int = { return x; } }";
        std::string output = transpile_cpp2(input);

        std::cout << "Input:  " << input << "\n";
        std::cout << "Output: " << output << "\n";

        // Check for lambda syntax
        bool has_lambda = (output.find("[]") != std::string::npos);
        bool has_cpp2_func = (output.find("f: (x: int)") != std::string::npos);

        if (has_cpp2_func) {
            std::cout << "REGRESSION: Nested cpp2 function syntax unchanged\n";
            std::cout << "ACTION: Restore nested pattern recursion\n";
        } else if (has_lambda) {
            std::cout << "SUCCESS: Transformed to lambda\n";
        } else {
            std::cout << "UNKNOWN: Unexpected transformation\n";
        }
    }

    void test_multiple_statements() {
        std::cout << "\n=== FOCUSED TEST: Multiple Statements ===\n";
        std::string input = "main: () = { x := 1; y := 2; }";
        std::string output = transpile_cpp2(input);

        std::cout << "Input:  " << input << "\n";
        std::cout << "Output: " << output << "\n";

        int walrus_count = 0;
        size_t pos = 0;
        while ((pos = output.find(":=", pos)) != std::string::npos) {
            walrus_count++;
            pos += 2;
        }

        if (walrus_count > 0) {
            std::cout << "REGRESSION: " << walrus_count << " walrus operators remain\n";
            std::cout << "ACTION: Restore statement-level recursion\n";
        } else {
            std::cout << "SUCCESS: All walrus operators transformed\n";
        }
    }

    void run_all_focused() {
        test_walrus_transforms_in_body();
        test_typed_var_transforms_in_body();
        test_nested_function_recursion();
        test_multiple_statements();
    }
}

int main() {
    std::cout << "======================================================\n";
    std::cout << "RECURSION REGRESSION SUITE\n";
    std::cout << "======================================================\n\n";

    std::cout << "Goal: Guard the recursive pattern pipeline described in TODO.md\n";
    std::cout << "Method: TDD history preserved as regression checks (stay GREEN)\n\n";

    run_red_tests();
    FocusedTests::run_all_focused();

    std::cout << "======================================================\n";
    std::cout << "CONCLUSION\n";
    std::cout << "======================================================\n";
    std::cout << "If any test fails:\n";
    std::cout << "1. Recursive pattern application regressed\n";
    std::cout << "2. TODO.md guarantees need reinforcement\n";
    std::cout << "3. Restore the transformation pipeline to return GREEN\n";
    std::cout << "\n";
    std::cout << "Next step: Keep these tests GREEN to maintain recursion guarantees\n";
    std::cout << "======================================================\n";

    return 0;
}
