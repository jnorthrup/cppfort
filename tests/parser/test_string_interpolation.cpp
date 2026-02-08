// Test: String Interpolation Parsing and Emission
// Verifies that Cpp2 string interpolation `"(expr)$"` is correctly emitted as
// `"" + cpp2::to_string(expr) + ""` in the generated C++ output.
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <iostream>
#include <string>
#include <cassert>

#include "lexer.hpp"
#include "slim_ast.hpp"
#include "emitter.hpp"

// Include implementations
#include "../../src/lexer.cpp"
#include "../../src/parser.cpp"
#include "../../src/emitter.cpp"

namespace test_string_interpolation {

bool contains(const std::string& str, const std::string& substr) {
    return str.find(substr) != std::string::npos;
}

void check_emit(const std::string& input, const std::string& expected_contains, const char* test_name) {
    cpp2_transpiler::Lexer lexer(input);
    auto tokens = lexer.tokenize();
    auto tree = cpp2::parser::parse(tokens);

    std::string output = generate_from_tree(tree, tokens);

    if (output.find(expected_contains) == std::string::npos) {
        std::cerr << "FAIL: " << test_name << "\n";
        std::cerr << "Input: " << input << "\n";
        std::cerr << "Expected to contain: " << expected_contains << "\n";
        std::cerr << "Actual output:\n" << output << "\n";
        std::exit(1);
    }

    std::cout << "PASS: " << test_name << "\n";
}

void check_not_emit(const std::string& input, const std::string& not_expected, const char* test_name) {
    cpp2_transpiler::Lexer lexer(input);
    auto tokens = lexer.tokenize();
    auto tree = cpp2::parser::parse(tokens);

    std::string output = generate_from_tree(tree, tokens);

    if (output.find(not_expected) != std::string::npos) {
        std::cerr << "FAIL: " << test_name << "\n";
        std::cerr << "Input: " << input << "\n";
        std::cerr << "Should NOT contain: " << not_expected << "\n";
        std::cerr << "Actual output:\n" << output << "\n";
        std::exit(1);
    }

    std::cout << "PASS: " << test_name << "\n";
}

// Test 1: Basic variable interpolation
void test_basic_interpolation() {
    std::cout << "Running test_basic_interpolation...\n";
    // Input: f: () = { x := 42; std::cout << "(x)$"; }
    // Expected: output contains cpp2::to_string(x)
    check_emit(
        R"(f: () = { x := 42; std::cout << "(x)$"; })",
        "cpp2::to_string(x)",
        "Basic variable interpolation (x)$ -> cpp2::to_string(x)"
    );
}

// Test 2: Interpolation with surrounding text
void test_interpolation_with_text() {
    std::cout << "Running test_interpolation_with_text...\n";
    // Input: f: () = { x := 42; std::cout << "value=(x)$!"; }
    // Expected: "value=" + cpp2::to_string(x) + "!"
    check_emit(
        R"(f: () = { x := 42; std::cout << "value=(x)$!"; })",
        R"("value=" + cpp2::to_string(x) + "!")",
        "Interpolation with surrounding text"
    );
}

// Test 3: Multiple interpolations in one string
void test_multiple_interpolations() {
    std::cout << "Running test_multiple_interpolations...\n";
    // Input: f: () = { x := 1; y := 2; std::cout << "(x)$ and (y)$"; }
    // Expected: cpp2::to_string(x) and cpp2::to_string(y) both present
    check_emit(
        R"(f: () = { x := 1; y := 2; std::cout << "(x)$ and (y)$"; })",
        "cpp2::to_string(x)",
        "Multiple interpolations - first"
    );
    check_emit(
        R"(f: () = { x := 1; y := 2; std::cout << "(x)$ and (y)$"; })",
        "cpp2::to_string(y)",
        "Multiple interpolations - second"
    );
}

// Test 4: No interpolation - plain string unchanged
void test_no_interpolation() {
    std::cout << "Running test_no_interpolation...\n";
    // A string without )$ should not be changed
    check_emit(
        R"(f: () = { std::cout << "hello world"; })",
        R"("hello world")",
        "No interpolation - plain string unchanged"
    );
    check_not_emit(
        R"(f: () = { std::cout << "hello world"; })",
        "cpp2::to_string",
        "No interpolation - no to_string call"
    );
}

// Test 5: Interpolation at string start
void test_interpolation_at_start() {
    std::cout << "Running test_interpolation_at_start...\n";
    // "(x)$end" -> "" + cpp2::to_string(x) + "end"
    check_emit(
        R"(f: () = { x := 42; std::cout << "(x)$end"; })",
        "cpp2::to_string(x)",
        "Interpolation at string start"
    );
}

// Test 6: Interpolation at string end
void test_interpolation_at_end() {
    std::cout << "Running test_interpolation_at_end...\n";
    // "start(x)$" -> "start" + cpp2::to_string(x) + ""
    check_emit(
        R"(f: () = { x := 42; std::cout << "start(x)$"; })",
        "cpp2::to_string(x)",
        "Interpolation at string end"
    );
}

// Test 7: Expression interpolation (not just variables)
void test_expression_interpolation() {
    std::cout << "Running test_expression_interpolation...\n";
    // "(x + 1)$" should contain cpp2::to_string with the expression
    check_emit(
        R"(f: () = { x := 42; std::cout << "(x + 1)$"; })",
        "cpp2::to_string(x + 1)",
        "Expression interpolation"
    );
}

// Test 8: Unmatched parentheses (not an interpolation)
void test_unmatched_parens() {
    std::cout << "Running test_unmatched_parens...\n";
    // "pl(ug$h" should NOT be treated as interpolation (no closing )$)
    check_not_emit(
        R"(f: () = { std::cout << "pl(ug$h"; })",
        "cpp2::to_string",
        "Unmatched parens - no interpolation"
    );
}

// Test 9: Format specifier interpolation
void test_format_specifier() {
    std::cout << "Running test_format_specifier...\n";
    // "(x:20)$" -> cpp2::to_string(x, "{:20}")
    check_emit(
        R"(f: () = { x := "hello"; std::cout << "(x:20)$"; })",
        R"(cpp2::to_string(x, "{:20}"))",
        "Format specifier interpolation"
    );
}

} // namespace test_string_interpolation

int main() {
    std::cout << "=== String Interpolation Tests ===\n\n";

    test_string_interpolation::test_basic_interpolation();
    test_string_interpolation::test_interpolation_with_text();
    test_string_interpolation::test_multiple_interpolations();
    test_string_interpolation::test_no_interpolation();
    test_string_interpolation::test_interpolation_at_start();
    test_string_interpolation::test_interpolation_at_end();
    test_string_interpolation::test_expression_interpolation();
    test_string_interpolation::test_unmatched_parens();
    test_string_interpolation::test_format_specifier();

    std::cout << "\nAll string interpolation tests passed.\n";
    return 0;
}
