// Test: Template Parsing and Emission
// Verifies that Cpp2 template syntax is correctly emitted.
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

namespace test_template {

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

// Test 1: Simple template function with type parameter
void test_template_function_type_param() {
    std::cout << "Running test_template_function_type_param...\n";
    // Cpp2: identity: <T> (x: T) -> T = { return x; }
    check_emit(
        R"(identity: <T> (x: T) -> T = { return x; })",
        "template",
        "Template function emits 'template' keyword"
    );
}

// Test 2: Template function generates proper syntax
void test_template_function_generation() {
    std::cout << "Running test_template_function_generation...\n";
    check_emit(
        R"(identity: <T> (x: T) -> T = { return x; })",
        "identity",
        "Template function name is preserved"
    );
}

// Test 3: Template arg in type usage - std::vector<int>
void test_template_type_usage() {
    std::cout << "Running test_template_type_usage...\n";
    check_emit(
        R"(f: () = { v: std::vector<int> = (); })",
        "std::vector<int>",
        "Template type usage: std::vector<int>"
    );
}

// Test 4: Nested template types
void test_nested_template_types() {
    std::cout << "Running test_nested_template_types...\n";
    check_emit(
        R"(f: () = { m: std::map<std::string, int> = (); })",
        "std::map<std::string",
        "Nested template types: std::map<std::string, int>"
    );
}

// Test 5: Function with multiple template parameters
void test_multiple_template_params() {
    std::cout << "Running test_multiple_template_params...\n";
    check_emit(
        R"(add: <T, U> (a: T, b: U) -> T = { return a; })",
        "template",
        "Multiple template params emit template keyword"
    );
}

// Test 6: Template with auto return type
void test_template_auto_return() {
    std::cout << "Running test_template_auto_return...\n";
    check_emit(
        R"(wrap: <T> (x: T) = { return x; })",
        "template",
        "Template with deduced return type emits template"
    );
}

} // namespace test_template

int main() {
    std::cout << "=== Template Tests ===\n\n";

    test_template::test_template_function_type_param();
    test_template::test_template_function_generation();
    test_template::test_template_type_usage();
    test_template::test_nested_template_types();
    test_template::test_multiple_template_params();
    test_template::test_template_auto_return();

    std::cout << "\nAll template tests passed.\n";
    return 0;
}
