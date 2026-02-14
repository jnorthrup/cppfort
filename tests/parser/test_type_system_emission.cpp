// Test: Type System Parsing and Emission
// Verifies that Cpp2 type definitions `Name: type = { ... }` are correctly emitted.
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

namespace test_type_system {

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

// Test 1: Basic type definition emits class
void test_basic_type() {
    std::cout << "Running test_basic_type...\n";
    check_emit(
        R"(Widget: type = { val: int = 0; })",
        "class Widget",
        "Basic type definition emits 'class Widget'"
    );
}

// Test 2: Type with member variable
void test_type_with_member() {
    std::cout << "Running test_type_with_member...\n";
    check_emit(
        R"(Point: type = { x: int = 0; y: int = 0; })",
        "class Point",
        "Type with members emits class"
    );
}

// Test 3: Type with method
void test_type_with_method() {
    std::cout << "Running test_type_with_method...\n";
    check_emit(
        R"(Widget: type = { get_val: (this) -> int = { return 42; } })",
        "Widget",
        "Type with method emits class name"
    );
}

// Test 4: @value metafunction generates comparison
void test_value_type() {
    std::cout << "Running test_value_type...\n";
    check_emit(
        R"(Widget: @value type = { val: int = 0; })",
        "operator<=>",
        "@value type generates operator<=>"
    );
}

// Test 5: @interface metafunction
void test_interface_type() {
    std::cout << "Running test_interface_type...\n";
    check_emit(
        R"(Shape: @interface type = { draw: (this) -> void; area: (this) -> double; })",
        "Shape",
        "@interface type emits class name"
    );
}

// Test 6: Type with public/private sections
void test_type_class_keyword() {
    std::cout << "Running test_type_class_keyword...\n";
    // Verify basic type definition is emitted as class (not struct by default)
    check_emit(
        R"(Animal: type = { name: std::string = ""; })",
        "class Animal",
        "Type emits as 'class' by default"
    );
}

} // namespace test_type_system

int main() {
    std::cout << "=== Type System Tests ===\n\n";

    test_type_system::test_basic_type();
    test_type_system::test_type_with_member();
    test_type_system::test_type_with_method();
    test_type_system::test_value_type();
    test_type_system::test_interface_type();
    test_type_system::test_type_class_keyword();

    std::cout << "\nAll type system tests passed.\n";
    return 0;
}
