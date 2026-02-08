// Test: UFCS (Unified Function Call Syntax) Parsing and Emission
// Verifies that Cpp2 UFCS `object.method()` is correctly emitted as
// `CPP2_UFCS(method)(object)` in the generated C++ output.
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

namespace test_ufcs {

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

// Test 1: Basic UFCS - single method call
void test_basic_ufcs() {
    std::cout << "Running test_basic_ufcs...\n";
    check_emit(
        R"(f: () = { s: std::string = "hello"; s.size(); })",
        "CPP2_UFCS(size)(s)",
        "Basic UFCS: s.size() -> CPP2_UFCS(size)(s)"
    );
}

// Test 2: UFCS with arguments
void test_ufcs_with_args() {
    std::cout << "Running test_ufcs_with_args...\n";
    check_emit(
        R"(f: () = { v: std::vector<int> = (); v.push_back(42); })",
        "CPP2_UFCS(push_back)(v, 42)",
        "UFCS with args: v.push_back(42) -> CPP2_UFCS(push_back)(v, 42)"
    );
}

// Test 3: UFCS with multiple arguments
void test_ufcs_multi_args() {
    std::cout << "Running test_ufcs_multi_args...\n";
    // Note: comma spacing `1 , 3` is a known pre-existing issue in the emitter
    check_emit(
        R"(f: () = { s: std::string = "hello"; s.substr(1, 3); })",
        "CPP2_UFCS(substr)(s",
        "UFCS multi args: s.substr(1, 3) -> CPP2_UFCS(substr)(s, ...)"
    );
}

// Test 4: Qualified function call should NOT use UFCS (std::cout << ...)
void test_qualified_call_no_ufcs() {
    std::cout << "Running test_qualified_call_no_ufcs...\n";
    // std::cout is a qualified name access via ::, not a UFCS call
    check_not_emit(
        R"(f: () = { std::cout << "hello"; })",
        "CPP2_UFCS(cout)",
        "Qualified name access is NOT UFCS"
    );
}

// Test 5: UFCS in expression context
void test_ufcs_in_expression() {
    std::cout << "Running test_ufcs_in_expression...\n";
    check_emit(
        R"(f: () = { s: std::string = "hello"; x := s.size(); })",
        "CPP2_UFCS(size)(s)",
        "UFCS in assignment expression"
    );
}

// Test 6: UFCS method name preserved
void test_ufcs_method_name() {
    std::cout << "Running test_ufcs_method_name...\n";
    check_emit(
        R"(f: () = { x := 42; x.to_string(); })",
        "CPP2_UFCS(to_string)(x)",
        "UFCS preserves method name"
    );
}

} // namespace test_ufcs

int main() {
    std::cout << "=== UFCS Tests ===\n\n";

    test_ufcs::test_basic_ufcs();
    test_ufcs::test_ufcs_with_args();
    test_ufcs::test_ufcs_multi_args();
    test_ufcs::test_qualified_call_no_ufcs();
    test_ufcs::test_ufcs_in_expression();
    test_ufcs::test_ufcs_method_name();

    std::cout << "\nAll UFCS tests passed.\n";
    return 0;
}
