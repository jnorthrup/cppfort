// Test: Function Declaration Emission
// Verifies that function declarations are correctly emitted as C++.
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <iostream>
#include <string>
#include <vector>
#include <cassert>
#include <sstream>

#include "lexer.hpp"
#include "slim_ast.hpp"
#include "emitter.hpp"

// Include implementations
#include "../../src/lexer.cpp"
#include "../../src/parser.cpp"
#include "../../src/emitter.cpp"

namespace test_emission {

// Helper to check if output contains substring
bool contains(const std::string& str, const std::string& substr) {
    return str.find(substr) != std::string::npos;
}

void test_basic_emission() {
    std::cout << "Running test_basic_emission..." << std::endl;
    std::string code = "my_func: () -> int = { return 42; }";
    
    cpp2_transpiler::Lexer lexer(code);
    auto tokens = lexer.tokenize();
    auto tree = cpp2::parser::parse(tokens);
    
    std::string output = generate_from_tree(tree, tokens);
    std::cout << "Output: " << output << "\n";
    
    assert(contains(output, "auto my_func() -> int {"));
    assert(contains(output, "return 42;"));
    
    std::cout << "  PASS\n";
}

void test_void_emission() {
    std::cout << "Running test_void_emission..." << std::endl;
    std::string code = "my_void: (x: int) -> void = { }";
    
    cpp2_transpiler::Lexer lexer(code);
    auto tokens = lexer.tokenize();
    auto tree = cpp2::parser::parse(tokens);
    
    std::string output = generate_from_tree(tree, tokens);
    std::cout << "Output: " << output << "\n";
    
    assert(contains(output, "auto my_void(int x) -> void {"));
    
    std::cout << "  PASS\n";
}

void test_expression_body_emission() {
    std::cout << "Running test_expression_body_emission..." << std::endl;
    std::string code = "expr_func: () = 123;";
    
    cpp2_transpiler::Lexer lexer(code);
    auto tokens = lexer.tokenize();
    auto tree = cpp2::parser::parse(tokens);
    
    std::string output = generate_from_tree(tree, tokens);
    std::cout << "Output: " << output << "\n";
    
    // Implicit return type should be auto
    assert(contains(output, "auto expr_func() -> auto {"));
    assert(contains(output, "return 123;"));
    
    std::cout << "  PASS\n";
}

void test_main_special_case() {
    std::cout << "Running test_main_special_case..." << std::endl;
    std::string code = "main: () -> int = { return 0; }";
    
    cpp2_transpiler::Lexer lexer(code);
    auto tokens = lexer.tokenize();
    auto tree = cpp2::parser::parse(tokens);
    
    std::string output = generate_from_tree(tree, tokens);
    std::cout << "Output: " << output << "\n";
    
    assert(contains(output, "int main(int argc, char* argv[]) {"));
    assert(contains(output, "return 0;"));
    
    std::cout << "  PASS\n";
}

} // namespace test_emission

int main() {
    std::cout << "=== Function Emission Tests ===\n";
    try {
        test_emission::test_basic_emission();
        test_emission::test_void_emission();
        test_emission::test_expression_body_emission();
        test_emission::test_main_special_case();
    } catch (const std::exception& e) {
        std::cerr << "Test FAILED with exception: " << e.what() << "\n";
        return 1;
    }
    std::cout << "=== All Tests Passed ===\n";
    return 0;
}
