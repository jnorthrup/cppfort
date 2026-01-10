// Test: Parameter Emission
// Verifies that parameter qualifiers are correctly emitted as C++ references/modifiers.
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

void test_inout_emission() {
    std::cout << "Running test_inout_emission..." << std::endl;
    std::string code = "my_func: (inout s: std::string) -> void = { }";
    
    cpp2_transpiler::Lexer lexer(code);
    auto tokens = lexer.tokenize();
    auto tree = cpp2::parser::parse(tokens);
    
    std::string output = generate_from_tree(tree, tokens);
    std::cout << "Output: " << output << "\n";
    
    // Expect: void my_func(std::string& s)
    assert(contains(output, "std::string& s") && "Should emit reference for inout");
    
    std::cout << "  PASS\n";
}

void test_out_emission() {
    std::cout << "Running test_out_emission..." << std::endl;
    std::string code = "my_func: (out result: int) -> void = { }";
    
    cpp2_transpiler::Lexer lexer(code);
    auto tokens = lexer.tokenize();
    auto tree = cpp2::parser::parse(tokens);
    
    std::string output = generate_from_tree(tree, tokens);
    std::cout << "Output: " << output << "\n";
    
    // Expect: void my_func(int& result)
    // Note: 'out' might imply more (initialization check), but signature should be reference.
    assert(contains(output, "int& result") && "Should emit reference for out");
    
    std::cout << "  PASS\n";
}

void test_move_emission() {
    std::cout << "Running test_move_emission..." << std::endl;
    std::string code = "my_func: (move x: Widget) -> void = { }";
    
    cpp2_transpiler::Lexer lexer(code);
    auto tokens = lexer.tokenize();
    auto tree = cpp2::parser::parse(tokens);
    
    std::string output = generate_from_tree(tree, tokens);
    std::cout << "Output: " << output << "\n";
    
    // Expect: void my_func(Widget&& x)
    assert(contains(output, "Widget&& x") && "Should emit rvalue reference for move");
    
    std::cout << "  PASS\n";
}

void test_forward_emission() {
    std::cout << "Running test_forward_emission..." << std::endl;
    std::string code = "my_func: (forward x: T) -> void = { }";
    
    cpp2_transpiler::Lexer lexer(code);
    auto tokens = lexer.tokenize();
    auto tree = cpp2::parser::parse(tokens);
    
    std::string output = generate_from_tree(tree, tokens);
    std::cout << "Output: " << output << "\n";
    
    // Expect: void my_func(T&& x)
    assert(contains(output, "T&& x") && "Should emit forwarding reference (T&&) for forward");
    
    std::cout << "  PASS\n";
}

} // namespace test_emission

int main() {
    std::cout << "=== Parameter Emission Tests ===\n";
    try {
        test_emission::test_inout_emission();
        test_emission::test_out_emission();
        test_emission::test_move_emission();
        test_emission::test_forward_emission();
    } catch (const std::exception& e) {
        std::cerr << "Test FAILED with exception: " << e.what() << "\n";
        return 1;
    }
    std::cout << "=== All Tests Passed ===\n";
    return 0;
}
