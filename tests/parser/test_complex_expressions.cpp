// Test: Complex Expressions Parsing and Emission
#include <iostream>
#include <string>
#include <cassert>
#include <algorithm>

#include "lexer.hpp"
#include "combinator_parser.hpp"
#include "emitter.hpp"

using namespace cpp2_transpiler;

void check_emit(const std::string& input, const std::string& expected_contains, const char* test_name) {
    cpp2_transpiler::Lexer lexer(input);
    auto tokens = lexer.tokenize();
    auto tree = cpp2::parser::parse(tokens);
    
    std::string output = generate_from_tree(tree, tokens);
    
    // Simple contains check
    if (output.find(expected_contains) == std::string::npos) {
        std::cerr << "FAIL: " << test_name << "\n";
        std::cerr << "Input: " << input << "\n";
        std::cerr << "Expected to contain: " << expected_contains << "\n";
        std::cerr << "Actual output:\n" << output << "\n";
        std::exit(1);
    }
    
    std::cout << "PASS: " << test_name << "\n";
}

int main() {
    std::cout << "=== Complex Expression Tests ===\n\n";

    // Function Calls
    check_emit("x: int = f();", "int x = f();", "Simple call 0 args");
    check_emit("x: int = f(1);", "int x = f(1);", "Simple call 1 arg");
    check_emit("x: int = f(1, 2);", "int x = f(1, 2);", "Simple call 2 args");
    check_emit("x: int = f(g(1));", "int x = f(g(1));", "Nested call");

    // Member Access
    check_emit("x: int = obj.field;", "int x = obj.field;", "Member access field");
    check_emit("x: int = obj.method();", "int x = obj.method();", "Member access method");
    check_emit("x: int = obj.inner.field;", "int x = obj.inner.field;", "Nested member access");

    // Subscripts
    check_emit("x: int = arr[0];", "int x = arr[0];", "Subscript 0");
    check_emit("x: int = map[\"key\"];", "int x = map[\"key\"];", "Subscript string");
    check_emit("x: int = vec[i + 1];", "int x = vec[i + 1];", "Subscript expression");

    // Binary Operators
    check_emit("x: bool = a && b || c;", "bool x = a && b || c;", "Logical ops");
    check_emit("x: int = (a << 1) & 0xFF;", "int x = (a << 1) & 0xFF;", "Bitwise and shift");

    // Ternary Operator
    check_emit("x: int = cond ? 1 : 0;", "int x = cond ? 1 : 0;", "Ternary simple");
    check_emit("x: int = (a > b) ? a : b;", "int x = (a > b) ? a : b;", "Ternary expression");

    // Complex Combinations
    check_emit("x: int = arr[i].method(a, b ? c : d);", "int x = arr[i].method(a, b ? c : d);", "Complex combination");

    std::cout << "\nAll complex expression tests passed.\n";
    return 0;
}
