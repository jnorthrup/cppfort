// Test: Variable Declarations Parsing and Emission
#include <iostream>
#include <string>
#include <cassert>

#include "lexer.hpp"
#include "combinator_parser.hpp"
#include "emitter.hpp"

using namespace cpp2_transpiler;

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

int main() {
    std::cout << "=== Variable Declaration Tests ===\n\n";

    // Typed variable with init
    check_emit("x: int = 42;", "int x = 42;", "Typed init");
    check_emit("s: std::string = \"hello\";", "std::string s = \"hello\";", "Typed string init");

    // Typed variable without init
    check_emit("x: int;", "int x;", "Typed no init");

    // Type deduction (:=)
    check_emit("x := 42;", "auto x = 42;", "Deduction int");
    check_emit("s := \"hello\";", "auto s = \"hello\";", "Deduction string");
    check_emit("f := 3.14;", "auto f = 3.14;", "Deduction float");

    // Local variable vs Global (check inside function)
    check_emit("f: () = { x: int = 42; }", "int x = 42;", "Local variable");
    check_emit("f: () = { x := 42; }", "auto x = 42;", "Local deduction");

    std::cout << "\nAll variable declaration tests passed.\n";
    return 0;
}
