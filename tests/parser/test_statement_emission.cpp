#include "combinator_parser.hpp"
#include "lexer.hpp"
#include "slim_ast.hpp"
#include "emitter.hpp"
#include <iostream>
#include <cassert>
#include <string>

using namespace cpp2::ast;

// Helper to emit code from Cpp2 source
std::string emit_from_source(const std::string& code) {
    cpp2_transpiler::Lexer lexer(code);
    auto tokens = lexer.tokenize();
    auto tree = cpp2::parser::parse(tokens);
    return generate_from_tree(tree, tokens);
}

void test_if_emission() {
    std::cout << "Testing If Statement Emission...\n";
    std::string code = R"(
main: () -> void = {
    if (true) {
        return 42;
    }
}
)";
    std::string output = emit_from_source(code);

    // Check for expected output (allow extra parentheses)
    if (output.find("if") == std::string::npos || output.find("true") == std::string::npos) {
        std::cerr << "FAIL: if statement not emitted correctly\n";
        std::cerr << "Output:\n" << output << "\n";
        std::exit(1);
    }
    if (output.find("return 42") == std::string::npos) {
        std::cerr << "FAIL: return statement not emitted correctly\n";
        std::cerr << "Output:\n" << output << "\n";
        std::exit(1);
    }
    std::cout << "PASS: If statement emission\n";
}

void test_while_emission() {
    std::cout << "Testing While Statement Emission...\n";
    std::string code = R"(
main: () -> void = {
    while (true) {
        return 42;
    }
}
)";
    std::string output = emit_from_source(code);

    // Check for expected output (allow extra parentheses)
    if (output.find("while") == std::string::npos || output.find("true") == std::string::npos) {
        std::cerr << "FAIL: while statement not emitted correctly\n";
        std::cerr << "Output:\n" << output << "\n";
        std::exit(1);
    }
    std::cout << "PASS: While statement emission\n";
}

void test_for_cpp2_emission() {
    std::cout << "Testing For Statement (Cpp2 style) Emission...\n";
    std::string code = R"(
main: () -> void = {
    for items do (x) {
        return x;
    }
}
)";
    std::string output = emit_from_source(code);

    // Should emit C++ style range-based for
    if (output.find("for (auto x : items)") == std::string::npos &&
        output.find("for(auto x:items)") == std::string::npos) {
        std::cerr << "FAIL: Cpp2 for statement not emitted correctly\n";
        std::cerr << "Output:\n" << output << "\n";
        std::exit(1);
    }
    std::cout << "PASS: Cpp2 For statement emission\n";
}

void test_for_cpp1_emission() {
    std::cout << "Testing For Statement (C++1 style) Emission...\n";
    std::string code = R"(
main: () -> void = {
    for (x : items) {
        return x;
    }
}
)";
    std::string output = emit_from_source(code);

    // Should emit C++ style range-based for
    if (output.find("for (auto x : items)") == std::string::npos &&
        output.find("for(auto x:items)") == std::string::npos) {
        std::cerr << "FAIL: C++1 for statement not emitted correctly\n";
        std::cerr << "Output:\n" << output << "\n";
        std::exit(1);
    }
    std::cout << "PASS: C++1 For statement emission\n";
}

void test_return_emission() {
    std::cout << "Testing Return Statement Emission...\n";
    std::string code = R"(
main: () -> int = {
    return 42;
}
)";
    std::string output = emit_from_source(code);

    if (output.find("return 42") == std::string::npos) {
        std::cerr << "FAIL: return statement not emitted correctly\n";
        std::cerr << "Output:\n" << output << "\n";
        std::exit(1);
    }
    std::cout << "PASS: Return statement emission\n";
}

int main() {
    test_if_emission();
    test_while_emission();
    test_for_cpp2_emission();
    test_for_cpp1_emission();
    test_return_emission();
    std::cout << "All statement emission tests passed\n";
    return 0;
}
