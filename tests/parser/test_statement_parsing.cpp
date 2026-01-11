#include "combinator_parser.hpp"
#include "lexer.hpp"
#include "slim_ast.hpp"
#include <iostream>
#include <cassert>
#include <string>
#include <vector>

using namespace cpp2::ast;

// Helper to check if a statement of specific kind exists in the parsed tree
bool has_statement(const ParseTree& tree, NodeKind kind) {
    for (const auto& node : tree.nodes) {
        if (node.kind == kind) return true;
    }
    return false;
}

void test_if_statement() {
    std::cout << "Testing If Statement...\n";
    std::string code = "main: () -> void = { if (true) { } }";
    cpp2_transpiler::Lexer lexer(code);
    auto tokens = lexer.tokenize();
    auto tree = cpp2::parser::parse(tokens);
    
    if (!has_statement(tree, NodeKind::IfStatement)) {
        std::cerr << "FAIL: IfStatement node not found in AST\n";
        std::exit(1);
    }
    std::cout << "PASS: If statement\n";
}

void test_while_statement() {
    std::cout << "Testing While Statement...\n";
    std::string code = "main: () -> void = { while (true) { } }";
    cpp2_transpiler::Lexer lexer(code);
    auto tokens = lexer.tokenize();
    auto tree = cpp2::parser::parse(tokens);
    
    if (!has_statement(tree, NodeKind::WhileStatement)) {
        std::cerr << "FAIL: WhileStatement node not found in AST\n";
        std::exit(1);
    }
    std::cout << "PASS: While statement\n";
}

void test_for_statement() {
    std::cout << "Testing For Statement (Cpp2)...\n";
    // Cpp2 style: for items do (x) { }
    std::string code_cpp2 = "main: () -> void = { for items do (x) { } }";
    cpp2_transpiler::Lexer lexer2(code_cpp2);
    auto tokens2 = lexer2.tokenize();
    auto tree2 = cpp2::parser::parse(tokens2);
    
    if (!has_statement(tree2, NodeKind::ForStatement)) {
        std::cerr << "FAIL: Cpp2 ForStatement node not found\n";
        std::exit(1);
    }
    std::cout << "PASS: Cpp2 For statement\n";

    std::cout << "Testing For Statement (C++1)...\n";
    // C++1 style: for (x : items) { }
    // Note: 'x : items' might need to be 'int x : items' or 'auto x : items' for validity, 
    // but parser should handle 'x : items' if we support implied type or just Identifier.
    std::string code_cpp1 = "main: () -> void = { for (x : items) { } }";
    cpp2_transpiler::Lexer lexer1(code_cpp1);
    auto tokens1 = lexer1.tokenize();
    auto tree1 = cpp2::parser::parse(tokens1);
    
    if (!has_statement(tree1, NodeKind::ForStatement)) {
        std::cerr << "FAIL: C++1 ForStatement node not found\n";
        std::exit(1);
    }
    std::cout << "PASS: C++1 For statement\n";
}

void test_return_statement() {
    std::cout << "Testing Return Statement...\n";
    std::string code = "main: () -> int = { return 42; }";
    cpp2_transpiler::Lexer lexer(code);
    auto tokens = lexer.tokenize();
    auto tree = cpp2::parser::parse(tokens);
    
    if (!has_statement(tree, NodeKind::ReturnStatement)) {
        std::cerr << "FAIL: ReturnStatement node not found in AST\n";
        std::exit(1);
    }
    std::cout << "PASS: Return statement\n";
}

int main() {
    test_if_statement();
    test_while_statement();
    test_for_statement();
    test_return_statement();
    std::cout << "All statement tests passed\n";
    return 0;
}