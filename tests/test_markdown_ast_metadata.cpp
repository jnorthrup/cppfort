#include <cassert>
#include <iostream>
#include <string>
#include "../include/lexer.hpp"
#include "../include/parser.hpp"
#include "../include/ast.hpp"

using namespace cpp2_transpiler;

void test_markdown_block_attached_to_function() {
    std::cout << "Testing markdown block attached to function..." << std::endl;

    std::string source = R"(
```docs
# Function documentation
This function does something.
```
func example() -> i32 = 42;
)";

    Lexer lexer(source);
    auto tokens = lexer.tokenize();
    Parser parser(tokens);
    auto ast = parser.parse();

    assert(ast->declarations.size() == 1);
    auto* func = static_cast<FunctionDeclaration*>(ast->declarations[0].get());

    assert(func->name == "example");
    assert(func->markdown_blocks.size() == 1);

    const auto& md = func->markdown_blocks[0];
    assert(md.name == "docs");
    assert(md.content.find("Function documentation") != std::string::npos);
    assert(md.sha256.length() == 64);

    std::cout << "Markdown block attached to function test passed!" << std::endl;
}

void test_multiple_markdown_blocks() {
    std::cout << "Testing multiple markdown blocks attached..." << std::endl;

    std::string source = R"(
```first
First block
```
```second
Second block
```
func example() -> i32 = 42;
)";

    Lexer lexer(source);
    auto tokens = lexer.tokenize();
    Parser parser(tokens);
    auto ast = parser.parse();

    assert(ast->declarations.size() == 1);
    auto* func = static_cast<FunctionDeclaration*>(ast->declarations[0].get());

    assert(func->markdown_blocks.size() == 2);
    assert(func->markdown_blocks[0].name == "first");
    assert(func->markdown_blocks[1].name == "second");

    std::cout << "Multiple markdown blocks test passed!" << std::endl;
}

void test_markdown_block_without_name() {
    std::cout << "Testing markdown block without name..." << std::endl;

    std::string source = R"(
```
Anonymous documentation
```
func example() -> i32 = 42;
)";

    Lexer lexer(source);
    auto tokens = lexer.tokenize();
    Parser parser(tokens);
    auto ast = parser.parse();

    assert(ast->declarations.size() == 1);
    auto* func = static_cast<FunctionDeclaration*>(ast->declarations[0].get());

    assert(func->markdown_blocks.size() == 1);
    assert(func->markdown_blocks[0].name.empty()); // No name

    std::cout << "Markdown block without name test passed!" << std::endl;
}

void test_markdown_block_sha256_computation() {
    std::cout << "Testing SHA256 computation in markdown block..." << std::endl;

    std::string source = R"(
```test
Hello world
```
func example() -> i32 = 42;
)";

    Lexer lexer(source);
    auto tokens = lexer.tokenize();
    Parser parser(tokens);
    auto ast = parser.parse();

    assert(ast->declarations.size() == 1);
    auto* func = static_cast<FunctionDeclaration*>(ast->declarations[0].get());

    assert(func->markdown_blocks.size() == 1);
    const auto& md = func->markdown_blocks[0];

    // SHA256 of "Hello world" = 64ec88ca00b268e5ba1a35678a1b5316d212f4f366b2477232534a8aeca37f3c
    assert(md.sha256 == "64ec88ca00b268e5ba1a35678a1b5316d212f4f366b2477232534a8aeca37f3c");

    std::cout << "SHA256 computation test passed!" << std::endl;
}

void test_variable_declaration_with_markdown() {
    std::cout << "Testing markdown block attached to variable..." << std::endl;

    std::string source = R"(
```docs
# Variable documentation
```
let x: i32 = 42;
)";

    Lexer lexer(source);
    auto tokens = lexer.tokenize();
    Parser parser(tokens);
    auto ast = parser.parse();

    assert(ast->declarations.size() == 1);
    auto* var = static_cast<VariableDeclaration*>(ast->declarations[0].get());

    assert(var->name == "x");
    assert(var->markdown_blocks.size() == 1);
    assert(var->markdown_blocks[0].name == "docs");

    std::cout << "Variable declaration with markdown test passed!" << std::endl;
}

void test_no_markdown_blocks() {
    std::cout << "Testing declaration without markdown blocks..." << std::endl;

    std::string source = R"(
func example() -> i32 = 42;
)";

    Lexer lexer(source);
    auto tokens = lexer.tokenize();
    Parser parser(tokens);
    auto ast = parser.parse();

    assert(ast->declarations.size() == 1);
    auto* func = static_cast<FunctionDeclaration*>(ast->declarations[0].get());

    assert(func->markdown_blocks.empty());

    std::cout << "No markdown blocks test passed!" << std::endl;
}

int main() {
    std::cout << "Running markdown AST metadata tests...\n" << std::endl;

    test_markdown_block_attached_to_function();
    test_multiple_markdown_blocks();
    test_markdown_block_without_name();
    test_markdown_block_sha256_computation();
    test_variable_declaration_with_markdown();
    test_no_markdown_blocks();

    std::cout << "\nAll markdown AST metadata tests passed!" << std::endl;
    return 0;
}
