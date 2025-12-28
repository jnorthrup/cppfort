#include <cassert>
#include <iostream>
#include <string>
#include "../include/lexer.hpp"

using namespace cpp2_transpiler;

void test_basic_markdown_block() {
    std::cout << "Testing basic markdown block recognition..." << std::endl;

    std::string source = R"(
```
Hello world
Foo bar
```
)";

    Lexer lexer(source);
    auto tokens = lexer.tokenize();

    // Should have: MarkdownBlock, EOF
    assert(tokens.size() >= 2);
    assert(tokens[0].type == TokenType::MarkdownBlock);
    assert(tokens[tokens.size() - 1].type == TokenType::EndOfFile);

    // Content should include the markdown text
    std::string content(tokens[0].lexeme);
    assert(content.find("Hello world") != std::string::npos);
    assert(content.find("Foo bar") != std::string::npos);

    std::cout << "Basic markdown block test passed!" << std::endl;
}

void test_markdown_block_with_name() {
    std::cout << "Testing markdown block with name..." << std::endl;

    std::string source = R"(
```module_name
This is content
```
)";

    Lexer lexer(source);
    auto tokens = lexer.tokenize();

    assert(tokens.size() >= 2);
    assert(tokens[0].type == TokenType::MarkdownBlock);

    std::string content(tokens[0].lexeme);
    assert(content.find("module_name") != std::string::npos);
    assert(content.find("This is content") != std::string::npos);

    std::cout << "Markdown block with name test passed!" << std::endl;
}

void test_markdown_block_empty() {
    std::cout << "Testing empty markdown block..." << std::endl;

    std::string source = R"(
```
```
)";

    Lexer lexer(source);
    auto tokens = lexer.tokenize();

    assert(tokens.size() >= 2);
    assert(tokens[0].type == TokenType::MarkdownBlock);

    std::cout << "Empty markdown block test passed!" << std::endl;
}

void test_markdown_block_with_code() {
    std::cout << "Testing markdown block interleaved with code..." << std::endl;

    std::string source = R"(
```docs
# Documentation
This is a doc block
```
func main() -> i32 = 42;
)";

    Lexer lexer(source);
    auto tokens = lexer.tokenize();

    // Should have: MarkdownBlock, Func, Identifier(main), ..., EOF
    assert(tokens.size() > 3);
    assert(tokens[0].type == TokenType::MarkdownBlock);

    // Find the func token
    bool found_func = false;
    for (const auto& token : tokens) {
        if (token.type == TokenType::Func) {
            found_func = true;
            break;
        }
    }
    assert(found_func);

    std::cout << "Markdown block with code test passed!" << std::endl;
}

void test_multiple_markdown_blocks() {
    std::cout << "Testing multiple markdown blocks..." << std::endl;

    std::string source = R"(
```first
First block
```
```second
Second block
```
)";

    Lexer lexer(source);
    auto tokens = lexer.tokenize();

    // Should have: MarkdownBlock, MarkdownBlock, EOF
    int markdown_count = 0;
    for (const auto& token : tokens) {
        if (token.type == TokenType::MarkdownBlock) {
            markdown_count++;
        }
    }
    assert(markdown_count == 2);

    std::cout << "Multiple markdown blocks test passed!" << std::endl;
}

int main() {
    std::cout << "Running markdown block lexer tests...\n" << std::endl;

    test_basic_markdown_block();
    test_markdown_block_with_name();
    test_markdown_block_empty();
    test_markdown_block_with_code();
    test_multiple_markdown_blocks();

    std::cout << "\nAll markdown block lexer tests passed!" << std::endl;
    return 0;
}
