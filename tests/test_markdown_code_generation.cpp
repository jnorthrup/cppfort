#include <cassert>
#include <iostream>
#include <string>
#include "../include/lexer.hpp"
#include "../include/parser.hpp"
#include "../include/code_generator.hpp"

using namespace cpp2_transpiler;

void test_basic_module_stub_generation() {
    std::cout << "Testing basic module stub generation..." << std::endl;

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

    CodeGenerator codegen;
    std::string output = codegen.generate(*ast);

    // Should contain the module stub
    assert(output.find("export module test;") != std::string::npos);
    assert(output.find("inline constexpr char cas_sha256[]") != std::string::npos);
    // SHA256 of "Hello world" = 64ec88ca00b268e5ba1a35678a1b5316d212f4f366b2477232534a8aeca37f3c
    assert(output.find("64ec88ca00b268e5ba1a35678a1b5316d212f4f366b2477232534a8aeca37f3c") != std::string::npos);

    std::cout << "Basic module stub generation test passed!" << std::endl;
}

void test_anonymous_module_stub_generation() {
    std::cout << "Testing anonymous module stub generation..." << std::endl;

    std::string source = R"(
```
Content here
```
func example() -> i32 = 42;
)";

    Lexer lexer(source);
    auto tokens = lexer.tokenize();
    Parser parser(tokens);
    auto ast = parser.parse();

    CodeGenerator codegen;
    std::string output = codegen.generate(*ast);

    // Should contain __cas_ prefixed module name
    assert(output.find("export module __cas_") != std::string::npos);
    assert(output.find("inline constexpr char cas_sha256[]") != std::string::npos);
    // SHA256 of "Content here" = cf39e7e538a45ec0c2cc730c90f32c65e2ffe57a7ff036f1bddca477e1b7ab3
    assert(output.find("cf39e7e538a45ec0c2cc730c90f32c65e2ffe57a7ff036f1bddca477e1b7ab3") != std::string::npos);

    std::cout << "Anonymous module stub generation test passed!" << std::endl;
}

void test_multiple_module_stubs() {
    std::cout << "Testing multiple module stubs..." << std::endl;

    std::string source = R"(
```first
First content
```
```second
Second content
```
func example() -> i32 = 42;
)";

    Lexer lexer(source);
    auto tokens = lexer.tokenize();
    Parser parser(tokens);
    auto ast = parser.parse();

    CodeGenerator codegen;
    std::string output = codegen.generate(*ast);

    // Should have two module declarations
    assert(output.find("export module first;") != std::string::npos);
    assert(output.find("export module second;") != std::string::npos);

    // Should have two SHA256 constants
    size_t sha256_count = 0;
    size_t pos = 0;
    while ((pos = output.find("cas_sha256[]", pos)) != std::string::npos) {
        sha256_count++;
        pos += 13;
    }
    assert(sha256_count == 2);

    std::cout << "Multiple module stubs test passed!" << std::endl;
}

void test_module_stub_format() {
    std::cout << "Testing module stub format..." << std::endl;

    std::string source = R"(
```docs
# Documentation
```
func example() -> i32 = 42;
)";

    Lexer lexer(source);
    auto tokens = lexer.tokenize();
    Parser parser(tokens);
    auto ast = parser.parse();

    CodeGenerator codegen;
    std::string output = codegen.generate(*ast);

    // Check format: export module <name>;
    assert(output.find("export module docs;") != std::string::npos);

    // Check format: inline constexpr char cas_sha256[] = "<hash>";
    assert(output.find("inline constexpr char cas_sha256[] = \"") != std::string::npos);
    assert(output.find("\";") != std::string::npos);

    // Module should be at the end (after function definition)
    size_t module_pos = output.find("export module");
    size_t func_pos = output.find("int example()");
    assert(module_pos > func_pos); // Module should come after function

    std::cout << "Module stub format test passed!" << std::endl;
}

void test_no_markdown_no_module() {
    std::cout << "Testing declaration without markdown..." << std::endl;

    std::string source = R"(
func example() -> i32 = 42;
)";

    Lexer lexer(source);
    auto tokens = lexer.tokenize();
    Parser parser(tokens);
    auto ast = parser.parse();

    CodeGenerator codegen;
    std::string output = codegen.generate(*ast);

    // Should NOT contain any module stubs
    assert(output.find("export module") == std::string::npos);
    assert(output.find("cas_sha256") == std::string::npos);

    std::cout << "No markdown no module test passed!" << std::endl;
}

void test_module_stub_with_function() {
    std::cout << "Testing complete output with function and module..." << std::endl;

    std::string source = R"(
```api
Public API documentation
```
func calculate(x: i32) -> i32 = x * 2;
)";

    Lexer lexer(source);
    auto tokens = lexer.tokenize();
    Parser parser(tokens);
    auto ast = parser.parse();

    CodeGenerator codegen;
    std::string output = codegen.generate(*ast);

    // Should have the function
    assert(output.find("int calculate") != std::string::npos);
    assert(output.find("x * 2") != std::string::npos);

    // Should have the module stub
    assert(output.find("export module api;") != std::string::npos);
    assert(output.find("inline constexpr char cas_sha256[]") != std::string::npos);

    std::cout << "Module stub with function test passed!" << std::endl;
}

int main() {
    std::cout << "Running markdown code generation tests...\n" << std::endl;

    test_basic_module_stub_generation();
    test_anonymous_module_stub_generation();
    test_multiple_module_stubs();
    test_module_stub_format();
    test_no_markdown_no_module();
    test_module_stub_with_function();

    std::cout << "\nAll markdown code generation tests passed!" << std::endl;
    return 0;
}
