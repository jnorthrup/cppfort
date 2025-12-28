#include <cassert>
#include <iostream>
#include <string>
#include <fstream>
#include <cstdlib>
#include "../include/lexer.hpp"
#include "../include/parser.hpp"
#include "../include/code_generator.hpp"

using namespace cpp2_transpiler;

// Test helper: compile C++20 code and check if it succeeds
bool test_compilation(const std::string& cpp_code, const std::string& test_name) {
    std::string filename = "/tmp/test_" + test_name + ".cpp";

    // Write the C++ code to a file
    std::ofstream out(filename);
    out << cpp_code;
    out.close();

    // Try to compile with clang++
    std::string cmd = "clang++ -std=c++20 -c " + filename + " -o /tmp/test_" + test_name + ".o 2>&1";
    FILE* pipe = popen(cmd.c_str(), "r");
    if (!pipe) return false;

    char buffer[256];
    std::string output;
    while (fgets(buffer, sizeof(buffer), pipe) != nullptr) {
        output += buffer;
    }
    int result = pclose(pipe);

    // Clean up
    std::remove(("/tmp/test_" + test_name + ".cpp").c_str());
    std::remove(("/tmp/test_" + test_name + ".o").c_str());

    return result == 0;
}

void test_end_to_end_compilation() {
    std::cout << "Testing end-to-end compilation..." << std::endl;

    std::string source = R"(
```module_docs
This is a module documentation block
```
func add(a: i32, b: i32) -> i32 = a + b;
)";

    Lexer lexer(source);
    auto tokens = lexer.tokenize();
    Parser parser(tokens);
    auto ast = parser.parse();

    CodeGenerator codegen;
    std::string output = codegen.generate(*ast);

    std::cout << "Generated C++ code:\n" << output << std::endl;

    // Verify the generated code compiles
    bool compiles = test_compilation(output, "end_to_end");
    assert(compiles);

    std::cout << "End-to-end compilation test passed!" << std::endl;
}

void test_module_stub_compilation() {
    std::cout << "Testing module stub compilation..." << std::endl;

    // Test that the module stub itself compiles
    std::string module_stub = R"(
export module test_module;

inline constexpr char cas_sha256[] = "64ec88ca00b268e5ba1a35678a1b5316d212f4f366b2477232534a8aeca37f3c";
)";

    bool compiles = test_compilation(module_stub, "module_stub");
    assert(compiles);

    std::cout << "Module stub compilation test passed!" << std::endl;
}

void test_multiple_markdown_blocks() {
    std::cout << "Testing multiple markdown blocks..." << std::endl;

    std::string source = R"(
```docs1
First documentation
```
```docs2
Second documentation
```
func process() -> i32 = 42;
)";

    Lexer lexer(source);
    auto tokens = lexer.tokenize();
    Parser parser(tokens);
    auto ast = parser.parse();

    CodeGenerator codegen;
    std::string output = codegen.generate(*ast);

    // Should have two module declarations
    size_t module_count = 0;
    size_t pos = 0;
    while ((pos = output.find("export module", pos)) != std::string::npos) {
        module_count++;
        pos += 13;
    }
    assert(module_count == 2);

    // Should compile
    bool compiles = test_compilation(output, "multiple_blocks");
    assert(compiles);

    std::cout << "Multiple markdown blocks test passed!" << std::endl;
}

void test_empty_markdown_block() {
    std::cout << "Testing empty markdown block..." << std::endl;

    std::string source = R"(
```
```
func example() -> i32 = 0;
)";

    Lexer lexer(source);
    auto tokens = lexer.tokenize();
    Parser parser(tokens);
    auto ast = parser.parse();

    CodeGenerator codegen;
    std::string output = codegen.generate(*ast);

    // SHA256 of empty string: e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855
    assert(output.find("e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855") != std::string::npos);

    // Should compile
    bool compiles = test_compilation(output, "empty_block");
    assert(compiles);

    std::cout << "Empty markdown block test passed!" << std::endl;
}

void test_unicode_markdown_content() {
    std::cout << "Testing Unicode markdown content..." << std::endl;

    std::string source = R"(
```unicode
Hello 世界
こんにちは
```
func test() -> i32 = 1;
)";

    Lexer lexer(source);
    auto tokens = lexer.tokenize();
    Parser parser(tokens);
    auto ast = parser.parse();

    CodeGenerator codegen;
    std::string output = codegen.generate(*ast);

    // Should have a valid SHA256 hash (64 hex chars)
    assert(output.find("inline constexpr char cas_sha256[]") != std::string::npos);

    // Should compile
    bool compiles = test_compilation(output, "unicode_content");
    assert(compiles);

    std::cout << "Unicode markdown content test passed!" << std::endl;
}

void test_complex_program_with_markdown() {
    std::cout << "Testing complex program with markdown..." << std::endl;

    std::string source = R"(
```api
Public API for math operations
```
func square(x: i32) -> i32 = x * x;

func cube(x: i32) -> i32 = x * x * x;
)";

    Lexer lexer(source);
    auto tokens = lexer.tokenize();
    Parser parser(tokens);
    auto ast = parser.parse();

    CodeGenerator codegen;
    std::string output = codegen.generate(*ast);

    // Should have function definitions
    assert(output.find("int square") != std::string::npos);
    assert(output.find("int cube") != std::string::npos);

    // Should have module stub
    assert(output.find("export module api;") != std::string::npos);

    // Should compile
    bool compiles = test_compilation(output, "complex_program");
    assert(compiles);

    std::cout << "Complex program with markdown test passed!" << std::endl;
}

void test_markdown_with_special_characters() {
    std::cout << "Testing markdown with special characters..." << std::endl;

    std::string source = R"(
```special
# Title with *bold* and _italic_
Code: `printf("Hello\n")`
Links: [text](url)
```
func main() -> i32 = 0;
)";

    Lexer lexer(source);
    auto tokens = lexer.tokenize();
    Parser parser(tokens);
    auto ast = parser.parse();

    CodeGenerator codegen;
    std::string output = codegen.generate(*ast);

    // Should compile (markdown content doesn't affect C++ compilation)
    bool compiles = test_compilation(output, "special_chars");
    assert(compiles);

    std::cout << "Markdown with special characters test passed!" << std::endl;
}

int main() {
    std::cout << "Running markdown integration tests...\n" << std::endl;

    test_module_stub_compilation();
    test_end_to_end_compilation();
    test_multiple_markdown_blocks();
    test_empty_markdown_block();
    test_unicode_markdown_content();
    test_complex_program_with_markdown();
    test_markdown_with_special_characters();

    std::cout << "\nAll markdown integration tests passed!" << std::endl;
    return 0;
}
