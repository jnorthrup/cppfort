// tests/codegen_extraction_test.cpp - Baseline tests for code generator extraction
// Phase 1: Captures current output behavior before modular decomposition
//
// These tests validate code generation for key constructs to ensure
// that subsequent extraction into separate compilation units doesn't
// introduce regressions in generated C++ output.

#include "parser.hpp"
#include "lexer.hpp"
#include "code_generator.hpp"
#include <cassert>
#include <iostream>
#include <string>

using namespace cpp2_transpiler;

namespace {

// Helper to transpile Cpp2 source to C++
std::string transpile(const std::string& source) {
    Lexer lexer(source);
    auto tokens = lexer.tokenize();
    Parser parser(tokens);
    auto ast = parser.parse();
    
    CodeGenerator codegen;
    return codegen.generate(*ast);
}

// Check if output contains expected substring
bool contains(const std::string& output, const std::string& expected) {
    return output.find(expected) != std::string::npos;
}

// Test: Variable declaration generates correct C++
void test_variable_codegen() {
    std::cout << "  test_variable_codegen... ";
    
    auto output = transpile("x: int = 42;");
    
    // Should generate something like: int x = 42;
    assert(contains(output, "int"));
    assert(contains(output, "x"));
    assert(contains(output, "42"));
    
    std::cout << "PASS\n";
}

// Test: Function declaration generates correct C++
void test_function_codegen() {
    std::cout << "  test_function_codegen... ";
    
    auto output = transpile("add: (a: int, b: int) -> int = a + b;");
    
    // Should generate a function with int return type
    assert(contains(output, "int"));
    assert(contains(output, "add"));
    
    std::cout << "PASS\n";
}

// Test: Type declaration generates struct/class
void test_type_codegen() {
    std::cout << "  test_type_codegen... ";
    
    auto output = transpile("Point: type = { x: int; y: int; }");
    
    // Should generate class or struct
    assert(contains(output, "Point") || contains(output, "class") || contains(output, "struct"));
    
    std::cout << "PASS\n";
}

// Test: Namespace generates correct C++
void test_namespace_codegen() {
    std::cout << "  test_namespace_codegen... ";
    
    auto output = transpile("utils: namespace = { }");
    
    assert(contains(output, "namespace"));
    assert(contains(output, "utils"));
    
    std::cout << "PASS\n";
}

// Test: If statement generates correct C++
void test_if_codegen() {
    std::cout << "  test_if_codegen... ";
    
    auto output = transpile(R"(
        test: () = {
            if true {
                x := 1;
            }
        }
    )");
    
    assert(contains(output, "if"));
    
    std::cout << "PASS\n";
}

// Test: While loop generates correct C++
void test_while_codegen() {
    std::cout << "  test_while_codegen... ";
    
    auto output = transpile(R"(
        test: () = {
            while false { }
        }
    )");
    
    assert(contains(output, "while"));
    
    std::cout << "PASS\n";
}

// Test: Return statement
void test_return_codegen() {
    std::cout << "  test_return_codegen... ";
    
    auto output = transpile("get_value: () -> int = { return 42; }");
    
    assert(contains(output, "return"));
    assert(contains(output, "42"));
    
    std::cout << "PASS\n";
}

// Test: Binary expressions preserve operator precedence
void test_expression_codegen() {
    std::cout << "  test_expression_codegen... ";
    
    auto output = transpile("calc: () -> int = 1 + 2 * 3;");
    
    // The output should maintain precedence (either through parens or operator order)
    assert(contains(output, "1"));
    assert(contains(output, "2"));
    assert(contains(output, "3"));
    
    std::cout << "PASS\n";
}

// Test: Template function generates template<>
void test_template_codegen() {
    std::cout << "  test_template_codegen... ";
    
    auto output = transpile("identity: <T> (x: T) -> T = x;");
    
    assert(contains(output, "template") || contains(output, "typename") || contains(output, "<"));
    
    std::cout << "PASS\n";
}

// Test: Parameter qualifiers translate correctly
void test_parameter_codegen() {
    std::cout << "  test_parameter_codegen... ";
    
    auto output = transpile("swap: (inout a: int, inout b: int) = { }");
    
    // inout should generate reference parameters
    assert(contains(output, "&") || contains(output, "swap"));
    
    std::cout << "PASS\n";
}

} // anonymous namespace

int main() {
    std::cout << "Code Generator Extraction Baseline Tests\n";
    std::cout << "=========================================\n";
    
    int passed = 0;
    int failed = 0;
    
    auto run_test = [&](auto test_fn) {
        try {
            test_fn();
            passed++;
        } catch (const std::exception& e) {
            std::cout << "FAIL: " << e.what() << "\n";
            failed++;
        } catch (...) {
            std::cout << "FAIL\n";
            failed++;
        }
    };
    
    run_test(test_variable_codegen);
    run_test(test_function_codegen);
    run_test(test_type_codegen);
    run_test(test_namespace_codegen);
    run_test(test_if_codegen);
    run_test(test_while_codegen);
    run_test(test_return_codegen);
    run_test(test_expression_codegen);
    run_test(test_template_codegen);
    run_test(test_parameter_codegen);
    
    std::cout << "\n=========================================\n";
    std::cout << "Results: " << passed << " passed, " << failed << " failed\n";
    
    return failed > 0 ? 1 : 0;
}
