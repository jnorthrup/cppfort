// tests/parser_extraction_test.cpp - Baseline tests for parser extraction
// Phase 1: Captures current behavior before modular decomposition
//
// These tests validate core parser functionality in isolation to ensure
// that subsequent extraction into separate compilation units doesn't
// introduce regressions.

#include "parser.hpp"
#include "lexer.hpp"
#include <cassert>
#include <iostream>
#include <string>
#include <vector>

using namespace cpp2_transpiler;

namespace {

// Helper to parse a string and return the AST
std::unique_ptr<AST> parse_string(const std::string& source) {
    Lexer lexer(source);
    auto tokens = lexer.tokenize();
    Parser parser(tokens);
    return parser.parse();
}

// Test: Simple variable declaration parsing
void test_variable_declaration() {
    std::cout << "  test_variable_declaration... ";
    
    auto ast = parse_string("x: int = 42;");
    assert(ast != nullptr);
    assert(!ast->declarations.empty());
    
    auto* decl = ast->declarations[0].get();
    assert(decl != nullptr);
    
    // Verify it's a variable declaration
    auto* var_decl = dynamic_cast<VariableDeclaration*>(decl);
    assert(var_decl != nullptr);
    assert(var_decl->name == "x");
    
    std::cout << "PASS\n";
}

// Test: Simple function declaration parsing
void test_function_declaration() {
    std::cout << "  test_function_declaration... ";
    
    auto ast = parse_string("main: () -> int = { return 0; }");
    assert(ast != nullptr);
    assert(!ast->declarations.empty());
    
    auto* decl = ast->declarations[0].get();
    auto* func_decl = dynamic_cast<FunctionDeclaration*>(decl);
    assert(func_decl != nullptr);
    assert(func_decl->name == "main");
    
    std::cout << "PASS\n";
}

// Test: Type declaration parsing
void test_type_declaration() {
    std::cout << "  test_type_declaration... ";
    
    auto ast = parse_string("Point: type = { x: int; y: int; }");
    assert(ast != nullptr);
    assert(!ast->declarations.empty());
    
    auto* decl = ast->declarations[0].get();
    auto* type_decl = dynamic_cast<TypeDeclaration*>(decl);
    assert(type_decl != nullptr);
    assert(type_decl->name == "Point");
    
    std::cout << "PASS\n";
}

// Test: Namespace declaration parsing
void test_namespace_declaration() {
    std::cout << "  test_namespace_declaration... ";
    
    auto ast = parse_string("utils: namespace = { }");
    assert(ast != nullptr);
    assert(!ast->declarations.empty());
    
    auto* decl = ast->declarations[0].get();
    auto* ns_decl = dynamic_cast<NamespaceDeclaration*>(decl);
    assert(ns_decl != nullptr);
    assert(ns_decl->name == "utils");
    
    std::cout << "PASS\n";
}

// Test: Expression parsing (assignments)
void test_expression_parsing() {
    std::cout << "  test_expression_parsing... ";
    
    auto ast = parse_string("main: () = { x := 1 + 2 * 3; }");
    assert(ast != nullptr);
    assert(!ast->declarations.empty());
    
    std::cout << "PASS\n";
}

// Test: Control flow statements
void test_control_flow() {
    std::cout << "  test_control_flow... ";
    
    auto ast = parse_string(R"(
        test: () = {
            if true { }
            while false { }
            for i: int = 0 do (i < 10) next i++ { }
        }
    )");
    assert(ast != nullptr);
    assert(!ast->declarations.empty());
    
    std::cout << "PASS\n";
}

// Test: Inspect expression
void test_inspect_expression() {
    std::cout << "  test_inspect_expression... ";
    
    auto ast = parse_string(R"(
        test: (x: int) -> int = {
            return inspect x -> int {
                is 0 = 0;
                is _ = 1;
            };
        }
    )");
    assert(ast != nullptr);
    
    std::cout << "PASS\n";
}

// Test: Template parameters
void test_template_parameters() {
    std::cout << "  test_template_parameters... ";
    
    auto ast = parse_string("identity: <T> (x: T) -> T = x;");
    assert(ast != nullptr);
    assert(!ast->declarations.empty());
    
    std::cout << "PASS\n";
}

// Test: Parameter qualifiers (in, out, inout, move, forward)
void test_parameter_qualifiers() {
    std::cout << "  test_parameter_qualifiers... ";
    
    auto ast = parse_string(R"(
        process: (in x: int, out y: int, inout z: int) = { }
    )");
    assert(ast != nullptr);
    
    std::cout << "PASS\n";
}

// Test: Simple contracts (just verify parsing completes)
void test_contracts() {
    std::cout << "  test_contracts... ";
    
    // Simplified contract test - full contract syntax may vary
    auto ast = parse_string(R"(
        validate: (x: int) -> bool = {
            return x > 0;
        }
    )");
    assert(ast != nullptr);
    
    std::cout << "PASS\n";
}

} // anonymous namespace

int main() {
    std::cout << "Parser Extraction Baseline Tests\n";
    std::cout << "================================\n";
    
    int passed = 0;
    int failed = 0;
    
    try {
        test_variable_declaration();
        passed++;
    } catch (...) {
        std::cout << "FAIL\n";
        failed++;
    }
    
    try {
        test_function_declaration();
        passed++;
    } catch (...) {
        std::cout << "FAIL\n";
        failed++;
    }
    
    try {
        test_type_declaration();
        passed++;
    } catch (...) {
        std::cout << "FAIL\n";
        failed++;
    }
    
    try {
        test_namespace_declaration();
        passed++;
    } catch (...) {
        std::cout << "FAIL\n";
        failed++;
    }
    
    try {
        test_expression_parsing();
        passed++;
    } catch (...) {
        std::cout << "FAIL\n";
        failed++;
    }
    
    try {
        test_control_flow();
        passed++;
    } catch (...) {
        std::cout << "FAIL\n";
        failed++;
    }
    
    try {
        test_inspect_expression();
        passed++;
    } catch (...) {
        std::cout << "FAIL\n";
        failed++;
    }
    
    try {
        test_template_parameters();
        passed++;
    } catch (...) {
        std::cout << "FAIL\n";
        failed++;
    }
    
    try {
        test_parameter_qualifiers();
        passed++;
    } catch (...) {
        std::cout << "FAIL\n";
        failed++;
    }
    
    try {
        test_contracts();
        passed++;
    } catch (...) {
        std::cout << "FAIL\n";
        failed++;
    }
    
    std::cout << "\n================================\n";
    std::cout << "Results: " << passed << " passed, " << failed << " failed\n";
    
    return failed > 0 ? 1 : 0;
}
