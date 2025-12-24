#include <iostream>
#include <string>
#include <cassert>
#include <fstream>
#include <sstream>

#include "../include/lexer.hpp"
#include "../include/parser.hpp"
#include "../include/semantic_analyzer.hpp"
#include "../include/code_generator.hpp"
#include "../include/safety_checker.hpp"
#include "../include/metafunction_processor.hpp"
#include "../include/contract_processor.hpp"

using namespace cpp2_transpiler;

// Test utilities
static std::string read_file(const std::string& filename) {
    std::ifstream file(filename);
    return std::string(std::istreambuf_iterator<char>(file),
                       std::istreambuf_iterator<char>());
}

bool transpile_and_compare(const std::string& input_file, const std::string& expected_file) {
    try {
        // Read input
        std::string source = read_file(input_file);
        std::string expected = read_file(expected_file);

        // Transpile
        Lexer lexer(source);
        auto tokens = lexer.tokenize();

        Parser parser(tokens);
        auto ast = parser.parse();

        SemanticAnalyzer semantic_analyzer;
        semantic_analyzer.analyze(*ast);

        SafetyChecker safety_checker;
        safety_checker.check(*ast);

        MetafunctionProcessor meta_processor;
        meta_processor.process(*ast);

        ContractProcessor contract_processor;
        contract_processor.process(*ast);

        CodeGenerator code_generator;
        auto result = code_generator.generate(*ast);

        // Compare
        return result == expected;
    }
    catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return false;
    }
}

// Test cases
void test_lexer() {
    std::cout << "Testing lexer..." << std::endl;

    std::string source = R"(
        func main() -> i32 = {
            let x: i32 = 42;
            return x;
        }
    )";

    Lexer lexer(source);
    auto tokens = lexer.tokenize();

    assert(tokens[0].type == TokenType::Func);
    assert(tokens[1].type == TokenType::Identifier);
    assert(tokens[1].lexeme == "main");
    assert(tokens[2].type == TokenType::LeftParen);
    assert(tokens[3].type == TokenType::RightParen);
    assert(tokens[4].type == TokenType::Arrow);
    assert(tokens[5].type == TokenType::Identifier);
    assert(tokens[5].lexeme == "i32");
    assert(tokens[6].type == TokenType::Equal);
    assert(tokens[7].type == TokenType::LeftBrace);
    assert(tokens[8].type == TokenType::Let);
    assert(tokens[9].type == TokenType::Identifier);
    assert(tokens[9].lexeme == "x");

    std::cout << "Lexer tests passed!" << std::endl;
}

void test_parser() {
    std::cout << "Testing parser..." << std::endl;

    std::string source = R"(
        let x: i32 = 42;
        func add(a: i32, b: i32) -> i32 = a + b;
    )";

    Lexer lexer(source);
    auto tokens = lexer.tokenize();

    Parser parser(tokens);
    auto ast = parser.parse();

    assert(!ast->declarations.empty());
    assert(ast->declarations[0]->kind == Declaration::Kind::Variable);
    assert(ast->declarations[1]->kind == Declaration::Kind::Function);

    auto var = static_cast<VariableDeclaration*>(ast->declarations[0].get());
    assert(var->name == "x");
    assert(var->type != nullptr);
    assert(var->type->name == "i32");

    auto func = static_cast<FunctionDeclaration*>(ast->declarations[1].get());
    assert(func->name == "add");
    assert(func->parameters.size() == 2);
    assert(func->parameters[0].name == "a");
    assert(func->parameters[1].name == "b");

    std::cout << "Parser tests passed!" << std::endl;
}

void test_simple_transpilation() {
    std::cout << "Testing simple transpilation..." << std::endl;

    std::string source = R"(
        func main() -> i32 {
            let x: i32 = 42;
            return x;
        }
    )";

    Lexer lexer(source);
    auto tokens = lexer.tokenize();

    Parser parser(tokens);
    auto ast = parser.parse();

    SemanticAnalyzer semantic_analyzer;
    semantic_analyzer.analyze(*ast);

    CodeGenerator code_generator;
    auto result = code_generator.generate(*ast);

    // Basic checks
    assert(result.find("int main()") != std::string::npos);
    assert(result.find("int x = 42") != std::string::npos);
    assert(result.find("return x") != std::string::npos);

    std::cout << "Simple transpilation tests passed!" << std::endl;
}

void test_type_deduction() {
    std::cout << "Testing type deduction..." << std::endl;

    std::string source = R"(
        let x = 42;
        let y = 3.14;
        let z = true;
    )";

    Lexer lexer(source);
    auto tokens = lexer.tokenize();

    Parser parser(tokens);
    auto ast = parser.parse();

    SemanticAnalyzer semantic_analyzer;
    semantic_analyzer.analyze(*ast);

    CodeGenerator code_generator;
    auto result = code_generator.generate(*ast);

    // Check that types were deduced
    assert(result.find("auto x = 42") != std::string::npos);
    assert(result.find("auto y = 3.14") != std::string::npos);
    assert(result.find("auto z = true") != std::string::npos);

    std::cout << "Type deduction tests passed!" << std::endl;
}

void test_ufcs() {
    std::cout << "Testing Unified Function Call Syntax..." << std::endl;

    std::string source = R"(
        func length(s: string) -> i32 = s.len();

        func test() {
            let s: string = "hello";
            let x = s.length();
            let y = length(s);
        }
    )";

    Lexer lexer(source);
    auto tokens = lexer.tokenize();

    Parser parser(tokens);
    auto ast = parser.parse();

    SemanticAnalyzer semantic_analyzer;
    semantic_analyzer.analyze(*ast);

    CodeGenerator code_generator;
    auto result = code_generator.generate(*ast);

    // Both should generate the same C++ code
    assert(result.find("length(s)") != std::string::npos);

    std::cout << "UFCS tests passed!" << std::endl;
}

void test_postfix_operators() {
    std::cout << "Testing postfix operators..." << std::endl;

    std::string source = R"(
        func test(p: i32*) {
            let x = p*;
            let addr = x&;
            x++;
            x--;
        }
    )";

    Lexer lexer(source);
    auto tokens = lexer.tokenize();

    Parser parser(tokens);
    auto ast = parser.parse();

    SemanticAnalyzer semantic_analyzer;
    semantic_analyzer.analyze(*ast);

    CodeGenerator code_generator;
    auto result = code_generator.generate(*ast);

    // Postfix operators should be converted to prefix
    assert(result.find("*p") != std::string::npos);
    assert(result.find("&x") != std::string::npos);

    std::cout << "Postfix operator tests passed!" << std::endl;
}

void test_contracts() {
    std::cout << "Testing contracts..." << std::endl;

    std::string source = R"(
        func divide(a: i32, b: i32) -> i32
            pre: b != 0
            post: result != 0
        = a / b;
    )";

    Lexer lexer(source);
    auto tokens = lexer.tokenize();

    Parser parser(tokens);
    auto ast = parser.parse();

    SemanticAnalyzer semantic_analyzer;
    semantic_analyzer.analyze(*ast);

    ContractProcessor contract_processor;
    contract_processor.process(*ast);

    CodeGenerator code_generator;
    auto result = code_generator.generate(*ast);

    // Contract checks should be generated
    assert(result.find("assert(b != 0)") != std::string::npos);

    std::cout << "Contract tests passed!" << std::endl;
}

void test_safety_checks() {
    std::cout << "Testing safety checks..." << std::endl;

    std::string source = R"(
        func test(arr: [i32], index: i32) -> i32 {
            return arr[index];
        }
    )";

    Lexer lexer(source);
    auto tokens = lexer.tokenize();

    Parser parser(tokens);
    auto ast = parser.parse();

    SemanticAnalyzer semantic_analyzer;
    semantic_analyzer.analyze(*ast);

    SafetyChecker safety_checker;
    safety_checker.check(*ast);

    CodeGenerator code_generator;
    auto result = code_generator.generate(*ast);

    // Bounds checking should be added
    assert(result.find("index >= 0") != std::string::npos ||
           result.find("static_cast<size_t>(index)") != std::string::npos);

    std::cout << "Safety check tests passed!" << std::endl;
}

void test_string_interpolation() {
    std::cout << "Testing string interpolation..." << std::endl;

    std::string source = R"(
        func test() {
            let name = "world";
            let message = "Hello $(name)!";
        }
    )";

    Lexer lexer(source);
    auto tokens = lexer.tokenize();

    Parser parser(tokens);
    auto ast = parser.parse();

    SemanticAnalyzer semantic_analyzer;
    semantic_analyzer.analyze(*ast);

    CodeGenerator code_generator;
    auto result = code_generator.generate(*ast);

    // String interpolation should be converted to format
    assert(result.find("std::format") != std::string::npos ||
           result.find("fmt::format") != std::string::npos);

    std::cout << "String interpolation tests passed!" << std::endl;
}

void test_range_operators() {
    std::cout << "Testing range operators..." << std::endl;

    std::string source = R"(
        func test() {
            for i in 0..<10 {
                // loop body
            }
            for i in 0..=10 {
                // loop body
            }
        }
    )";

    Lexer lexer(source);
    auto tokens = lexer.tokenize();

    Parser parser(tokens);
    auto ast = parser.parse();

    SemanticAnalyzer semantic_analyzer;
    semantic_analyzer.analyze(*ast);

    CodeGenerator code_generator;
    auto result = code_generator.generate(*ast);

    // Range operators should be converted
    assert(result.find("std::views::iota") != std::string::npos ||
           result.find("range") != std::string::npos);

    std::cout << "Range operator tests passed!" << std::endl;
}

void test_inspect_pattern_matching() {
    std::cout << "Testing inspect pattern matching..." << std::endl;

    std::string source = R"(
        func test(value: i32) -> i32 {
            inspect value {
                0 => 1,
                1 => 2,
                _ => 0,
            }
        }
    )";

    Lexer lexer(source);
    auto tokens = lexer.tokenize();

    Parser parser(tokens);
    auto ast = parser.parse();

    SemanticAnalyzer semantic_analyzer;
    semantic_analyzer.analyze(*ast);

    CodeGenerator code_generator;
    auto result = code_generator.generate(*ast);

    // Pattern matching should be converted to switch/if
    assert(result.find("switch") != std::string::npos ||
           result.find("if") != std::string::npos);

    std::cout << "Pattern matching tests passed!" << std::endl;
}

// Runner used by combined test harness
int test_main() {
    try {
        test_lexer();
        test_parser();
        test_simple_transpilation();
        test_type_deduction();
        test_ufcs();
        test_postfix_operators();
        test_contracts();
        test_safety_checks();
        test_string_interpolation();
        test_range_operators();
        test_inspect_pattern_matching();
        return 0;
    } catch (...) {
        return 1;
    }
}

void test_metafunctions() {
    std::cout << "Testing metafunctions..." << std::endl;

    std::string source = R"(
        @value
        @ordered
        @copyable
        type Point = {
            x: i32,
            y: i32,
        };
    )";

    Lexer lexer(source);
    auto tokens = lexer.tokenize();

    Parser parser(tokens);
    auto ast = parser.parse();

    SemanticAnalyzer semantic_analyzer;
    semantic_analyzer.analyze(*ast);

    MetafunctionProcessor meta_processor;
    meta_processor.process(*ast);

    CodeGenerator code_generator;
    auto result = code_generator.generate(*ast);

    // Metafunctions should generate additional members
    assert(result.find("operator==") != std::string::npos ||
           result.find("operator<") != std::string::npos);

    std::cout << "Metafunction tests passed!" << std::endl;
}

void test_templates() {
    std::cout << "Testing templates..." << std::endl;

    std::string source = R"(
        func max<T>(a: T, b: T) -> T = if a > b then a else b;
    )";

    Lexer lexer(source);
    auto tokens = lexer.tokenize();

    Parser parser(tokens);
    auto ast = parser.parse();

    SemanticAnalyzer semantic_analyzer;
    semantic_analyzer.analyze(*ast);

    CodeGenerator code_generator;
    auto result = code_generator.generate(*ast);

    // Template syntax should be converted
    assert(result.find("template<typename T>") != std::string::npos);

    std::cout << "Template tests passed!" << std::endl;
}

void test_integration() {
    std::cout << "Running integration tests..." << std::endl;

    // Create test files
    std::ofstream input_file("test_input.cpp2");
    input_file << R"(
        import std;

        @value
        @ordered
        type Vec2 = {
            x: f32,
            y: f32,
        };

        func dot(v1: Vec2, v2: Vec2) -> f32
            pre: v1.x != 0.0 || v1.y != 0.0
            post: result >= 0.0
        = v1.x * v2.x + v1.y * v2.y;

        func main() -> i32 {
            let v1 = Vec2{ x = 1.0, y = 2.0 };
            let v2 = Vec2{ x = 3.0, y = 4.0 };
            let result = v1.dot(v2);

            if result > 10.0 {
                return 1;
            }

            return 0;
        }
    )";
    input_file.close();

    // Test transpilation
    Lexer lexer(read_file("test_input.cpp2"));
    auto tokens = lexer.tokenize();

    Parser parser(tokens);
    auto ast = parser.parse();

    SemanticAnalyzer semantic_analyzer;
    semantic_analyzer.analyze(*ast);

    SafetyChecker safety_checker;
    safety_checker.check(*ast);

    MetafunctionProcessor meta_processor;
    meta_processor.process(*ast);

    ContractProcessor contract_processor;
    contract_processor.process(*ast);

    CodeGenerator code_generator;
    auto result = code_generator.generate(*ast);

    // Basic validation
    assert(!result.empty());
    assert(result.find("struct Vec2") != std::string::npos);
    assert(result.find("float dot") != std::string::npos);
    assert(result.find("int main") != std::string::npos);

    // Clean up
    std::remove("test_input.cpp2");

    std::cout << "Integration tests passed!" << std::endl;
}

 #ifndef COMBINED_TESTS
int main() {
    try {
        std::cout << "Running Cpp2 Transpiler Tests\n" << std::endl;

        test_lexer();
        test_parser();
        test_simple_transpilation();
        test_type_deduction();
        test_ufcs();
        test_postfix_operators();
        test_contracts();
        test_safety_checks();
        test_string_interpolation();
        test_range_operators();
        test_inspect_pattern_matching();
        test_metafunctions();
        test_templates();
        test_integration();

        std::cout << "\nAll tests passed! ✓" << std::endl;
        return 0;
    }
    catch (const std::exception& e) {
        std::cerr << "Test failed: " << e.what() << std::endl;
        return 1;
    }
}
#endif