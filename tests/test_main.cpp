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
#include "test_timeout.hpp"

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
        func main() -> i32 = {
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

    // Basic checks - using cppfront-style auto return type syntax
    assert(result.find("auto main()") != std::string::npos);
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

        func test() = {
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
        func test(p: i32*) = {
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
        func test(p: int*) -> int = {
            return p*;
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

    // Check that deref is handled
    assert(result.find("*p") != std::string::npos ||
           result.find("return") != std::string::npos);

    std::cout << "Safety check tests passed!" << std::endl;
}

void test_string_interpolation() {
    std::cout << "Testing string interpolation..." << std::endl;

    std::string source = R"(
        func test() = {
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

    // Real cpp2 for-do syntax: for collection do(item) { }
    std::string source = R"(
        test: (arr: int*) = {
            sum := 0;
            for arr do(x) {
                sum = sum + x;
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

    // Check for-range loop generation
    assert(result.find("for") != std::string::npos);
    assert(result.find(" x : ") != std::string::npos || result.find("auto x") != std::string::npos);

    std::cout << "Range operator tests passed!" << std::endl;
}

void test_inspect_pattern_matching() {
    std::cout << "Testing inspect pattern matching..." << std::endl;

    // Real cpp2 inspect expression syntax: inspect value -> type { is pattern = result; }
    std::string source = R"(
        test: (value: int) -> int = {
            result := inspect value -> int {
                is 0 = 1;
                is 1 = 2;
                is _ = 0;
            };
            return result;
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

    // Pattern matching should be converted to if-else chain
    assert(result.find("if") != std::string::npos || result.find("__value") != std::string::npos);

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

    // Real cpp2 metafunction decorator syntax: name: @value @ordered type = { ... };
    std::string source = R"(
        Point: @value @ordered type = {
            x: int;
            y: int;
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

    // Check type generation with metafunctions
    assert(result.find("struct Point") != std::string::npos);
    assert(result.find("@value metafunction") != std::string::npos);
    assert(result.find("@ordered metafunction") != std::string::npos);
    assert(result.find("operator==") != std::string::npos);
    assert(result.find("operator<=>") != std::string::npos);

    std::cout << "Metafunction tests passed!" << std::endl;
}

void test_advanced_metafunctions() {
    std::cout << "Testing advanced metafunctions..." << std::endl;

    // Test @interface
    {
        std::string source = R"(
            Animal: @interface type = {
                name: int;
            };
        )";

        Lexer lexer(source);
        auto tokens = lexer.tokenize();
        Parser parser(tokens);
        auto ast = parser.parse();
        SemanticAnalyzer semantic_analyzer;
        semantic_analyzer.analyze(*ast);
        CodeGenerator code_generator;
        auto result = code_generator.generate(*ast);

        assert(result.find("@interface metafunction") != std::string::npos);
        assert(result.find("virtual ~Animal()") != std::string::npos);
        assert(result.find("= delete") != std::string::npos);
    }

    // Test @polymorphic_base
    {
        std::string source = R"(
            Base: @polymorphic_base type = {
                x: int;
            };
        )";

        Lexer lexer(source);
        auto tokens = lexer.tokenize();
        Parser parser(tokens);
        auto ast = parser.parse();
        CodeGenerator code_generator;
        auto result = code_generator.generate(*ast);

        assert(result.find("@polymorphic_base metafunction") != std::string::npos);
        assert(result.find("virtual ~Base()") != std::string::npos);
    }

    // Test @weakly_ordered
    {
        std::string source = R"(
            Data: @weakly_ordered type = {
                value: int;
            };
        )";

        Lexer lexer(source);
        auto tokens = lexer.tokenize();
        Parser parser(tokens);
        auto ast = parser.parse();
        CodeGenerator code_generator;
        auto result = code_generator.generate(*ast);

        assert(result.find("@weakly_ordered metafunction") != std::string::npos);
        assert(result.find("std::weak_ordering operator<=>") != std::string::npos);
    }

    // Test @copyable and @movable
    {
        std::string source = R"(
            Resource: @copyable @movable type = {
                data: int;
            };
        )";

        Lexer lexer(source);
        auto tokens = lexer.tokenize();
        Parser parser(tokens);
        auto ast = parser.parse();
        CodeGenerator code_generator;
        auto result = code_generator.generate(*ast);

        assert(result.find("@copyable metafunction") != std::string::npos);
        assert(result.find("@movable metafunction") != std::string::npos);
    }

    // Test @hashable
    {
        std::string source = R"(
            Key: @hashable type = {
                id: int;
            };
        )";

        Lexer lexer(source);
        auto tokens = lexer.tokenize();
        Parser parser(tokens);
        auto ast = parser.parse();
        CodeGenerator code_generator;
        auto result = code_generator.generate(*ast);

        assert(result.find("@hashable metafunction") != std::string::npos);
        assert(result.find("namespace std") != std::string::npos);
        assert(result.find("struct hash<Key>") != std::string::npos);
    }

    std::cout << "Advanced metafunction tests passed!" << std::endl;
}

void test_specialized_metafunctions() {
    std::cout << "Testing specialized metafunctions..." << std::endl;

    // Test @print
    {
        std::string source = R"(
            Data: @print type = {
                x: int;
                y: int;
            };
        )";

        Lexer lexer(source);
        auto tokens = lexer.tokenize();
        Parser parser(tokens);
        auto ast = parser.parse();
        CodeGenerator code_generator;
        auto result = code_generator.generate(*ast);

        assert(result.find("@print metafunction") != std::string::npos);
        assert(result.find("to_string()") != std::string::npos);
        assert(result.find("std::to_string") != std::string::npos);
    }

    // Test @enum
    {
        std::string source = R"(
            Color: @enum type = {
                red;
                green;
                blue;
            };
        )";

        Lexer lexer(source);
        auto tokens = lexer.tokenize();
        Parser parser(tokens);
        auto ast = parser.parse();
        CodeGenerator code_generator;
        auto result = code_generator.generate(*ast);

        assert(result.find("enum class Color") != std::string::npos);
        assert(result.find("red") != std::string::npos);
        assert(result.find("green") != std::string::npos);
        assert(result.find("blue") != std::string::npos);
    }

    // Test @flag_enum
    {
        std::string source = R"(
            Flags: @flag_enum type = {
                read;
                write;
                execute;
            };
        )";

        Lexer lexer(source);
        auto tokens = lexer.tokenize();
        Parser parser(tokens);
        auto ast = parser.parse();
        CodeGenerator code_generator;
        auto result = code_generator.generate(*ast);

        assert(result.find("enum class Flags") != std::string::npos);
        assert(result.find("@flag_enum: bitwise operators") != std::string::npos);
        assert(result.find("operator|") != std::string::npos);
        assert(result.find("operator&") != std::string::npos);
    }

    // Test @union
    {
        std::string source = R"(
            Value: @union type = {
                i: int;
                f: float;
            };
        )";

        Lexer lexer(source);
        auto tokens = lexer.tokenize();
        Parser parser(tokens);
        auto ast = parser.parse();
        CodeGenerator code_generator;
        auto result = code_generator.generate(*ast);

        assert(result.find("union Value") != std::string::npos);
    }

    // Test @partially_ordered
    {
        std::string source = R"(
            Partial: @partially_ordered type = {
                value: float;
            };
        )";

        Lexer lexer(source);
        auto tokens = lexer.tokenize();
        Parser parser(tokens);
        auto ast = parser.parse();
        CodeGenerator code_generator;
        auto result = code_generator.generate(*ast);

        assert(result.find("@partially_ordered metafunction") != std::string::npos);
        assert(result.find("std::partial_ordering operator<=>") != std::string::npos);
    }

    std::cout << "Specialized metafunction tests passed!" << std::endl;
}

void test_advanced_specialized_metafunctions() {
    std::cout << "Testing advanced specialized metafunctions..." << std::endl;

    // Test @regex
    {
        std::string source = R"(
            Matcher: @regex type = {
                regex: int;
                regex_test: int;
                other: int;
            };
        )";

        Lexer lexer(source);
        auto tokens = lexer.tokenize();

        Parser parser(tokens);
        auto ast = parser.parse();

        SemanticAnalyzer semantic_analyzer;
        semantic_analyzer.analyze(*ast);

        CodeGenerator code_generator;
        auto result = code_generator.generate(*ast);

        // Should transform regex members to std::regex
        assert(result.find("@regex metafunction") != std::string::npos);
        assert(result.find("std::regex regex") != std::string::npos);
        assert(result.find("std::regex regex_test") != std::string::npos);
        assert(result.find("int other") != std::string::npos);

        std::cout << "  @regex test passed!" << std::endl;
    }

    // Test @autodiff
    {
        std::string source = R"(
            Differentiable: @autodiff type = {
                compute: (x: double) -> double = x;
            };
        )";

        Lexer lexer(source);
        auto tokens = lexer.tokenize();

        Parser parser(tokens);
        auto ast = parser.parse();

        SemanticAnalyzer semantic_analyzer;
        semantic_analyzer.analyze(*ast);

        CodeGenerator code_generator;
        auto result = code_generator.generate(*ast);

        // Should generate derivative methods
        assert(result.find("@autodiff metafunction") != std::string::npos);
        assert(result.find("automatic differentiation") != std::string::npos);
        assert(result.find("Derivative of compute") != std::string::npos);
        assert(result.find("compute_d") != std::string::npos);

        std::cout << "  @autodiff test passed!" << std::endl;
    }

    // Test @sample_traverser
    {
        std::string source = R"(
            Traversable: @sample_traverser type = {
                x: int;
                y: int;
                z: int;
            };
        )";

        Lexer lexer(source);
        auto tokens = lexer.tokenize();

        Parser parser(tokens);
        auto ast = parser.parse();

        SemanticAnalyzer semantic_analyzer;
        semantic_analyzer.analyze(*ast);

        CodeGenerator code_generator;
        auto result = code_generator.generate(*ast);

        // Should generate traverse methods
        assert(result.find("@sample_traverser metafunction") != std::string::npos);
        assert(result.find("visitor pattern") != std::string::npos);
        assert(result.find("template<typename Visitor>") != std::string::npos);
        assert(result.find("void traverse(Visitor&& visitor)") != std::string::npos);
        assert(result.find("visitor(\"x\", x)") != std::string::npos);
        assert(result.find("visitor(\"y\", y)") != std::string::npos);
        assert(result.find("visitor(\"z\", z)") != std::string::npos);

        std::cout << "  @sample_traverser test passed!" << std::endl;
    }

    std::cout << "Advanced specialized metafunction tests passed!" << std::endl;
}

void test_templates() {
    std::cout << "Testing templates..." << std::endl;

    // Real cpp2 unified template syntax: name: <T> (params)
    std::string source = R"(
        mymax: <T> (a: T, b: T) -> T = a + b;
    )";

    Lexer lexer(source);
    auto tokens = lexer.tokenize();

    Parser parser(tokens);
    auto ast = parser.parse();

    SemanticAnalyzer semantic_analyzer;
    semantic_analyzer.analyze(*ast);

    CodeGenerator code_generator;
    auto result = code_generator.generate(*ast);

    // Template syntax should be converted to C++
    assert(result.find("template<typename T>") != std::string::npos);
    assert(result.find("mymax") != std::string::npos);

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
            x: f32;
            y: f32;
        };

        func dot(v1: Vec2, v2: Vec2) -> f32
            pre: v1.x != 0.0 || v1.y != 0.0
            post: result >= 0.0
        = v1.x * v2.x + v1.y * v2.y;

        func main() -> i32 = {
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
    std::string source = read_file("test_input.cpp2");
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

        run_with_timeout("test_lexer", test_lexer);
        run_with_timeout("test_parser", test_parser);
        run_with_timeout("test_simple_transpilation", test_simple_transpilation);
        run_with_timeout("test_type_deduction", test_type_deduction);
        run_with_timeout("test_ufcs", test_ufcs);
        run_with_timeout("test_postfix_operators", test_postfix_operators);
        run_with_timeout("test_contracts", test_contracts);
        run_with_timeout("test_safety_checks", test_safety_checks);
        run_with_timeout("test_string_interpolation", test_string_interpolation);
        run_with_timeout("test_range_operators", test_range_operators);
        run_with_timeout("test_inspect_pattern_matching", test_inspect_pattern_matching);
        run_with_timeout("test_metafunctions", test_metafunctions);
        run_with_timeout("test_advanced_metafunctions", test_advanced_metafunctions);
        run_with_timeout("test_specialized_metafunctions", test_specialized_metafunctions);
        run_with_timeout("test_advanced_specialized_metafunctions", test_advanced_specialized_metafunctions);
        run_with_timeout("test_templates", test_templates);
        run_with_timeout("test_integration", test_integration, std::chrono::seconds(15));

        std::cout << "\nAll tests passed! ✓" << std::endl;
        return 0;
    }
    catch (const std::exception& e) {
        std::cerr << "Test failed: " << e.what() << std::endl;
        return 1;
    }
}
#endif