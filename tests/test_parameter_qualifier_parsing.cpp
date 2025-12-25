#include <iostream>
#include <cassert>
#include <string>
#include <string_view>
#include <memory>

#include "../include/lexer.hpp"
#include "../include/parser.hpp"
#include "../include/ast.hpp"

using namespace cpp2_transpiler;

// Helper struct to hold AST and function pointer
struct ParseResult {
    std::unique_ptr<AST> ast;
    FunctionDeclaration* func;
};

// Helper to parse code and get the first function declaration
ParseResult parse_first_function(const std::string& code) {
    Lexer lexer(code);
    auto tokens = lexer.tokenize();

    Parser parser(tokens);
    auto ast = parser.parse();

    FunctionDeclaration* func = nullptr;

    if (!ast->declarations.empty()) {
        auto* decl = ast->declarations[0].get();
        if (decl->kind == Declaration::Kind::Function) {
            func = static_cast<FunctionDeclaration*>(decl);
        }
    }

    return {std::move(ast), func};
}

// Test 1: Parse single qualifier 'inout'
void test_parse_inout_qualifier() {
    std::cout << "Test: Parse 'inout' qualifier\n";

    std::string code = "func foo: (inout x: int) -> int { return 0; }";
    auto [ast, func] = parse_first_function(code);

    assert(func != nullptr && "Failed to parse function");
    assert(func->parameters.size() == 1 && "Expected 1 parameter");

    const auto& param = func->parameters[0];
    assert(param.name == "x" && "Parameter name should be 'x'");
    assert(param.qualifiers.size() == 1 && "Expected 1 qualifier");
    assert(param.qualifiers[0] == ParameterQualifier::InOut && "Qualifier should be InOut");

    std::cout << "  Parameter: " << param.name << "\n";
    std::cout << "  Qualifiers: " << param.qualifiers.size() << "\n";
    std::cout << "✓ Test passed\n";
}

// Test 2: Parse single qualifier 'out'
void test_parse_out_qualifier() {
    std::cout << "Test: Parse 'out' qualifier\n";

    std::string code = "func bar: (out result: int) -> int { return 0; }";
    auto [ast, func] = parse_first_function(code);

    assert(func != nullptr && "Failed to parse function");
    assert(func->parameters.size() == 1 && "Expected 1 parameter");

    const auto& param = func->parameters[0];
    assert(param.name == "result" && "Parameter name should be 'result'");
    assert(param.qualifiers.size() == 1 && "Expected 1 qualifier");
    assert(param.qualifiers[0] == ParameterQualifier::Out && "Qualifier should be Out");

    std::cout << "  Parameter: " << param.name << "\n";
    std::cout << "  Qualifiers: " << param.qualifiers.size() << "\n";
    std::cout << "✓ Test passed\n";
}

// Test 3: Parse single qualifier 'move'
void test_parse_move_qualifier() {
    std::cout << "Test: Parse 'move' qualifier\n";

    std::string code = "func baz: (move data: std::string) -> int { return 0; }";
    auto [ast, func] = parse_first_function(code);

    assert(func != nullptr && "Failed to parse function");
    assert(func->parameters.size() == 1 && "Expected 1 parameter");

    const auto& param = func->parameters[0];
    assert(param.name == "data" && "Parameter name should be 'data'");
    assert(param.qualifiers.size() == 1 && "Expected 1 qualifier");
    assert(param.qualifiers[0] == ParameterQualifier::Move && "Qualifier should be Move");

    std::cout << "  Parameter: " << param.name << "\n";
    std::cout << "  Qualifiers: " << param.qualifiers.size() << "\n";
    std::cout << "✓ Test passed\n";
}

// Test 4: Parse single qualifier 'forward'
void test_parse_forward_qualifier() {
    std::cout << "Test: Parse 'forward' qualifier\n";

    std::string code = "func qux: (forward arg: auto) -> int { return 0; }";
    auto [ast, func] = parse_first_function(code);

    assert(func != nullptr && "Failed to parse function");
    assert(func->parameters.size() == 1 && "Expected 1 parameter");

    const auto& param = func->parameters[0];
    assert(param.name == "arg" && "Parameter name should be 'arg'");
    assert(param.qualifiers.size() == 1 && "Expected 1 qualifier");
    assert(param.qualifiers[0] == ParameterQualifier::Forward && "Qualifier should be Forward");

    std::cout << "  Parameter: " << param.name << "\n";
    std::cout << "  Qualifiers: " << param.qualifiers.size() << "\n";
    std::cout << "✓ Test passed\n";
}

// Test 5: Parse multiple parameters with different qualifiers
void test_parse_multiple_qualifiers() {
    std::cout << "Test: Parse multiple parameters with different qualifiers\n";

    std::string code = "func test: (inout a: int, out b: int, move c: int, forward d: int) -> int { return 0; }";
    auto [ast, func] = parse_first_function(code);

    assert(func != nullptr && "Failed to parse function");
    assert(func->parameters.size() == 4 && "Expected 4 parameters");

    // Check parameter 1: inout a
    assert(func->parameters[0].name == "a");
    assert(func->parameters[0].qualifiers.size() == 1);
    assert(func->parameters[0].qualifiers[0] == ParameterQualifier::InOut);

    // Check parameter 2: out b
    assert(func->parameters[1].name == "b");
    assert(func->parameters[1].qualifiers.size() == 1);
    assert(func->parameters[1].qualifiers[0] == ParameterQualifier::Out);

    // Check parameter 3: move c
    assert(func->parameters[2].name == "c");
    assert(func->parameters[2].qualifiers.size() == 1);
    assert(func->parameters[2].qualifiers[0] == ParameterQualifier::Move);

    // Check parameter 4: forward d
    assert(func->parameters[3].name == "d");
    assert(func->parameters[3].qualifiers.size() == 1);
    assert(func->parameters[3].qualifiers[0] == ParameterQualifier::Forward);

    std::cout << "  Parameter 1: " << func->parameters[0].name << " with " << func->parameters[0].qualifiers.size() << " qualifier(s)\n";
    std::cout << "  Parameter 2: " << func->parameters[1].name << " with " << func->parameters[1].qualifiers.size() << " qualifier(s)\n";
    std::cout << "  Parameter 3: " << func->parameters[2].name << " with " << func->parameters[2].qualifiers.size() << " qualifier(s)\n";
    std::cout << "  Parameter 4: " << func->parameters[3].name << " with " << func->parameters[3].qualifiers.size() << " qualifier(s)\n";
    std::cout << "✓ Test passed\n";
}

// Test 6: Parse parameter without qualifiers
void test_parse_no_qualifiers() {
    std::cout << "Test: Parse parameter without qualifiers\n";

    std::string code = "func normal: (x: int, y: int) -> int { return 0; }";
    auto [ast, func] = parse_first_function(code);

    assert(func != nullptr && "Failed to parse function");
    assert(func->parameters.size() == 2 && "Expected 2 parameters");

    assert(func->parameters[0].name == "x");
    assert(func->parameters[0].qualifiers.size() == 0 && "Expected 0 qualifiers for 'x'");

    assert(func->parameters[1].name == "y");
    assert(func->parameters[1].qualifiers.size() == 0 && "Expected 0 qualifiers for 'y'");

    std::cout << "  Parameter 1: " << func->parameters[0].name << " with " << func->parameters[0].qualifiers.size() << " qualifier(s)\n";
    std::cout << "  Parameter 2: " << func->parameters[1].name << " with " << func->parameters[1].qualifiers.size() << " qualifier(s)\n";
    std::cout << "✓ Test passed\n";
}

// Test 7: Parse unified syntax with qualifiers
void test_parse_unified_syntax_qualifiers() {
    std::cout << "Test: Parse unified syntax with qualifiers\n";

    // Unified syntax: name: (params) -> return { body }
    std::string code = "myfunc: (inout x: int) -> int { return 0; };";
    auto [ast, func] = parse_first_function(code);

    assert(func != nullptr && "Failed to parse function");
    assert(func->parameters.size() == 1 && "Expected 1 parameter");

    const auto& param = func->parameters[0];
    assert(param.name == "x" && "Parameter name should be 'x'");
    assert(param.qualifiers.size() == 1 && "Expected 1 qualifier");
    assert(param.qualifiers[0] == ParameterQualifier::InOut && "Qualifier should be InOut");

    std::cout << "  Parameter: " << param.name << "\n";
    std::cout << "  Qualifiers: " << param.qualifiers.size() << "\n";
    std::cout << "✓ Test passed\n";
}

int main() {
    std::cout << "=== Parameter Qualifier Parsing Tests ===\n\n";

    test_parse_inout_qualifier();
    test_parse_out_qualifier();
    test_parse_move_qualifier();
    test_parse_forward_qualifier();
    test_parse_multiple_qualifiers();
    test_parse_no_qualifiers();
    test_parse_unified_syntax_qualifiers();

    std::cout << "\n=== All tests passed ===\n";
    return 0;
}
